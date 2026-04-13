"""
server.py
FastAPI 後端 — Polymarket Weather Arbitrage Dashboard
啟動: python server.py
訪問: http://localhost:8000
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# 內部模組
from fetcher_polymarket import fetch_all_weather_markets, MOCK_WEATHER_MARKETS
from parser_market      import parse_all_markets
from comparator         import compare_all_markets, EDGE_THRESHOLD
from consensus          import get_consensus
from latency_tracker    import get_latency_summary
from history            import (log_prediction, get_brier_score, get_recent_predictions,
                                init_db, get_resolved_split, get_all_predictions,
                                auto_resolve_past_markets)
from fetcher_weather    import resolve_location

app = FastAPI(title="Weather Arb Dashboard", version="2.0")

# ── 靜態檔案 ─────────────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── In-memory 快取（30 分鐘，對齊 GFS 更新週期）────────────────────────────
_cache: dict = {}
_cache_ts:   float = 0.0
CACHE_TTL   = 30 * 60   # 30 minutes
_scan_lock  = threading.Lock()  # prevent concurrent scans


def _is_cache_valid() -> bool:
    return (time.time() - _cache_ts) < CACHE_TTL


def _run_scan() -> list[dict]:
    """完整掃描：fetch → parse → compare → consensus → log。"""
    today = date.today()

    # 1. 抓市場 (mock only if no live markets found)
    markets = fetch_all_weather_markets(use_mock_fallback=True)

    # 1b. Pre-filter: keep only actionable markets to limit API calls.
    live = [m for m in markets if m.get("source") != "mock"]
    mock = [m for m in markets if m.get("source") == "mock"]
    if live:
        # Filter to competitive prices (3–97%)
        live = [m for m in live if 0.03 <= (m.get("market_price_yes") or 0) <= 0.97]
        # For temp-bucket events: keep all buckets per city+date so we find the
        # best one — but cap total at 200 to stay within ~30s scan budget.
        # Sort by volume so we prioritise the most liquid markets.
        live.sort(key=lambda m: m.get("volume_24h") or m.get("liquidity") or 0, reverse=True)
        # Prefer markets ending tomorrow (today+1) as primary betting window.
        # Include today if still early, and day-after-tomorrow if budget allows.
        from datetime import timedelta
        tomorrow = (today + timedelta(days=1)).isoformat()
        day2     = (today + timedelta(days=2)).isoformat()
        today_s  = today.isoformat()

        def _end_day(m):
            return (m.get("end_date") or "")[:10]

        tomorrow_m = [m for m in live if _end_day(m) == tomorrow]
        today_m    = [m for m in live if _end_day(m) == today_s]
        day2_m     = [m for m in live if _end_day(m) == day2]
        rest       = [m for m in live if _end_day(m) not in (today_s, tomorrow, day2)]

        # Assemble: tomorrow first (best betting window), then today, day2, rest
        live = (tomorrow_m + today_m + day2_m + rest)[:200]
        markets = live
        print(f"[Scan] Pre-filtered to {len(markets)} competitive live markets")
    else:
        markets = mock

    # 2. 解析
    markets = parse_all_markets(markets, reference_date=today)

    # 2b. Pre-warm weather cache: collect unique (location, date) pairs and
    #     fetch them with controlled concurrency (6 workers) before the main
    #     compare step. The cross-run cache means subsequent scans reuse results.
    from comparator import _get_weather_cached

    unique_loc_dates: set[tuple] = set()
    for m in markets:
        p = m.get("parsed", {})
        loc = m.get("location_hint") or p.get("location")
        d_str = p.get("target_date")
        if loc and d_str:
            try:
                unique_loc_dates.add((loc, date.fromisoformat(d_str)))
            except ValueError:
                pass

    print(f"[Scan] Pre-fetching weather for {len(unique_loc_dates)} unique city+date combos...")
    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(lambda ld: _get_weather_cached(ld[0], ld[1]), unique_loc_dates))
    print(f"[Scan] Weather pre-fetch done.")

    # 3. 基本比較（Open-Meteo Ensemble — uses cached weather）
    results = compare_all_markets(markets)

    # 4. 三模型共識（並行處理，只對有機會/有 edge 的市場）
    def _enrich_one(r: dict) -> dict:
        if r.get("is_opportunity") or r.get("edge") is not None:
            parsed   = r.get("parsed", {})
            location = parsed.get("location", "")
            coords   = resolve_location(location)
            if coords:
                try:
                    target_str = parsed.get("target_date")
                    target_date = date.fromisoformat(target_str) if target_str else today
                    cons = get_consensus(
                        lat=coords[0], lon=coords[1],
                        target_date=target_date,
                        event_type=parsed.get("event_type", "rain"),
                        threshold=parsed.get("threshold"),
                        direction=parsed.get("direction", "any"),
                    )
                    r["consensus"]      = cons
                    r["consensus_prob"] = cons.get("consensus")
                    r["conviction"]     = cons.get("conviction")
                    r["models_agree"]   = cons.get("models_agree", 0)

                    # Kelly 建議注額：edge × conviction_factor × max $10
                    edge = abs(r.get("edge") or 0)
                    conv_factor = {"high": 1.0, "medium": 0.6, "low": 0.3}.get(
                        cons.get("conviction", "low"), 0.3
                    )
                    kelly_size = round(edge * conv_factor * 10, 2)
                    r["kelly_bet"] = min(kelly_size, 10.0)
                except Exception:
                    r["consensus"] = None
            try:
                log_prediction(r)
            except Exception:
                pass
        return r

    with ThreadPoolExecutor(max_workers=8) as executor:
        enriched = list(executor.map(_enrich_one, results))

    return enriched


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/opportunities")
def api_opportunities(refresh: bool = False):
    global _cache, _cache_ts
    if not refresh and _is_cache_valid() and _cache:
        return JSONResponse({**_cache, "cached": True})

    # Only one scan at a time — if already scanning, wait and return result
    if not _scan_lock.acquire(timeout=120):
        if _cache:
            return JSONResponse({**_cache, "cached": True, "note": "scan_in_progress"})
        return JSONResponse({"error": "scan timeout", "opportunities": [], "no_edge": [], "skipped": []}, status_code=503)

    try:
        # Double-check cache after acquiring lock (another thread may have just scanned)
        if not refresh and _is_cache_valid() and _cache:
            return JSONResponse({**_cache, "cached": True})

        results = _run_scan()
        scanned_at = datetime.now(timezone.utc).isoformat()

        raw_opps = [r for r in results if r.get("is_opportunity")]
        no_edge  = [r for r in results if not r.get("is_opportunity") and not r.get("skip_reason")]
        skipped  = [r for r in results if r.get("skip_reason")]

        # ── Sort opportunities: best → worst ────────────────────────────────
        # Score = abs_edge × conviction_weight × direction_bonus
        # BUY YES > BUY NO (buying is more natural, shorting needs margin)
        # high conviction > medium > low
        _conv_w = {"high": 1.0, "medium": 0.65, "low": 0.35}
        _dir_w  = {"BUY YES": 1.0, "BUY NO": 0.85, "HOLD": 0.0}

        def _score(r: dict) -> float:
            edge    = abs(r.get("abs_edge") or r.get("edge") or 0)
            conv    = _conv_w.get(r.get("conviction") or "low", 0.35)
            dirw    = _dir_w.get(r.get("action") or "HOLD", 0.85)
            return edge * conv * dirw

        opportunities = sorted(raw_opps, key=_score, reverse=True)

        # no_edge sorted by abs_edge descending so closest-to-opportunity first
        no_edge = sorted(no_edge, key=lambda r: abs(r.get("edge") or 0), reverse=True)

        # 序列化：移除不可 JSON 的欄位
        def _clean(r: dict) -> dict:
            out = {}
            for k, v in r.items():
                if k == "consensus" and isinstance(v, dict):
                    out[k] = v
                elif isinstance(v, (str, int, float, bool, type(None))):
                    out[k] = v
                elif isinstance(v, dict):
                    out[k] = {kk: vv for kk, vv in v.items()
                              if isinstance(vv, (str, int, float, bool, type(None)))}
            return out

        payload = {
            "scanned_at":    scanned_at,
            "edge_threshold": EDGE_THRESHOLD,
            "opportunities": [_clean(r) for r in opportunities],
            "no_edge":        [_clean(r) for r in no_edge],
            "skipped":        [_clean(r) for r in skipped],
            "summary": {
                "total":         len(results),
                "opportunities": len(opportunities),
                "no_edge":       len(no_edge),
                "skipped":       len(skipped),
            },
            "cached": False,
        }
        _cache    = payload
        _cache_ts = time.time()
        return JSONResponse(payload)

    except Exception as e:
        return JSONResponse({"error": str(e), "opportunities": [], "no_edge": [], "skipped": []}, status_code=500)
    finally:
        _scan_lock.release()


@app.get("/api/latency")
def api_latency():
    try:
        return JSONResponse(get_latency_summary())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/history")
def api_history(days: int = 30):
    try:
        init_db()
        brier = get_brier_score(days=days)
        recent = get_recent_predictions(limit=20)
        return JSONResponse({
            "brier_score": brier,
            "recent":      recent,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/history/resolved")
def api_history_resolved(days: int = 90):
    try:
        init_db()
        # Try to auto-resolve any past markets using historical weather data
        try:
            newly_resolved = auto_resolve_past_markets()
            if newly_resolved:
                print(f"[History] Auto-resolved {newly_resolved} past market(s)")
        except Exception:
            pass
        data = get_all_predictions(days=days, limit=300)
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/refresh")
def api_refresh():
    """強制清除快取並重新掃描。"""
    global _cache_ts
    _cache_ts = 0.0
    return api_opportunities(refresh=True)


@app.on_event("startup")
def _warm_cache():
    """Pre-warm the cache in a background thread so first page load is instant."""
    import threading
    def _bg():
        try:
            # Small delay so server is fully up before scan starts
            time.sleep(2)
            api_opportunities(refresh=False)
        except Exception:
            pass
    threading.Thread(target=_bg, daemon=True).start()


if __name__ == "__main__":
    import sys, os
    sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
    port = int(os.environ.get("PORT", 8000))
    print("Weather Arb Dashboard")
    print(f"   http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")

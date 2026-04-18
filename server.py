"""
server.py
FastAPI 後端 — Polymarket Weather Arbitrage Dashboard
啟動: python server.py
訪問: http://localhost:8000
"""

import asyncio
import time
from contextlib import asynccontextmanager
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

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """
    FastAPI lifespan: init aiohttp session, start background asyncio tasks,
    yield to app, then cancel tasks + close session on shutdown.
    """
    from _http import init_session, close_session
    await init_session()

    async def _bg_warm():
        await asyncio.sleep(2)
        try:
            await _run_scan_task()
        except Exception:
            pass

    async def _bg_resolver():
        await asyncio.sleep(30)
        while True:
            try:
                await asyncio.to_thread(init_db)
                n = await asyncio.to_thread(auto_resolve_past_markets)
                if n:
                    print(f"[Scheduler] Auto-resolved {n} settled market(s)")
            except Exception as e:
                print(f"[Scheduler] Resolution error: {e}")
            await asyncio.sleep(10 * 60)

    async def _bg_scanner():
        await asyncio.sleep(60)
        while True:
            try:
                await _run_scan_task()
            except Exception as e:
                print(f"[Scheduler] Auto-scan error: {e}")
            await asyncio.sleep(3 * 60)

    bg_tasks = [
        asyncio.create_task(_bg_warm(),     name="bg-warm"),
        asyncio.create_task(_bg_resolver(), name="bg-resolver"),
        asyncio.create_task(_bg_scanner(),  name="bg-scanner"),
    ]
    print("[Scheduler] Background asyncio tasks started")

    yield   # ← app runs here

    for t in bg_tasks:
        t.cancel()
    await asyncio.gather(*bg_tasks, return_exceptions=True)
    await close_session()
    print("[Shutdown] aiohttp session closed")


app = FastAPI(title="Weather Arb Dashboard", version="2.0", lifespan=_lifespan)

# ── 靜態檔案 ─────────────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── In-memory 快取（30 分鐘，對齊 GFS 更新週期）────────────────────────────
_cache: dict = {}
_cache_ts:   float = 0.0
CACHE_TTL   = 30 * 60   # 30 minutes
_scan_lock  = asyncio.Lock()    # prevent concurrent scans (asyncio-safe)
_scan_in_progress: bool = False  # true while a scan task is running


def _is_cache_valid() -> bool:
    return (time.time() - _cache_ts) < CACHE_TTL


async def _run_scan() -> list[dict]:
    """完整掃描：fetch → parse → compare → consensus → log。"""
    today = date.today()

    # 1. 抓市場 (mock only if no live markets found)
    markets = await fetch_all_weather_markets(use_mock_fallback=True)

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
        live = (tomorrow_m + today_m + day2_m + rest)[:400]
        markets = live
        print(f"[Scan] Pre-filtered to {len(markets)} competitive live markets")
    else:
        markets = mock

    # 2. 解析
    markets = parse_all_markets(markets, reference_date=today)

    # 3. 基本比較（Open-Meteo Ensemble — uses cached weather）
    results = await compare_all_markets(markets)

    # 4. 三模型共識（並行處理，只對有機會/有 edge 的市場）
    def _enrich_one(r: dict) -> dict:
        if r.get("is_opportunity"):
            parsed   = r.get("parsed", {})
            location = parsed.get("location", "")
            coords   = resolve_location(location)
            if coords:
                try:
                    target_str = parsed.get("target_date")
                    target_date = date.fromisoformat(target_str) if target_str else today
                    # For temperature_bucket markets, pass bucket bounds directly to
                    # consensus so it can compute P(lo ≤ max_temp ≤ hi) properly.
                    # The parser only extracts a single threshold value (e.g. the first
                    # number in "62-63°F"), so we must override with the actual bucket.
                    event_type_cons = parsed.get("event_type", "rain")
                    threshold_cons  = parsed.get("threshold")
                    temp_bucket = r.get("temp_bucket")
                    if temp_bucket and r.get("market_subtype") == "temperature_bucket":
                        event_type_cons = "temperature_bucket"
                        threshold_cons  = {
                            "lo_c": temp_bucket.get("lo_c"),
                            "hi_c": temp_bucket.get("hi_c"),
                        }

                    cons = get_consensus(
                        lat=coords[0], lon=coords[1],
                        target_date=target_date,
                        event_type=event_type_cons,
                        threshold=threshold_cons,
                        direction=parsed.get("direction", "any"),
                    )
                    r["consensus"]      = cons
                    r["consensus_prob"] = cons.get("consensus")
                    r["conviction"]     = cons.get("conviction")
                    r["models_agree"]   = cons.get("models_agree", 0)

                    # ── Bleed-stop patch (2026-04-18) ─────────────────────────
                    # Brier telemetry shows nowcast-adjusted model_probability
                    # (Kalman + METAR + WU override layer in comparator) scores
                    # WORSE than the pure 5-model consensus:
                    #   2-day: model_brier=0.368 vs consensus_brier=0.237
                    # The nowcast adjustment is actively degrading calibration,
                    # so override model_probability with consensus_prob and
                    # re-derive edge/action. Preserves wu_definitive overrides
                    # from comparator (those are physical certainties, not
                    # Kalman guesses) by only replacing values in the normal
                    # range — not the 0.01 death signal.
                    cp = cons.get("consensus")
                    mp_old = r.get("model_probability")
                    mkt_px = r.get("market_price_yes")
                    # Default source tag = "nowcast" (comparator path) unless
                    # upstream already tagged wu_definitive.
                    if not r.get("model_prob_source"):
                        r["model_prob_source"] = "nowcast"
                    if (cp is not None and mp_old is not None
                            and r.get("model_prob_source") != "wu_definitive"
                            and mkt_px is not None):
                        try:
                            mkt_px_f = float(mkt_px)
                            r["model_probability"] = cp
                            raw_edge = cp - mkt_px_f
                            r["edge"] = round(raw_edge, 4)
                            r["abs_edge"] = abs(round(raw_edge, 4))
                            if abs(raw_edge) >= 0.08:
                                r["action"] = "BUY YES" if raw_edge > 0 else "BUY NO"
                                r["is_opportunity"] = True
                            else:
                                r["action"] = "HOLD"
                                r["is_opportunity"] = False
                            r["model_prob_source"] = "consensus_override"
                        except (ValueError, TypeError):
                            pass

                    # Kelly 建議注額：使用已扣除 Spread 的 exec_edge（在 comparator 計算）
                    # exec_edge = paper_edge - bid/ask half-spread (dynamic, by price level)
                    exec_edge = abs(r.get("exec_edge") or r.get("edge") or 0)
                    conv_factor = {"high": 1.0, "medium": 0.6, "low": 0.3}.get(
                        cons.get("conviction", "low"), 0.3
                    )
                    kelly_size = round(exec_edge * conv_factor * 10, 2)
                    r["kelly_bet"] = min(kelly_size, 10.0)
                except Exception:
                    r["consensus"] = None
            try:
                log_prediction(r)
            except Exception:
                pass
        return r

    # Consensus enrichment only for top-10 opportunities (urllib sync calls are slow).
    # Non-opportunity markets get passed through unchanged.
    opps_sorted = sorted(
        [r for r in results if r.get("is_opportunity")],
        key=lambda r: abs(r.get("exec_edge") or r.get("edge") or 0),
        reverse=True,
    )
    top_opps  = opps_sorted[:10]
    rest_opps = opps_sorted[10:]
    non_opps  = [r for r in results if not r.get("is_opportunity")]

    enriched_top = list(await asyncio.gather(*[
        asyncio.to_thread(_enrich_one, r) for r in top_opps
    ]))
    enriched = enriched_top + rest_opps + non_opps

    # ── Portfolio Kelly: cap total exposure for mutually exclusive buckets ───────
    # Temperature buckets for the same (city, date) are mutually exclusive events
    # (exactly one bucket resolves YES).  Sizing each independently with Kelly
    # overstates total capital at risk by up to N× (one per bucket).
    # Fix: group by (city, date), sum raw Kelly bets, scale proportionally if
    # total exceeds the per-group cap.
    MAX_GROUP_KELLY = 10.0   # max total $-exposure per (city, date) group
    from collections import defaultdict as _defaultdict
    bucket_groups: dict = _defaultdict(list)
    for r in enriched:
        if r.get("is_opportunity") and r.get("market_subtype") == "temperature_bucket":
            parsed  = r.get("parsed", {})
            city    = (parsed.get("location") or "").strip().lower()
            date_s  = (parsed.get("target_date") or "")[:10]
            if city and date_s:
                bucket_groups[(city, date_s)].append(r)

    for (city, date_s), group in bucket_groups.items():
        total_kelly = sum(r.get("kelly_bet") or 0.0 for r in group)
        if total_kelly > MAX_GROUP_KELLY and total_kelly > 0:
            scale = MAX_GROUP_KELLY / total_kelly
            for r in group:
                raw_k = r.get("kelly_bet") or 0.0
                r["kelly_bet"]             = round(raw_k * scale, 2)
                r["kelly_portfolio_scaled"] = True
                r["kelly_group_total_raw"]  = round(total_kelly, 2)
            print(f"[Portfolio Kelly] {city} {date_s}: {len(group)} buckets "
                  f"${total_kelly:.2f} → ${MAX_GROUP_KELLY:.2f} (scale={scale:.3f})")

    # ── Background: resolve any past markets (non-blocking asyncio task) ────────
    async def _bg_resolve():
        try:
            n = await asyncio.to_thread(auto_resolve_past_markets)
            if n:
                print(f"[Scan] Background resolved {n} past market(s)")
        except Exception as e:
            print(f"[Scan] Background resolve error: {e}")
    asyncio.create_task(_bg_resolve())

    return enriched


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Lightweight health-check used by Railway.  Must return 200 fast."""
    return JSONResponse({"status": "ok"})


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


def _build_payload(results: list[dict]) -> dict:
    """將 _run_scan 結果序列化成 API payload（供 api_opportunities 和背景任務共用）。"""
    raw_opps = [r for r in results if r.get("is_opportunity")]
    no_edge  = [r for r in results if not r.get("is_opportunity") and not r.get("skip_reason")]
    skipped  = [r for r in results if r.get("skip_reason")]

    _conv_w = {"high": 1.0, "medium": 0.65, "low": 0.35}
    _dir_w  = {"BUY YES": 1.0, "BUY NO": 0.85, "HOLD": 0.0}

    def _score(r: dict) -> float:
        edge = abs(r.get("abs_edge") or r.get("edge") or 0)
        conv = _conv_w.get(r.get("conviction") or "low", 0.35)
        dirw = _dir_w.get(r.get("action") or "HOLD", 0.85)
        return edge * conv * dirw

    opportunities = sorted(raw_opps, key=_score, reverse=True)
    no_edge = sorted(no_edge, key=lambda r: abs(r.get("edge") or 0), reverse=True)

    def _clean(r: dict) -> dict:
        out = {}
        for k, v in r.items():
            if k == "consensus" and isinstance(v, dict):
                out[k] = v
            elif isinstance(v, (str, int, float, bool, type(None))):
                out[k] = v
            elif isinstance(v, list) and k in ("sources_used",):
                out[k] = [x for x in v if isinstance(x, str)]
            elif isinstance(v, dict):
                out[k] = {kk: vv for kk, vv in v.items()
                          if isinstance(vv, (str, int, float, bool, type(None)))}
        for extra_k in ("wu_temp_c", "wu_temp_f", "wu_age_min", "wu_source",
                        "wu_definitive", "wu_definitive_result", "latency_note",
                        "in_latency_arb_zone", "exec_edge", "exec_edge_vwap",
                        "half_spread", "kelly_portfolio_scaled", "kelly_group_total_raw",
                        "regime_shift"):
            if extra_k in r and extra_k not in out:
                v = r[extra_k]
                if isinstance(v, (str, int, float, bool, type(None))):
                    out[extra_k] = v
        return out

    return {
        "scanned_at":     datetime.now(timezone.utc).isoformat(),
        "edge_threshold": EDGE_THRESHOLD,
        "opportunities":  [_clean(r) for r in opportunities],
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


async def _run_scan_task():
    """
    Background scan task: acquire lock → scan → update cache → release.
    Called by both the background scheduler and the non-blocking refresh endpoint.
    Never blocks an HTTP request — callers fire-and-forget with asyncio.create_task().
    """
    global _cache, _cache_ts, _scan_in_progress
    if _scan_in_progress:
        return  # already scanning, skip
    _scan_in_progress = True
    try:
        async with _scan_lock:
            results = await _run_scan()
            payload = _build_payload(results)
            _cache    = payload
            _cache_ts = time.time()
            print(f"[Scan] Done — {payload['summary']['opportunities']} opportunities")
    except Exception as e:
        print(f"[Scan] Error: {e}")
    finally:
        _scan_in_progress = False


@app.get("/api/opportunities")
async def api_opportunities(refresh: bool = False):
    """
    Returns cached scan results immediately.
    If cache is stale (or refresh=True), triggers a background scan and returns
    whatever is in cache (or an empty scaffold with scanning=True).
    The client should poll every few seconds until scanning=False.

    This design keeps Railway's 30-second HTTP timeout happy — the endpoint
    never waits for a full 60-120 second scan to complete.
    """
    global _cache_ts

    if refresh:
        _cache_ts = 0.0  # invalidate so next poll triggers a new scan

    # If cache is fresh, return it immediately
    if _is_cache_valid() and _cache and not refresh:
        return JSONResponse({**_cache, "cached": True, "scanning": False})

    # Trigger background scan if not already running
    if not _scan_in_progress:
        asyncio.create_task(_run_scan_task())

    # Return stale cache (if any) immediately — client polls until scanning=False
    if _cache:
        return JSONResponse({**_cache, "cached": True, "scanning": _scan_in_progress})

    # First-ever load: no cache yet, return empty scaffold
    return JSONResponse({
        "scanning":      True,
        "cached":        False,
        "opportunities": [],
        "no_edge":       [],
        "skipped":       [],
        "summary":       {"total": 0, "opportunities": 0, "no_edge": 0, "skipped": 0},
    })


@app.get("/api/latency")
def api_latency():
    try:
        return JSONResponse(get_latency_summary())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/history")
async def api_history(days: int = 30):
    try:
        await asyncio.to_thread(init_db)
        brier  = await asyncio.to_thread(get_brier_score, days)
        recent = await asyncio.to_thread(get_recent_predictions, 20)
        return JSONResponse({"brier_score": brier, "recent": recent})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/history/resolved")
async def api_history_resolved(days: int = 90):
    try:
        await asyncio.to_thread(init_db)
        try:
            newly_resolved = await asyncio.to_thread(auto_resolve_past_markets)
            if newly_resolved:
                print(f"[History] Auto-resolved {newly_resolved} past market(s)")
        except Exception:
            pass
        data = await asyncio.to_thread(get_all_predictions, days, 300)
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/refresh")
async def api_refresh():
    """
    立即回傳（不等待掃描完成），觸發背景掃描任務。
    客戶端應輪詢 GET /api/opportunities 直到 scanning=False。
    Railway HTTP timeout 為 30s，完整掃描需 60-120s，故不能同步等待。
    """
    global _cache_ts
    _cache_ts = 0.0
    if not _scan_in_progress:
        asyncio.create_task(_run_scan_task())
    return JSONResponse({
        "scanning": True,
        "message":  "Scan triggered — poll /api/opportunities for results",
        **({"cached": True, **{k: v for k, v in _cache.items() if k != "cached"}} if _cache else {}),
    })


# Startup logic moved to _lifespan context manager (defined near top of file).


if __name__ == "__main__":
    import sys, os
    sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
    port = int(os.environ.get("PORT", 8000))
    print(f"Weather Arb Dashboard — listening on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

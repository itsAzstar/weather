"""
history.py
SQLite 歷史記錄 + Brier Score 校準追蹤。
每次掃描時記錄預測，市場結算後記錄結果，計算滾動 Brier Score。
"""

import os
import sqlite3
import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import Optional

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


def _http_get_json(url: str, params: Optional[dict] = None, timeout: float = 10.0) -> Optional[dict]:
    full_url = url + "?" + urllib.parse.urlencode(params) if params else url
    try:
        with urllib.request.urlopen(full_url, timeout=timeout, context=_SSL_CTX) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode())
    except Exception:
        return None

# Railway / production: set DB_PATH env var to a persistent volume path
# e.g. DB_PATH=/data/weather_history.db  (Railway Volume mounted at /data)
# Local dev: defaults to project directory
_default_db = Path(__file__).parent / "weather_history.db"
DB_PATH = Path(os.environ.get("DB_PATH", str(_default_db)))


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    # WAL mode：讀寫不互斷
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """建表（幂等）。"""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                condition_id    TEXT NOT NULL,
                question        TEXT,
                location        TEXT,
                event_type      TEXT,
                target_date     TEXT,
                market_price    REAL,
                model_prob      REAL,
                consensus_prob  REAL,
                conviction      TEXT,
                models_agree    INTEGER,
                edge            REAL,
                action          TEXT,
                predicted_at    TEXT NOT NULL,
                outcome         INTEGER,    -- 1=YES, 0=NO, NULL=pending
                resolved_at     TEXT,
                temp_bucket_json TEXT,      -- JSON: {lo_c, hi_c, direction}
                market_url      TEXT,
                market_subtype  TEXT,
                temp_display    TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_condition ON predictions(condition_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_target_date ON predictions(target_date)
        """)
        # Migrate: add new columns if they don't exist (idempotent)
        for col, coltype in [
            ("temp_bucket_json",  "TEXT"),
            ("market_url",        "TEXT"),
            ("market_subtype",    "TEXT"),
            ("temp_display",      "TEXT"),
            ("resolution_source", "TEXT"),   # 'polymarket' | 'archive' | 'manual'
            # Nowcast-contamination tracking (Bug A fix).
            # Needed to separate "pure forecast" Brier from nowcast-leaked Brier.
            ("days_ahead",                "INTEGER"),
            ("obs_adjusted",              "INTEGER"),  # 0/1 — METAR/WU observation was folded in
            ("time_remaining_hours",      "REAL"),
            ("in_latency_arb_zone",       "INTEGER"),  # 0/1
            # Dedup key — UTC date of the prediction, populated at insert time
            # so a UNIQUE index can enforce one-row-per-market-per-day atomically.
            # Previous SELECT-then-INSERT dedup was racy: concurrent _enrich_one
            # workers all saw "no existing" and all INSERTed → 100× duplicates.
            ("predicted_date",  "TEXT"),
            # Which layer produced `model_prob`:
            #   "nowcast"             — comparator's Kalman/METAR/WU adjustment (pre-2026-04-18)
            #   "consensus_override"  — bleed-stop patch in server._enrich_one overrode with consensus_prob
            #   "wu_definitive"       — WU daily high outside bucket → model_prob forced to 0.01
            # Needed to measure whether the 2026-04-18 bleed-stop actually improved Brier.
            ("model_prob_source", "TEXT"),
            # Snapshot of weather data that drove the decision. Previously
            # everything was re-fetched at resolve-time, making it impossible
            # to debug "why did we say dead?" after the fact.
            ("wu_temp_c",         "REAL"),   # WU spike-filtered max at decision time
            ("wu_temp_c_raw",     "REAL"),   # WU true max (includes single-hour spikes)
            ("wu_data_source",    "TEXT"),   # dailysummary | observations | ...
            ("obs_temp_c",        "REAL"),   # METAR instantaneous reading at decision time
        ]:
            try:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {coltype}")
            except Exception:
                pass  # column already exists

        # Backfill predicted_date for legacy rows (UTC date of predicted_at)
        conn.execute("""
            UPDATE predictions
               SET predicted_date = substr(predicted_at, 1, 10)
             WHERE predicted_date IS NULL AND predicted_at IS NOT NULL
        """)

        # Dedupe existing rows (keep lowest id per condition_id+predicted_date)
        # before creating the UNIQUE index — otherwise index creation fails.
        conn.execute("""
            DELETE FROM predictions
             WHERE id NOT IN (
               SELECT MIN(id) FROM predictions
                GROUP BY condition_id, predicted_date
             )
        """)

        # UNIQUE index: atomic dedup. INSERT OR IGNORE becomes the one-liner
        # replacement for the SELECT-then-INSERT race.
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_condition_date
                ON predictions(condition_id, predicted_date)
        """)


def log_prediction(result: dict):
    """
    記錄一次預測掃描結果。atomically dedup 同一市場當日記錄 via
    UNIQUE(condition_id, predicted_date) + INSERT OR IGNORE.
    """
    init_db()
    condition_id = result.get("condition_id", "")
    if not condition_id:
        return
    now_utc = datetime.now(timezone.utc)
    predicted_at = now_utc.isoformat()
    predicted_date = now_utc.date().isoformat()

    with _connect() as conn:
        parsed = result.get("parsed", {})
        bucket = result.get("temp_bucket")
        bucket_json = json.dumps(bucket) if bucket else None

        # Nowcast-contamination flags — snapshot at prediction time.
        # days_ahead==0 with obs_adjusted=True means model_prob has
        # already "peeked" at realized weather; Brier on these is leaked.
        days_ahead = result.get("days_ahead")
        obs_adjusted = result.get("obs_adjusted")
        time_remaining_hours = result.get("time_remaining_hours")
        in_latency = result.get("in_latency_arb_zone")

        # Weather snapshot at decision time — used for post-hoc debugging
        # ("why did we flag this dead?") instead of re-fetching at resolve-time.
        def _f(v):
            try: return float(v) if v is not None else None
            except (ValueError, TypeError): return None

        conn.execute("""
            INSERT OR IGNORE INTO predictions
                (condition_id, question, location, event_type, target_date,
                 market_price, model_prob, consensus_prob, conviction, models_agree,
                 edge, action, predicted_at, predicted_date,
                 temp_bucket_json, market_url, market_subtype, temp_display,
                 days_ahead, obs_adjusted, time_remaining_hours, in_latency_arb_zone,
                 model_prob_source,
                 wu_temp_c, wu_temp_c_raw, wu_data_source, obs_temp_c)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            condition_id,
            result.get("question", ""),
            result.get("location_hint") or parsed.get("location", ""),
            parsed.get("event_type", ""),
            parsed.get("target_date", ""),
            result.get("market_price_yes"),
            result.get("model_probability"),
            result.get("consensus_prob"),
            result.get("conviction"),
            result.get("models_agree"),
            result.get("edge"),
            result.get("action", ""),
            predicted_at,
            predicted_date,
            bucket_json,
            result.get("url"),
            result.get("market_subtype"),
            result.get("temp_display"),
            int(days_ahead) if isinstance(days_ahead, (int, float)) else None,
            1 if obs_adjusted else (0 if obs_adjusted is False else None),
            float(time_remaining_hours) if isinstance(time_remaining_hours, (int, float)) else None,
            1 if in_latency else (0 if in_latency is False else None),
            result.get("model_prob_source"),
            _f(result.get("wu_temp_c")),
            _f(result.get("wu_temp_c_raw")),
            result.get("wu_data_source"),
            _f(result.get("obs_temp_c")),
        ))


def record_outcome(condition_id: str, outcome: bool, source: str = "manual"):
    """
    市場結算後記錄結果。
    outcome=True → YES 結算, False → NO 結算。
    source: 'polymarket' | 'archive' | 'manual'
    """
    init_db()
    with _connect() as conn:
        conn.execute("""
            UPDATE predictions
            SET outcome=?, resolved_at=?, resolution_source=?
            WHERE condition_id=? AND outcome IS NULL
        """, (
            1 if outcome else 0,
            datetime.now(timezone.utc).isoformat(),
            source,
            condition_id,
        ))


def _brier(rows, key: str) -> Optional[float]:
    vals = [r for r in rows if r[key] is not None]
    if not vals:
        return None
    return sum((r[key] - r["outcome"]) ** 2 for r in vals) / len(vals)


def get_brier_score(days: int = 30) -> Optional[dict]:
    """
    計算最近 N 天已結算預測的 Brier Score，**分三組**：
      - pure_forecast: days_ahead >= 2  (純預報，最乾淨)
      - near_term:     days_ahead == 1 且 obs_adjusted == 0  (明天預報)
      - nowcast:       days_ahead <= 1 且 obs_adjusted == 1  (已混入觀測，有洩漏疑慮)

    完美校準 = 0，隨機猜測 = 0.25。
    nowcast 組若「看起來太好」(< 0.05)，很可能是資料洩漏，不是模型強。
    """
    init_db()
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with _connect() as conn:
        rows = conn.execute("""
            SELECT model_prob, consensus_prob, outcome,
                   days_ahead, obs_adjusted, model_prob_source
            FROM predictions
            WHERE outcome IS NOT NULL AND predicted_at >= ?
        """, (since,)).fetchall()

    if not rows:
        return None

    pure     = [r for r in rows if (r["days_ahead"] is not None and r["days_ahead"] >= 2)]
    near     = [r for r in rows if (r["days_ahead"] == 1 and not r["obs_adjusted"])]
    nowcast  = [r for r in rows if (r["days_ahead"] is not None and r["days_ahead"] <= 1 and r["obs_adjusted"] == 1)]
    unknown  = [r for r in rows if r["days_ahead"] is None]  # 老資料沒記錄欄位

    total = len(rows)
    wins = sum(1 for r in rows if r["outcome"] == 1)

    def _group(rs):
        if not rs:
            return None
        return {
            "n":               len(rs),
            "brier_model":     round(_brier(rs, "model_prob"),     4) if _brier(rs, "model_prob")     is not None else None,
            "brier_consensus": round(_brier(rs, "consensus_prob"), 4) if _brier(rs, "consensus_prob") is not None else None,
            "win_rate":        round(sum(1 for r in rs if r["outcome"] == 1) / len(rs), 3),
        }

    # Headline Brier EXCLUDES the "unknown" group (rows logged before the
    # days_ahead/obs_adjusted columns existed AND before parser/fetcher
    # produced market_subtype + temp_bucket). Those rows have polluted
    # consensus_prob (bucket markets run through exceed formula) and would
    # drag the headline number into the "worse than random" zone.
    known = [r for r in rows if r["days_ahead"] is not None]
    bs_model_all     = _brier(known, "model_prob")     if known else None
    bs_consensus_all = _brier(known, "consensus_prob") if known else None
    # Fallback to all-rows if no post-fix data exists yet (fresh DB).
    if bs_model_all is None:
        bs_model_all     = _brier(rows, "model_prob")
        bs_consensus_all = _brier(rows, "consensus_prob")
    ref = bs_consensus_all if bs_consensus_all is not None else bs_model_all

    # 污染警示：若 nowcast 組 Brier 遠優於 pure，大概率是資料洩漏
    leakage_warning = None
    ng = _group(nowcast)
    pg = _group(pure)
    if ng and pg and ng["brier_model"] is not None and pg["brier_model"] is not None:
        if pg["brier_model"] > 0.05 and ng["brier_model"] < pg["brier_model"] * 0.5:
            leakage_warning = (
                f"Nowcast Brier ({ng['brier_model']:.3f}) 遠優於 pure forecast "
                f"({pg['brier_model']:.3f}) — 疑似觀測資料洩漏至 model_prob"
            )

    return {
        "total_resolved":    total,
        "wins":              wins,
        "win_rate":          round(wins / total, 3),
        "brier_model":       round(bs_model_all, 4)     if bs_model_all     is not None else None,
        "brier_consensus":   round(bs_consensus_all, 4) if bs_consensus_all is not None else None,
        "days_window":       days,
        "groups": {
            "pure_forecast": _group(pure),     # days_ahead >= 2
            "near_term":     _group(near),     # days_ahead == 1, no obs
            "nowcast":       _group(nowcast),  # obs_adjusted (可能洩漏)
            "unknown":       _group(unknown),  # migration 前的老資料
        },
        # 2026-04-18 bleed-stop A/B: model_prob 來源分組。
        # 比較 nowcast (舊 Kalman/METAR 加工) vs consensus_override (純 5 模型)
        # 部署 24-48h 後若 consensus_override 的 Brier 明顯低 → 證明 override 有效。
        "by_source": {
            "nowcast":            _group([r for r in rows if r["model_prob_source"] == "nowcast"]),
            "consensus_override": _group([r for r in rows if r["model_prob_source"] == "consensus_override"]),
            "wu_definitive":      _group([r for r in rows if r["model_prob_source"] == "wu_definitive"]),
            "legacy_null":        _group([r for r in rows if r["model_prob_source"] is None]),
        },
        "leakage_warning": leakage_warning,
        # 解讀：< 0.10 優秀，< 0.20 良好，> 0.25 = 比隨機還差
        "rating": (
            "優秀" if ref is not None and ref < 0.10
            else "良好" if ref is not None and ref < 0.20
            else "需改進"
        ),
    }


def get_recent_predictions(limit: int = 20) -> list[dict]:
    """取得最近的預測記錄（含結算狀態）。"""
    init_db()
    with _connect() as conn:
        rows = conn.execute("""
            SELECT condition_id, question, location, target_date,
                   market_price, consensus_prob, edge, action,
                   predicted_at, outcome, resolved_at
            FROM predictions
            ORDER BY predicted_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def _resolve_from_polymarket(condition_id: str) -> Optional[bool]:
    """
    Query Polymarket CLOB API to get the actual market resolution.
    Returns True (YES won), False (NO won), None (still open / unknown).

    When a market resolves:
      YES wins → YES token price = 1.0, NO token price = 0.0
      NO wins  → YES token price = 0.0, NO token price = 1.0
    """
    data = _http_get_json(f"https://clob.polymarket.com/markets/{condition_id}", timeout=10)
    if not data:
        return None
    try:
        tokens = data.get("tokens", [])
        price_yes = None
        for token in tokens:
            outcome = (token.get("outcome") or "").lower()
            try:
                p = float(token.get("price", -1))
            except (ValueError, TypeError):
                continue
            if outcome == "yes":
                price_yes = p

        if price_yes is None:
            return None
        if price_yes >= 0.99:
            return True   # YES won
        if price_yes <= 0.01:
            return False  # NO won
        return None       # Still trading (not yet resolved)
    except Exception as e:
        # Schema-drift guard: Polymarket CLOB changing `tokens[].price` shape
        # would silently stall resolution. Log instead of hiding.
        print(f"[CLOB] Resolve parse failed for {condition_id[:10]}...: "
              f"{type(e).__name__}: {e}")
        return None


def _resolve_from_weather_archive(condition_id: str, location: str,
                                   target_date_str: str, temp_bucket_json: str) -> Optional[bool]:
    """
    Fallback: estimate resolution from Open-Meteo historical archive.
    Only works for temperature_bucket markets.
    NOTE: uses city-centre coords, not the exact station — may differ from Polymarket.

    Bug C fix: `timezone=auto` so daily_max covers the city's local calendar day.
    `timezone=UTC` split NYC's day across 20:00→20:00 local, making archive resolution
    disagree with Polymarket (which resolves on local-day high).
    """
    from fetcher_weather import resolve_location
    try:
        target_d = date.fromisoformat(target_date_str)
        bucket = json.loads(temp_bucket_json or "{}")
        if not bucket:
            return None
        # KNOWN LIMITATION: resolve_location() returns city-center coords, but
        # Polymarket resolves against the Wunderground ICAO station high (often
        # an airport 5-20 km from downtown). For cities like Austin (KAUS 10 km
        # south) or Dallas (KDAL downtown vs DFW airport) this can introduce a
        # 0.5-2°C bias in archive-fallback resolutions. Stage 1 (Polymarket CLOB)
        # is the truth-source and gets tried first, so this only affects stuck
        # pending rows where CLOB hasn't settled. TODO: add ICAO→coords table
        # in fetcher_stations.py and prefer station coords when icao is known.
        coords = resolve_location(location)
        if not coords:
            return None
        lat, lon = coords

        data = _http_get_json(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude":   lat,
                "longitude":  lon,
                "start_date": target_d.isoformat(),
                "end_date":   target_d.isoformat(),
                "daily":      "temperature_2m_max",
                "timezone":   "auto",
            },
            timeout=10,
        )
        if not data:
            return None
        temps = (data.get("daily") or {}).get("temperature_2m_max") or []
        if not temps or temps[0] is None:
            return None
        actual_c = temps[0]

        lo_c = bucket.get("lo_c")
        hi_c = bucket.get("hi_c")
        if lo_c is not None and hi_c is not None:
            return lo_c <= actual_c <= hi_c
        elif lo_c is not None:
            return actual_c >= lo_c
        elif hi_c is not None:
            return actual_c <= hi_c
        return None
    except Exception as e:
        # Archive fallback failure: log so we notice systematic Open-Meteo outages
        # or resolve_location() misses instead of silently leaving rows pending forever.
        print(f"[Archive] Resolve failed for {location}/{target_date_str}: "
              f"{type(e).__name__}: {e}")
        return None


def auto_resolve_past_markets():
    """
    Auto-resolve past predictions using a two-stage approach:

    Stage 1 — Polymarket CLOB API (all market types):
      Query the actual market resolution price. If YES token = 1.0 → YES won,
      0.0 → NO won. This is the ground truth. Works for rain, temp, wind etc.

    Stage 2 — Open-Meteo archive fallback (temperature_bucket only):
      If Polymarket hasn't resolved yet (market still showing mid prices),
      estimate from historical weather. Less accurate but better than nothing.

    Returns number of markets newly resolved.
    """
    init_db()
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    with _connect() as conn:
        rows = conn.execute("""
            SELECT id, condition_id, location, target_date,
                   temp_bucket_json, market_subtype, market_url
            FROM predictions
            WHERE outcome IS NULL
              AND target_date <= ?
            LIMIT 200
        """, (yesterday,)).fetchall()

    resolved = 0
    poly_hits = 0
    archive_hits = 0

    for r in rows:
        try:
            condition_id = r["condition_id"]
            # Skip mock markets
            if condition_id.startswith("mock-"):
                continue

            # ── Stage 1: Ask Polymarket directly ──────────────────────
            outcome = _resolve_from_polymarket(condition_id)
            if outcome is not None:
                record_outcome(condition_id, outcome, source="polymarket")
                resolved += 1
                poly_hits += 1
                continue

            # ── Stage 2: Estimate from weather archive (temp only) ────
            if (r["market_subtype"] == "temperature_bucket"
                    and r["temp_bucket_json"]
                    and r["target_date"]):
                outcome = _resolve_from_weather_archive(
                    condition_id, r["location"] or "",
                    r["target_date"], r["temp_bucket_json"],
                )
                if outcome is not None:
                    record_outcome(condition_id, outcome, source="archive")
                    resolved += 1
                    archive_hits += 1

        except Exception as e:
            print(f"[History] Resolve failed for {r['condition_id'][:10]}... "
                  f"({r['location']}/{r['target_date']}): {type(e).__name__}: {e}")
            continue

    if resolved:
        print(f"[History] Resolved {resolved} markets "
              f"(Polymarket: {poly_hits}, archive-estimate: {archive_hits})")
    return resolved


def get_all_predictions(days: int = 90, limit: int = 200) -> dict:
    """
    返回所有預測記錄（含待結算），分成三組：
      correct  — BUY YES + outcome=1，或 BUY NO + outcome=0
      wrong    — 結算了但預測錯誤
      pending  — outcome IS NULL（尚未結算）
    """
    init_db()
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with _connect() as conn:
        rows = conn.execute("""
            SELECT condition_id, question, location, target_date,
                   market_price, model_prob, consensus_prob, edge, action,
                   conviction, predicted_at, outcome, resolved_at,
                   temp_bucket_json, market_url, market_subtype, temp_display,
                   model_prob_source,
                   wu_temp_c, wu_temp_c_raw, wu_data_source, obs_temp_c
            FROM predictions
            WHERE predicted_at >= ?
            ORDER BY predicted_at DESC
            LIMIT ?
        """, (since, limit)).fetchall()

    correct, wrong, pending = [], [], []
    for r in rows:
        d = dict(r)
        action  = d.get("action", "")
        outcome = d.get("outcome")

        if outcome is None:
            d["verdict"] = "pending"
            pending.append(d)
        elif (action == "BUY YES" and outcome == 1) or (action == "BUY NO" and outcome == 0):
            d["verdict"] = "correct"
            correct.append(d)
        else:
            d["verdict"] = "wrong"
            wrong.append(d)

    total_settled = len(correct) + len(wrong)
    success_rate = round(len(correct) / total_settled, 4) if total_settled > 0 else None
    failure_rate = round(len(wrong)   / total_settled, 4) if total_settled > 0 else None

    return {
        "correct":       correct,
        "wrong":         wrong,
        "pending":       pending,
        "total_settled": total_settled,
        "total":         len(correct) + len(wrong) + len(pending),
        "success_count": len(correct),
        "failure_count": len(wrong),
        "pending_count": len(pending),
        "success_rate":  success_rate,
        "failure_rate":  failure_rate,
        "days_window":   days,
    }


def get_resolved_split(days: int = 90) -> dict:
    """
    返回已結算預測，分成「正確」和「失敗」兩組。
    判斷邏輯：
      - BUY YES + outcome=1 → 正確
      - BUY NO  + outcome=0 → 正確
      - 其他 → 失敗（包含 HOLD/SKIP 有結算者）
    """
    init_db()
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with _connect() as conn:
        rows = conn.execute("""
            SELECT condition_id, question, location, target_date,
                   market_price, model_prob, consensus_prob, edge, action,
                   conviction, predicted_at, outcome, resolved_at
            FROM predictions
            WHERE outcome IS NOT NULL AND predicted_at >= ?
            ORDER BY resolved_at DESC
        """, (since,)).fetchall()

    correct = []
    wrong   = []
    pending_rows = []

    for r in rows:
        d = dict(r)
        action  = d.get("action", "")
        outcome = d.get("outcome")

        if action == "BUY YES" and outcome == 1:
            d["verdict"] = "correct"
            correct.append(d)
        elif action == "BUY NO" and outcome == 0:
            d["verdict"] = "correct"
            correct.append(d)
        elif outcome is not None:
            d["verdict"] = "wrong"
            wrong.append(d)

    total = len(correct) + len(wrong)
    success_rate = round(len(correct) / total, 4) if total > 0 else None
    failure_rate = round(len(wrong)   / total, 4) if total > 0 else None

    return {
        "correct":      correct,
        "wrong":        wrong,
        "total":        total,
        "success_count": len(correct),
        "failure_count": len(wrong),
        "success_rate":  success_rate,
        "failure_rate":  failure_rate,
        "days_window":   days,
    }


if __name__ == "__main__":
    init_db()
    score = get_brier_score()
    if score:
        print(f"Brier Score ({score['days_window']}d): {score['brier_model']:.4f} — {score['rating']}")
        print(f"Win rate: {score['win_rate']:.1%} ({score['wins']}/{score['total_resolved']})")
    else:
        print("尚無結算記錄。掃描幾天後再查。")

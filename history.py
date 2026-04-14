"""
history.py
SQLite 歷史記錄 + Brier Score 校準追蹤。
每次掃描時記錄預測，市場結算後記錄結果，計算滾動 Brier Score。
"""

import sqlite3
import json
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "weather_history.db"


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
        ]:
            try:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} {coltype}")
            except Exception:
                pass  # column already exists


def log_prediction(result: dict):
    """記錄一次預測掃描結果。若已存在同 condition_id 今日記錄則略過。"""
    init_db()
    condition_id = result.get("condition_id", "")
    today = date.today().isoformat()

    with _connect() as conn:
        # 避免同一天重複記錄同一市場
        existing = conn.execute(
            "SELECT id FROM predictions WHERE condition_id=? AND DATE(predicted_at)=?",
            (condition_id, today)
        ).fetchone()
        if existing:
            return

        parsed = result.get("parsed", {})
        bucket = result.get("temp_bucket")
        bucket_json = json.dumps(bucket) if bucket else None
        conn.execute("""
            INSERT INTO predictions
                (condition_id, question, location, event_type, target_date,
                 market_price, model_prob, consensus_prob, conviction, models_agree,
                 edge, action, predicted_at,
                 temp_bucket_json, market_url, market_subtype, temp_display)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
            datetime.now(timezone.utc).isoformat(),
            bucket_json,
            result.get("url"),
            result.get("market_subtype"),
            result.get("temp_display"),
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


def get_brier_score(days: int = 30) -> Optional[dict]:
    """
    計算最近 N 天已結算預測的 Brier Score。
    Brier Score = mean((predicted_prob - outcome)^2)
    完美校準 = 0，隨機猜測 = 0.25
    """
    init_db()
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with _connect() as conn:
        rows = conn.execute("""
            SELECT model_prob, consensus_prob, outcome
            FROM predictions
            WHERE outcome IS NOT NULL AND predicted_at >= ?
        """, (since,)).fetchall()

    if not rows:
        return None

    total = len(rows)
    # 優先用 consensus_prob，否則用 model_prob
    bs_model     = sum((r["model_prob"] - r["outcome"])**2 for r in rows if r["model_prob"] is not None) / total
    bs_consensus = None
    consensus_rows = [r for r in rows if r["consensus_prob"] is not None]
    if consensus_rows:
        bs_consensus = sum((r["consensus_prob"] - r["outcome"])**2 for r in consensus_rows) / len(consensus_rows)

    wins = sum(1 for r in rows if r["outcome"] == 1)
    return {
        "total_resolved":    total,
        "wins":              wins,
        "win_rate":          round(wins / total, 3),
        "brier_model":       round(bs_model, 4),
        "brier_consensus":   round(bs_consensus, 4) if bs_consensus else None,
        "days_window":       days,
        # 解讀：< 0.10 優秀，< 0.20 良好，> 0.25 = 比隨機還差
        "rating": (
            "優秀" if (bs_consensus or bs_model) < 0.10
            else "良好" if (bs_consensus or bs_model) < 0.20
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
    import requests
    try:
        resp = requests.get(
            f"https://clob.polymarket.com/markets/{condition_id}",
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
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
    except Exception:
        return None


def _resolve_from_weather_archive(condition_id: str, location: str,
                                   target_date_str: str, temp_bucket_json: str) -> Optional[bool]:
    """
    Fallback: estimate resolution from Open-Meteo historical archive.
    Only works for temperature_bucket markets.
    NOTE: uses city-centre coords, not the exact station — may differ from Polymarket.
    """
    import requests
    from fetcher_weather import resolve_location
    try:
        target_d = date.fromisoformat(target_date_str)
        bucket = json.loads(temp_bucket_json or "{}")
        if not bucket:
            return None
        coords = resolve_location(location)
        if not coords:
            return None
        lat, lon = coords

        resp = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude":   lat,
                "longitude":  lon,
                "start_date": target_d.isoformat(),
                "end_date":   target_d.isoformat(),
                "daily":      "temperature_2m_max",
                "timezone":   "UTC",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        temps = (resp.json().get("daily") or {}).get("temperature_2m_max") or []
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
    except Exception:
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

        except Exception:
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
                   temp_bucket_json, market_url, market_subtype, temp_display
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

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
                resolved_at     TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_condition ON predictions(condition_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_target_date ON predictions(target_date)
        """)


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
        conn.execute("""
            INSERT INTO predictions
                (condition_id, question, location, event_type, target_date,
                 market_price, model_prob, consensus_prob, conviction, models_agree,
                 edge, action, predicted_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            condition_id,
            result.get("question", ""),
            parsed.get("location", ""),
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
        ))


def record_outcome(condition_id: str, outcome: bool):
    """
    市場結算後記錄結果。
    outcome=True → YES 結算, False → NO 結算。
    """
    init_db()
    with _connect() as conn:
        conn.execute("""
            UPDATE predictions
            SET outcome=?, resolved_at=?
            WHERE condition_id=? AND outcome IS NULL
        """, (
            1 if outcome else 0,
            datetime.now(timezone.utc).isoformat(),
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

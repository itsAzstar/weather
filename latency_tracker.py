"""
latency_tracker.py
追蹤 GFS / ECMWF / ICON 模型更新時間，計算資料新鮮度。
核心邏輯：新一輪 model run 落地後，Polymarket 市場定價需 2-12 小時更新。
這個窗口就是 Latency 套利機會。
"""

from datetime import datetime, timezone, timedelta


# 各模型 run 時刻（UTC）及資料可用延遲
MODEL_SCHEDULE = {
    "GFS": {
        "run_hours_utc": [0, 6, 12, 18],   # 每 6 小時一次
        "lag_hours":     4,                 # run 後約 4 小時資料才可用
        "description":   "NOAA GFS (每 6h)",
    },
    "ECMWF": {
        "run_hours_utc": [0, 12],           # 每 12 小時一次
        "lag_hours":     6,                 # run 後約 6 小時資料才可用
        "description":   "ECMWF IFS (每 12h)",
    },
    "ICON": {
        "run_hours_utc": [0, 6, 12, 18],   # 每 6 小時一次
        "lag_hours":     3,                 # run 後約 3 小時資料才可用
        "description":   "DWD ICON (每 6h)",
    },
}

# 若新 run 落地後不到這個小時數，視為「套利窗口開啟」
LATENCY_WINDOW_HOURS = 3.0

# 若超過這個小時數，資料過舊警示
STALE_THRESHOLD_HOURS = 9.0


def get_model_status(now_utc: datetime = None) -> dict:
    """
    計算每個模型的當前狀態。
    回傳 dict:
    {
        "GFS": {
            "last_available":  datetime,   # 最近一次資料可用時刻
            "hours_ago":       float,      # 距今幾小時
            "next_available":  datetime,   # 下次資料可用時刻
            "is_fresh":        bool,       # < LATENCY_WINDOW_HOURS
            "is_stale":        bool,       # > STALE_THRESHOLD_HOURS
            "alert":           str | None, # 套利窗口提示
        },
        ...
        "latency_alert":   bool,   # 任一模型剛更新（套利窗口）
        "alert_models":    list,   # 哪些模型觸發
    }
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    result = {}
    latency_alert = False
    alert_models = []

    for model_name, cfg in MODEL_SCHEDULE.items():
        run_hours = cfg["run_hours_utc"]
        lag = cfg["lag_hours"]

        # 找最近一個已可用的 run
        last_avail = None
        next_avail = None
        for i in range(48):  # 回看最多 48 小時
            candidate_dt = now_utc.replace(minute=0, second=0, microsecond=0) - timedelta(hours=i)
            if candidate_dt.hour in run_hours:
                avail_dt = candidate_dt + timedelta(hours=lag)
                if avail_dt <= now_utc:
                    if last_avail is None:
                        last_avail = avail_dt
                else:
                    if last_avail is None:
                        # 還沒到，這個 run 是下一個
                        if next_avail is None:
                            next_avail = avail_dt

        # 找下一個可用時間
        if next_avail is None:
            for i in range(1, 25):
                candidate_dt = now_utc.replace(minute=0, second=0, microsecond=0) + timedelta(hours=i)
                if candidate_dt.hour in run_hours:
                    next_avail = candidate_dt + timedelta(hours=lag)
                    break

        hours_ago = None
        is_fresh = False
        is_stale = False
        alert = None

        if last_avail:
            hours_ago = (now_utc - last_avail).total_seconds() / 3600
            is_fresh = hours_ago <= LATENCY_WINDOW_HOURS
            is_stale = hours_ago >= STALE_THRESHOLD_HOURS

            if is_fresh:
                alert = f"⚡ {model_name} 新 run 上線 {hours_ago:.1f}h 前 — 市場可能尚未重新定價！"
                latency_alert = True
                alert_models.append(model_name)

        result[model_name] = {
            "description":    cfg["description"],
            "last_available": last_avail.isoformat() if last_avail else None,
            "hours_ago":      round(hours_ago, 1) if hours_ago is not None else None,
            "next_available": next_avail.isoformat() if next_avail else None,
            "is_fresh":       is_fresh,
            "is_stale":       is_stale,
            "alert":          alert,
        }

    result["latency_alert"] = latency_alert
    result["alert_models"]  = alert_models
    result["checked_at"]    = now_utc.isoformat()
    return result


def get_latency_summary(now_utc: datetime = None) -> dict:
    """簡化版，給 API 端點用。"""
    status = get_model_status(now_utc)
    models = []
    for name in ["GFS", "ECMWF", "ICON"]:
        s = status[name]
        models.append({
            "name":           name,
            "description":    s["description"],
            "hours_ago":      s["hours_ago"],
            "is_fresh":       s["is_fresh"],
            "is_stale":       s["is_stale"],
            "alert":          s["alert"],
            "next_available": s["next_available"],
        })
    return {
        "models":        models,
        "latency_alert": status["latency_alert"],
        "alert_models":  status["alert_models"],
        "checked_at":    status["checked_at"],
    }


if __name__ == "__main__":
    summary = get_latency_summary()
    print(f"Checked at: {summary['checked_at']}")
    for m in summary["models"]:
        fresh_tag = " ⚡ FRESH" if m["is_fresh"] else (" ⚠ STALE" if m["is_stale"] else "")
        ago = f"{m['hours_ago']:.1f}h ago" if m["hours_ago"] is not None else "unknown"
        print(f"  {m['name']:6} {ago:12}{fresh_tag}")
    if summary["latency_alert"]:
        print(f"\n🔔 LATENCY ALERT: {', '.join(summary['alert_models'])} 剛更新！")

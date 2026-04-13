"""
consensus.py
三模型共識計算：GFS + ECMWF + ICON via Open-Meteo
對每個市場取得三組預測，計算共識分數和 ensemble spread。
"""

import requests
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from typing import Optional

# Open-Meteo 各模型 API
GFS_API     = "https://api.open-meteo.com/v1/gfs"
ECMWF_API   = "https://api.open-meteo.com/v1/ecmwf"
ICON_API    = "https://api.open-meteo.com/v1/dwd-icon"
ENSEMBLE_API = "https://ensemble-api.open-meteo.com/v1/ensemble"

COMMON_PARAMS = {
    "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,windspeed_10m_max,precipitation_probability_max",
    "timezone": "UTC",
    "forecast_days": 10,
}


def _safe_get(url: str, params: dict, timeout: int = 8) -> Optional[dict]:
    for attempt in range(2):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == 1:
                return None
            time.sleep(0.5)
    return None


def _extract_day(data: dict, target_date: date) -> Optional[dict]:
    """從 Open-Meteo daily response 中取出指定日期的數據。"""
    if not data or "daily" not in data:
        return None
    dates = data["daily"].get("time", [])
    target_str = target_date.isoformat()
    if target_str not in dates:
        return None
    idx = dates.index(target_str)
    d = data["daily"]
    return {
        "precip_mm":     _safe_idx(d, "precipitation_sum", idx),
        "precip_prob":   _safe_idx(d, "precipitation_probability_max", idx),
        "temp_max_c":    _safe_idx(d, "temperature_2m_max", idx),
        "temp_min_c":    _safe_idx(d, "temperature_2m_min", idx),
        "wind_max_kph":  _safe_idx(d, "windspeed_10m_max", idx),
    }


def _safe_idx(d: dict, key: str, idx: int):
    vals = d.get(key)
    if vals and idx < len(vals):
        return vals[idx]
    return None


def _rain_prob_from_day(day: dict) -> float:
    """從 daily 數據估算降雨概率（0-1）。"""
    if day is None:
        return 0.5
    # 優先用 precip_prob（百分比 0-100）
    pp = day.get("precip_prob")
    if pp is not None:
        return pp / 100.0
    # fallback：用降水量
    mm = day.get("precip_mm") or 0.0
    if mm >= 10:  return 0.90
    if mm >= 5:   return 0.75
    if mm >= 1:   return 0.55
    if mm > 0:    return 0.35
    return 0.10


def _temp_exceed_prob(day: dict, thresh_c: float) -> float:
    """估算最高溫超過閾值的概率（0-1）。"""
    if day is None:
        return 0.5
    t_max = day.get("temp_max_c")
    t_min = day.get("temp_min_c")
    if t_max is None:
        return 0.5
    # 用 max 和 min 的線性插值
    if t_max >= thresh_c:
        return 0.85
    if t_min is not None and t_min >= thresh_c:
        return 0.55
    gap = thresh_c - t_max
    if gap <= 2:   return 0.35
    if gap <= 5:   return 0.15
    return 0.05


def _wind_exceed_prob(day: dict, thresh_kph: float) -> float:
    """估算最大風速超過閾值的概率（0-1）。"""
    if day is None:
        return 0.5
    w = day.get("wind_max_kph")
    if w is None:
        return 0.5
    if w >= thresh_kph:     return 0.85
    gap = thresh_kph - w
    if gap <= 10:  return 0.35
    if gap <= 25:  return 0.15
    return 0.05


def _model_prob_from_day(day: dict, event_type: str, threshold: Optional[dict], direction: str) -> Optional[float]:
    """從單一模型的 daily 數據計算 YES 概率。"""
    if day is None:
        return None

    if event_type in ("rain", "flood"):
        prob = _rain_prob_from_day(day)
        if direction == "below":
            prob = 1.0 - prob
        return round(prob, 3)

    elif event_type == "snow":
        t_max = day.get("temp_max_c")
        rain_p = _rain_prob_from_day(day)
        if t_max is not None and t_max < 2:
            prob = rain_p * 0.9
        elif t_max is not None and t_max < 5:
            prob = rain_p * 0.5
        else:
            prob = rain_p * 0.1
        return round(prob, 3)

    elif event_type == "temperature":
        if threshold is None:
            return 0.50
        thresh_c = None
        if threshold.get("value_c") is not None:
            thresh_c = threshold["value_c"]
        elif threshold.get("value_f") is not None:
            thresh_c = (threshold["value_f"] - 32) * 5 / 9
        if thresh_c is None:
            return None
        prob = _temp_exceed_prob(day, thresh_c)
        if direction == "below":
            prob = 1.0 - prob
        return round(prob, 3)

    elif event_type in ("wind", "storm"):
        thresh_mph = (threshold or {}).get("value_mph", 25.0)
        thresh_kph = thresh_mph * 1.60934
        prob = _wind_exceed_prob(day, thresh_kph)
        if direction == "below":
            prob = 1.0 - prob
        return round(prob, 3)

    elif event_type == "sunny":
        prob = 1.0 - _rain_prob_from_day(day)
        return round(prob, 3)

    return None


def get_consensus(
    lat: float,
    lon: float,
    target_date: date,
    event_type: str,
    threshold: Optional[dict],
    direction: str = "any",
) -> dict:
    """
    三模型各自查詢，回傳共識結果：
    {
        "gfs":          float | None,
        "ecmwf":        float | None,
        "icon":         float | None,
        "consensus":    float | None,   # 有效模型的平均值
        "models_agree": int,            # 幾個模型方向一致
        "spread":       float | None,   # max - min（不確定度）
        "conviction":   str,            # "high" / "medium" / "low"
    }
    """
    base_params = {**COMMON_PARAMS, "latitude": lat, "longitude": lon}

    # 並行查詢 GFS / ECMWF / ICON
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_gfs   = ex.submit(_safe_get, GFS_API,   base_params)
        f_ecmwf = ex.submit(_safe_get, ECMWF_API, base_params)
        f_icon  = ex.submit(_safe_get, ICON_API,  base_params)
    gfs_raw   = f_gfs.result()
    ecmwf_raw = f_ecmwf.result()
    icon_raw  = f_icon.result()

    gfs_day   = _extract_day(gfs_raw,   target_date)
    ecmwf_day = _extract_day(ecmwf_raw, target_date)
    icon_day  = _extract_day(icon_raw,  target_date)

    gfs_p   = _model_prob_from_day(gfs_day,   event_type, threshold, direction)
    ecmwf_p = _model_prob_from_day(ecmwf_day, event_type, threshold, direction)
    icon_p  = _model_prob_from_day(icon_day,  event_type, threshold, direction)

    probs = [p for p in [gfs_p, ecmwf_p, icon_p] if p is not None]
    consensus = round(sum(probs) / len(probs), 3) if probs else None

    # 計算方向一致性（都 > 0.5 或都 < 0.5）
    models_agree = 0
    if len(probs) >= 2:
        above = sum(1 for p in probs if p > 0.5)
        below = sum(1 for p in probs if p <= 0.5)
        models_agree = max(above, below)

    spread = round(max(probs) - min(probs), 3) if len(probs) >= 2 else None

    # Conviction: 3 models agree + spread < 0.15 = high
    if models_agree == 3 and (spread or 1.0) < 0.15:
        conviction = "high"
    elif models_agree >= 2:
        conviction = "medium"
    else:
        conviction = "low"

    return {
        "gfs":          gfs_p,
        "ecmwf":        ecmwf_p,
        "icon":         icon_p,
        "consensus":    consensus,
        "models_agree": models_agree,
        "spread":       spread,
        "conviction":   conviction,
        "model_count":  len(probs),
    }

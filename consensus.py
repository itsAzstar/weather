"""
consensus.py
多模型共識計算：GFS + ECMWF + ICON (Open-Meteo) + NWS + Met.no
對每個市場取得五組預測，計算共識分數和 ensemble spread。
"""

import json as _json
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from typing import Optional

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode    = ssl.CERT_NONE

from fetcher_nws   import get_nws_forecast
from fetcher_metno import get_metno_forecast

# ── Model data cache (keyed by (lat, lon, date)) ─────────────────────────────
# Raw API responses are the same for all temperature buckets of the same city+date.
# Caching avoids 10-15x duplicate API calls per scan (one per bucket).
_model_data_cache: dict = {}
_model_data_lock = threading.Lock()
MODEL_DATA_CACHE_TTL = 25 * 60  # 25 minutes — matches weather cache TTL

# ── Open-Meteo rate limiter ───────────────────────────────────────────────────
# Caps concurrent in-flight requests to api.open-meteo.com at 3.
# Without this, cache expiry triggers 8 outer workers × 3 models = 24 simultaneous
# requests → burst triggers 429 → all markets skip for the full cache window.
_openmeteo_sem = threading.Semaphore(3)

# Open-Meteo 各模型 API
GFS_API     = "https://api.open-meteo.com/v1/gfs"
ECMWF_API   = "https://api.open-meteo.com/v1/ecmwf"
ICON_API    = "https://api.open-meteo.com/v1/dwd-icon"
ENSEMBLE_API = "https://ensemble-api.open-meteo.com/v1/ensemble"

COMMON_PARAMS = {
    "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,windspeed_10m_max,precipitation_probability_max",
    # Use local timezone so temperature_2m_max covers the LOCAL calendar day.
    # "UTC" caused Tokyo/Seoul/etc. max temps to span the wrong local-day window.
    "timezone": "auto",
    "forecast_days": 10,
}


def _safe_get(url: str, params: dict, timeout: int = 8) -> Optional[dict]:
    # consensus.py runs inside asyncio.to_thread (sync context) so urllib is fine.
    full_url = url + "?" + urllib.parse.urlencode(params) if params else url
    for attempt in range(2):
        try:
            with urllib.request.urlopen(full_url, timeout=timeout, context=_SSL_CTX) as resp:
                return _json.loads(resp.read().decode())
        except Exception:
            if attempt == 1:
                return None
            time.sleep(0.5)
    return None


def _safe_get_openmeteo(url: str, params: dict, timeout: int = 8) -> Optional[dict]:
    """Rate-limited wrapper for api.open-meteo.com — max 3 concurrent in-flight."""
    with _openmeteo_sem:
        return _safe_get(url, params, timeout=timeout)


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


def _sigma_for_days_ahead(days_ahead: Optional[int]) -> float:
    """
    GFS daily-max temperature RMSE scales with forecast lead time.
    Empirical (NOAA verification stats, 2m temp daily max):
        day 0 (today, 00Z run):   ~1.0 °C
        day 1:                    ~1.5 °C
        day 2:                    ~2.0 °C
        day 3+:                   ~2.5 °C

    Using a fixed sigma=2.5 for 1°C-wide buckets mathematically caps peak
    P(YES) at ~16% regardless of forecast accuracy — which was the root
    cause of systematic BUY NO losses on narrow-bucket markets (the
    2026-04-18 Brier investigation traced 28.9% win rate to this).

    Linear scale: sigma = max(0.8, 0.6 * days_ahead + 0.8)
        day 0: 0.8 → 1° bucket peak ~47%
        day 1: 1.4 → peak ~28%
        day 2: 2.0 → peak ~20%
        day 3: 2.6 → peak ~15%
    Floor 0.8 °C to prevent overconfidence when model is effectively
    observing live temperature and sigma → 0.
    """
    if days_ahead is None or days_ahead < 0:
        return 2.5  # fallback: old behaviour for safety
    return max(0.8, 0.6 * days_ahead + 0.8)


def _temp_exceed_prob(day: dict, thresh_c: float, days_ahead: Optional[int] = None) -> float:
    """
    P(daily_max >= thresh_c) via normal CDF with sigma scaled by forecast
    lead time (see _sigma_for_days_ahead).

    Previous step function (0.85/0.55/0.35/0.15/0.05) produced systematic
    miscalibration: every forecast that beat the threshold was clipped to
    exactly 0.85 regardless of margin, so a 1 °C buffer and a 10 °C buffer
    got the same probability. NCDF scales smoothly with the gap.
    """
    if day is None:
        return 0.5
    t_max = day.get("temp_max_c")
    if t_max is None:
        return 0.5
    import math
    sigma = _sigma_for_days_ahead(days_ahead)
    z = (t_max - thresh_c) / sigma
    p = 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
    return max(0.02, min(0.98, round(p, 3)))


def _wind_exceed_prob(day: dict, thresh_kph: float) -> float:
    """
    P(max_wind >= thresh_kph) via normal CDF with sigma = 8 kph
    (approx GFS wind-max RMSE at 1-3 day lead). Replaces step function
    to avoid systematic miscalibration near the threshold.
    """
    if day is None:
        return 0.5
    w = day.get("wind_max_kph")
    if w is None:
        return 0.5
    import math
    sigma = 8.0
    z = (w - thresh_kph) / sigma
    p = 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
    return max(0.02, min(0.98, round(p, 3)))


def _model_prob_from_day(day: dict, event_type: str, threshold: Optional[dict], direction: str,
                         days_ahead: Optional[int] = None) -> Optional[float]:
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
        # Defensive: if threshold carries BOTH a lo and hi bound, this is
        # really a bucket market that was mislabeled as "temperature".
        # Route to bucket NCDF logic instead of exceed logic — the latter
        # systematically reports ~0.85 for any threshold the city typically
        # beats, producing Brier > 0.35 on narrow-bucket markets.
        has_lo = threshold.get("lo_c") is not None or threshold.get("lo_f") is not None
        has_hi = threshold.get("hi_c") is not None or threshold.get("hi_f") is not None
        if has_lo and has_hi:
            lo_c = threshold.get("lo_c")
            if lo_c is None and threshold.get("lo_f") is not None:
                lo_c = (threshold["lo_f"] - 32) * 5 / 9
            hi_c = threshold.get("hi_c")
            if hi_c is None and threshold.get("hi_f") is not None:
                hi_c = (threshold["hi_f"] - 32) * 5 / 9
            return _model_prob_from_day(
                day, "temperature_bucket",
                {"lo_c": lo_c, "hi_c": hi_c},
                direction,
                days_ahead=days_ahead,
            )

        thresh_c = None
        if threshold.get("value_c") is not None:
            thresh_c = threshold["value_c"]
        elif threshold.get("value_f") is not None:
            thresh_c = (threshold["value_f"] - 32) * 5 / 9
        if thresh_c is None:
            return None
        prob = _temp_exceed_prob(day, thresh_c, days_ahead=days_ahead)
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

    elif event_type == "temperature_bucket":
        # Temperature bucket: P(lo_c ≤ max_temp ≤ hi_c) via normal CDF
        # Uses same approach as comparator._temp_bucket_model_prob but without obs adjustment
        import math
        def _ncdf(x: float) -> float:
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

        lo_c = (threshold or {}).get("lo_c")
        hi_c = (threshold or {}).get("hi_c")
        t_max = day.get("temp_max_c")
        if t_max is None or (lo_c is None and hi_c is None):
            return None
        # Sigma scales with forecast lead time so narrow 1°C buckets aren't
        # mathematically capped at ~16% peak probability. See _sigma_for_days_ahead.
        sigma = _sigma_for_days_ahead(days_ahead)
        lo_cdf = _ncdf((lo_c - t_max) / sigma) if lo_c is not None else 0.0
        hi_cdf = _ncdf((hi_c - t_max) / sigma) if hi_c is not None else 1.0
        return max(0.01, min(0.99, round(hi_cdf - lo_cdf, 3)))

    return None


def _nws_day_to_synthetic(nws: Optional[dict]) -> Optional[dict]:
    """Convert NWS forecast dict to synthetic 'day' format for _model_prob_from_day."""
    if nws is None:
        return None
    return {
        "temp_max_c":   nws.get("temp_max_c"),
        "temp_min_c":   nws.get("temp_min_c"),
        "precip_prob":  (nws.get("precip_pct") or 0) * 100,  # 0-100
        "precip_mm":    None,
        "wind_max_kph": None,
    }


def _metno_day_to_synthetic(metno: Optional[dict]) -> Optional[dict]:
    """Convert Met.no forecast dict to synthetic 'day' format for _model_prob_from_day."""
    if metno is None:
        return None
    return {
        "temp_max_c":   metno.get("temp_max_c"),
        "temp_min_c":   metno.get("temp_min_c"),
        "precip_prob":  (metno.get("precip_pct") or 0) * 100,  # 0-100
        "precip_mm":    None,
        "wind_max_kph": None,
    }


def _fetch_model_data(lat: float, lon: float, target_date: date) -> dict:
    """
    Fetch raw daily data from all 5 models for (lat, lon, date) with caching.
    Returns dict with gfs_day, ecmwf_day, icon_day, nws_day, metno_day.
    Cached for 25 min so all temperature buckets of the same city+date share one fetch.
    """
    key = (round(lat, 3), round(lon, 3), target_date.isoformat())
    now = time.time()

    with _model_data_lock:
        entry = _model_data_cache.get(key)
        if entry is not None:
            data, ts = entry
            if now - ts < MODEL_DATA_CACHE_TTL:
                return data

    base_params = {**COMMON_PARAMS, "latitude": lat, "longitude": lon}
    with ThreadPoolExecutor(max_workers=5) as ex:
        f_gfs   = ex.submit(_safe_get_openmeteo, GFS_API,   base_params)
        f_ecmwf = ex.submit(_safe_get_openmeteo, ECMWF_API, base_params)
        f_icon  = ex.submit(_safe_get_openmeteo, ICON_API,  base_params)
        f_nws   = ex.submit(get_nws_forecast,   lat, lon, target_date)
        f_metno = ex.submit(get_metno_forecast, lat, lon, target_date)

    data = {
        "gfs_day":   _extract_day(f_gfs.result(),   target_date),
        "ecmwf_day": _extract_day(f_ecmwf.result(), target_date),
        "icon_day":  _extract_day(f_icon.result(),  target_date),
        "nws_day":   _nws_day_to_synthetic(f_nws.result()),
        "metno_day": _metno_day_to_synthetic(f_metno.result()),
    }

    with _model_data_lock:
        _model_data_cache[key] = (data, time.time())

    return data


def get_consensus(
    lat: float,
    lon: float,
    target_date: date,
    event_type: str,
    threshold: Optional[dict],
    direction: str = "any",
    days_ahead: Optional[int] = None,
) -> dict:
    """
    五模型各自查詢，回傳共識結果：
    {
        "gfs":          float | None,
        "ecmwf":        float | None,
        "icon":         float | None,
        "nws":          float | None,   # US only
        "metno":        float | None,   # global
        "consensus":    float | None,   # 有效模型的平均值
        "models_agree": int,            # 幾個模型方向一致
        "spread":       float | None,   # max - min（不確定度）
        "conviction":   str,            # "high" / "medium" / "low"
        "sources_used": list[str],      # names of sources that returned data
    }
    """
    # Use cached model data — all buckets for the same city+date share one API fetch.
    model_data = _fetch_model_data(lat, lon, target_date)
    gfs_day   = model_data["gfs_day"]
    ecmwf_day = model_data["ecmwf_day"]
    icon_day  = model_data["icon_day"]
    nws_day   = model_data["nws_day"]
    metno_day = model_data["metno_day"]

    gfs_p   = _model_prob_from_day(gfs_day,   event_type, threshold, direction, days_ahead=days_ahead)
    ecmwf_p = _model_prob_from_day(ecmwf_day, event_type, threshold, direction, days_ahead=days_ahead)
    icon_p  = _model_prob_from_day(icon_day,  event_type, threshold, direction, days_ahead=days_ahead)
    nws_p   = _model_prob_from_day(nws_day,   event_type, threshold, direction, days_ahead=days_ahead)
    metno_p = _model_prob_from_day(metno_day, event_type, threshold, direction, days_ahead=days_ahead)

    # Build sources list
    source_map = [
        ("GFS",    gfs_p),
        ("ECMWF",  ecmwf_p),
        ("ICON",   icon_p),
        ("NWS",    nws_p),
        ("Met.no", metno_p),
    ]
    sources_used = [name for name, p in source_map if p is not None]
    all_probs    = [p for _, p in source_map if p is not None]

    consensus = round(sum(all_probs) / len(all_probs), 3) if all_probs else None

    # 計算方向一致性（都 > 0.5 或都 < 0.5）
    models_agree = 0
    if len(all_probs) >= 2:
        above = sum(1 for p in all_probs if p > 0.5)
        below = sum(1 for p in all_probs if p <= 0.5)
        models_agree = max(above, below)

    spread = round(max(all_probs) - min(all_probs), 3) if len(all_probs) >= 2 else None

    # Conviction: majority models agree + spread < 0.15 = high
    total_sources = len(all_probs)
    majority = (total_sources // 2) + 1
    if models_agree >= majority and (spread or 1.0) < 0.15:
        conviction = "high"
    elif models_agree >= max(2, majority - 1):
        conviction = "medium"
    else:
        conviction = "low"

    return {
        "gfs":          gfs_p,
        "ecmwf":        ecmwf_p,
        "icon":         icon_p,
        "nws":          nws_p,
        "metno":        metno_p,
        "consensus":    consensus,
        "models_agree": models_agree,
        "spread":       spread,
        "conviction":   conviction,
        "model_count":  len(all_probs),
        "sources_used": sources_used,
    }

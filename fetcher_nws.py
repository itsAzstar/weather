"""
fetcher_nws.py
Fetches NWS (National Weather Service) daily forecasts for US locations.
Returns None for non-US locations (NWS returns 404 outside CONUS/AK/HI).
"""

import json as _json
import ssl
import threading
import time
import urllib.error
import urllib.request
from datetime import date, datetime
from typing import Optional

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode    = ssl.CERT_NONE


def _urllib_get(url: str, headers: dict, timeout: int = 15) -> Optional[dict]:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
            return _json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise  # let caller handle 404 specifically
        return None
    except Exception:
        return None

NWS_HEADERS = {"User-Agent": "WeatherArb/2.0 (contact@example.com)"}

# Cache: (lat_rounded, lon_rounded, date_str) → (result, timestamp)
_nws_cache: dict[tuple, tuple[Optional[dict], float]] = {}
_nws_lock  = threading.Lock()
NWS_CACHE_TTL = 60 * 60  # 60 minutes


def _cache_key(lat: float, lon: float, target_date: date) -> tuple:
    # Round to 2 decimal places to allow nearby coordinates to share cache
    return (round(lat, 2), round(lon, 2), target_date.isoformat())


def _f_to_c(f: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return round((f - 32) * 5 / 9, 1)


def get_nws_forecast(lat: float, lon: float, target_date: date) -> Optional[dict]:
    """
    Fetch NWS daily forecast for the given coordinates and target date.

    Step 1: GET /points/{lat},{lon} → get forecast URL
    Step 2: GET {forecast_url} → parse daily periods for target_date

    Returns:
      {temp_max_c, temp_min_c, precip_pct, short_forecast, wind_speed, source: "NWS"}
    or None if unavailable (non-US coords, network error, etc.)

    Note: NWS forecast endpoint returns temperatures in Fahrenheit.
    """
    key = _cache_key(lat, lon, target_date)
    now = time.time()

    with _nws_lock:
        entry = _nws_cache.get(key)
        if entry is not None:
            result, ts = entry
            if now - ts < NWS_CACHE_TTL:
                return result

    try:
        result = _fetch_nws(lat, lon, target_date)
    except Exception as e:
        print(f"[NWS] Unexpected error for ({lat},{lon}) {target_date}: {e}")
        result = None

    with _nws_lock:
        _nws_cache[key] = (result, now)
    return result


def _fetch_nws(lat: float, lon: float, target_date: date) -> Optional[dict]:
    """Internal: fetch and parse NWS forecast (no cache logic)."""
    # Step 1: Points API → get gridpoint / forecast URL
    points_url = f"https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
    print(f"[NWS] Points lookup: {points_url}")
    try:
        points_data = _urllib_get(points_url, NWS_HEADERS, timeout=15)
        if points_data is None:
            print(f"[NWS] Location ({lat},{lon}) not in NWS coverage or request failed")
            return None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"[NWS] Location ({lat},{lon}) not in NWS coverage (404) — skipping")
        else:
            print(f"[NWS] HTTP error on points lookup: {e}")
        return None
    except Exception as e:
        print(f"[NWS] Error on points lookup: {e}")
        return None

    forecast_url = (points_data.get("properties") or {}).get("forecast")
    if not forecast_url:
        print(f"[NWS] No forecast URL in points response for ({lat},{lon})")
        return None

    # Step 2: Daily forecast
    print(f"[NWS] Fetching forecast: {forecast_url}")
    try:
        fc_data = _urllib_get(forecast_url, NWS_HEADERS, timeout=15)
        if fc_data is None:
            print(f"[NWS] Failed to fetch forecast from {forecast_url}")
            return None
    except Exception as e:
        print(f"[NWS] Error on forecast fetch: {e}")
        return None

    periods = (fc_data.get("properties") or {}).get("periods") or []
    if not periods:
        print(f"[NWS] No forecast periods in response")
        return None

    # NWS returns daytime/nighttime periods. Find the ones matching target_date.
    target_str = target_date.isoformat()
    day_period   = None
    night_period = None

    for period in periods:
        start_raw = period.get("startTime", "")
        # startTime is ISO8601 like "2026-04-15T06:00:00-05:00"
        period_date = start_raw[:10]
        if period_date == target_str:
            if period.get("isDaytime", True):
                day_period = period
            else:
                night_period = period

    # Fallback: match by name (e.g., "Monday", "Monday Night")
    if day_period is None and night_period is None:
        # Try to find closest matching period by date
        for period in periods:
            start_raw = period.get("startTime", "")
            period_date = start_raw[:10]
            if period_date == target_str:
                if period.get("isDaytime", True) and day_period is None:
                    day_period = period
                elif night_period is None:
                    night_period = period

    if day_period is None and night_period is None:
        print(f"[NWS] No periods found for {target_str}")
        return None

    # Extract temp (NWS forecast returns Fahrenheit)
    temp_max_c = None
    temp_min_c = None

    if day_period:
        t = day_period.get("temperature")
        unit = day_period.get("temperatureUnit", "F")
        if t is not None:
            t_c = _f_to_c(float(t)) if unit == "F" else round(float(t), 1)
            temp_max_c = t_c

    if night_period:
        t = night_period.get("temperature")
        unit = night_period.get("temperatureUnit", "F")
        if t is not None:
            t_c = _f_to_c(float(t)) if unit == "F" else round(float(t), 1)
            temp_min_c = t_c

    # Short forecast text and precip
    short_forecast = (day_period or night_period or {}).get("shortForecast", "")
    wind_speed     = (day_period or {}).get("windSpeed", "")  # e.g., "10 to 15 mph"

    # Estimate precip probability from NWS period
    precip_pct = (day_period or night_period or {}).get("probabilityOfPrecipitation", {})
    if isinstance(precip_pct, dict):
        precip_pct = precip_pct.get("value")
    if precip_pct is not None:
        precip_pct = float(precip_pct) / 100.0
    else:
        # Infer from short forecast text
        sf_lower = short_forecast.lower()
        if "thunder" in sf_lower or "heavy rain" in sf_lower:
            precip_pct = 0.80
        elif "rain" in sf_lower or "showers" in sf_lower:
            precip_pct = 0.60
        elif "chance" in sf_lower and ("rain" in sf_lower or "shower" in sf_lower):
            precip_pct = 0.40
        elif "slight chance" in sf_lower:
            precip_pct = 0.20
        elif "sunny" in sf_lower or "clear" in sf_lower or "mostly sunny" in sf_lower:
            precip_pct = 0.05
        else:
            precip_pct = 0.15

    result = {
        "temp_max_c":     temp_max_c,
        "temp_min_c":     temp_min_c,
        "precip_pct":     round(precip_pct, 3) if precip_pct is not None else None,
        "short_forecast": short_forecast,
        "wind_speed":     wind_speed,
        "source":         "NWS",
    }
    print(f"[NWS] ({lat},{lon}) {target_str}: max={temp_max_c}°C min={temp_min_c}°C precip={precip_pct}")
    return result

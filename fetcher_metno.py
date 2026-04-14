"""
fetcher_metno.py
Fetches forecasts from the Norwegian Meteorological Institute (Met.no)
LocationForecast 2.0 API. Works globally, unlike NWS.
"""

import threading
import time
import requests
from datetime import date, datetime, timezone
from typing import Optional

METNO_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
METNO_HEADERS = {
    "User-Agent": "WeatherArb/2.0 github.com/Azstarrr/weather",
}

# Cache: (lat_rounded, lon_rounded, date_str) → (result, timestamp)
_metno_cache: dict[tuple, tuple[Optional[dict], float]] = {}
_metno_lock  = threading.Lock()
METNO_CACHE_TTL = 60 * 60  # 60 minutes


def _cache_key(lat: float, lon: float, target_date: date) -> tuple:
    return (round(lat, 2), round(lon, 2), target_date.isoformat())


def get_metno_forecast(lat: float, lon: float, target_date: date) -> Optional[dict]:
    """
    Fetch Met.no compact locationforecast for the given coordinates and date.

    Returns:
      {temp_max_c, temp_min_c, precip_pct, source: "Met.no"}
    or None on failure.

    Met.no returns temperatures in Celsius.
    """
    key = _cache_key(lat, lon, target_date)
    now = time.time()

    with _metno_lock:
        entry = _metno_cache.get(key)
        if entry is not None:
            result, ts = entry
            if now - ts < METNO_CACHE_TTL:
                return result

    try:
        result = _fetch_metno(lat, lon, target_date)
    except Exception as e:
        print(f"[MetNo] Unexpected error for ({lat},{lon}) {target_date}: {e}")
        result = None

    with _metno_lock:
        _metno_cache[key] = (result, now)
    return result


def _fetch_metno(lat: float, lon: float, target_date: date) -> Optional[dict]:
    """Internal: fetch and parse Met.no hourly data for target_date."""
    params = {"lat": round(lat, 4), "lon": round(lon, 4)}
    print(f"[MetNo] Fetching forecast for ({lat},{lon}) on {target_date} ...")
    try:
        resp = requests.get(METNO_URL, params=params, headers=METNO_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        print(f"[MetNo] Timeout for ({lat},{lon})")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"[MetNo] HTTP error for ({lat},{lon}): {e}")
        return None
    except Exception as e:
        print(f"[MetNo] Error for ({lat},{lon}): {e}")
        return None

    try:
        timeseries = (data.get("properties") or {}).get("timeseries") or []
        if not timeseries:
            print(f"[MetNo] No timeseries in response")
            return None

        target_str = target_date.isoformat()
        temps = []
        precip_probs = []
        precip_amounts = []

        for entry in timeseries:
            ts_str = entry.get("time", "")
            # Met.no returns times in UTC ISO8601: "2026-04-15T06:00:00Z"
            entry_date = ts_str[:10]
            if entry_date != target_str:
                continue

            instant = (entry.get("data") or {}).get("instant") or {}
            details = (instant.get("details") or {})

            t = details.get("air_temperature")
            if t is not None:
                temps.append(float(t))

            # next_1_hours or next_6_hours for precip
            for window in ("next_1_hours", "next_6_hours"):
                window_data = (entry.get("data") or {}).get(window) or {}
                wd = (window_data.get("details") or {})
                pp = wd.get("probability_of_precipitation")
                if pp is not None:
                    precip_probs.append(float(pp) / 100.0)
                pa = wd.get("precipitation_amount")
                if pa is not None:
                    precip_amounts.append(float(pa))
                break  # only use one window per hour

        if not temps:
            print(f"[MetNo] No temperature data for {target_str}")
            return None

        temp_max_c = round(max(temps), 1)
        temp_min_c = round(min(temps), 1)

        # Precip probability: max over the day (most pessimistic = most informative)
        if precip_probs:
            precip_pct = round(max(precip_probs), 3)
        elif precip_amounts:
            # Fallback: estimate from amounts
            total = sum(precip_amounts)
            if total >= 10:   precip_pct = 0.85
            elif total >= 5:  precip_pct = 0.70
            elif total >= 1:  precip_pct = 0.50
            elif total > 0:   precip_pct = 0.30
            else:             precip_pct = 0.05
        else:
            precip_pct = 0.10

        result = {
            "temp_max_c": temp_max_c,
            "temp_min_c": temp_min_c,
            "precip_pct": precip_pct,
            "source":     "Met.no",
        }
        print(f"[MetNo] ({lat},{lon}) {target_str}: max={temp_max_c}°C min={temp_min_c}°C precip={precip_pct}")
        return result

    except Exception as e:
        print(f"[MetNo] Error parsing response for ({lat},{lon}): {e}")
        return None

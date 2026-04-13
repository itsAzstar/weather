"""
fetcher_weather.py
Fetches ensemble weather forecasts from Open-Meteo for a given location and date.
"""

import requests
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional

ENSEMBLE_API   = "https://ensemble-api.open-meteo.com/v1/ensemble"
FORECAST_API   = "https://api.open-meteo.com/v1/forecast"    # fast daily, no ensemble

# Supported city → (lat, lon) lookup
CITY_COORDS: dict[str, tuple[float, float]] = {
    # Americas
    "new york":    (40.7128, -74.0060),
    "nyc":         (40.7128, -74.0060),
    "new york city": (40.7128, -74.0060),
    "los angeles": (34.0522, -118.2437),
    "la":          (34.0522, -118.2437),
    "chicago":     (41.8781, -87.6298),
    "houston":     (29.7604, -95.3698),
    "dallas":      (32.7767, -96.7970),
    "atlanta":     (33.7490, -84.3880),
    "miami":       (25.7617, -80.1918),
    "seattle":     (47.6062, -122.3321),
    "san francisco": (37.7749, -122.4194),
    "sf":          (37.7749, -122.4194),
    "denver":      (39.7392, -104.9903),
    "boston":      (42.3601, -71.0589),
    "phoenix":     (33.4484, -112.0740),
    "las vegas":   (36.1699, -115.1398),
    "minneapolis": (44.9778, -93.2650),
    "detroit":     (42.3314, -83.0458),
    "philadelphia": (39.9526, -75.1652),
    "washington dc": (38.9072, -77.0369),
    "austin":      (30.2672, -97.7431),
    "toronto":     (43.6532, -79.3832),
    "sao paulo":   (-23.5505, -46.6333),
    "buenos aires": (-34.6118, -58.4173),
    "mexico city": (19.4326, -99.1332),
    "panama city": (8.9943, -79.5188),
    # Europe
    "london":      (51.5074, -0.1278),
    "paris":       (48.8566, 2.3522),
    "berlin":      (52.5200, 13.4050),
    "madrid":      (40.4168, -3.7038),
    "rome":        (41.9028, 12.4964),
    "amsterdam":   (52.3676, 4.9041),
    "moscow":      (55.7558, 37.6173),
    "milan":       (45.4642, 9.1900),
    "warsaw":      (52.2297, 21.0122),
    "helsinki":    (60.1699, 24.9384),
    "istanbul":    (41.0082, 28.9784),
    "ankara":      (39.9334, 32.8597),
    "munich":      (48.1351, 11.5820),
    # Asia
    "tokyo":       (35.6762, 139.6503),
    "beijing":     (39.9042, 116.4074),
    "shanghai":    (31.2304, 121.4737),
    "hong kong":   (22.3193, 114.1694),
    "seoul":       (37.5665, 126.9780),
    "singapore":   (1.3521, 103.8198),
    "dubai":       (25.2048, 55.2708),
    "mumbai":      (19.0760, 72.8777),
    "bangkok":     (13.7563, 100.5018),
    "jakarta":     (-6.2088, 106.8456),
    "taipei":      (25.0330, 121.5654),
    "kuala lumpur": (3.1390, 101.6869),
    "kl":          (3.1390, 101.6869),
    "shenzhen":    (22.5431, 114.0579),
    "wuhan":       (30.5928, 114.3055),
    "chengdu":     (30.5728, 104.0668),
    "chongqing":   (29.4316, 106.9123),
    "lucknow":     (26.8467, 80.9462),
    "busan":       (35.1796, 129.0756),
    "tel aviv":    (32.0853, 34.7818),
    "jeddah":      (21.5433, 39.1728),
    "cairo":       (30.0444, 31.2357),
    "nairobi":     (-1.2921, 36.8219),
    # Oceania / Africa
    "sydney":      (-33.8688, 151.2093),
    "melbourne":   (-37.8136, 144.9631),
    "auckland":    (-36.8485, 174.7633),
    "wellington":  (-41.2865, 174.7762),
    "johannesburg": (-26.2041, 28.0473),
    "cape town":   (-33.9249, 18.4241),
    "lagos":       (6.5244, 3.3792),
}


def resolve_location(location: str) -> Optional[tuple[float, float]]:
    """
    Resolve a city name string to (lat, lon).
    Returns None if not found.
    """
    key = location.strip().lower()
    if key in CITY_COORDS:
        return CITY_COORDS[key]
    # Partial match fallback
    for city, coords in CITY_COORDS.items():
        if key in city or city in key:
            return coords
    return None


def _safe_get(url: str, params: dict, retries: int = 2, timeout: int = 8) -> Optional[dict]:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            print(f"    [HTTP error] {e} — attempt {attempt + 1}/{retries}")
        except requests.exceptions.ConnectionError as e:
            print(f"    [Connection error] {e} — attempt {attempt + 1}/{retries}")
        except requests.exceptions.Timeout:
            print(f"    [Timeout] — attempt {attempt + 1}/{retries}")
        except Exception as e:
            print(f"    [Unexpected] {e} — attempt {attempt + 1}/{retries}")
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    return None


def fetch_daily_forecast(
    lat: float,
    lon: float,
    target_date: date,
) -> Optional[dict]:
    """
    Fast path: fetch a single day's forecast from the regular Open-Meteo API
    (not ensemble). Returns the same schema as fetch_ensemble_forecast.
    No rate limit issues.  Used as primary source; ensemble is fallback.
    """
    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": target_date.isoformat(),
        "end_date":   (target_date + timedelta(days=1)).isoformat(),
        "daily": (
            "temperature_2m_max,temperature_2m_min,"
            "precipitation_sum,precipitation_probability_max,"
            "windspeed_10m_max"
        ),
        "wind_speed_unit":      "mph",
        "temperature_unit":     "celsius",
        "precipitation_unit":   "mm",
        "timezone":             "UTC",
    }
    data = _safe_get(FORECAST_API, params, retries=2, timeout=6)
    if data is None:
        return None

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    target_str = target_date.isoformat()
    if target_str not in dates:
        return None

    idx = dates.index(target_str)
    def _v(key):
        arr = daily.get(key)
        return arr[idx] if arr and idx < len(arr) else None

    temp_max = _v("temperature_2m_max")
    temp_min = _v("temperature_2m_min")
    precip   = _v("precipitation_sum") or 0.0
    precip_p = _v("precipitation_probability_max")
    wind_max = _v("windspeed_10m_max")

    if precip_p is not None:
        rain_prob = max(0.0, min(1.0, precip_p / 100.0))
    elif precip > 0:
        if precip >= 10: rain_prob = 0.90
        elif precip >= 5: rain_prob = 0.75
        elif precip >= 1: rain_prob = 0.55
        else: rain_prob = 0.35
    else:
        rain_prob = 0.10

    return {
        "lat":              lat,
        "lon":              lon,
        "target_date":      target_date.isoformat(),
        "rain_probability": round(rain_prob, 3),
        "total_precip_mm":  round(precip, 2),
        "temp_min_c":       round(temp_min, 1) if temp_min is not None else None,
        "temp_max_c":       round(temp_max, 1) if temp_max is not None else None,
        "temp_mean_c":      round((temp_max + temp_min) / 2, 1) if temp_max is not None and temp_min is not None else None,
        "temp_max_f":       round(temp_max * 9 / 5 + 32, 1) if temp_max is not None else None,
        "wind_max_mph":     round(wind_max, 1) if wind_max is not None else None,
        "wind_mean_mph":    None,
        "ensemble_members": 0,
        "data_points":      1,
        "source":           "daily_forecast",
    }


def fetch_ensemble_forecast(
    lat: float,
    lon: float,
    target_date: date,
) -> Optional[dict]:
    """
    Fetch ensemble forecast from Open-Meteo for the given coordinates and date.
    Returns a dict with precipitation, temperature, and wind data,
    or None on failure.
    """
    # Fetch a window around the target date
    start = target_date
    end = target_date + timedelta(days=1)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": (
            "precipitation,temperature_2m,windspeed_10m,precipitation_probability"
        ),
        "models": "icon_seamless",   # reliable global ensemble
        "wind_speed_unit": "mph",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "timezone": "UTC",
    }

    data = _safe_get(ENSEMBLE_API, params)
    if data is None:
        # Fallback: try gfs_seamless model
        params["models"] = "gfs_seamless"
        data = _safe_get(ENSEMBLE_API, params)

    if data is None:
        return None

    return _process_ensemble_data(data, target_date, lat, lon)


def _process_ensemble_data(data: dict, target_date: date, lat: float, lon: float) -> dict:
    """
    Parse raw Open-Meteo ensemble JSON into a clean summary dict.
    """
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])

    # Collect all variable arrays (may have multiple members like precipitation_member01 etc.)
    precip_arrays = []
    temp_arrays = []
    wind_arrays = []
    precip_prob_array = []

    for key, values in hourly.items():
        if key == "time":
            continue
        if key.startswith("precipitation") and "probability" not in key:
            precip_arrays.append(values)
        elif key.startswith("precipitation_probability"):
            precip_prob_array = values
        elif key.startswith("temperature"):
            temp_arrays.append(values)
        elif key.startswith("windspeed") or key.startswith("wind_speed"):
            wind_arrays.append(values)

    # Filter indices to the target date
    target_str = target_date.isoformat()
    target_indices = [i for i, t in enumerate(times) if t and t.startswith(target_str)]

    if not target_indices:
        # Fallback — use all data
        target_indices = list(range(len(times)))

    def slice_and_flatten(arrays, indices):
        vals = []
        for arr in arrays:
            for i in indices:
                if i < len(arr) and arr[i] is not None:
                    vals.append(float(arr[i]))
        return vals

    precip_vals = slice_and_flatten(precip_arrays, target_indices)
    temp_vals = slice_and_flatten(temp_arrays, target_indices)
    wind_vals = slice_and_flatten(wind_arrays, target_indices)

    # Precipitation probability from deterministic forecast if available
    if precip_prob_array:
        prob_vals = [precip_prob_array[i] for i in target_indices if i < len(precip_prob_array) and precip_prob_array[i] is not None]
    else:
        prob_vals = []

    # -- Rain probability --
    # Method 1: use precipitation_probability field directly
    if prob_vals:
        rain_prob = max(float(v) for v in prob_vals) / 100.0
    elif precip_vals:
        # Method 2: fraction of ensemble members / hours with precip > 0.1 mm
        rain_prob = sum(1 for v in precip_vals if v > 0.1) / len(precip_vals)
    else:
        rain_prob = 0.3  # fallback default

    # Clamp to valid range
    rain_prob = max(0.0, min(1.0, rain_prob))

    # -- Temperature summary --
    if temp_vals:
        temp_min = min(temp_vals)
        temp_max = max(temp_vals)
        temp_mean = sum(temp_vals) / len(temp_vals)
    else:
        temp_min = temp_max = temp_mean = None

    # -- Wind summary --
    if wind_vals:
        wind_max = max(wind_vals)
        wind_mean = sum(wind_vals) / len(wind_vals)
    else:
        wind_max = wind_mean = None

    # -- Total precipitation (mean across ensemble members if multiple, else sum over day) --
    if precip_vals and len(precip_arrays) > 1:
        # multiple members: average max daily total per member
        member_totals = []
        for arr in precip_arrays:
            member_day = [float(arr[i]) for i in target_indices if i < len(arr) and arr[i] is not None]
            if member_day:
                member_totals.append(sum(member_day))
        total_precip_mm = sum(member_totals) / len(member_totals) if member_totals else 0.0
    elif precip_vals:
        total_precip_mm = sum(precip_vals)
    else:
        total_precip_mm = 0.0

    return {
        "lat": lat,
        "lon": lon,
        "target_date": target_date.isoformat(),
        "rain_probability": rain_prob,
        "total_precip_mm": round(total_precip_mm, 2),
        "temp_min_c": round(temp_min, 1) if temp_min is not None else None,
        "temp_max_c": round(temp_max, 1) if temp_max is not None else None,
        "temp_mean_c": round(temp_mean, 1) if temp_mean is not None else None,
        "temp_max_f": round(temp_max * 9 / 5 + 32, 1) if temp_max is not None else None,
        "wind_max_mph": round(wind_max, 1) if wind_max is not None else None,
        "wind_mean_mph": round(wind_mean, 1) if wind_mean is not None else None,
        "ensemble_members": len(precip_arrays) if precip_arrays else 0,
        "data_points": len(target_indices),
    }


def get_weather_for_location_date(
    location: str,
    target_date: date,
) -> Optional[dict]:
    """
    High-level wrapper: resolve location name → coordinates → fetch forecast.
    Uses fast daily forecast first; falls back to ensemble if daily fails.
    """
    coords = resolve_location(location)
    if coords is None:
        print(f"    [Weather] Unknown location: '{location}'")
        return None
    lat, lon = coords
    print(f"    [Weather] Fetching forecast for {location} ({lat:.2f},{lon:.2f}) on {target_date}")
    # Try fast daily API first (no rate limit, <1s)
    result = fetch_daily_forecast(lat, lon, target_date)
    if result is None:
        # Fall back to ensemble
        result = fetch_ensemble_forecast(lat, lon, target_date)
    if result:
        result["location"] = location
    return result


def get_rain_probability(weather: dict) -> float:
    """Extract rain probability (0–1) from a weather forecast dict."""
    return weather.get("rain_probability", 0.0)


def get_temp_exceed_probability(weather: dict, threshold_c: float) -> float:
    """
    Estimate probability that max temperature exceeds `threshold_c` degrees Celsius.
    Uses a simple linear interpolation between temp_min and temp_max.
    """
    t_min = weather.get("temp_min_c")
    t_max = weather.get("temp_max_c")
    if t_min is None or t_max is None:
        return 0.5

    if t_max <= threshold_c:
        return 0.0
    if t_min >= threshold_c:
        return 1.0

    # Linear: what fraction of the [min, max] range is above threshold?
    span = t_max - t_min
    if span == 0:
        return 0.0
    above = t_max - threshold_c
    return max(0.0, min(1.0, above / span))


def get_wind_exceed_probability(weather: dict, threshold_mph: float) -> float:
    """
    Estimate probability that max wind speed exceeds `threshold_mph`.
    Simple heuristic based on mean and max wind.
    """
    w_max = weather.get("wind_max_mph")
    w_mean = weather.get("wind_mean_mph")
    if w_max is None:
        return 0.3  # unknown

    if w_max <= threshold_mph:
        return 0.05
    if w_mean is not None and w_mean >= threshold_mph:
        return 0.90

    # Interpolate
    if w_mean is not None and w_max > w_mean:
        frac = (w_max - threshold_mph) / (w_max - w_mean + 1e-9)
        return max(0.05, min(0.95, frac))

    return 0.50


if __name__ == "__main__":
    import json
    today = date.today()
    for city in ["London", "New York", "Tokyo", "Dubai"]:
        w = get_weather_for_location_date(city, today + timedelta(days=1))
        if w:
            print(json.dumps(w, indent=2))
        else:
            print(f"Failed to fetch weather for {city}")

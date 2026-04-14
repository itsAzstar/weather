"""
comparator.py
Compares Polymarket weather market prices against Open-Meteo ensemble model probabilities.
Flags opportunities where the divergence exceeds the threshold (default 5%).
Also fetches station observations and applies obs-adjustment for intraday markets.
"""

import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone, timedelta
from typing import Optional

from fetcher_weather import (
    get_weather_for_location_date,
    get_rain_probability,
    get_temp_exceed_probability,
    get_wind_exceed_probability,
)
from fetcher_stations import resolve_station, get_station_obs

MAX_DAYS_AHEAD = 10       # Only consider markets resolving within this window
EDGE_THRESHOLD = 0.05     # 5% minimum divergence to flag as opportunity

# ── Timezone offset helpers (lon-based approximation) ────────────────────────
# Major US cities hardcoded for accuracy; others use lon/15 heuristic.
_CITY_UTC_OFFSET: dict[str, float] = {
    # Eastern US (UTC-5/-4)
    "new york": -5, "nyc": -5, "new york city": -5, "boston": -5,
    "philadelphia": -5, "washington dc": -5, "miami": -5, "atlanta": -5,
    "charlotte": -5, "orlando": -5, "tampa": -5, "baltimore": -5,
    "pittsburgh": -5, "cleveland": -5, "detroit": -5, "columbus": -5,
    "indianapolis": -5, "nashville": -5, "memphis": -5, "jacksonville": -5,
    "raleigh": -5, "richmond": -5,
    # Central US (UTC-6/-5)
    "chicago": -6, "minneapolis": -6, "milwaukee": -6, "st. louis": -6,
    "kansas city": -6, "omaha": -6, "des moines": -6, "dallas": -6,
    "houston": -6, "austin": -6, "san antonio": -6, "new orleans": -6,
    "oklahoma city": -6, "little rock": -6,
    # Mountain US (UTC-7/-6)
    "denver": -7, "phoenix": -7, "salt lake city": -7, "albuquerque": -7,
    "el paso": -7, "boise": -7, "tucson": -7,
    # Pacific US (UTC-8/-7)
    "los angeles": -8, "la": -8, "san francisco": -8, "sf": -8,
    "san jose": -8, "seattle": -8, "portland": -8, "las vegas": -8,
    "sacramento": -8, "san diego": -8, "spokane": -8, "reno": -8,
    # Alaska (UTC-9)
    "anchorage": -9, "juneau": -9, "fairbanks": -9,
    # Canada
    "toronto": -5, "montreal": -5, "ottawa": -5, "halifax": -4,
    "vancouver": -8, "calgary": -7, "edmonton": -7, "winnipeg": -6,
    # Europe
    "london": 0, "lisbon": 0, "reykjavik": 0,
    "paris": 1, "berlin": 1, "madrid": 1, "rome": 1, "amsterdam": 1,
    "brussels": 1, "zurich": 1, "vienna": 1, "prague": 1, "warsaw": 1,
    "budapest": 1, "stockholm": 2, "oslo": 1, "copenhagen": 1,
    "helsinki": 2, "athens": 2, "istanbul": 3,
    "moscow": 3, "dubai": 4,
    # Asia
    "tel aviv": 2, "cairo": 2, "jeddah": 3,
    "mumbai": 5.5, "delhi": 5.5, "kolkata": 5.5, "lucknow": 5.5,
    "bangkok": 7, "jakarta": 7, "kuala lumpur": 8, "kl": 8,
    "singapore": 8, "hong kong": 8, "shenzhen": 8, "beijing": 8,
    "shanghai": 8, "taipei": 8, "seoul": 9, "tokyo": 9, "osaka": 9,
    # Oceania
    "sydney": 10, "melbourne": 10, "brisbane": 10, "auckland": 12,
    "wellington": 12,
    # Americas (non-US)
    "sao paulo": -3, "rio de janeiro": -3, "buenos aires": -3,
    "mexico city": -6, "panama city": -5, "bogota": -5, "lima": -5,
    "santiago": -4,
    # Africa
    "nairobi": 3, "lagos": 1, "johannesburg": 2, "cape town": 2,
}


def _get_utc_offset(city: str, lon: float) -> float:
    """
    Return approximate UTC offset in hours for a city.
    Uses hardcoded table for major cities; falls back to lon/15 heuristic.
    Does NOT account for DST (within 1h accuracy is sufficient).
    """
    key = city.strip().lower()
    if key in _CITY_UTC_OFFSET:
        return _CITY_UTC_OFFSET[key]
    # Partial match
    for city_key, offset in _CITY_UTC_OFFSET.items():
        if key in city_key or city_key in key:
            return offset
    # Fallback: longitude-based approximation (lon/15)
    return round(lon / 15.0)


def _time_remaining_hours(city: str, lon: float, target_date: date) -> float:
    """
    Estimate how many hours remain in the local day at target_date.
    Returns 0.0 if the day has already passed, 24.0 if it hasn't started.
    """
    utc_offset = _get_utc_offset(city, lon)
    now_utc = datetime.now(timezone.utc)
    # Local time approximation (no DST)
    local_now_h = now_utc.hour + now_utc.minute / 60.0 + utc_offset
    # Wrap to 0-24
    local_now_h = local_now_h % 24

    today_utc = now_utc.date()
    # If target_date is today (local), hours_remaining = 24 - local_hour
    # If target_date is in the future, full 24 hours remain
    # If target_date is in the past, 0 hours remain
    days_diff = (target_date - today_utc).days

    # Account for timezone shift (if local day is already tomorrow)
    local_date_offset = int((now_utc.hour + utc_offset) // 24)
    effective_today = today_utc + timedelta(days=local_date_offset)
    days_diff_local = (target_date - effective_today).days

    if days_diff_local < 0:
        return 0.0
    elif days_diff_local > 0:
        return 24.0
    else:
        # Same local day
        return max(0.0, 24.0 - local_now_h)

# ── Weather result cache (persistent across scan runs, 30-min TTL) ────────────
# Stores (result, timestamp) so weather isn't re-fetched every 30-min scan.
import time as _time

_weather_cache: dict = {}          # key → (result_or_future, timestamp_or_None)
_weather_cache_lock = threading.Lock()
WEATHER_CACHE_TTL = 25 * 60       # 25 minutes — refresh before server cache expires


def _get_weather_cached(location: str, target_date: date) -> Optional[dict]:
    """Fetch weather with cross-run cache (25-min TTL) and per-run deduplication."""
    import concurrent.futures as cf
    key = (location.lower(), target_date.isoformat())
    now = _time.time()

    with _weather_cache_lock:
        entry = _weather_cache.get(key)
        if entry is not None:
            val, ts = entry
            if ts is None:
                # In-flight Future — wait for it
                is_first = False
                fut = val
            elif now - ts < WEATHER_CACHE_TTL:
                # Valid cached result
                return val
            else:
                # Stale — re-fetch
                fut = cf.Future()
                _weather_cache[key] = (fut, None)
                is_first = True
        else:
            fut = cf.Future()
            _weather_cache[key] = (fut, None)
            is_first = True

    if is_first:
        try:
            result = get_weather_for_location_date(location, target_date)
            with _weather_cache_lock:
                _weather_cache[key] = (result, _time.time())
            fut.set_result(result)
            return result
        except Exception as e:
            with _weather_cache_lock:
                _weather_cache.pop(key, None)
            fut.set_exception(e)
            return None

    try:
        return fut.result(timeout=30)
    except Exception:
        return None


def clear_weather_cache():
    """Evict only stale entries (keep fresh ones for next scan)."""
    now = _time.time()
    with _weather_cache_lock:
        stale = [k for k, (v, ts) in _weather_cache.items()
                 if ts is None or (now - ts) >= WEATHER_CACHE_TTL]
        for k in stale:
            del _weather_cache[k]


# ── Normal distribution helpers ───────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """CDF of the standard normal distribution using math.erf."""
    return (1.0 + math.erf(x / math.sqrt(2))) / 2.0


def _temp_bucket_model_prob(
    weather: dict,
    temp_bucket: dict,
    days_ahead: int,
    obs_temp_c: Optional[float] = None,
    time_remaining_hours: Optional[float] = None,
) -> tuple[Optional[float], bool]:
    """
    Compute the probability (0-1) that the actual max temperature falls in
    the given bucket, using a Normal distribution centred on the forecast max temp.

    Forecast uncertainty σ grows with lead time:
      day 0-1 → 1.5 °C, +0.4 °C per extra day.

    If current obs + time_remaining are provided, applies obs-adjustment:
    - obs_weight = 1 - time_remaining/24  (0 at midnight, ~1 at end of day)
    - If obs already exceeds bucket hi_c → P(YES) collapses toward 0
    - If obs is below bucket lo_c and < 3h left → P(YES) collapses toward 0
    - If obs is inside bucket and < 6h left → P(YES) is boosted

    Returns: (probability, obs_adjusted_bool)
    """
    mu = weather.get("temp_max_c")
    if mu is None:
        return None, False

    sigma = max(1.5, 1.5 + (days_ahead - 1) * 0.4)
    lo_c = temp_bucket.get("lo_c")
    hi_c = temp_bucket.get("hi_c")

    lo_cdf = _norm_cdf((lo_c - mu) / sigma) if lo_c is not None else 0.0
    hi_cdf = _norm_cdf((hi_c - mu) / sigma) if hi_c is not None else 1.0
    forecast_prob = max(0.0, min(1.0, hi_cdf - lo_cdf))

    # Obs adjustment
    obs_adjusted = False
    if obs_temp_c is not None and time_remaining_hours is not None:
        obs_weight = max(0.0, 1.0 - time_remaining_hours / 24.0)

        if obs_weight > 0.05:  # Only adjust if meaningful time has passed today
            obs_adjusted = True

            # Current obs EXCEEDS bucket ceiling → max for today is already above hi_c
            # This means YES is IMPOSSIBLE (temp_max > hi_c → outside bucket above)
            if hi_c is not None and obs_temp_c > hi_c:
                # Obs already above bucket top → bucket is definitely missed from above
                # P(YES) collapses toward 0 proportional to obs_weight
                adjusted = forecast_prob * (1.0 - obs_weight)
                forecast_prob = max(0.01, round(adjusted, 3))

            # Current obs is already ABOVE bucket floor with < 4h left → likely inside or above
            elif lo_c is not None and obs_temp_c >= lo_c and time_remaining_hours < 4.0:
                if hi_c is None or obs_temp_c < hi_c:
                    # Obs is inside bucket and day nearly over → boost P(YES)
                    boosted = forecast_prob + (1.0 - forecast_prob) * obs_weight * 0.7
                    forecast_prob = min(0.97, round(boosted, 3))

            # Obs is BELOW bucket floor and very little time left → won't reach bucket
            elif lo_c is not None and obs_temp_c < lo_c and time_remaining_hours < 3.0:
                # Temperature unlikely to rise enough to enter bucket
                gap_c = lo_c - obs_temp_c
                if gap_c > 3.0:  # very large gap → nearly impossible
                    adjusted = forecast_prob * (1.0 - obs_weight * 0.9)
                    forecast_prob = max(0.01, round(adjusted, 3))
                elif gap_c > 1.0:
                    adjusted = forecast_prob * (1.0 - obs_weight * 0.5)
                    forecast_prob = max(0.02, round(adjusted, 3))

    return max(0.0, min(1.0, round(forecast_prob, 3))), obs_adjusted


def _celsius_to_fahrenheit(c: float) -> float:
    return c * 9 / 5 + 32


def _calculate_model_probability(
    weather: dict,
    event_type: str,
    threshold: Optional[dict],
    direction: str,
) -> Optional[float]:
    """
    Given a weather forecast dict and parsed market metadata,
    compute the model's probability for the YES outcome.

    Returns a float [0, 1] or None if the event type cannot be evaluated.
    """
    if event_type == "rain" or event_type == "flood":
        prob = get_rain_probability(weather)
        # If there's a precipitation threshold, adjust
        if threshold and threshold.get("value_mm") is not None:
            thresh_mm = threshold["value_mm"]
            total_mm = weather.get("total_precip_mm", 0.0)
            # Rough adjustment: if model expects more precip than threshold, increase prob
            if thresh_mm > 0:
                ratio = total_mm / thresh_mm
                if ratio >= 2.0:
                    prob = min(0.95, prob * 1.2)
                elif ratio <= 0.3:
                    prob = max(0.05, prob * 0.6)
        if direction == "below":
            prob = 1.0 - prob
        return round(prob, 3)

    elif event_type == "snow":
        # Snow probability is correlated with rain in cold conditions
        temp_max = weather.get("temp_max_c")
        rain_prob = get_rain_probability(weather)
        if temp_max is not None and temp_max < 2:
            snow_prob = rain_prob * 0.9
        elif temp_max is not None and temp_max < 5:
            snow_prob = rain_prob * 0.5
        else:
            snow_prob = rain_prob * 0.1
        return round(snow_prob, 3)

    elif event_type == "temperature":
        if threshold is None:
            # No threshold: just return ~50% (unknown)
            return 0.50

        # Determine threshold in Celsius
        if threshold.get("value_c") is not None:
            thresh_c = threshold["value_c"]
        elif threshold.get("value_f") is not None:
            thresh_c = (threshold["value_f"] - 32) * 5 / 9
        else:
            return None

        prob = get_temp_exceed_probability(weather, thresh_c)
        if direction == "below":
            prob = 1.0 - prob
        return round(prob, 3)

    elif event_type == "wind" or event_type == "storm":
        if threshold and threshold.get("value_mph") is not None:
            thresh_mph = threshold["value_mph"]
        else:
            # Default: any notable wind event (>25 mph)
            thresh_mph = 25.0

        prob = get_wind_exceed_probability(weather, thresh_mph)
        if direction == "below":
            prob = 1.0 - prob
        return round(prob, 3)

    elif event_type == "humidity":
        # No good humidity data from this API; return None
        return None

    elif event_type == "sunny":
        # Invert rain probability
        prob = 1.0 - get_rain_probability(weather)
        return round(prob, 3)

    else:
        # Unknown event type
        return None


def _get_station_data(location: str, weather: Optional[dict]) -> dict:
    """
    Resolve ICAO station for a location and fetch current obs.
    Returns a dict with station_icao, station_name, obs_temp_c, obs_temp_f, obs_age_min.
    All fields may be None if unavailable.
    """
    result = {
        "station_icao":  None,
        "station_name":  None,
        "obs_temp_c":    None,
        "obs_temp_f":    None,
        "obs_age_min":   None,
    }
    try:
        icao = resolve_station(location)
        if icao is None:
            return result
        result["station_icao"] = icao
        obs = get_station_obs(icao)
        if obs:
            t_c = obs.get("temp_c")
            result["station_name"] = obs.get("station_name") or icao
            result["obs_temp_c"]   = t_c
            result["obs_temp_f"]   = round(t_c * 9 / 5 + 32, 1) if t_c is not None else None
            result["obs_age_min"]  = obs.get("obs_age_minutes")
        else:
            result["station_name"] = icao
    except Exception as e:
        print(f"[Stations] Error resolving station for '{location}': {e}")
    return result


def compare_market(market: dict) -> Optional[dict]:
    """
    Evaluate a single parsed Polymarket market against weather model data.

    Returns a result dict with:
      - All market fields
      - model_probability
      - edge (difference between model and market)
      - action (BUY YES / BUY NO / SKIP)
      - is_opportunity (bool)
      - station_icao, station_name, obs_temp_c, obs_temp_f, obs_age_min (station obs)
      - time_remaining_hours (hours left in local day)
      - obs_adjusted (bool: whether obs changed the probability)
    Or None if the market cannot be evaluated.
    """
    parsed = market.get("parsed", {})
    location = parsed.get("location")
    target_date_str = parsed.get("target_date")
    event_type = parsed.get("event_type", "unknown")
    threshold = parsed.get("threshold")
    direction = parsed.get("direction", "any")

    if not location or not target_date_str:
        return None

    try:
        target_date = date.fromisoformat(target_date_str)
    except (ValueError, TypeError):
        return None

    # Compute today fresh each call (avoid stale module-level date if server runs overnight)
    today = date.today()
    # Skip markets too far in the future or already past
    days_ahead = (target_date - today).days
    if days_ahead < 0:
        return {
            **market,
            "model_probability": None,
            "weather_data": None,
            "edge": None,
            "action": "SKIP (past)",
            "is_opportunity": False,
            "skip_reason": f"Market resolved {abs(days_ahead)} day(s) ago",
            "days_ahead": days_ahead,
        }
    if days_ahead > MAX_DAYS_AHEAD:
        return {
            **market,
            "model_probability": None,
            "weather_data": None,
            "edge": None,
            "action": "SKIP (too far ahead)",
            "is_opportunity": False,
            "skip_reason": f"Resolves in {days_ahead} days (>{MAX_DAYS_AHEAD} day window)",
        }

    # Fetch weather (cached per location+date to avoid duplicate API calls)
    weather = _get_weather_cached(location, target_date)
    _ = today  # suppress unused warning
    if weather is None:
        return {
            **market,
            "model_probability": None,
            "weather_data": None,
            "edge": None,
            "action": "SKIP (no weather data)",
            "is_opportunity": False,
            "skip_reason": "Could not fetch weather forecast",
        }

    # ── Resolve station + fetch obs (for today's / near-term markets) ─────────
    station_data = {"station_icao": None, "station_name": None,
                    "obs_temp_c": None, "obs_temp_f": None, "obs_age_min": None}
    time_remaining_hours = 24.0  # default: full day ahead
    obs_adjusted = False

    if days_ahead <= 1:  # Only fetch obs for today/tomorrow markets
        station_data = _get_station_data(location, weather)
        # Compute lon for time-remaining calculation
        lon = weather.get("lon") or 0.0
        time_remaining_hours = _time_remaining_hours(location, lon, target_date)

    # ── Temperature bucket markets (from Polymarket daily temp events) ─────────
    if market.get("market_subtype") == "temperature_bucket":
        temp_bucket = market.get("temp_bucket", {})
        model_prob, obs_adjusted = _temp_bucket_model_prob(
            weather, temp_bucket, days_ahead,
            obs_temp_c=station_data.get("obs_temp_c"),
            time_remaining_hours=time_remaining_hours,
        )
        if model_prob is None:
            return {
                **market,
                "model_probability": None,
                "weather_data": weather,
                "edge": None,
                "action": "SKIP (no temp data)",
                "is_opportunity": False,
                "skip_reason": "No temperature data from weather model",
                **station_data,
                "time_remaining_hours": time_remaining_hours,
                "obs_adjusted": obs_adjusted,
            }
        market_price_yes = market.get("market_price_yes")
        if market_price_yes is None:
            return {**market, "model_probability": model_prob, "weather_data": weather,
                    "edge": None, "action": "SKIP (no market price)", "is_opportunity": False,
                    "skip_reason": "Market price unavailable",
                    **station_data,
                    "time_remaining_hours": time_remaining_hours,
                    "obs_adjusted": obs_adjusted}
        try:
            market_price_yes = float(market_price_yes)
        except (ValueError, TypeError):
            return None
        edge = round(model_prob - market_price_yes, 4)
        abs_edge = abs(edge)
        is_opp = abs_edge > EDGE_THRESHOLD
        action = ("BUY YES" if edge > 0 else "BUY NO") if is_opp else "HOLD"
        return {
            **market,
            "model_probability": model_prob,
            "weather_data": weather,
            "weather_temp_max_c": weather.get("temp_max_c"),
            "weather_temp_max_f": weather.get("temp_max_f"),
            "edge": edge,
            "abs_edge": abs_edge,
            "action": action,
            "is_opportunity": is_opp,
            "skip_reason": None,
            "days_ahead": days_ahead,
            **station_data,
            "time_remaining_hours": round(time_remaining_hours, 2),
            "obs_adjusted": obs_adjusted,
        }

    # Calculate model probability (standard binary markets)
    model_prob = _calculate_model_probability(weather, event_type, threshold, direction)
    if model_prob is None:
        return {
            **market,
            "model_probability": None,
            "weather_data": weather,
            "edge": None,
            "action": "SKIP (unsupported event type)",
            "is_opportunity": False,
            "skip_reason": f"Cannot model '{event_type}' event",
            **station_data,
            "time_remaining_hours": round(time_remaining_hours, 2),
            "obs_adjusted": obs_adjusted,
        }

    # Get market price for YES
    market_price_yes = market.get("market_price_yes")
    if market_price_yes is None:
        return {
            **market,
            "model_probability": model_prob,
            "weather_data": weather,
            "edge": None,
            "action": "SKIP (no market price)",
            "is_opportunity": False,
            "skip_reason": "Market price unavailable",
            **station_data,
            "time_remaining_hours": round(time_remaining_hours, 2),
            "obs_adjusted": obs_adjusted,
        }

    try:
        market_price_yes = float(market_price_yes)
    except (ValueError, TypeError):
        return None

    # Calculate edge
    edge = model_prob - market_price_yes
    abs_edge = abs(edge)
    is_opportunity = abs_edge > EDGE_THRESHOLD

    if not is_opportunity:
        action = "SKIP"
    elif edge > 0:
        action = "BUY YES"   # Model says more likely than market thinks
    else:
        action = "BUY NO"    # Model says less likely than market thinks

    return {
        **market,
        "model_probability": model_prob,
        "weather_data": weather,
        "edge": round(edge, 4),
        "abs_edge": round(abs_edge, 4),
        "action": action,
        "is_opportunity": is_opportunity,
        "skip_reason": None,
        "days_ahead": days_ahead,
        **station_data,
        "time_remaining_hours": round(time_remaining_hours, 2),
        "obs_adjusted": obs_adjusted,
    }


def compare_all_markets(markets: list[dict]) -> list[dict]:
    """
    Run comparisons for all markets in parallel and return sorted results.
    Opportunities are sorted by abs_edge descending.
    Non-opportunities follow.
    """
    clear_weather_cache()   # fresh cache per scan run
    results = []
    skipped = 0

    # Parallel weather fetches — 8 workers; cache deduplicates same city+date
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_market = {executor.submit(compare_market, m): m for m in markets}
        for future in as_completed(future_to_market):
            try:
                result = future.result()
            except Exception:
                skipped += 1
                continue
            if result is None:
                skipped += 1
                continue
            results.append(result)

    print(f"\n[Comparator] Evaluated {len(results)} markets, skipped {skipped} (no location/date/price)")

    # Sort: opportunities first (by edge size), then the rest
    opportunities = sorted(
        [r for r in results if r.get("is_opportunity")],
        key=lambda r: r.get("abs_edge", 0),
        reverse=True,
    )
    no_edge = [r for r in results if not r.get("is_opportunity")]

    return opportunities + no_edge


if __name__ == "__main__":
    # Quick self-test with one mock market
    from parser_market import parse_market

    mock = {
        "question": "Will it rain in London on April 10, 2026?",
        "condition_id": "test-london-rain",
        "market_price_yes": 0.35,
        "market_price_no": 0.65,
        "end_date": "2026-04-10T23:59:00Z",
        "description": "",
        "url": "https://polymarket.com/event/test",
        "source": "mock",
    }
    parse_market(mock, reference_date=date(2026, 4, 9))
    result = compare_market(mock)
    if result:
        print(f"Market: {result['question']}")
        print(f"Market price YES: {result['market_price_yes']:.0%}")
        print(f"Model probability: {result['model_probability']:.0%}")
        print(f"Edge: {result['edge']:+.1%}")
        print(f"Action: {result['action']}")

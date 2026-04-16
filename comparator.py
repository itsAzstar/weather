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
from fetcher_wu import get_wu_temp_cached

MAX_DAYS_AHEAD = 10       # Only consider markets resolving within this window
EDGE_THRESHOLD = 0.08     # 8% minimum divergence to flag as opportunity (5% had too much noise)

# ── Latency arbitrage zone ────────────────────────────────────────────────────
# Markets within this many hours of resolution are dominated by HFT bots
# polling WU every few minutes.  Our GFS/ECMWF forecast cannot compete on
# LATENCY — but WU's current obs IS the resolution anchor.  Treat these
# markets differently: if WU high matches a bucket we have HIGH confidence;
# if WU high is outside we know the answer. Flag them but with a note.
LATENCY_ARB_HOURS = 12.0

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

# ── Weather result cache (persistent across scan runs, 2-hour TTL) ────────────
# Stores (result, timestamp) so weather isn't re-fetched too often.
# Stale-while-revalidate: if refetch fails (e.g. rate limit), keep using the
# last known good value for up to WEATHER_STALE_TTL (6 h) rather than returning
# None and skipping every market.
import time as _time

_weather_cache: dict = {}          # key → (result_or_future, timestamp_or_None)
_weather_cache_lock = threading.Lock()
WEATHER_CACHE_TTL   = 2 * 60 * 60   # 2 hours — weather doesn't change minute-to-minute
WEATHER_STALE_TTL   = 6 * 60 * 60   # 6 hours — stale-while-revalidate grace window


def _get_weather_cached(location: str, target_date: date) -> Optional[dict]:
    """Fetch weather with cross-run cache (2-h TTL) and per-run deduplication.
    Stale-while-revalidate: returns last-known-good if refetch fails and cache
    is younger than WEATHER_STALE_TTL (6 h), rather than dropping the market."""
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
                stale_val = None
            elif now - ts < WEATHER_CACHE_TTL:
                # Valid cached result
                return val
            else:
                # Stale — re-fetch, but keep val as fallback
                stale_val = val
                fut = cf.Future()
                _weather_cache[key] = (fut, None)
                is_first = True
        else:
            stale_val = None
            fut = cf.Future()
            _weather_cache[key] = (fut, None)
            is_first = True

    if is_first:
        try:
            result = get_weather_for_location_date(location, target_date)
        except Exception as e:
            result = None
            print(f"[Weather] Exception fetching {location} {target_date}: {e}")

        if result is not None:
            # Success — cache for the full TTL
            with _weather_cache_lock:
                _weather_cache[key] = (result, _time.time())
            fut.set_result(result)
            return result
        else:
            # Failed fetch — do NOT cache None (allow retry on next scan)
            # Stale-while-revalidate: serve last-known-good if available
            with _weather_cache_lock:
                if stale_val is not None:
                    # Restore stale entry; mark as expired so next scan retries
                    _weather_cache[key] = (stale_val, now - WEATHER_CACHE_TTL)
                    print(f"[Weather] Fetch failed for {location} {target_date} — serving stale data")
                    fut.set_result(stale_val)
                    return stale_val
                else:
                    _weather_cache.pop(key, None)
            fut.set_result(None)
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
    Compute P(T_max ∈ [lo_c, hi_c]) using the correct distribution given evidence.

    Two regimes:

    A. No intraday obs: T_max ~ N(forecast_mu, sigma²)
       sigma calibrated to Open-Meteo daily-max MAE (~1.2°C at d1, +0.5/day).

    B. Obs available: T_max | T_max >= T_now ~ TruncatedNormal(mu, sigma, lower=T_now)
       T_max is MONOTONICALLY INCREASING, so P(T_max < T_now) = 0 exactly.
       The truncated distribution concentrates probability to the right of T_now.

       Formula for P(lo ≤ T_max ≤ hi | T_max ≥ T_now):
         eff_lo = max(lo_c, T_now)
         Z      = 1 - Φ((T_now - mu) / sigma)   [normalising constant]
         prob   = [Φ((hi - mu)/σ) - Φ((eff_lo - mu)/σ)] / Z

       This correctly produces:
         - P = 0 when T_now > hi_c (obs already exceeds ceiling — permanent)
         - Probability shifts rightward as T_now rises within the bucket
         - NO artificial boost toward certainty while time remains — tail risk
           for temperature continuing to rise is always present until resolution

    sigma shrinks as the day progresses (less residual movement possible).
    """
    forecast_mu = weather.get("temp_max_c")
    if forecast_mu is None:
        return None, False

    lo_c = temp_bucket.get("lo_c")
    hi_c = temp_bucket.get("hi_c")

    # ── σ: ensemble spread is ground truth; heuristic is fallback ─────────────
    # Open-Meteo Ensemble API returns σ across member daily maxes.
    # This reflects actual atmospheric uncertainty — non-linear, not heuristic.
    # Only fall back to the lead-time heuristic when ensemble data is absent.
    ensemble_spread = weather.get("temp_spread_c")
    if ensemble_spread is not None and ensemble_spread > 0.2:
        sigma_base = float(ensemble_spread)
    else:
        # Fallback heuristic: Open-Meteo daily-max MAE ~1.2°C at d1, +0.5°C/day
        sigma_base = max(1.2, 1.2 + (days_ahead - 1) * 0.5)

    # ── Regime A: pure forecast, no obs ───────────────────────────────────────
    if obs_temp_c is None or time_remaining_hours is None:
        lo_cdf = _norm_cdf((lo_c - forecast_mu) / sigma_base) if lo_c is not None else 0.0
        hi_cdf = _norm_cdf((hi_c - forecast_mu) / sigma_base) if hi_c is not None else 1.0
        return round(max(0.0, min(1.0, hi_cdf - lo_cdf)), 3), False

    # ── Regime B: truncated normal given T_now ────────────────────────────────
    obs_adjusted = True
    hours_elapsed = max(0.0, 24.0 - time_remaining_hours)
    day_progress  = min(1.0, hours_elapsed / 24.0)

    # Physical hard constraint: T_max already exceeded bucket ceiling → dead.
    if hi_c is not None and obs_temp_c > hi_c:
        return 0.0, True

    # ── Bayesian μ update ─────────────────────────────────────────────────────
    # When T_now > forecast_mu the NWP was wrong about today's atmospheric state
    # (e.g. unexpected foehn, early cloud clearance).  Anchoring on the stale
    # forecast μ makes the truncated distribution pathologically right-skewed.
    #
    # Fix: estimate μ_adjusted by projecting T_now forward along the observed
    # warming slope to the expected daily peak (~14:00 local = hour 14).
    # Then blend prior (forecast) and posterior (trajectory-projected) with
    # weights that increase with both deviation magnitude and day progress.
    if obs_temp_c > forecast_mu and hours_elapsed > 1.0:
        t_min = weather.get("temp_min_c")
        if t_min is None:
            t_min = forecast_mu - 8.0   # typical diurnal range fallback

        # Observed warming slope from morning minimum to now
        slope = (obs_temp_c - t_min) / hours_elapsed   # °C/hour

        PEAK_HOUR = 14.0   # typical max temp ~14:00 local
        if hours_elapsed < PEAK_HOUR:
            hours_to_peak = PEAK_HOUR - hours_elapsed
            # Slope dampens as peak approaches (~45% of slope for remaining hours)
            t_projected = obs_temp_c + slope * hours_to_peak * 0.45
        else:
            # Post-peak: daily max is approximately T_now
            t_projected = obs_temp_c

        # Blend weights
        # - deviation weight: large anomaly = strong evidence of structural shift
        # - time weight: later in day = obs is more informative about final max
        dev_weight  = min(0.5, (obs_temp_c - forecast_mu) / 4.0)
        time_weight = min(0.5, day_progress)
        blend = min(0.85, dev_weight + time_weight)

        mu = (1.0 - blend) * forecast_mu + blend * t_projected
        print(f"[Comparator] Bayes μ: fc={forecast_mu:.1f} obs={obs_temp_c:.1f} "
              f"proj={t_projected:.1f} blend={blend:.2f} → μ={mu:.1f}")
    else:
        mu = forecast_mu

    # Residual sigma: ensemble spread × (1 - fraction of day that has solidified).
    # The ensemble σ still reflects full-day uncertainty; shrink proportionally
    # as T_now pins down the lower bound of T_max.
    sigma = max(0.5, sigma_base * (1.0 - day_progress * 0.55))

    # Normalising constant: P(T_max >= T_now) in N(μ, σ)
    z_a    = (obs_temp_c - mu) / sigma
    p_above = 1.0 - _norm_cdf(z_a)

    if p_above < 1e-7:
        # Numerical edge: T_now >> μ.  All mass concentrated at T_now.
        in_lo = lo_c is None or obs_temp_c >= lo_c
        in_hi = hi_c is None or obs_temp_c <= hi_c
        return (0.93 if (in_lo and in_hi) else 0.02), True

    # P(lo ≤ T_max ≤ hi | T_max ≥ T_now)
    eff_lo = max(lo_c, obs_temp_c) if lo_c is not None else obs_temp_c
    eff_hi = hi_c   # upper limit unchanged — temps CAN still rise

    z_lo = (eff_lo - mu) / sigma
    z_hi = (eff_hi - mu) / sigma if eff_hi is not None else math.inf
    p_hi = _norm_cdf(z_hi) if eff_hi is not None else 1.0
    p_lo = _norm_cdf(z_lo)

    p_in_range = p_hi - p_lo
    prob = p_in_range / p_above
    return round(max(0.0, min(0.97, prob)), 3), True


def _celsius_to_fahrenheit(c: float) -> float:
    return c * 9 / 5 + 32


def _estimate_half_spread(price_yes: float, days_ahead: int) -> float:
    """
    Estimate the one-way execution cost (half bid-ask spread) for a Polymarket
    weather market.  This is the cost you pay vs the mid-price when hitting
    a market order — the number that must be deducted from paper edge to get
    executable edge.

    Empirical Polymarket weather market spreads:
    - Deep in-the-money / out-of-the-money (>80% or <20%): wide spreads 8-15¢
    - Remote markets (day-ahead, competitive): 3-6¢ half-spread
    - Micro-cap / thin books: can be 10-25¢ wide

    We use price distance from 0.5 as proxy for liquidity / spread.
    Far-from-mid = less liquid = wider spread.
    """
    mid_distance = abs(price_yes - 0.5)

    # Base half-spread by lead time (day-ahead markets thinner than same-day)
    if days_ahead >= 2:
        base = 0.05    # typical day+2 half-spread
    elif days_ahead == 1:
        base = 0.04
    else:
        base = 0.03    # intraday: tighter (more activity)

    # Widen for extreme prices (less liquidity at tails)
    if mid_distance > 0.35:   # price < 0.15 or > 0.85
        spread_mult = 2.5
    elif mid_distance > 0.25:  # price < 0.25 or > 0.75
        spread_mult = 1.8
    elif mid_distance > 0.15:  # price < 0.35 or > 0.65
        spread_mult = 1.3
    else:
        spread_mult = 1.0

    return min(0.15, base * spread_mult)


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
        # Use Open-Meteo's native snowfall_sum → snow_probability field.
        # The old rain_prob × temp_coeff hack ignored atmospheric vertical profile.
        snow_prob = weather.get("snow_probability")
        if snow_prob is not None:
            return round(snow_prob, 3)
        # Fallback if older cache entry lacks the field
        temp_max  = weather.get("temp_max_c")
        rain_prob = get_rain_probability(weather)
        if temp_max is not None and temp_max < 2:
            return round(rain_prob * 0.9, 3)
        elif temp_max is not None and temp_max < 5:
            return round(rain_prob * 0.5, 3)
        return round(rain_prob * 0.1, 3)

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


def _get_station_data(location: str, weather: Optional[dict], target_date: Optional[date] = None) -> dict:
    """
    Resolve ICAO station for a location and fetch current obs + WU daily high.

    Returns a dict with:
      station_icao, station_name, obs_temp_c, obs_temp_f, obs_age_min
      wu_temp_c, wu_temp_f, wu_age_min, wu_source  (WU daily high — matches Polymarket resolution)
    All fields may be None if unavailable.
    """
    result = {
        "station_icao":  None,
        "station_name":  None,
        "obs_temp_c":    None,
        "obs_temp_f":    None,
        "obs_age_min":   None,
        "wu_temp_c":     None,
        "wu_temp_f":     None,
        "wu_age_min":    None,
        "wu_source":     None,
    }
    try:
        icao = resolve_station(location)
        if icao is None:
            return result
        result["station_icao"] = icao

        # ── METAR / NWS obs (current temperature, intraday tracking) ─────────
        obs = get_station_obs(icao)
        if obs:
            t_c = obs.get("temp_c")
            result["station_name"] = obs.get("station_name") or icao
            result["obs_temp_c"]   = t_c
            result["obs_temp_f"]   = round(t_c * 9 / 5 + 32, 1) if t_c is not None else None
            result["obs_age_min"]  = obs.get("obs_age_minutes")
        else:
            result["station_name"] = icao

        # ── WU daily high (matches Polymarket's official resolution source) ───
        # Fetch for today and tomorrow markets.  WU may not have data for future
        # dates, but will have today's running high immediately.
        if target_date is not None:
            try:
                wu = get_wu_temp_cached(icao, target_date)
                if wu:
                    result["wu_temp_c"]  = wu.get("wu_temp_c")
                    result["wu_temp_f"]  = wu.get("wu_temp_f")
                    result["wu_age_min"] = wu.get("wu_age_min")
                    result["wu_source"]  = wu.get("wu_source")
            except Exception as wu_e:
                print(f"[WU] Skipping WU fetch for {location}: {wu_e}")

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
        station_data = _get_station_data(location, weather, target_date=target_date)
        # Compute lon for time-remaining calculation
        lon = weather.get("lon") or 0.0
        time_remaining_hours = _time_remaining_hours(location, lon, target_date)

    # ── Temperature bucket markets (from Polymarket daily temp events) ─────────
    if market.get("market_subtype") == "temperature_bucket":
        temp_bucket = market.get("temp_bucket", {})

        # ── WU authoritative temperature ──────────────────────────────────────
        # WU daily high is what Polymarket resolves against.
        # If WU has data, use it as the observation instead of METAR.
        # METAR and WU can diverge 1-2°C — fatal for narrow buckets.
        wu_temp_c   = station_data.get("wu_temp_c")
        metar_temp  = station_data.get("obs_temp_c")
        # Prefer WU high (resolution anchor).  Fall back to METAR current.
        effective_obs_c = wu_temp_c if wu_temp_c is not None else metar_temp

        # ── Latency arb zone detection ────────────────────────────────────────
        # < LATENCY_ARB_HOURS: WU's running high IS the market outcome.
        # HFT bots poll WU every few minutes.  We cannot compete on speed.
        # However: WU high tells us the ANSWER if it's outside the bucket.
        # If WU high > hi_c → bucket is dead → BUY NO with certainty.
        # If WU high inside bucket and < 2h left → BUY YES with high confidence.
        in_latency_arb_zone = time_remaining_hours < LATENCY_ARB_HOURS
        wu_definitive = False
        wu_definitive_result = None  # "dead" or "alive"

        if wu_temp_c is not None and in_latency_arb_zone:
            lo_c_b = temp_bucket.get("lo_c")
            hi_c_b = temp_bucket.get("hi_c")
            # Only the "dead" case is physically certain:
            #   T_max is monotonically increasing.  Once WU daily high > hi_c,
            #   the bucket ceiling has already been breached — P(YES) = 0.
            #
            # "alive" (WU inside bucket) is NOT certain: temperature can still
            #   rise above hi_c before the day ends.  Tail risk persists.
            if hi_c_b is not None and wu_temp_c > hi_c_b:
                wu_definitive = True
                wu_definitive_result = "dead"
            elif lo_c_b is not None and wu_temp_c < lo_c_b and time_remaining_hours < 3.0:
                # WU daily high below floor with <3h left: won't reach lo_c
                wu_definitive = True
                wu_definitive_result = "dead"

        model_prob, obs_adjusted = _temp_bucket_model_prob(
            weather, temp_bucket, days_ahead,
            obs_temp_c=effective_obs_c,
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

        # ── WU definitive override ────────────────────────────────────────
        # Only the "dead" case is physically certain (monotonic max temp).
        # "alive" (inside bucket) is NOT overridden — temperatures can still
        # rise above hi_c before resolution, so tail risk must remain in prob.
        if wu_definitive and wu_definitive_result == "dead":
            model_prob = 0.01  # effectively zero; tiny non-zero for display

        # ── Market certainty guard ─────────────────────────────────────────
        # When the market has already priced certainty (YES>90% or NO>90%),
        # the day is nearly done and live traders have observed real temps.
        # Our GFS forecast is stale — defer to the market, don't fight it.
        # Exception: if WU is definitive, trust WU over the crowd.
        market_certainty = market_price_yes >= 0.90 or market_price_yes <= 0.10
        if market_certainty and time_remaining_hours < 8.0 and not wu_definitive:
            return {
                **market,
                "model_probability": model_prob,
                "weather_data": weather,
                "weather_temp_max_c": weather.get("temp_max_c"),
                "weather_temp_max_f": weather.get("temp_max_f"),
                "edge": round(model_prob - market_price_yes, 4),
                "abs_edge": abs(model_prob - market_price_yes),
                "action": "HOLD",
                "is_opportunity": False,
                "skip_reason": f"Market certainty >{int(max(market_price_yes, 1-market_price_yes)*100)}% with {time_remaining_hours:.1f}h left — market knows more than model",
                "days_ahead": days_ahead,
                **station_data,
                "time_remaining_hours": round(time_remaining_hours, 2),
                "obs_adjusted": obs_adjusted,
                "in_latency_arb_zone": in_latency_arb_zone,
                "wu_definitive": wu_definitive,
            }

        # ── Spread-adjusted edge ───────────────────────────────────────────
        # market_price_yes is the mid-price (or last trade).
        # To BUY YES you pay ask = mid + half_spread.
        # To BUY NO you pay ask_no = 1 - (mid - half_spread) = 1 - bid_yes.
        # The EDGE_THRESHOLD gate must see executable edge, not paper edge.
        half_spread = _estimate_half_spread(market_price_yes, days_ahead)
        raw_edge    = model_prob - market_price_yes
        # Executable edge deducts execution cost in the direction of the trade
        if raw_edge > 0:
            exec_edge = raw_edge - half_spread   # BUY YES: pay ask
        else:
            exec_edge = raw_edge + half_spread   # BUY NO: pay ask_no

        edge     = round(raw_edge, 4)
        abs_edge = abs(exec_edge)

        # Cap: edge >55% almost certainly means model error, not real arb.
        # Exception: WU definitive cases legitimately produce >90% edges.
        EDGE_CAP = 0.55
        if abs(raw_edge) > EDGE_CAP and not wu_definitive:
            return {
                **market,
                "model_probability": model_prob,
                "weather_data": weather,
                "weather_temp_max_c": weather.get("temp_max_c"),
                "weather_temp_max_f": weather.get("temp_max_f"),
                "edge": edge,
                "abs_edge": round(abs(raw_edge), 4),
                "exec_edge": round(exec_edge, 4),
                "half_spread": round(half_spread, 4),
                "action": "HOLD",
                "is_opportunity": False,
                "skip_reason": f"Edge {abs(raw_edge):.0%} exceeds cap {EDGE_CAP:.0%} — likely model/forecast error",
                "days_ahead": days_ahead,
                **station_data,
                "time_remaining_hours": round(time_remaining_hours, 2),
                "obs_adjusted": obs_adjusted,
                "in_latency_arb_zone": in_latency_arb_zone,
                "wu_definitive": wu_definitive,
            }
        # is_opportunity: based on EXECUTABLE edge, not paper edge
        is_opp = exec_edge > EDGE_THRESHOLD if raw_edge > 0 else (-exec_edge) > EDGE_THRESHOLD
        action = ("BUY YES" if raw_edge > 0 else "BUY NO") if is_opp else "HOLD"

        # ── Latency arb zone warning ───────────────────────────────────────────
        latency_note = None
        if in_latency_arb_zone and not wu_definitive and wu_temp_c is None:
            latency_note = f"HFT zone ({time_remaining_hours:.0f}h left, no WU data) — model edge not executable"
        elif in_latency_arb_zone and not wu_definitive and wu_temp_c is not None:
            latency_note = f"WU high {wu_temp_c:.1f}°C observed but not definitive — {time_remaining_hours:.0f}h left"

        fc_max_c = weather.get("temp_max_c")
        fc_max_f = weather.get("temp_max_f")
        return {
            **market,
            "model_probability": model_prob,
            "weather_data": weather,
            "weather_temp_max_c": fc_max_c,
            "weather_temp_max_f": fc_max_f,
            "edge": edge,
            "abs_edge": round(abs(raw_edge), 4),
            "exec_edge": round(exec_edge, 4),   # what you actually capture after spread
            "half_spread": round(half_spread, 4),
            "action": action,
            "is_opportunity": is_opp,
            "skip_reason": None,
            "days_ahead": days_ahead,
            **station_data,
            "time_remaining_hours": round(time_remaining_hours, 2),
            "obs_adjusted": obs_adjusted,
            "in_latency_arb_zone": in_latency_arb_zone,
            "wu_definitive": wu_definitive,
            "wu_definitive_result": wu_definitive_result,
            "latency_note": latency_note,
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

    # ── Spread-adjusted edge ─────────────────────────────────────────────────
    raw_edge    = model_prob - market_price_yes
    half_spread = _estimate_half_spread(market_price_yes, days_ahead)
    exec_edge   = raw_edge - half_spread if raw_edge > 0 else raw_edge + half_spread

    is_opportunity = (exec_edge > EDGE_THRESHOLD if raw_edge > 0
                      else -exec_edge > EDGE_THRESHOLD)

    if not is_opportunity:
        action = "SKIP"
    elif raw_edge > 0:
        action = "BUY YES"
    else:
        action = "BUY NO"

    return {
        **market,
        "model_probability": model_prob,
        "weather_data": weather,
        "edge": round(raw_edge, 4),
        "abs_edge": round(abs(raw_edge), 4),
        "exec_edge": round(exec_edge, 4),
        "half_spread": round(half_spread, 4),
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

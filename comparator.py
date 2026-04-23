"""
comparator.py
Compares Polymarket weather market prices against Open-Meteo ensemble model probabilities.
Flags opportunities where the divergence exceeds the threshold (default 5%).
Also fetches station observations and applies obs-adjustment for intraday markets.
"""

import asyncio
import calendar
import math
import threading
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
from fetcher_polymarket import fetch_clob_book, sweep_book as _sweep_book

MAX_DAYS_AHEAD = 10       # Only consider markets resolving within this window
EDGE_THRESHOLD = 0.08     # 8% minimum divergence to flag as opportunity (5% had too much noise)
# Asymmetric: BUY YES remains structurally underperforming vs BUY NO.
# 2026-04-18→21 rolling sample (n=262): BUY YES 29.2% (14/48) vs BUY NO 67.8% (145/214).
# Even after lifting YES threshold 0.15→0.20, YES still ~21pp below break-even.
# Push to 0.25 — forces YES bets only when model is *very* confident the bucket hits.
EDGE_THRESHOLD_YES = 0.25
EDGE_THRESHOLD_NO  = EDGE_THRESHOLD

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


try:
    from zoneinfo import ZoneInfo
    _ZI_AVAILABLE = True
except ImportError:
    _ZI_AVAILABLE = False

# City -> IANA timezone. Used by _get_utc_offset when zoneinfo is available
# for accurate DST handling (handles cities that don't observe DST even if
# their offset range suggests they might, e.g. Tokyo, Lagos, São Paulo).
_CITY_TZ: dict[str, str] = {
    "new york": "America/New_York", "nyc": "America/New_York", "new york city": "America/New_York",
    "boston": "America/New_York", "philadelphia": "America/New_York", "washington dc": "America/New_York",
    "miami": "America/New_York", "atlanta": "America/New_York", "charlotte": "America/New_York",
    "orlando": "America/New_York", "tampa": "America/New_York", "baltimore": "America/New_York",
    "pittsburgh": "America/New_York", "cleveland": "America/New_York", "detroit": "America/Detroit",
    "columbus": "America/New_York", "indianapolis": "America/Indiana/Indianapolis",
    "nashville": "America/Chicago", "memphis": "America/Chicago", "jacksonville": "America/New_York",
    "raleigh": "America/New_York", "richmond": "America/New_York",
    "chicago": "America/Chicago", "minneapolis": "America/Chicago", "milwaukee": "America/Chicago",
    "st. louis": "America/Chicago", "kansas city": "America/Chicago", "omaha": "America/Chicago",
    "des moines": "America/Chicago", "dallas": "America/Chicago", "houston": "America/Chicago",
    "austin": "America/Chicago", "san antonio": "America/Chicago", "new orleans": "America/Chicago",
    "oklahoma city": "America/Chicago", "little rock": "America/Chicago",
    "denver": "America/Denver", "phoenix": "America/Phoenix", "salt lake city": "America/Denver",
    "albuquerque": "America/Denver", "el paso": "America/Denver", "boise": "America/Boise",
    "tucson": "America/Phoenix",
    "los angeles": "America/Los_Angeles", "la": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles", "sf": "America/Los_Angeles",
    "san jose": "America/Los_Angeles", "seattle": "America/Los_Angeles",
    "portland": "America/Los_Angeles", "las vegas": "America/Los_Angeles",
    "sacramento": "America/Los_Angeles", "san diego": "America/Los_Angeles",
    "spokane": "America/Los_Angeles", "reno": "America/Los_Angeles",
    "anchorage": "America/Anchorage", "juneau": "America/Juneau", "fairbanks": "America/Anchorage",
    "toronto": "America/Toronto", "montreal": "America/Montreal", "ottawa": "America/Toronto",
    "halifax": "America/Halifax", "vancouver": "America/Vancouver", "calgary": "America/Edmonton",
    "edmonton": "America/Edmonton", "winnipeg": "America/Winnipeg",
    "london": "Europe/London", "lisbon": "Europe/Lisbon", "reykjavik": "Atlantic/Reykjavik",
    "paris": "Europe/Paris", "berlin": "Europe/Berlin", "madrid": "Europe/Madrid",
    "rome": "Europe/Rome", "amsterdam": "Europe/Amsterdam", "brussels": "Europe/Brussels",
    "zurich": "Europe/Zurich", "vienna": "Europe/Vienna", "prague": "Europe/Prague",
    "warsaw": "Europe/Warsaw", "budapest": "Europe/Budapest", "stockholm": "Europe/Stockholm",
    "oslo": "Europe/Oslo", "copenhagen": "Europe/Copenhagen", "helsinki": "Europe/Helsinki",
    "athens": "Europe/Athens", "istanbul": "Europe/Istanbul", "moscow": "Europe/Moscow",
    "dubai": "Asia/Dubai",
    "tel aviv": "Asia/Tel_Aviv", "cairo": "Africa/Cairo", "jeddah": "Asia/Riyadh",
    "mumbai": "Asia/Kolkata", "delhi": "Asia/Kolkata", "kolkata": "Asia/Kolkata",
    "lucknow": "Asia/Kolkata",
    "bangkok": "Asia/Bangkok", "jakarta": "Asia/Jakarta",
    "kuala lumpur": "Asia/Kuala_Lumpur", "kl": "Asia/Kuala_Lumpur",
    "singapore": "Asia/Singapore", "hong kong": "Asia/Hong_Kong",
    "shenzhen": "Asia/Shanghai", "beijing": "Asia/Shanghai", "shanghai": "Asia/Shanghai",
    "taipei": "Asia/Taipei", "seoul": "Asia/Seoul", "tokyo": "Asia/Tokyo", "osaka": "Asia/Tokyo",
    "sydney": "Australia/Sydney", "melbourne": "Australia/Melbourne",
    "brisbane": "Australia/Brisbane", "auckland": "Pacific/Auckland",
    "wellington": "Pacific/Auckland",
    "sao paulo": "America/Sao_Paulo", "rio de janeiro": "America/Sao_Paulo",
    "buenos aires": "America/Argentina/Buenos_Aires", "mexico city": "America/Mexico_City",
    "panama city": "America/Panama", "bogota": "America/Bogota", "lima": "America/Lima",
    "santiago": "America/Santiago",
    "nairobi": "Africa/Nairobi", "lagos": "Africa/Lagos",
    "johannesburg": "Africa/Johannesburg", "cape town": "Africa/Johannesburg",
}


def _is_dst_active(dt: datetime, base_offset: float) -> bool:
    """
    Rough DST check — fallback when zoneinfo isn't available or city unknown.
    Covers US (2nd Sun Mar → 1st Sun Nov), EU (last Sun Mar → last Sun Oct),
    and AU/NZ (1st Sun Oct → 1st Sun Apr).

    Known imperfect: falsely flags DST for non-DST regions in the covered
    offset ranges (Japan, China, India, most of Africa, most of South America).
    _get_utc_offset uses zoneinfo first when possible to avoid this.
    """
    month = dt.month
    day   = dt.day

    # ── Northern Hemisphere DST (UTC-11 … UTC+3): roughly Apr–Oct ────────────
    if -11 <= base_offset <= 3:
        return 3 < month < 11 or (month == 3 and day >= 8) or (month == 11 and day < 7)

    # ── Southern Hemisphere (AU/NZ/SA: UTC+8 … UTC+13) ───────────────────────
    if base_offset >= 8:
        return month >= 10 or month <= 3

    return False


def _get_utc_offset(city: str, lon: float, dt: Optional[datetime] = None) -> float:
    """
    Return UTC offset in hours for a city at a given moment.
    Prefers zoneinfo (handles DST exactly, including cities that don't
    observe DST in otherwise-DST-ranged offsets). Falls back to hardcoded
    offset table + approximate DST rule, then to lon/15 heuristic.
    """
    key = city.strip().lower()

    # ── Preferred: real tz lookup via zoneinfo ──────────────────────────────
    if _ZI_AVAILABLE and dt is not None:
        tz_name = _CITY_TZ.get(key)
        if tz_name is None:
            for ck, tn in _CITY_TZ.items():
                if key in ck or ck in key:
                    tz_name = tn
                    break
        if tz_name:
            try:
                tz = ZoneInfo(tz_name)
                moment = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                off = moment.astimezone(tz).utcoffset()
                if off is not None:
                    return off.total_seconds() / 3600.0
            except Exception:
                pass

    # ── Fallback: hardcoded base offset + approximate DST rule ──────────────
    base = None
    if key in _CITY_UTC_OFFSET:
        base = _CITY_UTC_OFFSET[key]
    else:
        for city_key, offset in _CITY_UTC_OFFSET.items():
            if key in city_key or city_key in key:
                base = offset
                break
    if base is None:
        base = round(lon / 15.0)

    if dt is not None and _is_dst_active(dt, base):
        return base + 1.0
    return base


def _time_remaining_hours(city: str, lon: float, target_date: date) -> float:
    """
    Estimate how many hours remain in the local day at target_date.
    Returns 0.0 if the day has already passed, 24.0 if it hasn't started.
    """
    now_utc = datetime.now(timezone.utc)
    utc_offset = _get_utc_offset(city, lon, dt=now_utc)
    # Local time with DST correction
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

# ── Kalman posterior state (persistent across 30-min scans, 26h TTL) ─────────
# Key: (location_lower, date_iso, "lo_c-hi_c")
# Value: (p_mm_post, p_aa_post, p_ma_post)  — posterior covariance
# Without this, P resets every scan → σ never converges → phantom spillover.
_kalman_state: dict = {}
_kalman_state_ts: dict = {}
_kalman_state_lock = threading.Lock()
_KALMAN_STATE_MAX_AGE_S = 26 * 3600   # purge next-day stale entries


def _kalman_state_get(key: tuple) -> Optional[tuple[float, float, float]]:
    """Return saved (p_mm, p_aa, p_ma) or None if missing/stale."""
    with _kalman_state_lock:
        ts = _kalman_state_ts.get(key)
        if ts is None:
            return None
        import time
        if time.time() - ts > _KALMAN_STATE_MAX_AGE_S:
            _kalman_state.pop(key, None)
            _kalman_state_ts.pop(key, None)
            return None
        return _kalman_state.get(key)


def _kalman_state_set(key: tuple, p_mm: float, p_aa: float, p_ma: float) -> None:
    """Persist posterior covariance for next scan."""
    import time
    with _kalman_state_lock:
        _kalman_state[key] = (p_mm, p_aa, p_ma)
        _kalman_state_ts[key] = time.time()


def _kalman_state_reset(key: tuple) -> None:
    """Wipe state on regime shift so the next scan restarts from prior."""
    with _kalman_state_lock:
        _kalman_state.pop(key, None)
        _kalman_state_ts.pop(key, None)


def _kalman_state_sweep() -> int:
    """
    Active TTL sweep — deletes every entry older than _KALMAN_STATE_MAX_AGE_S.

    _kalman_state_get() only purges on read, so keys for markets that have
    resolved and are never scanned again linger forever. On a Railway
    container with weeks of uptime that's hundreds of stale entries per
    day → slow memory creep + a noisy regime-shift baseline.

    Called from compare_all_markets() at scan start, next to
    clear_weather_cache(). Returns count of purged entries for log.
    """
    import time
    now = time.time()
    with _kalman_state_lock:
        stale = [k for k, ts in _kalman_state_ts.items()
                 if now - ts > _KALMAN_STATE_MAX_AGE_S]
        for k in stale:
            _kalman_state.pop(k, None)
            _kalman_state_ts.pop(k, None)
    return len(stale)


# ── Weather result cache (persistent across scan runs, 2-hour TTL) ────────────
# Stores (result, timestamp) so weather isn't re-fetched too often.
# Stale-while-revalidate: if refetch fails (e.g. rate limit), keep using the
# last known good value for up to WEATHER_STALE_TTL (6 h) rather than returning
# None and skipping every market.
import time as _time

_weather_cache: dict = {}       # key → (result, timestamp)
_weather_inflight: dict = {}    # key → asyncio.Event (stampede prevention)
WEATHER_CACHE_TTL = 2 * 60 * 60
WEATHER_STALE_TTL = 6 * 60 * 60


async def _get_weather_cached(location: str, target_date: date) -> Optional[dict]:
    key = (location.lower(), target_date.isoformat())
    now = _time.time()

    entry = _weather_cache.get(key)
    stale_val = None
    if entry is not None:
        result, ts = entry
        if now - ts < WEATHER_CACHE_TTL:
            return result
        stale_val = result

    event = _weather_inflight.get(key)
    if event is not None:
        await event.wait()
        entry = _weather_cache.get(key)
        return entry[0] if entry else stale_val

    event = asyncio.Event()
    _weather_inflight[key] = event
    try:
        result = await get_weather_for_location_date(location, target_date)
    except Exception as e:
        print(f"[Weather] Exception fetching {location} {target_date}: {e}")
        result = None
    finally:
        _weather_inflight.pop(key, None)
        event.set()

    if result is not None:
        _weather_cache[key] = (result, _time.time())
        return result
    elif stale_val is not None:
        _weather_cache[key] = (stale_val, now - WEATHER_CACHE_TTL)
        print(f"[Weather] Fetch failed for {location} {target_date} — serving stale")
        return stale_val
    return None


def clear_weather_cache():
    now = _time.time()
    stale = [k for k, (v, ts) in list(_weather_cache.items())
             if now - ts >= WEATHER_CACHE_TTL]
    for k in stale:
        _weather_cache.pop(k, None)


# ── Normal distribution helpers ───────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """CDF of the standard normal distribution using math.erf."""
    return (1.0 + math.erf(x / math.sqrt(2))) / 2.0


_REGIME_SHIFT_SIGMA = 3.0    # innovation threshold in units of σ_y
_REGIME_P_BOOST    = 10.0   # multiply prior P by this when regime shift detected


def _kalman_diurnal_update(
    T_mean_prior: float,
    A_prior: float,
    p_mm: float,
    p_aa: float,
    p_ma: float,
    obs_temp_c: float,
    cos_phase: float,
    R: float,
) -> tuple[float, float, float, float, bool, float, float, float]:
    """
    1-observation 2-state Kalman filter update with regime-shift detection.

    State x = [T_mean, A], where T_max = T_mean + A.
    Observation model: z = T_now = T_mean + A·cos(φ), i.e. H = [1, c].

    When cos(φ) ≈ 0 (near midnight), the Kalman gain K → 0 and the observation
    barely updates the state — no singularity.  This replaces the algebraic
    A_obs = (T_now − T_mean) / cos(φ) which explodes at small cos_phase values.

    Regime-shift detection (Innovation Check):
        y_norm = |y| / σ_y where σ_y = sqrt(S)
        If y_norm > _REGIME_SHIFT_SIGMA (3σ), the residual is non-Gaussian —
        a weather event (cold front, convective downdraft) has invalidated the
        diurnal model.  In that case P is boosted by _REGIME_P_BOOST so the
        Kalman gain K → 1, forcing the posterior to trust the observation
        rather than the prior.  This prevents the model from dismissing a
        real 5°C regime change as an outlier.

    Returns (T_mean_post, A_post, T_max_post, sigma_T_max_post, regime_shift_flag,
             p_mm_post, p_aa_post, p_ma_post).
    The posterior P values are returned so callers can persist them across scans
    (sequential Kalman: σ converges over repeated observations).
    """
    c = cos_phase

    # Innovation
    y = obs_temp_c - (T_mean_prior + c * A_prior)

    # Innovation covariance S = H·P·H' + R  (scalar since H is 1×2)
    S = p_mm + 2.0 * c * p_ma + c * c * p_aa + R
    if S < 1e-9:
        # Degenerate: near-zero innovation variance → no update
        T_max_post = T_mean_prior + A_prior
        sigma_post = math.sqrt(max(1e-4, p_mm + 2.0 * p_ma + p_aa))
        return T_mean_prior, A_prior, T_max_post, sigma_post, False, p_mm, p_aa, p_ma

    # ── Regime-shift check ────────────────────────────────────────────────────
    # A standard Kalman filter assumes Gaussian noise.  A convective downdraft
    # or cold front can drop temperature 5°C in 30 min — not Gaussian.
    # As the day progresses, σ_prior stays fixed (we recompute P fresh each
    # call), but if we ran a sequential filter its P would converge → small K.
    # This check defends against that: any innovation > 3σ_y triggers a P reset
    # that forces K → near 1 so the model immediately trusts the anomaly.
    sigma_inno   = math.sqrt(S)
    regime_shift = abs(y) > _REGIME_SHIFT_SIGMA * sigma_inno

    if regime_shift:
        # Boost P: multiply prior variances by _REGIME_P_BOOST.
        # This makes K large → state update leans heavily toward obs.
        p_mm = p_mm * _REGIME_P_BOOST
        p_aa = p_aa * _REGIME_P_BOOST
        p_ma = p_ma * _REGIME_P_BOOST
        S    = p_mm + 2.0 * c * p_ma + c * c * p_aa + R
        print(f"[Kalman] ⚡ Regime shift: innov={y:+.2f}°C "
              f"({abs(y)/sigma_inno:.1f}σ > {_REGIME_SHIFT_SIGMA}σ) — P×{_REGIME_P_BOOST}")

    # Kalman gains:  K = P·H' / S
    k_m = (p_mm + c * p_ma) / S
    k_a = (p_ma + c * p_aa) / S

    # State update
    T_mean_post = T_mean_prior + k_m * y
    A_post      = A_prior      + k_a * y
    T_max_post  = T_mean_post  + A_post

    # Posterior covariance  P_post = (I − K·H)·P
    p_mm_post = p_mm - k_m * (p_mm  + c * p_ma)
    p_aa_post = p_aa - k_a * (p_ma  + c * p_aa)
    p_ma_post = p_ma - k_m * (p_ma  + c * p_aa)

    # Var(T_max_post) = Var(T_mean + A) = p_mm + 2·p_ma + p_aa  (posterior)
    var_T_max = p_mm_post + 2.0 * p_ma_post + p_aa_post
    sigma_post = math.sqrt(max(1e-4, var_T_max))

    return T_mean_post, A_post, T_max_post, sigma_post, regime_shift, p_mm_post, p_aa_post, p_ma_post


# Standard Kelly test size used for VWAP depth-impact check ($20 notional)
_VWAP_TEST_USD = 20.0


def _temp_bucket_model_prob(
    weather: dict,
    temp_bucket: dict,
    days_ahead: int,
    obs_temp_c: Optional[float] = None,
    time_remaining_hours: Optional[float] = None,
    wu_high_c: Optional[float] = None,
    state_key: Optional[tuple] = None,
) -> tuple[Optional[float], bool, bool]:
    """
    Compute P(T_max ∈ [lo_c, hi_c]) using the correct distribution given evidence.
    Returns (prob, obs_adjusted, regime_shift).
    regime_shift=True when the Kalman innovation exceeded 3σ — a weather event
    (cold front, convective downdraft) invalidated the diurnal model mid-day.

    Regime A (no obs): T_max ~ N(μ_forecast, σ²)
      σ = ensemble member std-dev if available; else lead-time heuristic.

    Regime B (obs available): Diurnal Bayesian update + Truncated Normal.

      Step 1 — 2-state Kalman filter infers T_max from T_now:
        State x = [T_mean, A], T_max = T_mean + A.
        Observation: T_now = T_mean + A × cos(φ),  H = [1, cos(φ)].
        Prior splits σ_prior² equally: Var(T_mean) = Var(A) = σ_prior²/2.
        When cos(φ) ≈ 0 (near midnight), Kalman gain K → 0 — no singularity.
        This replaces the algebraic A_obs = (T_now − T_mean)/cos(φ) that
        explodes at small cos_phase (the old cosine-singularity bug).

      Step 2 — posterior σ from Kalman covariance:
        σ_post = sqrt(Var(T_mean_post) + 2·Cov + Var(A_post))
        Used directly — NO additional time-decay.

      Step 3 — Truncated Normal given T_max ≥ T_now:
        P(lo ≤ T_max ≤ hi | T_max ≥ T_now)
          = [Φ((hi − μ)/σ) − Φ((eff_lo − μ)/σ)] / [1 − Φ((T_now − μ)/σ)]
        eff_lo = max(lo_c, T_now).

      The ONLY hard certainties:
        wu_high_c > hi_c → P(YES) = 0  (WU daily high already exceeded ceiling).
        obs_temp_c > hi_c → P(YES) = 0  (current temp already above ceiling).
      wu_high_c is the WU running daily maximum — resolution anchor for Polymarket.
      obs_temp_c is the instantaneous METAR reading — Kalman diurnal model input.
      These must NOT be conflated: using wu_high_c as Kalman input inflates T_max
      by up to 2°C and causes phantom spillover into adjacent buckets.
    """
    forecast_mu = weather.get("temp_max_c")
    if forecast_mu is None:
        return None, False, False

    lo_c = temp_bucket.get("lo_c")
    hi_c = temp_bucket.get("hi_c")

    # ── σ_prior: ensemble member spread is ground truth ───────────────────────
    # The ensemble σ already encodes all temporal uncertainty across members.
    # Do NOT apply additional time-decay to it — that would double-count the
    # information already captured by the ensemble spread narrowing near resolution.
    ensemble_spread = weather.get("temp_spread_c")
    if ensemble_spread is not None and ensemble_spread > 0.2:
        sigma_prior = float(ensemble_spread)
    else:
        # Fallback: Open-Meteo daily-max MAE ~1.2°C at d1, +0.5°C/day
        sigma_prior = max(1.2, 1.2 + (days_ahead - 1) * 0.5)

    # ── Regime A: pure forecast, no obs ───────────────────────────────────────
    if obs_temp_c is None or time_remaining_hours is None:
        lo_cdf = _norm_cdf((lo_c - forecast_mu) / sigma_prior) if lo_c is not None else 0.0
        hi_cdf = _norm_cdf((hi_c - forecast_mu) / sigma_prior) if hi_c is not None else 1.0
        return round(max(0.0, min(1.0, hi_cdf - lo_cdf)), 3), False, False

    # ── Regime B: truncated normal + diurnal Bayes ────────────────────────────
    obs_adjusted = True
    hours_elapsed = max(0.0, 24.0 - time_remaining_hours)

    # Physical hard constraint: T_max already exceeded bucket ceiling → dead.
    # Use wu_high_c (WU running daily max = resolution anchor) if available,
    # otherwise fall back to obs_temp_c (instantaneous METAR reading).
    ceiling_obs = wu_high_c if wu_high_c is not None else obs_temp_c
    if hi_c is not None and ceiling_obs > hi_c:
        return 0.0, True, False

    # ── 2-state Kalman filter μ update ───────────────────────────────────────
    # obs_temp_c = instantaneous METAR temperature → diurnal model input.
    # wu_high_c  = running daily maximum → used ONLY for ceiling check above.
    # Conflating them inflates T_max_post by up to 2°C (phantom spillover).
    #
    # State x = [T_mean, A], observation z = T_now = T_mean + A·cos(φ).
    # P is persisted across 30-min scans via state_key so σ converges.
    # When cos(φ) ≈ 0 (midnight–6am), K → 0 → minimal state update.

    T_PEAK_HOUR = 14.0
    phase     = math.pi * (hours_elapsed - T_PEAK_HOUR) / 12.0
    cos_phase = math.cos(phase)

    t_min_fc = weather.get("temp_min_c")

    if hours_elapsed > 0:
        # Prior mean for [T_mean, A] from forecast
        T_mean_fc = (forecast_mu + t_min_fc) / 2.0 if t_min_fc is not None else forecast_mu * 0.9
        A_fc      = forecast_mu - T_mean_fc   # amplitude: T_max − T_mean

        # Load persisted posterior P (sequential Kalman — σ converges over scans).
        # On first scan or after regime reset, fall back to the uninformed prior.
        saved_p = _kalman_state_get(state_key) if state_key else None
        if saved_p is not None:
            p_mm, p_aa, p_ma = saved_p
        else:
            p_mm = sigma_prior ** 2 / 2.0
            p_aa = sigma_prior ** 2 / 2.0
            p_ma = 0.0
        R = 0.5 ** 2   # METAR measurement noise: ±0.5°C

        _, _, t_max_inferred, sigma_post, regime_shift, p_mm_post, p_aa_post, p_ma_post = (
            _kalman_diurnal_update(T_mean_fc, A_fc, p_mm, p_aa, p_ma, obs_temp_c, cos_phase, R)
        )

        if state_key:
            if regime_shift:
                # Regime shift → reset state so next scan restarts from prior
                _kalman_state_reset(state_key)
            else:
                _kalman_state_set(state_key, p_mm_post, p_aa_post, p_ma_post)

        mu    = t_max_inferred
        sigma = sigma_post

        print(f"[Comparator] Kalman: fc={forecast_mu:.1f} T_now={obs_temp_c:.1f} "
              f"cosφ={cos_phase:.2f} → T_max_post={mu:.1f} σ={sigma:.2f}"
              + (" ⚡REGIME" if regime_shift else "")
              + (" [resumed P]" if saved_p is not None else " [fresh P]"))
    else:
        regime_shift = False
        # t=0 (no elapsed time): pure forecast, no obs information yet
        mu, sigma = forecast_mu, sigma_prior

    # Normalising constant: P(T_max >= T_now) in N(μ, σ)
    z_a    = (obs_temp_c - mu) / sigma
    p_above = 1.0 - _norm_cdf(z_a)

    if p_above < 1e-7:
        # Numerical edge: T_now >> μ.  All mass concentrated at T_now.
        in_lo = lo_c is None or obs_temp_c >= lo_c
        in_hi = hi_c is None or obs_temp_c <= hi_c
        return (0.93 if (in_lo and in_hi) else 0.02), True, regime_shift

    # P(lo ≤ T_max ≤ hi | T_max ≥ T_now)
    eff_lo = max(lo_c, obs_temp_c) if lo_c is not None else obs_temp_c
    eff_hi = hi_c   # upper limit unchanged — temps CAN still rise

    z_lo = (eff_lo - mu) / sigma
    z_hi = (eff_hi - mu) / sigma if eff_hi is not None else math.inf
    p_hi = _norm_cdf(z_hi) if eff_hi is not None else 1.0
    p_lo = _norm_cdf(z_lo)

    p_in_range = p_hi - p_lo
    prob = p_in_range / p_above
    return round(max(0.0, min(0.97, prob)), 3), True, regime_shift


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


async def _get_station_data(location: str, weather: Optional[dict], target_date: Optional[date] = None) -> dict:
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
        obs = await get_station_obs(icao)
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
                wu = await get_wu_temp_cached(icao, target_date)
                if wu:
                    result["wu_temp_c"]       = wu.get("wu_temp_c")
                    result["wu_temp_f"]       = wu.get("wu_temp_f")
                    result["wu_temp_c_raw"]   = wu.get("wu_temp_c_raw")
                    result["wu_temp_f_raw"]   = wu.get("wu_temp_f_raw")
                    result["wu_age_min"]      = wu.get("wu_age_min")
                    result["wu_source"]       = wu.get("wu_source")
                    result["wu_data_source"]  = wu.get("wu_data_source")
            except Exception as wu_e:
                print(f"[WU] Skipping WU fetch for {location}: {wu_e}")

    except Exception as e:
        print(f"[Stations] Error resolving station for '{location}': {e}")
    return result


async def compare_market(market: dict) -> Optional[dict]:
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

    # ── Local-date expiry check (timezone-aware) ─────────────────────────────
    # date.today() returns UTC date on Railway.  A NYC market ending April 16
    # is still live at UTC April 17 01:00 (= NYC April 16 21:00 EDT).
    # Fix: recalculate days_ahead using the city's local date so western
    # markets aren't prematurely skipped when viewed from Asia/UTC.
    now_utc = datetime.now(timezone.utc)
    today_utc = now_utc.date()
    days_ahead = (target_date - today_utc).days

    if days_ahead < 0:
        # UTC says expired — double-check with city's local calendar.
        # e.g. NYC (UTC-4): still April 16 until 04:00 UTC April 17.
        local_offset = _get_utc_offset(location, 0.0, dt=now_utc)
        local_date = (now_utc + timedelta(hours=local_offset)).date()
        days_ahead = (target_date - local_date).days

    today = today_utc   # keep for downstream compatibility

    if days_ahead < 0:
        return {
            **market,
            "model_probability": None,
            "weather_data": None,
            "edge": None,
            "action": "SKIP (past)",
            "is_opportunity": False,
            "skip_reason": f"Market resolved (local date past)",
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
    weather = await _get_weather_cached(location, target_date)
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
        station_data = await _get_station_data(location, weather, target_date=target_date)
        # Compute lon for time-remaining calculation
        lon = weather.get("lon") or 0.0
        time_remaining_hours = _time_remaining_hours(location, lon, target_date)

    # ── Temperature bucket markets (from Polymarket daily temp events) ─────────
    if market.get("market_subtype") == "temperature_bucket":
        temp_bucket = market.get("temp_bucket", {})

        # ── WU vs METAR: two separate roles ──────────────────────────────────
        # wu_temp_c  = WU running daily maximum → resolution anchor (ceiling check).
        # metar_temp = instantaneous METAR reading → Kalman diurnal model input.
        # NEVER conflate them: using wu_high as Kalman obs inflates T_max_post
        # by ~2°C at 10am and causes phantom spillover into adjacent buckets.
        wu_temp_c       = station_data.get("wu_temp_c")       # spike-robust max
        wu_temp_c_raw   = station_data.get("wu_temp_c_raw")   # true max incl. spikes
        wu_data_source  = station_data.get("wu_data_source")  # dailysummary | observations | ...
        if wu_temp_c_raw is None:
            wu_temp_c_raw = wu_temp_c
        metar_temp = station_data.get("obs_temp_c")
        # Kalman input: METAR current temp (instantaneous).
        # Ceiling check: WU daily high (monotonically non-decreasing).
        kalman_obs_c = metar_temp  # Kalman sees instantaneous temperature

        # ── Latency arb zone detection ────────────────────────────────────────
        # < LATENCY_ARB_HOURS: WU's running high IS the market outcome.
        # HFT bots poll WU every few minutes.  We cannot compete on speed.
        # However: WU high tells us the ANSWER if it's outside the bucket.
        # If WU high > hi_c → bucket is dead → BUY NO with certainty.
        # If WU high inside bucket and < 2h left → BUY YES with high confidence.
        in_latency_arb_zone = time_remaining_hours < LATENCY_ARB_HOURS
        wu_definitive = False
        # Only "dead" is ever set today. "alive" was speculative — removed from
        # the type to prevent future dead-code confusion. Any unreached `== "alive"`
        # check in consumers is a latent bug, not a feature.
        wu_definitive_result = None  # "dead" | None

        # ── Data-source gate for wu_definitive ───────────────────────────
        # Only the QC'd dailysummary path aligns with what Polymarket resolves.
        # The observations-array path leaks single-hour sensor spikes that get
        # filtered out of the final daily summary. In 2026-04-21→22 telemetry,
        # every wu_definitive *loss* (Wellington x2, Madrid x1) came from
        # the observations fallback. Refuse to go definitive on that path.
        wu_source_trusted = (wu_data_source == "dailysummary")

        if wu_temp_c is not None and in_latency_arb_zone and wu_source_trusted:
            lo_c_b = temp_bucket.get("lo_c")
            hi_c_b = temp_bucket.get("hi_c")
            # Only the "dead" case is physically certain:
            #   T_max is monotonically increasing.  Once WU daily high > hi_c,
            #   the bucket ceiling has already been breached — P(YES) = 0.
            #
            # "alive" (WU inside bucket) is NOT certain: temperature can still
            #   rise above hi_c before the day ends.  Tail risk persists.
            #
            # Spike buffer: WU hourly observations occasionally include bad
            # sensor spikes that get QC-filtered from the daily summary
            # Polymarket resolves against.  Houston 2026-04-18 was flagged
            # dead at bucket 82-83°F but actual daily high was 82.7°F — an
            # intraday spike > 83.5°F pushed the WU reading above ceiling
            # before QC.  Require 0.5°C (≈1°F) overshoot before trusting.
            WU_SPIKE_BUFFER_C = 0.5
            # Ceiling check uses spike-robust max: a single bad hourly obs
            # shouldn't flip a market dead when QC'd daily summary disagrees.
            if hi_c_b is not None and wu_temp_c > hi_c_b + WU_SPIKE_BUFFER_C:
                wu_definitive = True
                wu_definitive_result = "dead"
            # Floor check uses RAW max (true max across all obs). Tokyo
            # 2026-04-19 regression: spike filter dropped a legitimate
            # single-hour 22°C peak, making robust max read 20°C and
            # falsely triggering "dead below 22.5°C floor". If any hour
            # actually touched the bucket, we cannot declare dead.
            elif lo_c_b is not None and wu_temp_c_raw < lo_c_b - WU_SPIKE_BUFFER_C and time_remaining_hours < 3.0:
                wu_definitive = True
                wu_definitive_result = "dead"

        # Build state key for Kalman persistence: (location, date) ONLY.
        # The Kalman filter models the city's atmospheric state [T_mean, A] —
        # one physical reality regardless of how many buckets Polymarket opened.
        # Binding the key to "lo_c-hi_c" ran 5 independent Kalman chains for
        # the same city+date, each with identical inputs → redundant computation.
        # All buckets for the same city+date now share one converging σ.
        lo_c_b = temp_bucket.get("lo_c")
        hi_c_b = temp_bucket.get("hi_c")
        _state_key = (
            location.strip().lower(),
            target_date.isoformat(),
        ) if kalman_obs_c is not None else None

        model_prob, obs_adjusted, regime_shift_flag = _temp_bucket_model_prob(
            weather, temp_bucket, days_ahead,
            obs_temp_c=kalman_obs_c,
            time_remaining_hours=time_remaining_hours,
            wu_high_c=wu_temp_c,
            state_key=_state_key,
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
            market["model_prob_source"] = "wu_definitive"

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
        # Attempt to fetch real L2 bid/ask from CLOB.  Only pre-screened
        # candidates (|raw_edge| > half the threshold) get a book fetch to
        # limit API calls.  Fall back to estimated spread when unavailable.
        raw_edge    = model_prob - market_price_yes
        token_id_yes = market.get("token_id_yes")
        book = None
        if token_id_yes and abs(raw_edge) > EDGE_THRESHOLD * 0.4:
            try:
                book = await fetch_clob_book(token_id_yes)
            except Exception:
                pass

        exec_edge_vwap = None
        if book and book.get("best_ask") and book.get("best_bid"):
            # True execution cost from real order book
            if raw_edge > 0:
                exec_edge = model_prob - book["best_ask"]   # BUY YES: pay ask
            else:
                exec_edge = book["best_bid"] - model_prob   # BUY NO: pay 1-ask_no = bid_yes
            half_spread = book["half_spread"]

            # ── VWAP depth check: does a $20 order sweep above best_ask? ──────
            # exec_edge = model_prob - best_ask assumes infinite top-of-book liquidity.
            # A $20 Kelly bet may consume multiple levels, paying VWAP > best_ask.
            # Conservative edge: use min(exec_edge_best_ask, exec_edge_vwap).
            if raw_edge > 0:
                asks_full = book.get("asks_full", [])
                if asks_full:
                    sweep = _sweep_book(asks_full, _VWAP_TEST_USD)
                    vwap = sweep.get("vwap")
                    if vwap is not None:
                        exec_edge_vwap = model_prob - vwap
                        if exec_edge_vwap < exec_edge:
                            print(f"[VWAP] {token_id_yes[:10]}: best_ask={book['best_ask']:.3f} "
                                  f"vwap@${_VWAP_TEST_USD:.0f}={vwap:.4f} "
                                  f"edge_vwap={exec_edge_vwap:.4f} < edge_ask={exec_edge:.4f}")
                            exec_edge = exec_edge_vwap  # pay the real depth cost
        else:
            # Fallback: estimated spread (clearly worse than real book)
            half_spread = _estimate_half_spread(market_price_yes, days_ahead)
            if raw_edge > 0:
                exec_edge = raw_edge - half_spread
            else:
                exec_edge = raw_edge + half_spread

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
        # is_opportunity: based on EXECUTABLE edge (VWAP-adjusted), not paper edge
        # Asymmetric threshold: BUY YES needs more edge than BUY NO (see EDGE_THRESHOLD_YES).
        is_opp = exec_edge > EDGE_THRESHOLD_YES if raw_edge > 0 else (-exec_edge) > EDGE_THRESHOLD_NO
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
            "exec_edge": round(exec_edge, 4),   # VWAP-adjusted executable edge
            "exec_edge_vwap": round(exec_edge_vwap, 4) if exec_edge_vwap is not None else None,
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
            "regime_shift": regime_shift_flag,   # Kalman 3σ innovation flag
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

    # ── Spread-adjusted edge (real book preferred) ────────────────────────────
    raw_edge     = model_prob - market_price_yes
    token_id_yes = market.get("token_id_yes")
    book = None
    if token_id_yes and abs(raw_edge) > EDGE_THRESHOLD * 0.4:
        try:
            book = await fetch_clob_book(token_id_yes)
        except Exception:
            pass

    exec_edge_vwap = None
    if book and book.get("best_ask") and book.get("best_bid"):
        exec_edge   = model_prob - book["best_ask"] if raw_edge > 0 else book["best_bid"] - model_prob
        half_spread = book["half_spread"]
        # VWAP depth-impact check for BUY YES orders
        if raw_edge > 0:
            asks_full = book.get("asks_full", [])
            if asks_full:
                sweep = _sweep_book(asks_full, _VWAP_TEST_USD)
                vwap = sweep.get("vwap")
                if vwap is not None:
                    exec_edge_vwap = model_prob - vwap
                    if exec_edge_vwap < exec_edge:
                        exec_edge = exec_edge_vwap
    else:
        half_spread = _estimate_half_spread(market_price_yes, days_ahead)
        exec_edge   = raw_edge - half_spread if raw_edge > 0 else raw_edge + half_spread

    is_opportunity = (exec_edge > EDGE_THRESHOLD_YES if raw_edge > 0
                      else -exec_edge > EDGE_THRESHOLD_NO)

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
        "exec_edge_vwap": round(exec_edge_vwap, 4) if exec_edge_vwap is not None else None,
        "half_spread": round(half_spread, 4),
        "action": action,
        "is_opportunity": is_opportunity,
        "skip_reason": None,
        "days_ahead": days_ahead,
        **station_data,
        "time_remaining_hours": round(time_remaining_hours, 2),
        "obs_adjusted": obs_adjusted,
    }


_CHUNK_SIZE = 100  # markets per concurrent gather batch


async def compare_all_markets(markets: list[dict]) -> list[dict]:
    """
    Run comparisons for all markets concurrently using asyncio.gather.
    True async I/O — no thread pool, no blocking, no GIL contention.
    Opportunities are sorted by abs_edge descending.
    Non-opportunities follow.

    Markets are processed in chunks of _CHUNK_SIZE to prevent thundering-herd
    on Open-Meteo, Polymarket CLOB, and station APIs when 500+ markets arrive
    simultaneously.  Within each chunk all coroutines run concurrently; chunks
    run sequentially with a 50ms yield between them so the event loop can flush
    I/O callbacks and enforce per-host connection limits cleanly.
    """
    clear_weather_cache()   # evict stale entries before a fresh scan run
    _kalman_purged = _kalman_state_sweep()
    if _kalman_purged > 0:
        print(f"[Kalman] Purged {_kalman_purged} stale state entries "
              f"(>{_KALMAN_STATE_MAX_AGE_S//3600}h old). "
              f"Live entries: {len(_kalman_state)}")
    results = []
    skipped = 0

    total = len(markets)
    for chunk_start in range(0, total, _CHUNK_SIZE):
        chunk = markets[chunk_start : chunk_start + _CHUNK_SIZE]
        chunk_end = min(chunk_start + _CHUNK_SIZE, total)
        print(f"[Comparator] Processing markets {chunk_start+1}–{chunk_end} of {total}")

        raw = await asyncio.gather(
            *[compare_market(m) for m in chunk],
            return_exceptions=True,
        )
        for r in raw:
            if isinstance(r, Exception):
                print(f"[Comparator] Market error: {r}")
                skipped += 1
            elif r is None:
                skipped += 1
            else:
                results.append(r)

        # Yield between chunks so the event loop can flush I/O callbacks,
        # enforce connection limits, and avoid bursting all CLOB/Open-Meteo
        # requests at once.  Skip the sleep after the last chunk.
        if chunk_end < total:
            await asyncio.sleep(0.05)

    print(f"\n[Comparator] Evaluated {len(results)} markets, skipped {skipped} (no location/date/price)")

    # ── Inter-bucket normalization ────────────────────────────────────────────
    # Temperature buckets for the same (city, date) are mutually exclusive and
    # exhaustive.  The sum of model_probability across buckets MUST equal 1.0.
    # Independent CDF computation produces sums > 1 (they are not normalized).
    # Fix: group by (location, date), sum raw probs, rescale each to sum=1.
    # Recompute edge and is_opportunity after normalization.
    from collections import defaultdict
    bucket_groups: dict = defaultdict(list)
    for r in results:
        if r.get("market_subtype") == "temperature_bucket" and r.get("model_probability") is not None:
            loc  = (r.get("location_hint") or r.get("location") or "").strip().lower()
            date_str = r.get("target_date") or r.get("date") or ""
            if loc and date_str:
                bucket_groups[(loc, date_str)].append(r)

    for (loc, date_str), group in bucket_groups.items():
        total = sum(r["model_probability"] for r in group)
        if total <= 0:
            continue
        if abs(total - 1.0) < 0.01:
            continue  # already normalised — skip
        print(f"[Comparator] Normalizing buckets {loc} {date_str}: "
              f"raw_sum={total:.3f} ({len(group)} buckets)")
        for r in group:
            r["model_probability"] = round(r["model_probability"] / total, 3)
            # Recompute edge and action after normalization
            mp = r.get("market_price_yes")
            if mp is not None:
                try:
                    mp = float(mp)
                    new_edge = r["model_probability"] - mp
                    r["edge"] = round(new_edge, 4)
                    r["abs_edge"] = round(abs(new_edge), 4)
                    thr = EDGE_THRESHOLD_YES if new_edge > 0 else EDGE_THRESHOLD_NO
                    if abs(new_edge) >= thr and mp not in (0.0, 1.0):
                        r["is_opportunity"] = True
                        r["action"] = "BUY YES" if new_edge > 0 else "BUY NO"
                    else:
                        r["is_opportunity"] = False
                        if not r.get("action", "").startswith("SKIP"):
                            r["action"] = "NO EDGE"
                except (ValueError, TypeError):
                    pass

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

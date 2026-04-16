"""
fetcher_stations.py
Maps Polymarket weather cities to ASOS/AWOS ICAO station codes and fetches
current observations from the NWS API.

Polymarket markets resolve against specific airport weather stations
(ASOS/AWOS via Weather Underground / NOAA), not city-level forecasts.
"""

import asyncio
import time
from typing import Optional

from _http import get_session

# ── ICAO station code mapping ─────────────────────────────────────────────────
# Format: city_key → [primary_icao, ...]  (ordered by relevance)
# These are ASOS/AWOS stations used by Weather Underground / NOAA / Polymarket.
CITY_TO_ICAO: dict[str, list[str]] = {
    # USA — East
    "new york":        ["KJFK", "KLGA", "KEWR"],
    "nyc":             ["KJFK", "KLGA", "KEWR"],
    "new york city":   ["KJFK", "KLGA", "KEWR"],
    "boston":          ["KBOS"],
    "philadelphia":    ["KPHL"],
    "washington dc":   ["KDCA", "KIAD", "KBWI"],
    "miami":           ["KMIA", "KFLL"],
    "atlanta":         ["KATL"],
    "charlotte":       ["KCLT"],
    "orlando":         ["KMCO"],
    "tampa":           ["KTPA"],
    "baltimore":       ["KBWI"],
    "newark":          ["KEWR"],
    "pittsburgh":      ["KPIT"],
    "cleveland":       ["KCLE"],
    "detroit":         ["KDTW"],
    "columbus":        ["KCMH"],
    "indianapolis":    ["KIND"],
    "nashville":       ["KBNA"],
    "memphis":         ["KMEM"],
    "jacksonville":    ["KJAX"],
    "raleigh":         ["KRDU"],
    "richmond":        ["KRIC"],
    # USA — Midwest
    "chicago":         ["KORD", "KMDW"],
    "minneapolis":     ["KMSP"],
    "milwaukee":       ["KMKE"],
    "st. louis":       ["KSTL"],
    "kansas city":     ["KMCI"],
    "omaha":           ["KOMA"],
    "des moines":      ["KDSM"],
    # USA — South / Southwest
    "dallas":          ["KDFW", "KDAL"],
    "houston":         ["KIAH", "KHOU"],
    "austin":          ["KAUS"],
    "san antonio":     ["KSAT"],
    "new orleans":     ["KMSY"],
    "oklahoma city":   ["KOKC"],
    "little rock":     ["KLIT"],
    "el paso":         ["KELP"],
    "albuquerque":     ["KABQ"],
    # USA — West
    "los angeles":     ["KLAX", "KBUR", "KSNA"],
    "la":              ["KLAX", "KBUR", "KSNA"],
    "san francisco":   ["KSFO", "KOAK"],
    "sf":              ["KSFO", "KOAK"],
    "san jose":        ["KSJC"],
    "seattle":         ["KSEA"],
    "portland":        ["KPDX"],
    "denver":          ["KDEN"],
    "phoenix":         ["KPHX"],
    "las vegas":       ["KLAS"],
    "salt lake city":  ["KSLC"],
    "tucson":          ["KTUS"],
    "sacramento":      ["KSMF"],
    "san diego":       ["KSAN"],
    "boise":           ["KBOI"],
    "spokane":         ["KGEG"],
    "reno":            ["KRNO"],
    # Canada
    "toronto":         ["CYYZ", "CYTZ"],
    "montreal":        ["CYUL"],
    "vancouver":       ["CYVR"],
    "calgary":         ["CYYC"],
    "edmonton":        ["CYEG"],
    "ottawa":          ["CYOW"],
    "winnipeg":        ["CYWG"],
    # UK / Europe
    "london":          ["EGLL", "EGKK"],
    "paris":           ["LFPG", "LFPO"],
    "berlin":          ["EDDB"],
    "madrid":          ["LEMD"],
    "rome":            ["LIRF"],
    "amsterdam":       ["EHAM"],
    "munich":          ["EDDM"],
    "frankfurt":       ["EDDF"],
    "zurich":          ["LSZH"],
    "vienna":          ["LOWW"],
    "brussels":        ["EBBR"],
    "stockholm":       ["ESSA"],
    "oslo":            ["ENGM"],
    "copenhagen":      ["EKCH"],
    "helsinki":        ["EFHK"],
    "warsaw":          ["EPWA"],
    "prague":          ["LKPR"],
    "budapest":        ["LHBP"],
    "istanbul":        ["LTFM", "LTBA"],
    "athens":          ["LGAV"],
    "lisbon":          ["LPPT"],
    # Asia
    "tokyo":           ["RJTT", "RJAA"],
    "osaka":           ["RJOO", "RJBB"],
    "beijing":         ["ZBAA"],
    "shanghai":        ["ZSSS", "ZSPD"],
    "hong kong":       ["VHHH"],
    "seoul":           ["RKSS", "RKSI"],
    "taipei":          ["RCTP", "RCSS"],
    "singapore":       ["WSSS"],
    "dubai":           ["OMDB"],
    "mumbai":          ["VABB"],
    "bangkok":         ["VTBS", "VTBD"],
    "kuala lumpur":    ["WMKK"],
    "kl":              ["WMKK"],
    "jakarta":         ["WIII"],
    "tel aviv":        ["LLBG"],
    "jeddah":          ["OEJN"],
    "cairo":           ["HECA"],
    # Oceania
    "sydney":          ["YSSY"],
    "melbourne":       ["YMML"],
    "auckland":        ["NZAA"],
    "wellington":      ["NZWN"],
    # Africa
    "johannesburg":    ["FAOR"],
    "cape town":       ["FACT"],
    "nairobi":         ["HKNA"],
    "lagos":           ["DNMM"],
    # Americas (non-US)
    "sao paulo":       ["SBGR", "SBSP"],
    "buenos aires":    ["SAEZ", "SABE"],
    "mexico city":     ["MMMX"],
    "panama city":     ["MPTO"],
    "bogota":          ["SKBO"],
    "lima":            ["SPJC"],
    "santiago":        ["SCEL"],
}

# ── NWS observation cache ─────────────────────────────────────────────────────
_obs_cache: dict[str, tuple[Optional[dict], float]] = {}  # icao → (result, ts)
_obs_lock_inst: Optional[asyncio.Lock] = None
OBS_CACHE_TTL = 10 * 60  # 10 minutes (NWS obs update hourly)

NWS_HEADERS = {"User-Agent": "WeatherArb/2.0 (contact@example.com)"}


def _get_obs_lock() -> asyncio.Lock:
    global _obs_lock_inst
    if _obs_lock_inst is None:
        _obs_lock_inst = asyncio.Lock()
    return _obs_lock_inst


def resolve_station(city: str) -> Optional[str]:
    """
    Resolve a city name to its primary ICAO station code.
    Returns the first (primary) code if found, else None.
    """
    key = city.strip().lower()
    if key in CITY_TO_ICAO:
        codes = CITY_TO_ICAO[key]
        return codes[0] if codes else None
    # Partial / substring match
    for city_key, codes in CITY_TO_ICAO.items():
        if key in city_key or city_key in key:
            return codes[0] if codes else None
    return None


def _is_us_icao(icao: str) -> bool:
    """
    US ICAO codes start with K (lower 48 + HI/AK use K prefix mostly).
    Canadian start with C, others are clearly non-US.
    """
    if not icao:
        return False
    return icao.upper().startswith("K")


def _parse_obs_age(ts_str: str) -> tuple[Optional[str], Optional[float]]:
    """Parse ISO timestamp → (observed_at_utc, obs_age_minutes)."""
    if not ts_str:
        return None, None
    try:
        from datetime import datetime, timezone as tz
        obs_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        age_s  = (datetime.now(tz.utc) - obs_dt).total_seconds()
        return ts_str, round(age_s / 60, 1)
    except Exception:
        return ts_str, None


async def _get_nws_obs(icao: str) -> Optional[dict]:
    """Fetch current observation from NWS (US stations only)."""
    url = f"https://api.weather.gov/stations/{icao}/observations/latest"
    try:
        print(f"[Stations/NWS] Fetching obs for {icao} ...")
        async with get_session().get(url, headers=NWS_HEADERS) as resp:
            if resp.status == 404:
                return None
            resp.raise_for_status()
            data = await resp.json()
    except Exception as e:
        print(f"[Stations/NWS] Error for {icao}: {e}")
        return None

    try:
        props    = data.get("properties", {})
        temp_c   = (props.get("temperature",       {}) or {}).get("value")
        dew_c    = (props.get("dewpoint",           {}) or {}).get("value")
        hum      = (props.get("relativeHumidity",   {}) or {}).get("value")
        wind_kph = (props.get("windSpeed",          {}) or {}).get("value")
        wind_mph = round(wind_kph / 1.60934, 1) if wind_kph is not None else None
        station_id = props.get("station", "").split("/")[-1] or icao
        obs_at, age_min = _parse_obs_age(props.get("timestamp"))
        if temp_c is None:
            return None
        result = {
            "temp_c":          round(float(temp_c), 1),
            "humidity_pct":    round(float(hum), 1) if hum is not None else None,
            "wind_mph":        wind_mph,
            "dewpoint_c":      round(float(dew_c), 1) if dew_c is not None else None,
            "station_id":      station_id,
            "station_name":    icao,
            "observed_at_utc": obs_at,
            "obs_age_minutes": age_min,
        }
        print(f"[Stations/NWS] {icao}: {result['temp_c']}°C, age={age_min}min")
        return result
    except Exception as e:
        print(f"[Stations/NWS] Parse error for {icao}: {e}")
        return None


async def _get_metar_obs(icao: str) -> Optional[dict]:
    """
    Fetch current METAR for any ICAO station via aviationweather.gov.
    Covers worldwide stations (RJTT, RKSS, EGLL, VHHH, etc.) that NWS won't serve.
    This is the same underlying data source Weather Underground uses for international
    stations, matching Polymarket's official resolution data.
    """
    url = "https://aviationweather.gov/api/data/metar"
    params = {"ids": icao, "format": "json", "hours": 2}
    try:
        print(f"[Stations/METAR] Fetching obs for {icao} ...")
        async with get_session().get(url, params=params, headers=NWS_HEADERS) as resp:
            resp.raise_for_status()
            data = await resp.json()
        if not data:
            print(f"[Stations/METAR] No data for {icao}")
            return None
        obs = data[0]  # most recent report
        temp_c = obs.get("temp")
        if temp_c is None:
            return None
        # obsTime is Unix epoch (seconds)
        obs_ts = obs.get("obsTime")
        obs_age_min = round((time.time() - obs_ts) / 60, 1) if obs_ts else None
        obs_at = obs.get("reportTime") or obs.get("receiptTime")
        dew_c  = obs.get("dewp")
        wind_kts = obs.get("wspd")
        wind_mph = round(wind_kts * 1.15078, 1) if wind_kts is not None else None
        result = {
            "temp_c":          round(float(temp_c), 1),
            "humidity_pct":    None,
            "wind_mph":        wind_mph,
            "dewpoint_c":      round(float(dew_c), 1) if dew_c is not None else None,
            "station_id":      icao,
            "station_name":    obs.get("icaoId", icao),
            "observed_at_utc": obs_at,
            "obs_age_minutes": obs_age_min,
        }
        print(f"[Stations/METAR] {icao}: {result['temp_c']}°C, age={obs_age_min}min")
        return result
    except Exception as e:
        print(f"[Stations/METAR] Error for {icao}: {e}")
        return None


async def get_station_obs(icao: str) -> Optional[dict]:
    """
    Fetch current observation for any ICAO station.
    - US stations (K prefix): NWS API
    - International: aviationweather.gov METAR (same source as Weather Underground)

    Returns dict with:
      temp_c, humidity_pct, wind_mph, dewpoint_c,
      station_id, station_name, observed_at_utc, obs_age_minutes
    """
    if not icao:
        return None
    icao = icao.upper()

    # Check cache
    now = time.time()
    async with _get_obs_lock():
        entry = _obs_cache.get(icao)
        if entry is not None:
            result, ts = entry
            if now - ts < OBS_CACHE_TTL:
                return result

    # Dispatch to correct source
    if _is_us_icao(icao):
        result = await _get_nws_obs(icao)
    else:
        result = await _get_metar_obs(icao)

    async with _get_obs_lock():
        _obs_cache[icao] = (result, now)
    return result

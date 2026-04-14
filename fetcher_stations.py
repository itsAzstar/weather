"""
fetcher_stations.py
Maps Polymarket weather cities to ASOS/AWOS ICAO station codes and fetches
current observations from the NWS API.

Polymarket markets resolve against specific airport weather stations
(ASOS/AWOS via Weather Underground / NOAA), not city-level forecasts.
"""

import threading
import time
import requests
from typing import Optional

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
_obs_lock = threading.Lock()
OBS_CACHE_TTL = 10 * 60  # 10 minutes (NWS obs update hourly)

NWS_HEADERS = {"User-Agent": "WeatherArb/2.0 (contact@example.com)"}


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


def get_station_obs(icao: str) -> Optional[dict]:
    """
    Fetch current ASOS/AWOS observation for the given ICAO station from NWS API.
    NWS only serves US stations — returns None gracefully for international codes.

    Returns dict with:
      temp_c, humidity_pct, wind_mph, dewpoint_c,
      station_id, station_name, observed_at_utc, obs_age_minutes
    """
    if not icao:
        return None

    icao = icao.upper()

    # NWS only covers US stations
    if not _is_us_icao(icao):
        print(f"[Stations] {icao} is non-US — skipping NWS obs fetch")
        return None

    # Check cache
    now = time.time()
    with _obs_lock:
        entry = _obs_cache.get(icao)
        if entry is not None:
            result, ts = entry
            if now - ts < OBS_CACHE_TTL:
                return result

    # Fetch from NWS
    url = f"https://api.weather.gov/stations/{icao}/observations/latest"
    try:
        print(f"[Stations] Fetching obs for {icao} ...")
        resp = requests.get(url, headers=NWS_HEADERS, timeout=10)
        if resp.status_code == 404:
            print(f"[Stations] Station {icao} not found in NWS (404)")
            with _obs_lock:
                _obs_cache[icao] = (None, now)
            return None
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        print(f"[Stations] Timeout fetching obs for {icao}")
        return None
    except Exception as e:
        print(f"[Stations] Error fetching obs for {icao}: {e}")
        return None

    try:
        props = data.get("properties", {})
        temp_prop = props.get("temperature", {}) or {}
        dew_prop  = props.get("dewpoint", {}) or {}
        hum_prop  = props.get("relativeHumidity", {}) or {}
        wind_prop = props.get("windSpeed", {}) or {}
        ts_str    = props.get("timestamp")  # ISO8601 UTC

        temp_c = temp_prop.get("value")  # Celsius from NWS JSON
        dew_c  = dew_prop.get("value")
        hum    = hum_prop.get("value")
        # NWS returns wind in km/h (unit: wmoUnit:km_h-1 or similar)
        wind_kph = wind_prop.get("value")
        wind_mph = round(wind_kph / 1.60934, 1) if wind_kph is not None else None

        # Station identity
        station_id   = props.get("station", "").split("/")[-1] or icao
        station_name = (data.get("properties", {}).get("rawMessage") or "")[:0]  # placeholder
        # Try to get from station URL
        # The obs response embeds station name in "station" URL — we use ICAO as fallback
        station_name = icao  # will be overridden if we can parse it

        # Parse obs age
        obs_age_minutes = None
        observed_at_utc = None
        if ts_str:
            observed_at_utc = ts_str
            try:
                from datetime import datetime, timezone as tz
                obs_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                age_s  = (datetime.now(tz.utc) - obs_dt).total_seconds()
                obs_age_minutes = round(age_s / 60, 1)
            except Exception:
                pass

        if temp_c is None:
            print(f"[Stations] {icao}: obs has no temperature value")
            with _obs_lock:
                _obs_cache[icao] = (None, now)
            return None

        result = {
            "temp_c":           round(float(temp_c), 1),
            "humidity_pct":     round(float(hum), 1) if hum is not None else None,
            "wind_mph":         wind_mph,
            "dewpoint_c":       round(float(dew_c), 1) if dew_c is not None else None,
            "station_id":       station_id,
            "station_name":     station_name,
            "observed_at_utc":  observed_at_utc,
            "obs_age_minutes":  obs_age_minutes,
        }
        print(f"[Stations] {icao}: {result['temp_c']}°C, age={obs_age_minutes}min")
        with _obs_lock:
            _obs_cache[icao] = (result, now)
        return result

    except Exception as e:
        print(f"[Stations] Error parsing obs for {icao}: {e}")
        with _obs_lock:
            _obs_cache[icao] = (None, now)
        return None

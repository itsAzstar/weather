"""
fetcher_wu.py
Scrapes Weather Underground history pages to get the exact daily high
temperature that Polymarket uses for resolution.

WU is Polymarket's official resolution source — a 1-2°C discrepancy
between METAR and WU can flip a bucket market outcome.

URL format:
  https://www.wunderground.com/history/daily/{ICAO}/date/{YYYY-M-D}

The page embeds a Next.js __NEXT_DATA__ script tag with the full
observation data as JSON — no API key required.
"""

import asyncio
import html as _html
import json
import re
import time
from datetime import date, datetime, timezone
from typing import Optional

from _http import get_session

# ── Cache ─────────────────────────────────────────────────────────────────────
# WU obs are final once the day is over. During the day, refresh frequently.
_wu_cache: dict[str, tuple[Optional[dict], float]] = {}
_wu_lock_inst: Optional[asyncio.Lock] = None

WU_CACHE_TTL_ACTIVE  = 15 * 60   # 15 min: today's market still resolving
WU_CACHE_TTL_SETTLED = 24 * 3600  # 24 h: past date, data won't change

WU_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def _get_wu_lock() -> asyncio.Lock:
    global _wu_lock_inst
    if _wu_lock_inst is None:
        _wu_lock_inst = asyncio.Lock()
    return _wu_lock_inst


def _cache_ttl(target_date: date) -> float:
    """
    Return appropriate cache TTL. A date is only "settled" when it's
    unambiguously past in *every* timezone — i.e. at least 2 UTC days behind.
    Otherwise (today ± 1 day UTC), a station somewhere on earth still has that
    calendar day active, so use the short TTL.

    Bug D fix: the previous version used `target_date >= date.today()` with
    UTC today. From Taiwan (UTC+8) at noon, UTC today is already tomorrow for
    NYC, so `target_date == NYC's today` fell into SETTLED (24h) and served
    stale data for a market that was still actively resolving.
    """
    utc_today = datetime.now(timezone.utc).date()
    days_behind = (utc_today - target_date).days
    if days_behind >= 2:
        return WU_CACHE_TTL_SETTLED
    return WU_CACHE_TTL_ACTIVE


def _extract_temp_from_next_data(html: str) -> tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Parse the __NEXT_DATA__ JSON blob embedded in WU's Next.js page.
    Returns (high_robust_f, high_raw_f, data_source) where data_source is
    one of: "dailysummary" | "observations" | "history_nested" | None.

    The data_source tag is critical for wu_definitive decisions:
      - "dailysummary" = WU's own QC'd daily roll-up. Aligns with what
        Polymarket resolves against. Spike-free, reliable. SAFE to use for
        definitive dead/alive calls.
      - "observations" = raw hourly sensor stream pre-QC. Single-hour spikes
        routinely show temps 1-2°C above what dailysummary later reports.
        Wellington 2026-04-21 and 4-22, Madrid 4-22 all false-dead'd from
        this path. UNSAFE for definitive calls, only for monitoring.

    high_robust_f drops the single highest observation when N>=4 to filter
    transient sensor spikes. high_raw_f is the true max across all obs.
    """
    m = re.search(
        r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
        html, re.DOTALL
    )
    # ── Path 0 (2026-04 refresh): WU replaced __NEXT_DATA__ with `app-root-state`.
    # The new schema stores QC'd daily highs at the top level of a numerically-keyed
    # cache dict: data[<cache_id>].b.calendarDayTemperatureMax = [target_day_high, ...].
    # Index [0] corresponds to the URL's target date (verified against the rendered
    # HTML "High: XX°F" table — both return the same value for KDAL 2026-04-22 = 82).
    # This path is the new "dailysummary" source — safe for wu_definitive gating.
    m0 = re.search(
        r'<script[^>]*id=["\']app-root-state["\'][^>]*>(.*?)</script>',
        html, re.DOTALL
    )
    if m0:
        try:
            data0 = json.loads(_html.unescape(m0.group(1)))
            for _k, _v in data0.items():
                if not isinstance(_v, dict):
                    continue
                b = _v.get("b")
                if not isinstance(b, dict):
                    continue
                cdtm = b.get("calendarDayTemperatureMax") or b.get("temperatureMax")
                if isinstance(cdtm, list) and cdtm and cdtm[0] is not None:
                    try:
                        v = float(cdtm[0])
                        # calendarDayTemperatureMax is already in °F on imperial pages
                        return (v, v, "dailysummary")
                    except (ValueError, TypeError):
                        continue
        except (json.JSONDecodeError, ValueError, AttributeError):
            pass

    # ── Legacy __NEXT_DATA__ path (pre-2026-04 WU, still present on some routes) ──
    if not m:
        return (None, None, None)

    try:
        data = json.loads(m.group(1))
    except (json.JSONDecodeError, ValueError):
        return (None, None, None)

    # ── Path 1: historySummary.dailysummary[0].hightempi / hightempm ─────────
    # dailysummary is WU's own QC'd daily roll-up — no spike filtering needed.
    try:
        daily = (
            data["props"]["pageProps"]["historySummary"]["dailysummary"]
        )
        if daily:
            row = daily[0] if isinstance(daily, list) else daily
            hi_f = (
                row.get("hightempi")
                or row.get("maxtempi")
                or (row.get("hightemp", {}) or {}).get("imperial", {}).get("value")
            )
            if hi_f is not None:
                v = float(hi_f)
                return (v, v, "dailysummary")
            hi_c = row.get("hightempm") or row.get("maxtempm")
            if hi_c is not None:
                v = float(hi_c) * 9 / 5 + 32
                return (v, v, "dailysummary")
    except (KeyError, TypeError, IndexError):
        pass

    # ── Path 2: observations array — return both robust and raw max ──────────
    # Raw max = true max across all hourly obs (used for floor check — we
    #   want to be conservative about calling a bucket dead-below-floor when
    #   one legitimate hour may have peaked inside the bucket).
    # Robust max = drop single highest when N>=4 to filter sensor spikes
    #   (used for ceiling check — Houston 2026-04-18 spike case).
    try:
        obs_list = (
            data["props"]["pageProps"]["historySummary"]["observations"]
        )
        if obs_list:
            temps_f = []
            for obs in obs_list:
                t = (
                    (obs.get("temperatureMax") or {}).get("imperial", {}).get("value")
                    or (obs.get("temp") or {}).get("imperial", {}).get("value")
                    or obs.get("tempi")
                    or obs.get("temp_f")
                )
                if t is not None:
                    try:
                        temps_f.append(float(t))
                    except (ValueError, TypeError):
                        continue
            if temps_f:
                temps_f.sort(reverse=True)
                raw = temps_f[0]
                robust = temps_f[1] if len(temps_f) >= 4 else temps_f[0]
                return (robust, raw, "observations")
    except (KeyError, TypeError):
        pass

    # ── Path 3: nested under a different key structure ────────────────────────
    try:
        history = data["props"]["pageProps"]["history"]
        if history:
            daily = history.get("dailysummary") or history.get("observations", [])
            if isinstance(daily, list) and daily:
                row = daily[0]
                hi_f = (
                    row.get("hightempi")
                    or row.get("maxtempi")
                    or row.get("TemperatureMaxF")
                    or row.get("TemperatureHighF")
                )
                if hi_f is not None:
                    v = float(hi_f)
                    return (v, v, "history_nested")
    except (KeyError, TypeError, IndexError):
        pass

    return (None, None, None)


def _extract_temp_from_html_table(html: str) -> Optional[float]:
    """
    Fallback: extract daily high temperature directly from the HTML table.
    WU's history page renders a summary table with "High" / "Low" / "Avg" rows.
    """
    # Look for patterns like: "High" ... "82" ... "°F"
    # The table format: <td>High</td>...<td>82</td>
    patterns = [
        # Pattern: High ... number ... °F
        r'(?:High|Maximum|Max\s*Temp)[^<]*</(?:td|th)>[^<]*(?:<[^>]+>[^<]*</[^>]+>\s*)*<(?:td|th)[^>]*>\s*(-?\d+(?:\.\d+)?)',
        # Pattern: "82 °F" near "high" keyword
        r'high[^<]{0,200}?(-?\d{2,3})\s*°?\s*[Ff]',
        # Table row with "Maximum" label
        r'(?:Maximum|High)\s*Temperature[^<]*</[^>]+>\s*(?:<[^>]+>)*\s*(-?\d+(?:\.\d+)?)',
    ]
    for pat in patterns:
        m = re.search(pat, html, re.IGNORECASE | re.DOTALL)
        if m:
            try:
                return float(m.group(1))
            except (ValueError, TypeError):
                continue
    return None


async def get_wu_daily_high(
    icao: str,
    target_date: date,
    unit: str = "F",
) -> Optional[dict]:
    """
    Fetch Weather Underground daily high temperature for a station and date.

    Args:
        icao: ICAO station code (e.g. "KJFK", "EGLL")
        target_date: The calendar date to query
        unit: "F" (default) or "C" — unit of the returned value

    Returns dict with:
        wu_temp_f    (float) — daily high in °F
        wu_temp_c    (float) — daily high in °C
        wu_date      (str)   — date queried (ISO)
        wu_station   (str)   — ICAO code
        wu_age_min   (float) — minutes since last WU update
        wu_source    (str)   — "live" or "cache"
    or None on failure.
    """
    icao = icao.upper()
    key  = f"{icao}:{target_date.isoformat()}"
    now  = time.time()
    ttl  = _cache_ttl(target_date)

    async with _get_wu_lock():
        entry = _wu_cache.get(key)
        if entry is not None:
            result, ts = entry
            if now - ts < ttl:
                if result is not None:
                    return {**result, "wu_source": "cache"}
                # Cached None (404 / parse failure) — respect TTL before retry
                return None

    # Fetch WU history page
    date_str = f"{target_date.year}-{target_date.month}-{target_date.day}"
    url = f"https://www.wunderground.com/history/daily/{icao}/date/{date_str}"

    try:
        print(f"[WU] Fetching {icao} {target_date} ...")
        async with get_session().get(url, headers=WU_HEADERS, allow_redirects=True) as resp:
            if resp.status == 404:
                print(f"[WU] 404 for {icao} {target_date}")
                async with _get_wu_lock():
                    _wu_cache[key] = (None, now)
                return None
            if resp.status != 200:
                print(f"[WU] HTTP {resp.status} for {icao} {target_date}")
                return None   # Don't cache transient errors
            html = await resp.text()

        # Try __NEXT_DATA__ first (authoritative), then HTML table fallback
        temp_f, temp_f_raw, data_src = _extract_temp_from_next_data(html)
        if temp_f is None:
            temp_f = _extract_temp_from_html_table(html)
            temp_f_raw = temp_f   # table fallback: no obs list to compute raw
            data_src = "html_table" if temp_f is not None else None

        if temp_f is None:
            print(f"[WU] Could not parse temperature for {icao} {target_date}")
            # Don't cache parse failures — WU page structure may change
            return None

        temp_c      = round((temp_f - 32) * 5 / 9, 1)
        temp_f      = round(temp_f, 1)
        temp_f_raw  = round(temp_f_raw, 1) if temp_f_raw is not None else temp_f
        temp_c_raw  = round((temp_f_raw - 32) * 5 / 9, 1)

        # Estimate observation age: WU updates hourly for active days.
        # "Settled" only when target_date is >= 2 UTC days behind (past in every tz).
        utc_today = datetime.now(timezone.utc).date()
        if (utc_today - target_date).days >= 2:
            age_min = None   # Settled — age not meaningful
        else:
            # WU updates approx on the hour
            now_dt = datetime.now(timezone.utc)
            age_min = round(now_dt.minute + (now_dt.second / 60), 1)

        result = {
            "wu_temp_f":        temp_f,
            "wu_temp_c":        temp_c,
            "wu_temp_f_raw":    temp_f_raw,
            "wu_temp_c_raw":    temp_c_raw,
            "wu_date":          target_date.isoformat(),
            "wu_station":       icao,
            "wu_age_min":       age_min,
            "wu_data_source":   data_src,   # dailysummary | observations | ...
        }
        spike_note = "" if temp_f_raw == temp_f else f" (raw={temp_f_raw}°F)"
        print(f"[WU] {icao} {target_date}: high={temp_f}°F ({temp_c}°C) "
              f"src={data_src}{spike_note}")

        async with _get_wu_lock():
            _wu_cache[key] = (result, now)

        return {**result, "wu_source": "live"}

    except asyncio.TimeoutError:
        print(f"[WU] Timeout for {icao} {target_date}")
        return None
    except Exception as e:
        print(f"[WU] Error for {icao} {target_date}: {e}")
        return None


async def get_wu_temp_cached(icao: str, target_date: date) -> Optional[dict]:
    """Thin wrapper: always go through the main cache."""
    return await get_wu_daily_high(icao, target_date)

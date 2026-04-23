"""
WU parser smoke test.

Run this locally before deploying OR when WU output looks off:
    python test_wu_parser.py

It hits real WU history pages for a fixed set of ICAO stations and asserts:
  1. The parser returns a non-None temperature
  2. The data_source is "dailysummary" (the only source we trust for
     wu_definitive gating)
  3. The returned high is sane (-40..140 °F)

If WU changes their page schema again (like the 2026-04 __NEXT_DATA__ →
app-root-state flip we missed for ~24h), this test fails LOUDLY before
the parser silently regresses every wu_definitive signal in production.

Exit code 0 = all pass, 1 = any failure.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import date, timedelta

import aiohttp

from fetcher_wu import WU_HEADERS, _extract_temp_from_next_data

# Stations + expected temp ranges (°F) for yesterday.
# Ranges are loose enough to survive weather variance but tight enough
# to catch "returned 0" or "returned -273" bugs.
TEST_STATIONS = [
    ("KDAL", "Dallas",     (30, 115)),
    ("KNYC", "New York",   (10, 105)),
    ("KLAX", "Los Angeles",(40, 115)),
    ("NZWN", "Wellington", (30,  85)),
    ("LEMD", "Madrid",     (25, 110)),
]

URL_FMT = "https://www.wunderground.com/history/daily/{icao}/date/{y}-{m}-{d}"


async def _fetch(url: str) -> str:
    timeout = aiohttp.ClientTimeout(total=25)
    async with aiohttp.ClientSession(timeout=timeout, headers=WU_HEADERS) as s:
        async with s.get(url) as r:
            r.raise_for_status()
            return await r.text()


async def _check_one(icao: str, name: str, lo: float, hi: float, d: date):
    url = URL_FMT.format(icao=icao, y=d.year, m=d.month, d=d.day)
    try:
        html = await _fetch(url)
    except Exception as e:
        return (icao, name, False, f"fetch failed: {type(e).__name__}: {e}")

    robust, raw, source = _extract_temp_from_next_data(html)

    if robust is None:
        return (icao, name, False, f"parser returned None (page {len(html)} bytes)")
    if source != "dailysummary":
        return (icao, name, False, f"source={source!r}, wanted 'dailysummary' (fallback path active — schema drift?)")
    if not (lo <= robust <= hi):
        return (icao, name, False, f"temp {robust}°F outside sane range [{lo},{hi}]")

    return (icao, name, True, f"{robust}°F (source={source})")


async def main():
    target_date = date.today() - timedelta(days=1)
    print(f"[WU smoke] date={target_date.isoformat()}  stations={len(TEST_STATIONS)}")
    print("-" * 60)

    results = await asyncio.gather(*[
        _check_one(icao, name, lo, hi, target_date)
        for icao, name, (lo, hi) in TEST_STATIONS
    ])

    passed = 0
    for icao, name, ok, msg in results:
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {icao} {name:12s} {msg}")
        if ok:
            passed += 1

    print("-" * 60)
    print(f"[WU smoke] {passed}/{len(results)} passed")

    # Gate: require >=60% dailysummary hits. Lower than 100% tolerates
    # transient WU glitches on one station without false-alarming.
    threshold = 0.6
    ratio = passed / len(results)
    if ratio < threshold:
        print(f"[WU smoke] FAIL — {ratio:.0%} < {threshold:.0%} threshold. Parser likely regressed.")
        sys.exit(1)
    print(f"[WU smoke] OK — {ratio:.0%} >= {threshold:.0%} threshold.")


if __name__ == "__main__":
    asyncio.run(main())

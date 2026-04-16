"""
fetcher_polymarket.py
Fetches active Polymarket weather-related markets from CLOB and Gamma APIs.
"""

import re
import requests
import time
import json
import threading
from datetime import datetime, timezone, date, timedelta

CLOB_BASE = "https://clob.polymarket.com"
GAMMA_BASE = "https://gamma-api.polymarket.com"

# ── L2 order book cache ───────────────────────────────────────────────────────
_book_cache: dict[str, tuple] = {}   # token_id → (result_or_None, timestamp)
_book_lock  = threading.Lock()
BOOK_CACHE_TTL = 2 * 60   # 2 minutes: order books are volatile


def fetch_clob_book(token_id: str) -> dict | None:
    """
    Fetch the L2 order book for a YES token from Polymarket CLOB.

    Returns:
        {"best_bid": float, "best_ask": float,
         "spread": float, "half_spread": float, "mid": float}
    or None if unavailable.

    Cached 2 minutes.  The spread from this function is authoritative —
    use it instead of any estimated spread formula.
    """
    if not token_id:
        return None

    now = time.time()
    with _book_lock:
        entry = _book_cache.get(token_id)
        if entry:
            result, ts = entry
            if now - ts < BOOK_CACHE_TTL:
                return result

    try:
        data = _safe_get(f"{CLOB_BASE}/book", params={"token_id": token_id})
        if not data:
            with _book_lock:
                _book_cache[token_id] = (None, now)
            return None

        # Sort bids descending, asks ascending — pick best levels
        bids = sorted(data.get("bids", []),
                      key=lambda x: float(x.get("price", 0)), reverse=True)
        asks = sorted(data.get("asks", []),
                      key=lambda x: float(x.get("price", 999)))

        best_bid = float(bids[0]["price"]) if bids else None
        best_ask = float(asks[0]["price"]) if asks else None

        if best_bid is None or best_ask is None or best_ask <= best_bid:
            result = None
        else:
            spread = best_ask - best_bid
            result = {
                "best_bid":    round(best_bid, 4),
                "best_ask":    round(best_ask, 4),
                "spread":      round(spread, 4),
                "half_spread": round(spread / 2.0, 4),
                "mid":         round((best_ask + best_bid) / 2.0, 4),
            }
            print(f"[CLOB Book] {token_id[:10]}: bid={best_bid:.3f} ask={best_ask:.3f} "
                  f"spread={spread:.3f}")

        with _book_lock:
            _book_cache[token_id] = (result, now)
        return result

    except Exception as e:
        print(f"[CLOB Book] Error for {token_id}: {e}")
        return None

WEATHER_KEYWORDS = [
    "rain", "rainfall", "precipitation", "weather",
    "snow", "snowfall", "storm", "hurricane", "tornado", "flood",
    "humidity", "drought", "frost",
    "celsius", "fahrenheit", "typhoon", "cyclone", "blizzard",
    "will it rain", "will it snow", "temperature exceed", "degrees fahrenheit",
    "degrees celsius", "inches of rain", "mm of rain", "wind gust",
]

# ── Temperature bucket parsing ─────────────────────────────────────────────────

def _parse_temp_bucket(group_item_title: str) -> dict | None:
    """
    Parse temperature bucket from groupItemTitle (from Polymarket Gamma events API).
    Returns dict with lo_c, hi_c (Celsius bounds), direction, display, mid_c.
    Returns None if unparseable.

    Examples:
      "22°C"            → exact, lo_c=21.5, hi_c=22.5
      "27°C or below"   → below, lo_c=None, hi_c=27.5
      "24°C or higher"  → above, lo_c=23.5, hi_c=None
      "80-81°F"         → range, lo_c=26.4, hi_c=27.5
      "58°F or higher"  → above, lo_c=14.2, hi_c=None
    """
    s = group_item_title.strip()
    # Normalize unicode degree symbols (Polymarket uses \u7c1e sometimes)
    s = s.replace('\u7c1e', '°').replace('\uf8ff', '')

    # Pattern: range like "80-81°F" or "20-21°C"
    m = re.match(r'^(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*°?\s*([CF])\b', s, re.I)
    if m:
        lo_val = float(m.group(1))
        hi_val = float(m.group(2))
        unit = m.group(3).upper()
        if unit == 'F':
            lo_c = (lo_val - 0.5 - 32) * 5 / 9
            hi_c = (hi_val + 0.5 - 32) * 5 / 9
            mid_c = ((lo_val + hi_val) / 2 - 32) * 5 / 9
        else:
            lo_c = lo_val - 0.5
            hi_c = hi_val + 0.5
            mid_c = (lo_val + hi_val) / 2
        return {'lo_c': lo_c, 'hi_c': hi_c, 'lo': lo_val, 'hi': hi_val,
                'unit': unit, 'direction': 'range', 'display': s, 'mid_c': mid_c}

    # Pattern: single value with optional "or higher"/"or below"
    m = re.match(r'^(-?\d+(?:\.\d+)?)\s*°?\s*([CF])\s*(or\s+(?:higher|above|below|lower))?\s*$', s, re.I)
    if m:
        val = float(m.group(1))
        unit = m.group(2).upper()
        modifier = (m.group(3) or '').lower()
        val_c = (val - 32) * 5 / 9 if unit == 'F' else val

        if 'higher' in modifier or 'above' in modifier:
            return {'lo_c': val_c - 0.5, 'hi_c': None, 'val': val, 'unit': unit,
                    'direction': 'above', 'display': s, 'mid_c': val_c}
        elif 'lower' in modifier or 'below' in modifier:
            return {'lo_c': None, 'hi_c': val_c + 0.5, 'val': val, 'unit': unit,
                    'direction': 'below', 'display': s, 'mid_c': val_c}
        else:
            return {'lo_c': val_c - 0.5, 'hi_c': val_c + 0.5, 'val': val, 'unit': unit,
                    'direction': 'exact', 'display': s, 'mid_c': val_c}
    return None

# Terms that appear in non-weather markets containing weather words (NBA Heat, Cold War, etc.)
WEATHER_EXCLUSION_PATTERNS = [
    r"\bnba\b",
    r"\bnfl\b",
    r"\bnhl\b",
    r"\bmlb\b",
    r"\bsoccer\b",
    r"\bfootball\b",
    r"\bbasketball\b",
    r"\bheat\s+vs\b",       # Miami Heat vs
    r"\bvs\.\s+heat\b",
    r"\bvs\s+heat\b",
    r"\bheat\s+win\b",
    r"\bheat\s+loss\b",
    r"\bheat\s+cover\b",
    r"\bhawks\s+vs\b",
    r"\bceltics\b",
    r"\bbucks\b",
    r"\bnuggets\b",
    r"\bknicks\b",
    r"\bmavericks\b",
    r"\bthunder\b.*\bvs\b",  # OKC Thunder vs
    r"\blightning\b.*\bvs\b", # Tampa Bay Lightning vs
    r"\bstorm\s+vs\b",       # Seattle Storm (WNBA) vs
    r"\bcold\s+war\b",
    r"\bice\s+cream\b",
    r"\bice\s+age\b",
    r"\bold\s+spice\b",
    r"\bwind\s+(energy|power|turbine|farm)\b",
]

def _mk_mock_date(offset_days: int) -> str:
    """Generate an ISO date string offset_days from today."""
    d = date.today() + timedelta(days=offset_days)
    return d.strftime("%Y-%m-%d")


def _mock_date_label(offset_days: int) -> str:
    """Human-readable date label for mock market questions."""
    d = date.today() + timedelta(days=offset_days)
    return d.strftime("%B %-d, %Y") if hasattr(d, 'strftime') else str(d)


def _build_mock_markets() -> list[dict]:
    """
    Build mock weather markets with dates always relative to today.
    This ensures the pipeline always has data to demonstrate.
    Called at import time so MOCK_WEATHER_MARKETS is always fresh.
    """
    def fmt(offset):
        d = date.today() + timedelta(days=offset)
        try:
            label = d.strftime("%B {day}, %Y").replace("{day}", str(d.day))
        except Exception:
            label = str(d)
        return d.isoformat(), d.strftime("%B") + f" {d.day}, {d.year}", f"{d.isoformat()}T23:59:00Z"

    d0_iso, d0_lbl, d0_end = fmt(0)
    d1_iso, d1_lbl, d1_end = fmt(1)
    d2_iso, d2_lbl, d2_end = fmt(2)
    d3_iso, d3_lbl, d3_end = fmt(3)
    d4_iso, d4_lbl, d4_end = fmt(4)

    return [
        {
            "question": f"Will it rain in London on {d1_lbl}?",
            "condition_id": f"mock-london-rain-{d1_iso}",
            "market_price_yes": 0.42,
            "market_price_no": 0.58,
            "end_date": d1_end,
            "description": f"Resolves YES if London receives measurable precipitation on {d1_lbl}.",
            "location_hint": "London",
            "url": None,
            "source": "mock",
        },
        {
            "question": f"Will New York City temperature exceed 60°F on {d1_lbl}?",
            "condition_id": f"mock-nyc-temp-{d1_iso}",
            "market_price_yes": 0.55,
            "market_price_no": 0.45,
            "end_date": d1_end,
            "description": f"Resolves YES if NYC max temperature exceeds 60°F on {d1_lbl}.",
            "location_hint": "New York",
            "url": None,
            "source": "mock",
        },
        {
            "question": f"Will Paris receive more than 5mm of rain on {d2_lbl}?",
            "condition_id": f"mock-paris-rain-{d2_iso}",
            "market_price_yes": 0.31,
            "market_price_no": 0.69,
            "end_date": d2_end,
            "description": f"Resolves YES if Paris receives more than 5mm of precipitation on {d2_lbl}.",
            "location_hint": "Paris",
            "url": None,
            "source": "mock",
        },
        {
            "question": f"Will Tokyo experience rain on {d2_lbl}?",
            "condition_id": f"mock-tokyo-rain-{d2_iso}",
            "market_price_yes": 0.48,
            "market_price_no": 0.52,
            "end_date": d2_end,
            "description": f"Resolves YES if Tokyo receives measurable precipitation on {d2_lbl}.",
            "location_hint": "Tokyo",
            "url": None,
            "source": "mock",
        },
        {
            "question": f"Will Sydney temperature exceed 25°C on {d3_lbl}?",
            "condition_id": f"mock-sydney-temp-{d3_iso}",
            "market_price_yes": 0.25,
            "market_price_no": 0.75,
            "end_date": d3_end,
            "description": f"Resolves YES if Sydney max temperature exceeds 25°C on {d3_lbl}.",
            "location_hint": "Sydney",
            "url": None,
            "source": "mock",
        },
        {
            "question": f"Will Dubai temperature exceed 35°C on {d3_lbl}?",
            "condition_id": f"mock-dubai-temp-{d3_iso}",
            "market_price_yes": 0.71,
            "market_price_no": 0.29,
            "end_date": d3_end,
            "description": f"Resolves YES if Dubai max temperature exceeds 35°C on {d3_lbl}.",
            "location_hint": "Dubai",
            "url": None,
            "source": "mock",
        },
        {
            "question": f"Will Singapore receive rain on {d4_lbl}?",
            "condition_id": f"mock-singapore-rain-{d4_iso}",
            "market_price_yes": 0.60,
            "market_price_no": 0.40,
            "end_date": d4_end,
            "description": f"Resolves YES if Singapore receives measurable precipitation on {d4_lbl}.",
            "location_hint": "Singapore",
            "url": None,
            "source": "mock",
        },
        {
            "question": f"Will there be a wind gust above 40mph in Chicago on {d2_lbl}?",
            "condition_id": f"mock-chicago-wind-{d2_iso}",
            "market_price_yes": 0.38,
            "market_price_no": 0.62,
            "end_date": d2_end,
            "description": f"Resolves YES if Chicago records a wind gust above 40mph on {d2_lbl}.",
            "location_hint": "Chicago",
            "url": None,
            "source": "mock",
        },
        {
            "question": f"Will Los Angeles temperature exceed 75°F on {d1_lbl}?",
            "condition_id": f"mock-la-temp-{d1_iso}",
            "market_price_yes": 0.63,
            "market_price_no": 0.37,
            "end_date": d1_end,
            "description": f"Resolves YES if LA max temperature exceeds 75°F on {d1_lbl}.",
            "location_hint": "Los Angeles",
            "url": None,
            "source": "mock",
        },
        {
            "question": f"Will Berlin experience rain on {d3_lbl}?",
            "condition_id": f"mock-berlin-rain-{d3_iso}",
            "market_price_yes": 0.45,
            "market_price_no": 0.55,
            "end_date": d3_end,
            "description": f"Resolves YES if Berlin receives measurable precipitation on {d3_lbl}.",
            "location_hint": "Berlin",
            "url": None,
            "source": "mock",
        },
        {
            "question": f"Will Miami temperature exceed 80°F on {d4_lbl}?",
            "condition_id": f"mock-miami-temp-{d4_iso}",
            "market_price_yes": 0.78,
            "market_price_no": 0.22,
            "end_date": d4_end,
            "description": f"Resolves YES if Miami max temperature exceeds 80°F on {d4_lbl}.",
            "location_hint": "Miami",
            "url": None,
            "source": "mock",
        },
        {
            "question": f"Will Seoul receive rain on {d2_lbl}?",
            "condition_id": f"mock-seoul-rain-{d2_iso}",
            "market_price_yes": 0.33,
            "market_price_no": 0.67,
            "end_date": d2_end,
            "description": f"Resolves YES if Seoul receives measurable precipitation on {d2_lbl}.",
            "location_hint": "Seoul",
            "url": None,
            "source": "mock",
        },
    ]


# Module-level mock list — always relative to today
MOCK_WEATHER_MARKETS = _build_mock_markets()


def _is_weather_market(text: str) -> bool:
    """
    Return True if the text looks like a genuine weather market.
    Uses keyword matching with exclusion patterns to filter out sports markets
    that happen to contain words like 'heat', 'storm', 'thunder', etc.
    """
    lower = text.lower()

    # Must contain at least one weather keyword
    has_keyword = any(kw in lower for kw in WEATHER_KEYWORDS)
    if not has_keyword:
        return False

    # Reject if any exclusion pattern matches
    for pat in WEATHER_EXCLUSION_PATTERNS:
        if re.search(pat, lower):
            return False

    return True


def _safe_get(url: str, params: dict = None, retries: int = 2, timeout: int = 10) -> dict | list | None:
    """GET with retry logic."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            print(f"  [HTTP error] {e} — attempt {attempt + 1}/{retries}")
        except requests.exceptions.ConnectionError as e:
            print(f"  [Connection error] {e} — attempt {attempt + 1}/{retries}")
        except requests.exceptions.Timeout:
            print(f"  [Timeout] — attempt {attempt + 1}/{retries}")
        except Exception as e:
            print(f"  [Unexpected error] {e} — attempt {attempt + 1}/{retries}")
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    return None


def _parse_clob_market(raw: dict) -> dict | None:
    """Normalise a raw CLOB market dict into our schema."""
    question = raw.get("question", "")
    desc = raw.get("description", "") or ""
    if not _is_weather_market(question + " " + desc):
        return None

    # Try to get a market price and token_id from tokens
    price_yes = None
    price_no = None
    token_id_yes = None
    token_id_no  = None
    tokens = raw.get("tokens", [])
    for token in tokens:
        outcome = (token.get("outcome") or "").lower()
        price = token.get("price")
        tid   = token.get("token_id")
        if outcome == "yes":
            token_id_yes = tid
            if price is not None:
                try:
                    price_yes = float(price)
                except (ValueError, TypeError):
                    pass
        elif outcome == "no":
            token_id_no = tid
            if price is not None:
                try:
                    price_no = float(price)
                except (ValueError, TypeError):
                    pass

    if price_yes is None and price_no is not None:
        price_yes = 1.0 - price_no
    if price_no is None and price_yes is not None:
        price_no = 1.0 - price_yes

    condition_id = raw.get("condition_id", raw.get("id", ""))
    slug = raw.get("market_slug", condition_id)
    return {
        "question":        question,
        "condition_id":    condition_id,
        "market_price_yes": price_yes,
        "market_price_no":  price_no,
        "token_id_yes":    token_id_yes,
        "token_id_no":     token_id_no,
        "end_date":        raw.get("end_date_iso") or raw.get("end_date"),
        "description":     desc,
        "location_hint":   None,
        "url":             f"https://polymarket.com/event/{slug}",
        "source":          "clob",
    }


def _parse_outcome_prices(raw: dict) -> tuple[float | None, float | None]:
    """Extract (price_yes, price_no) from a Gamma market dict.
    outcomePrices can be a JSON string like '["0.34","0.66"]' or a list."""
    op = raw.get("outcomePrices", "[]")
    if isinstance(op, str):
        try:
            op = json.loads(op)
        except Exception:
            op = []
    price_yes = price_no = None
    try:
        if len(op) >= 1:
            price_yes = float(op[0])
        if len(op) >= 2:
            price_no = float(op[1])
    except (ValueError, TypeError):
        pass
    if price_yes is None and price_no is not None:
        price_yes = 1.0 - price_no
    if price_no is None and price_yes is not None:
        price_no = 1.0 - price_yes
    return price_yes, price_no


def _parse_gamma_market(raw: dict) -> dict | None:
    """Normalise a raw Gamma API market dict into our schema."""
    question = raw.get("question", "")
    desc = raw.get("description", "") or ""
    if not _is_weather_market(question + " " + desc):
        return None

    price_yes, price_no = _parse_outcome_prices(raw)

    slug = raw.get("slug", raw.get("id", ""))
    return {
        "question": question,
        "condition_id": raw.get("conditionId", raw.get("id", "")),
        "market_price_yes": price_yes,
        "market_price_no": price_no,
        "end_date": raw.get("endDate") or raw.get("endDateIso"),
        "description": desc,
        "location_hint": None,
        "url": f"https://polymarket.com/event/{slug}",
        "source": "gamma",
    }


def fetch_clob_markets(max_pages: int = 5) -> list[dict]:
    """Fetch weather markets from Polymarket CLOB API (paginated)."""
    print("[CLOB] Fetching markets from Polymarket CLOB API...")
    markets = []
    next_cursor = None

    for page in range(max_pages):
        params = {"active": "true", "closed": "false", "limit": 100}
        if next_cursor:
            params["next_cursor"] = next_cursor

        data = _safe_get(f"{CLOB_BASE}/markets", params=params)
        if data is None:
            print(f"  [CLOB] Failed to fetch page {page + 1}, stopping.")
            break

        raw_list = []
        if isinstance(data, list):
            raw_list = data
            next_cursor = None
        elif isinstance(data, dict):
            raw_list = data.get("data", [])
            next_cursor = data.get("next_cursor")

        print(f"  [CLOB] Page {page + 1}: got {len(raw_list)} markets")
        for raw in raw_list:
            parsed = _parse_clob_market(raw)
            if parsed:
                markets.append(parsed)

        if not next_cursor or not raw_list:
            break
        time.sleep(0.2)

    print(f"  [CLOB] Found {len(markets)} weather markets")
    return markets


def fetch_gamma_markets(max_pages: int = 5) -> list[dict]:
    """Fetch weather markets from Polymarket Gamma API."""
    print("[Gamma] Fetching markets from Polymarket Gamma API...")
    markets = []

    for offset in range(0, max_pages * 100, 100):
        params = {
            "active": "true",
            "closed": "false",
            "limit": 100,
            "offset": offset,
        }
        data = _safe_get(f"{GAMMA_BASE}/markets", params=params)
        if data is None:
            print(f"  [Gamma] Failed at offset {offset}, stopping.")
            break

        raw_list = data if isinstance(data, list) else data.get("markets", data.get("data", []))
        print(f"  [Gamma] Offset {offset}: got {len(raw_list)} markets")

        for raw in raw_list:
            parsed = _parse_gamma_market(raw)
            if parsed:
                markets.append(parsed)

        if len(raw_list) < 100:
            break
        time.sleep(0.2)

    print(f"  [Gamma] Found {len(markets)} weather markets")
    return markets


def fetch_gamma_weather_events(max_pages: int = 2) -> list[dict]:
    """
    Fetch real Polymarket daily temperature markets via Gamma events API.
    These are 'Highest temperature in [City] on [Date]?' markets with
    multiple temperature bucket sub-markets each.
    Returns flat list of market dicts, one per sub-market.
    """
    print("[Gamma Events] Fetching weather temperature events...")
    markets = []

    for page_num in range(max_pages):
        offset = page_num * 100
        params = {
            "active":     "true",
            "closed":     "false",
            "limit":      100,
            "offset":     offset,
            "tag_slug":   "weather",
            "order":      "volume24hr",
            "ascending":  "false",
        }
        data = _safe_get(f"{GAMMA_BASE}/events", params=params)
        if data is None:
            print(f"  [Gamma Events] Failed at offset {offset}, stopping.")
            break

        events = data if isinstance(data, list) else data.get("events", [])
        print(f"  [Gamma Events] Offset {offset}: got {len(events)} events")

        for event in events:
            title = event.get("title", "")
            # Only process "Highest temperature in ..." events
            if "highest temperature" not in title.lower():
                continue

            event_slug = event.get("slug", "")
            end_date_raw = event.get("endDate", "")

            # Extract city name from title: "Highest temperature in [City] on [Date]?"
            city_m = re.search(r"highest temperature in (.+?) on ", title, re.I)
            if not city_m:
                continue
            city = city_m.group(1).strip()

            # Process each temperature-bucket sub-market
            for sub in event.get("markets", []):
                if not sub.get("active", True):
                    continue

                group_title = sub.get("groupItemTitle", "")
                if not group_title:
                    continue

                bucket = _parse_temp_bucket(group_title)
                if bucket is None:
                    continue

                price_yes, price_no = _parse_outcome_prices(sub)

                if price_yes is None:
                    continue

                sub_slug = sub.get("slug", "")
                question = sub.get("question") or f"Will the highest temperature in {city} be {group_title}?"

                markets.append({
                    "question":       question,
                    "condition_id":   sub.get("conditionId") or sub.get("id", ""),
                    "market_price_yes": price_yes,
                    "market_price_no":  (price_no if price_no is not None else round(1.0 - price_yes, 4)),
                    "end_date":       sub.get("endDate") or end_date_raw,
                    "description":    sub.get("description", ""),
                    "location_hint":  city,
                    "url":            (f"https://polymarket.com/event/{event_slug}" if event_slug else None),
                    "source":         "gamma_event",
                    "market_subtype": "temperature_bucket",
                    "temp_bucket":    bucket,
                    "temp_display":   group_title,
                    "event_title":    title,
                    "liquidity":      sub.get("liquidityNum", 0),
                    "volume_24h":     sub.get("volume24hr", 0),
                })

        if len(events) < 100:
            break
        time.sleep(0.2)

    print(f"  [Gamma Events] Found {len(markets)} temperature bucket markets")
    return markets


def _deduplicate(markets: list[dict]) -> list[dict]:  # also exported as public alias below
    """Remove markets with duplicate condition_ids, preferring real data over mock."""
    seen: dict[str, dict] = {}
    for m in markets:
        cid = m["condition_id"]
        if cid not in seen or m["source"] != "mock":
            seen[cid] = m
    return list(seen.values())


def fetch_all_weather_markets(use_mock_fallback: bool = True) -> list[dict]:
    """
    Main entry point. Fetches CLOB, Gamma keyword search, and Gamma weather events in parallel.
    Falls back to mock data if nothing found.
    Returns a deduplicated list of weather market dicts.
    """
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_clob   = ex.submit(fetch_clob_markets, 1)         # keyword scan (1 page)
        f_gamma  = ex.submit(fetch_gamma_markets, 2)        # keyword scan (2 pages)
        f_events = ex.submit(fetch_gamma_weather_events, 2) # real temp markets (2 pages)
        clob   = f_clob.result()
        gamma  = f_gamma.result()
        events = f_events.result()

    combined = clob + gamma + events
    combined = _deduplicate(combined)

    live_count = sum(1 for m in combined if m.get("source") != "mock")

    if not live_count and use_mock_fallback:
        print("\n[INFO] No live weather markets found — using mock data to demonstrate the pipeline.\n")
        combined = MOCK_WEATHER_MARKETS[:]

    print(f"\n[Total] {len(combined)} weather markets loaded "
          f"({len(clob)} CLOB, {len(gamma)} Gamma-kw, {len(events)} Gamma-events, "
          f"{'mock fallback' if not live_count else 'live data'})")
    return combined


# Public alias so callers can do: from fetcher_polymarket import _deduplicate
deduplicate = _deduplicate


if __name__ == "__main__":
    markets = fetch_all_weather_markets()
    print(json.dumps(markets[:3], indent=2, default=str))

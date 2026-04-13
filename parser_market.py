"""
parser_market.py
Parses Polymarket weather market questions to extract structured information:
- Location (city/country)
- Event type (rain, temperature, wind, etc.)
- Target date(s)
- Threshold value (if any)
"""

import re
from datetime import date, datetime, timedelta
from dateutil import parser as dateutil_parser
from typing import Optional

# ─── Location aliases ────────────────────────────────────────────────────────

CITY_ALIASES: dict[str, str] = {
    # Americas
    "nyc": "New York",
    "new york city": "New York",
    "new york": "New York",
    "los angeles": "Los Angeles",
    "la": "Los Angeles",
    "chicago": "Chicago",
    "houston": "Houston",
    "dallas": "Dallas",
    "atlanta": "Atlanta",
    "miami": "Miami",
    "seattle": "Seattle",
    "san francisco": "San Francisco",
    "sf": "San Francisco",
    "denver": "Denver",
    "boston": "Boston",
    "phoenix": "Phoenix",
    "las vegas": "Las Vegas",
    "minneapolis": "Minneapolis",
    "detroit": "Detroit",
    "philadelphia": "Philadelphia",
    "washington dc": "Washington DC",
    "washington, dc": "Washington DC",
    "dc": "Washington DC",
    "austin": "Austin",
    "toronto": "Toronto",
    "sao paulo": "Sao Paulo",
    "buenos aires": "Buenos Aires",
    "mexico city": "Mexico City",
    "panama city": "Panama City",
    # Europe
    "london": "London",
    "paris": "Paris",
    "berlin": "Berlin",
    "madrid": "Madrid",
    "rome": "Rome",
    "amsterdam": "Amsterdam",
    "moscow": "Moscow",
    "milan": "Milan",
    "warsaw": "Warsaw",
    "helsinki": "Helsinki",
    "istanbul": "Istanbul",
    "ankara": "Ankara",
    # Asia
    "tokyo": "Tokyo",
    "beijing": "Beijing",
    "shanghai": "Shanghai",
    "hong kong": "Hong Kong",
    "seoul": "Seoul",
    "singapore": "Singapore",
    "dubai": "Dubai",
    "mumbai": "Mumbai",
    "bangkok": "Bangkok",
    "jakarta": "Jakarta",
    "taipei": "Taipei",
    "kuala lumpur": "Kuala Lumpur",
    "kl": "Kuala Lumpur",
    "shenzhen": "Shenzhen",
    "wuhan": "Wuhan",
    "chengdu": "Chengdu",
    "chongqing": "Chongqing",
    "lucknow": "Lucknow",
    "busan": "Busan",
    "tel aviv": "Tel Aviv",
    "jeddah": "Jeddah",
    "cairo": "Cairo",
    "nairobi": "Nairobi",
    # Oceania / Africa
    "sydney": "Sydney",
    "melbourne": "Melbourne",
    "auckland": "Auckland",
    "wellington": "Wellington",
    "johannesburg": "Johannesburg",
    "cape town": "Cape Town",
    "lagos": "Lagos",
    # Country aliases
    "uk": "London",
    "england": "London",
    "france": "Paris",
    "japan": "Tokyo",
    "australia": "Sydney",
    "uae": "Dubai",
}

# Known city names for regex scanning (longest first to avoid partial matches)
KNOWN_CITIES = sorted(CITY_ALIASES.keys(), key=len, reverse=True)

# ─── Event types ─────────────────────────────────────────────────────────────

EVENT_PATTERNS = [
    ("rain",        re.compile(r"\brain(fall|s|ed|ing)?\b|\bprecipitation\b|\bdownpour\b", re.I)),
    ("snow",        re.compile(r"\bsnow(fall|s|ed|ing|storm)?\b|\bblizzard\b|\bsleet\b", re.I)),
    ("temperature", re.compile(r"\btemperature\b|\btemp\b|\b°[fc]\b|\bdegrees?\b|\bheat\b|\bcold\b|\bfreezing\b|\bwarm\b", re.I)),
    ("wind",        re.compile(r"\bwind(s|y|speed|gusts?)?\b|\bgale\b|\bhurricane\b|\btyphoon\b|\bcyclone\b|\btornado\b", re.I)),
    ("storm",       re.compile(r"\bstorm(s|y)?\b|\bthunderstorm\b|\btempest\b", re.I)),
    ("humidity",    re.compile(r"\bhumidity\b|\bhumid\b", re.I)),
    ("flood",       re.compile(r"\bflood(ing|s)?\b|\bflash flood\b", re.I)),
    ("drought",     re.compile(r"\bdrought\b|\bdry spell\b", re.I)),
    ("fog",         re.compile(r"\bfog(gy)?\b|\bmist(y)?\b", re.I)),
    ("sunny",       re.compile(r"\bsunn(y|ier|shine)\b|\bclear sky\b|\bcloudless\b", re.I)),
]

# ─── Threshold extraction ─────────────────────────────────────────────────────

# Matches things like: "90°F", "32°C", "1 inch", "5mm", "40mph", "50 km/h", "50%"
THRESHOLD_PATTERNS = [
    # Temperature: 90°F or 32°C or -5 degrees C
    ("temperature_f", re.compile(r"(-?\d+(?:\.\d+)?)\s*°?\s*f\b", re.I)),
    ("temperature_c", re.compile(r"(-?\d+(?:\.\d+)?)\s*°?\s*c\b", re.I)),
    ("temperature_deg", re.compile(r"(-?\d+(?:\.\d+)?)\s*degrees?\s*(fahrenheit|celsius|f|c)\b", re.I)),
    # Precipitation: 1 inch, 0.5 inches, 5mm, 10 millimeters
    ("precip_inches", re.compile(r"(\d+(?:\.\d+)?)\s*inch(es)?\b", re.I)),
    ("precip_mm",     re.compile(r"(\d+(?:\.\d+)?)\s*mm\b", re.I)),
    ("precip_cm",     re.compile(r"(\d+(?:\.\d+)?)\s*cm\b", re.I)),
    # Wind: 40mph, 60 km/h, 50 knots
    ("wind_mph",      re.compile(r"(\d+(?:\.\d+)?)\s*mph\b", re.I)),
    ("wind_kmh",      re.compile(r"(\d+(?:\.\d+)?)\s*km/h\b", re.I)),
    ("wind_knots",    re.compile(r"(\d+(?:\.\d+)?)\s*knots?\b", re.I)),
    # Generic percentage
    ("percent",       re.compile(r"(\d+(?:\.\d+)?)\s*%", re.I)),
]

# ─── Date extraction ──────────────────────────────────────────────────────────

# Month names
MONTHS = (
    "january|february|march|april|may|june|july|august|september|"
    "october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec"
)
DATE_RE = re.compile(
    rf"(?:on\s+)?(?:({MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?"
    rf"(?:\s*,?\s*(\d{{4}}))?)|"
    rf"(?:(\d{{1,2}})/(\d{{1,2}})(?:/(\d{{2,4}}))?)",
    re.I,
)

RELATIVE_DATE_RE = re.compile(
    r"\b(today|tomorrow|this week|next week|this month|next month|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.I,
)


def _extract_location(text: str) -> Optional[str]:
    """Find the first matching city or country in the question text."""
    lower = text.lower()
    for alias in KNOWN_CITIES:
        # Word-boundary-ish check
        pattern = r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])"
        if re.search(pattern, lower):
            return CITY_ALIASES[alias]
    return None


def _extract_event_type(text: str) -> str:
    """Return the primary weather event type detected in the text."""
    for event_type, pattern in EVENT_PATTERNS:
        if pattern.search(text):
            return event_type
    return "unknown"


def _extract_threshold(text: str, event_type: str) -> Optional[dict]:
    """
    Extract the threshold value from the question.
    Returns a dict like:
      {"type": "temperature_f", "value": 90.0, "unit": "F", "value_c": 32.2}
    or None if no threshold found.
    """
    for th_type, pattern in THRESHOLD_PATTERNS:
        m = pattern.search(text)
        if m:
            raw_val = float(m.group(1))
            result = {"type": th_type, "raw_value": raw_val}

            if th_type == "temperature_f":
                result["unit"] = "F"
                result["value_f"] = raw_val
                result["value_c"] = round((raw_val - 32) * 5 / 9, 1)
            elif th_type == "temperature_c":
                result["unit"] = "C"
                result["value_c"] = raw_val
                result["value_f"] = round(raw_val * 9 / 5 + 32, 1)
            elif th_type == "temperature_deg":
                unit_str = m.group(2).lower() if m.group(2) else ""
                if unit_str.startswith("f"):
                    result["unit"] = "F"
                    result["value_f"] = raw_val
                    result["value_c"] = round((raw_val - 32) * 5 / 9, 1)
                else:
                    result["unit"] = "C"
                    result["value_c"] = raw_val
                    result["value_f"] = round(raw_val * 9 / 5 + 32, 1)
            elif th_type == "precip_inches":
                result["unit"] = "in"
                result["value_mm"] = round(raw_val * 25.4, 1)
            elif th_type == "precip_mm":
                result["unit"] = "mm"
                result["value_mm"] = raw_val
            elif th_type == "precip_cm":
                result["unit"] = "cm"
                result["value_mm"] = raw_val * 10
            elif th_type == "wind_mph":
                result["unit"] = "mph"
                result["value_mph"] = raw_val
            elif th_type == "wind_kmh":
                result["unit"] = "km/h"
                result["value_mph"] = round(raw_val / 1.609, 1)
            elif th_type == "wind_knots":
                result["unit"] = "knots"
                result["value_mph"] = round(raw_val * 1.151, 1)
            elif th_type == "percent":
                result["unit"] = "%"

            return result
    return None


def _extract_date(text: str, reference_date: Optional[date] = None) -> Optional[date]:
    """
    Try to extract an explicit or relative date from the question text.
    Returns a date object or None.
    """
    ref = reference_date or date.today()

    # Try explicit date patterns
    for m in DATE_RE.finditer(text):
        groups = m.groups()
        try:
            # Named month pattern: groups 0-2
            if groups[0]:
                month_str = groups[0]
                day = int(groups[1])
                year = int(groups[2]) if groups[2] else ref.year
                dt = dateutil_parser.parse(f"{month_str} {day} {year}")
                return dt.date()
            # Numeric pattern: groups 3-5
            elif groups[3]:
                month = int(groups[3])
                day = int(groups[4])
                year_raw = groups[5]
                if year_raw:
                    year = int(year_raw)
                    if year < 100:
                        year += 2000
                else:
                    year = ref.year
                return date(year, month, day)
        except (ValueError, TypeError):
            continue

    # Try relative dates
    rel = RELATIVE_DATE_RE.search(text)
    if rel:
        word = rel.group(1).lower()
        if word == "today":
            return ref
        if word == "tomorrow":
            return ref + timedelta(days=1)
        if word == "this week":
            return ref + timedelta(days=3)
        if word == "next week":
            return ref + timedelta(days=7)
        # Day of week
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        if word in days:
            target_dow = days.index(word)
            current_dow = ref.weekday()
            delta = (target_dow - current_dow) % 7
            if delta == 0:
                delta = 7
            return ref + timedelta(days=delta)

    # Try dateutil as final fallback
    # Look for date-like substrings
    date_like = re.search(
        r"\b(?:\d{1,2}\s+)?(?:" + MONTHS + r")\s+\d{1,2}(?:,?\s*\d{4})?\b",
        text, re.I
    )
    if date_like:
        try:
            dt = dateutil_parser.parse(date_like.group(), default=datetime(ref.year, ref.month, ref.day))
            return dt.date()
        except (ValueError, TypeError):
            pass

    return None


def _extract_direction(text: str) -> str:
    """
    Determine if the question is about exceeding/above or below a threshold.
    Returns 'above', 'below', or 'any'.
    """
    above = re.compile(
        r"\b(above|exceed|over|more than|greater than|at least|high(?:er)? than|surpass)\b",
        re.I,
    )
    below = re.compile(
        r"\b(below|under|less than|lower than|drop below|fall below|at most)\b",
        re.I,
    )
    if above.search(text):
        return "above"
    if below.search(text):
        return "below"
    return "any"


def parse_market(market: dict, reference_date: Optional[date] = None) -> dict:
    """
    Parse a Polymarket market dict and return an enriched dict with:
      - location, event_type, target_date, threshold, direction
    The input market dict is extended in-place and also returned.
    """
    question = market.get("question", "")
    description = market.get("description", "") or ""
    combined_text = question + " " + description

    ref = reference_date or date.today()

    # Location: prefer explicit hint, then parse from text
    location = market.get("location_hint") or _extract_location(combined_text)
    event_type = _extract_event_type(combined_text)
    target_date = _extract_date(combined_text, reference_date=ref)
    threshold = _extract_threshold(combined_text, event_type)
    direction = _extract_direction(combined_text)

    # If end_date available and no explicit target date found, use end_date
    if target_date is None and market.get("end_date"):
        try:
            end_dt = dateutil_parser.parse(str(market["end_date"]))
            target_date = end_dt.date()
        except (ValueError, TypeError):
            pass

    market["parsed"] = {
        "location": location,
        "event_type": event_type,
        "target_date": target_date.isoformat() if target_date else None,
        "target_date_obj": target_date,
        "threshold": threshold,
        "direction": direction,
    }
    return market


def parse_all_markets(markets: list[dict], reference_date: Optional[date] = None) -> list[dict]:
    """Parse a list of markets, returning only those with parseable location + date."""
    ref = reference_date or date.today()
    parsed = []
    for m in markets:
        parse_market(m, reference_date=ref)
        p = m.get("parsed", {})
        if p.get("location") and p.get("target_date"):
            parsed.append(m)
        else:
            # Still include but mark as unparseable
            m.setdefault("parsed", {})["parseable"] = False
            parsed.append(m)
    return parsed


if __name__ == "__main__":
    import json
    test_questions = [
        "Will it rain in London on April 10, 2026?",
        "Will New York City temperature exceed 60°F on April 11, 2026?",
        "Will Paris receive more than 5mm of rain on April 12, 2026?",
        "Will there be a wind gust above 40mph in Chicago on April 12, 2026?",
        "Will Dubai temperature exceed 35°C on April 11, 2026?",
        "Will NYC get snow this week?",
    ]
    for q in test_questions:
        m = {"question": q, "description": ""}
        parse_market(m)
        print(f"Q: {q}")
        print(f"   → {json.dumps(m['parsed'], default=str)}")
        print()

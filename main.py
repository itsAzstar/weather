"""
main.py
Polymarket Weather Arbitrage Scanner
Orchestrates: fetch → parse → compare → report
"""

import sys
from datetime import date, datetime, timezone

TODAY = date.today()

BANNER = """
=================================================================
   POLYMARKET WEATHER ARBITRAGE SCANNER
   Powered by Open-Meteo Ensemble Forecasts
=================================================================
"""

DIVIDER = "-" * 65


def _pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1%}"


def _format_threshold(threshold: dict | None) -> str:
    if not threshold:
        return ""
    th_type = threshold.get("type", "")
    raw = threshold.get("raw_value")
    unit = threshold.get("unit", "")
    if raw is None:
        return ""
    return f" (threshold: {raw}{unit})"


def _print_result(result: dict, index: int) -> None:
    """Pretty-print a single comparison result."""
    question = result.get("question", "Unknown")
    market_price = result.get("market_price_yes")
    model_prob = result.get("model_probability")
    edge = result.get("edge")
    action = result.get("action", "SKIP")
    is_opp = result.get("is_opportunity", False)
    url = result.get("url", "")
    skip_reason = result.get("skip_reason")
    parsed = result.get("parsed", {})
    location = parsed.get("location", "?")
    event_type = parsed.get("event_type", "?")
    target_date = parsed.get("target_date", "?")
    threshold = parsed.get("threshold")
    weather = result.get("weather_data") or {}

    label = "[OPPORTUNITY]" if is_opp else "[NO EDGE]"
    if skip_reason:
        label = "[SKIPPED]"

    print(f"\n{label} #{index}: {question}")
    print(f"  Location:       {location}  |  Event: {event_type}{_format_threshold(threshold)}")
    print(f"  Target Date:    {target_date}")
    print(f"  Market Price (YES): {_pct(market_price)}")
    print(f"  Model Probability:  {_pct(model_prob)}")

    if edge is not None:
        sign = "+" if edge >= 0 else ""
        print(f"  Edge:           {sign}{edge:.1%}  =>  {action}")
    else:
        print(f"  Status:         {action}")
        if skip_reason:
            print(f"  Reason:         {skip_reason}")

    # Weather details
    if weather:
        rain_pct = weather.get("rain_probability")
        temp_min = weather.get("temp_min_c")
        temp_max = weather.get("temp_max_c")
        wind_max = weather.get("wind_max_mph")
        precip_mm = weather.get("total_precip_mm")

        weather_parts = []
        if rain_pct is not None:
            weather_parts.append(f"Rain prob: {rain_pct:.0%}")
        if precip_mm is not None:
            weather_parts.append(f"Precip: {precip_mm:.1f}mm")
        if temp_min is not None and temp_max is not None:
            temp_min_f = temp_min * 9 / 5 + 32
            temp_max_f = temp_max * 9 / 5 + 32
            weather_parts.append(f"Temp: {temp_min:.0f} to {temp_max:.0f}C ({temp_min_f:.0f} to {temp_max_f:.0f}F)")
        if wind_max is not None:
            weather_parts.append(f"Max wind: {wind_max:.0f}mph")
        if weather_parts:
            print(f"  Forecast:       {' | '.join(weather_parts)}")

    if url:
        print(f"  URL:            {url}")


def print_report(results: list[dict]) -> None:
    """Print the full arbitrage report to stdout."""
    print(BANNER)
    print(f"Scan date: {TODAY.strftime('%B %d, %Y')}")
    print(f"Forecast window: next 7 days")
    print(f"Edge threshold: >10%")

    total = len(results)
    opportunities = [r for r in results if r.get("is_opportunity")]
    no_edge = [r for r in results if not r.get("is_opportunity") and not r.get("skip_reason")]
    skipped = [r for r in results if r.get("skip_reason")]

    print(f"\nTotal markets evaluated: {total}")
    print(f"  Opportunities found:   {len(opportunities)}")
    print(f"  No edge:               {len(no_edge)}")
    print(f"  Skipped:               {len(skipped)}")

    # ── Opportunities ──────────────────────────────────────────────────────
    if opportunities:
        print(f"\n{DIVIDER}")
        print("  ARBITRAGE OPPORTUNITIES")
        print(DIVIDER)
        for i, result in enumerate(opportunities, 1):
            _print_result(result, i)
    else:
        print(f"\n{DIVIDER}")
        print("  No arbitrage opportunities found above the 10% threshold.")

    # ── No edge ────────────────────────────────────────────────────────────
    if no_edge:
        print(f"\n{DIVIDER}")
        print("  MARKETS WITH SMALL / NO EDGE")
        print(DIVIDER)
        for i, result in enumerate(no_edge, 1):
            _print_result(result, i)

    # ── Skipped ────────────────────────────────────────────────────────────
    if skipped:
        print(f"\n{DIVIDER}")
        print("  SKIPPED MARKETS")
        print(DIVIDER)
        for i, result in enumerate(skipped, 1):
            _print_result(result, i)

    print(f"\n{DIVIDER}")
    print("Scan complete.")
    print(DIVIDER)


def run(
    use_mock_fallback: bool = True,
    always_include_mock: bool = False,
    edge_threshold: float = 0.10,
) -> list[dict]:
    """
    Main pipeline:
      1. Fetch Polymarket weather markets
      2. Parse market questions
      3. Fetch weather forecasts
      4. Compare and flag opportunities
      5. Print report

    Args:
        use_mock_fallback: If True, use mock markets when no live markets found.
        always_include_mock: If True, always append mock demo markets even if live
                             markets exist (useful for showing the pipeline end-to-end).
        edge_threshold: Minimum probability divergence to flag as opportunity (0-1).
    """
    # ── Step 1: Fetch Polymarket markets ───────────────────────────────────
    print("\n[Step 1/3] Fetching Polymarket weather markets...")
    from fetcher_polymarket import fetch_all_weather_markets, MOCK_WEATHER_MARKETS
    markets = fetch_all_weather_markets(use_mock_fallback=use_mock_fallback)

    live_count = sum(1 for m in markets if m.get("source") != "mock")
    mock_count = sum(1 for m in markets if m.get("source") == "mock")

    # If live markets exist but none have priceable data, also show mock for demo
    if always_include_mock or (live_count > 0 and mock_count == 0):
        existing_ids = {m["condition_id"] for m in markets}
        extras = [m for m in MOCK_WEATHER_MARKETS if m["condition_id"] not in existing_ids]
        if extras:
            print(f"\n[INFO] Appending {len(extras)} mock demo markets to illustrate the full pipeline.")
            markets = markets + extras

    if not markets:
        print("No markets to process. Exiting.")
        return []

    print(f"  → {len(markets)} markets loaded ({live_count} live, {len(markets)-live_count} mock)")

    # ── Step 2: Parse market questions ─────────────────────────────────────
    print("\n[Step 2/3] Parsing market questions...")
    from parser_market import parse_all_markets
    markets = parse_all_markets(markets, reference_date=TODAY)

    parseable = sum(1 for m in markets if m.get("parsed", {}).get("location"))
    print(f"  → {parseable}/{len(markets)} markets successfully parsed")

    # ── Step 3: Compare market prices vs weather model ─────────────────────
    print("\n[Step 3/3] Fetching weather forecasts and computing edges...")
    from comparator import compare_all_markets
    # Temporarily override threshold for comparator
    import comparator
    original_threshold = comparator.EDGE_THRESHOLD
    comparator.EDGE_THRESHOLD = edge_threshold

    results = compare_all_markets(markets)
    comparator.EDGE_THRESHOLD = original_threshold

    # ── Print report ───────────────────────────────────────────────────────
    print_report(results)

    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Polymarket Weather Arbitrage Scanner")
    ap.add_argument(
        "--no-mock", action="store_true",
        help="Disable mock fallback (only use live Polymarket data)",
    )
    ap.add_argument(
        "--edge", type=float, default=10.0,
        help="Minimum edge %% to flag as opportunity (default: 10)",
    )
    args = ap.parse_args()

    run(
        use_mock_fallback=not args.no_mock,
        always_include_mock=(not args.no_mock),
        edge_threshold=args.edge / 100.0,
    )

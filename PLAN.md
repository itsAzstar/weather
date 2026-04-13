# Weather Bot Website — Polymarket Weather Arbitrage Dashboard

## What We're Building

Turn the existing `polymarket-weather` CLI scanner into a real-time web dashboard.

Current state: Python CLI that runs on demand, prints a text report, exits.
Target state: Live website that auto-refreshes, shows model consensus visually, flags latency arbitrage windows, and makes position sizing obvious.

## Problem

The existing bot has real edge logic (Open-Meteo ensemble vs Polymarket price) but the interface is a terminal wall of text. You can't monitor it continuously, share it, or see trends over time. The edge windows (new model run drops → market reprices 2-12h later) are invisible in the current CLI.

## Reddit Research: What Actually Works

From `r/PredictionMarkets`, `r/Polymarket`, meteorology forums, and documented $24k+ profitable weather bots:

### 1. Forecast Latency Arbitrage (Highest ROI, Lowest Effort)
When a new GFS run (00/06/12/18 UTC) or ECMWF run (00/12 UTC) drops, Polymarket prices lag 2-12 hours. If a new model run shifts a temperature forecast by ±5°F, prices haven't adjusted. Entry window: the first 1-6 hours after new model runs.

### 2. Multi-Model Consensus (3+ Models = High Conviction)
When GFS + ECMWF + ICON all agree at 70-90% probability on an outcome AND market price contradicts, that's the highest-edge setup. One model alone is noise. Three models agreeing is signal.

### 3. Ensemble Spread = Uncertainty Price
ECMWF EPS (50 members) spread between 10th/90th percentile temperatures tells you how certain the forecast is. High spread + market pricing as certain = fade the market. Low spread + market uncertain = take the market's side.

### 4. Station-Specific Resolution
Polymarket contracts resolve on SPECIFIC weather stations (KJFK for NYC, EGLC for London). Generic grid forecasts miss local effects. ASOS station-level data gives a 1-3°F accuracy advantage.

### 5. Calibration (Brier Score Tracking)
Track every prediction. If your model says 80% and that outcome occurs 60% of the time, you're systematically overconfident. Reliability diagrams reveal this instantly.

### 6. Position Sizing (Kelly-ish Micro-bets)
Top profitable traders ($30k+ Meropi, $18.5k 1pixel) use $1-3 micro-bets at extreme conviction only. Bet size = function of (model agreement %) × (edge %). Never trade low-conviction setups.

## Current Codebase

```
polymarket-weather/
  main.py              — CLI orchestrator (fetch → parse → compare → report)
  fetcher_polymarket.py — Pulls active weather markets from Polymarket Gamma API
  fetcher_weather.py   — Open-Meteo ensemble API wrapper (single model source)
  parser_market.py     — NLP parser for market question text
  comparator.py        — Computes edge = model_prob - market_price
  requirements.txt
```

**Gaps vs Reddit research:**
- Only uses Open-Meteo (1 model). No GFS vs ECMWF comparison.
- No latency tracking (when did last model run drop vs current time?).
- No ensemble spread display.
- No Brier score / calibration tracking.
- No position sizing guidance.
- Terminal only. No persistence. No trend view.

## Target Architecture

### Backend: FastAPI
- `/api/opportunities` — current arbitrage opportunities with all signals
- `/api/models` — GFS + ECMWF + ICON consensus comparison
- `/api/latency` — model run freshness (hours since last GFS/ECMWF update)
- `/api/history` — past predictions + outcomes for calibration
- Auto-refresh every 30 minutes (aligned to GFS run times)

### Frontend: Single HTML file (Vanilla JS + Chart.js)
No build toolchain. Deploy anywhere. Auto-loads from FastAPI.

**Dashboard panels:**
1. **Live Opportunities** — market name, edge %, model consensus %, recommended bet size
2. **Model Freshness** — "GFS updated 2h ago | ECMWF updated 5h ago | ICON updated 1h ago"  
3. **Consensus Meter** — for each opportunity: how many models agree (1/2/3)
4. **Ensemble Spread** — temperature uncertainty band (10th-90th percentile)
5. **Calibration Score** — Brier score for past 30 days of predictions
6. **Latency Alert Banner** — "⚡ New GFS run 23 minutes ago — check for price lag"

### Deployment
- Backend: runs locally or Fly.io/Railway
- Frontend: serves from FastAPI static files OR any static host

## Multi-Model Data Sources

| Model | API | Update Freq | Best For |
|-------|-----|-------------|---------|
| GFS | Open-Meteo (free) | Every 6h (00,06,12,18 UTC) | 0-5 day global |
| ECMWF | Open-Meteo (free) | Every 12h (00,12 UTC) | 3-7 day medium range |
| ICON | Open-Meteo (free) | Every 6h | European precision |
| Ensemble | Open-Meteo Ensemble API | Every 12h | Uncertainty quantification |

All free via Open-Meteo. No API keys needed.

## Implementation Plan

### Phase 1: Backend (FastAPI + multi-model)
1. Add `fetcher_gfs.py` — fetch GFS forecast via Open-Meteo GFS API
2. Add `fetcher_ecmwf.py` — fetch ECMWF forecast via Open-Meteo ECMWF API
3. Add `consensus.py` — compare 3 model outputs, compute agreement score
4. Add `latency_tracker.py` — track model run timestamps, compute freshness
5. Add `history.py` — persist predictions to SQLite, compute Brier score
6. Add `server.py` — FastAPI app with all endpoints
7. Update `comparator.py` — use multi-model consensus instead of single model

### Phase 2: Frontend (HTML dashboard)
1. `static/index.html` — responsive dashboard
2. Real-time refresh every 5 minutes
3. Color-coded opportunity cards (green/yellow/red by edge %)
4. Model freshness indicator with ⚡ alert when new run < 1h old
5. Chart.js charts: ensemble spread per market, calibration curve

### Phase 3: Calibration + Position Sizing
1. Log every prediction with timestamp
2. After market resolution, record outcome
3. Compute Brier score rolling 30-day window
4. Kelly position sizing: `bet_size = edge × conviction × $10_max`

## Constraints

- Python 3.11+
- No paid APIs (all Open-Meteo free tier)
- Self-contained — no cloud DB required (SQLite local)
- Single command start: `python server.py`
- Works on Windows (current dev machine)

## Success Metrics

- Page loads in < 2s
- Opportunities refresh every 30 min (aligned to GFS release)
- At least 3 markets analyzed per scan
- Model consensus visible on every card
- Latency alert fires within 10 min of new GFS/ECMWF run

# MACD + RSI Divergence Screener

A Streamlit stock screener for regular and hidden divergence between price and two indicators: RSI and MACD histogram.

It scans one symbol at a time across selected timeframes, ranks the strongest setups, and visualizes each signal with chart overlays and diagnostics.

You can type either a ticker or a company name in the symbol field. The app shows matching suggestions (ticker + company name), and scans the selected ticker.

## What It Does

- Timeframes: `1h`, `4h`, `1d`, `1w`
- Divergence detection:
  - Regular bullish (price lower low, indicator higher low)
  - Regular bearish (price higher high, indicator lower high)
  - Hidden bullish (price higher low, indicator lower low)
  - Hidden bearish (price lower high, indicator higher high)
- Indicators:
  - MACD histogram
  - RSI
- Causal scoring model (historical scores are computed at signal time, no future leak)
- MA regime filter (daily/weekly 50/200 trend context)
- MA regime filter supports `soft` and `hard` modes
- Optional strict indicator-pivot mode (indicator pivots must be independently detected)
- Timeframe-specific scan thresholds and pivot prominence filter
- MACD histogram move threshold scales with the symbol's price (percentage-of-price), so filtering stays consistent across a $5 stock and a $2,000 stock instead of using one fixed absolute number
- Optional non-consecutive pivot pairing via per-timeframe pivot lookback
- Session-aware `4h` resampling aligned to US market hours
- Signal deduplication and directional conflict resolution
- Historical backtest entries are priced at the pivot's confirmation bar (pivot time + the timeframe's right-side pivot bars), not the pivot bar itself, to avoid look-ahead bias
- Streamlit UI with signal table and chart preview

## Why These Settings

- `soft` MA regime mode is the default because it keeps valid setups when daily and weekly trend context disagree.
- Strict indicator pivots are optional because they reduce false positives, but they also filter more aggressively.
- Non-consecutive pivot pairing is enabled because it can recover valid setups that consecutive-only pairing misses.
- Session-aware `4h` resampling keeps intraday bars aligned to regular US market hours.
- MACD histogram is priced in raw dollars (unlike RSI's bounded 0-100 scale), so its minimum-move filter (`macd_hist_min_move_pct` in `ScanRule`) is expressed as a fraction of price and scaled per symbol at scan time, instead of reusing RSI's fixed absolute threshold.
- Backtest entries use the pivot's confirmation bar rather than the pivot bar itself, because a pivot can't be identified as a local extreme until the timeframe's right-side confirmation bars have printed; pricing the entry any earlier would silently look ahead.

## Project Structure

- `app.py`: Streamlit application entrypoint
- `src/screener/config.py`: constants, timeframe settings, and scan rules
- `src/screener/data.py`: market data fetching + resampling
- `src/screener/indicators.py`: RSI and MACD calculations
- `src/screener/divergence.py`: pivot and divergence engine (regular + hidden)
- `src/screener/scanner.py`: scanner orchestration, MA gating, and scoring
- `tests/`: unit tests for RSI and divergence logic

## Quickstart

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

If your Windows Python install is broken (e.g. `python` resolves to the Microsoft Store stub instead of a real interpreter), [uv](https://github.com/astral-sh/uv) is a reliable alternative that manages both Python and the virtual environment:

```powershell
uv venv .venv --python 3.12
uv pip install -r requirements.txt --python .venv
.venv\Scripts\python.exe -m streamlit run app.py
```

3. Run the app:

```powershell
python -m streamlit run app.py
```

4. Stop the app:

In the same terminal where Streamlit is running, press `Ctrl+C`.

If needed on Windows, you can also stop all Streamlit processes with:

```powershell
taskkill /IM streamlit.exe /F
```

Alternative command (if `streamlit` is directly available on PATH):

```powershell
streamlit run app.py
```

## Operational Notes

- Data source currently uses Yahoo Finance (`yfinance`) for speed of prototyping.
- Price download uses adjusted OHLC by default for cleaner long-history indicator behavior.
- `4h` candles are resampled from `1h` data and aligned to regular US market hours (`09:30-16:00` ET).
- In the UI, you can choose divergence types, indicators, MA regime filtering, strict indicator pivots, and whether to keep opposite-direction conflicts.
- The symbol input supports search-style suggestions (for example: "Apple" -> `AAPL`, "Micro" -> `MSFT`, `MU`, etc.).
- The UI uses a single-symbol workflow for faster interactive scans.
- MA regime mode:
  - `soft`: reject only signals directly opposing regime
  - `hard`: require exact regime alignment
- The persistent scanner candle cache is session-scoped; use the UI button to clear it when you want a fresh scan session.
- The benchmark script uses synthetic same-session repeats, so it measures scanner cache behavior without network noise.
- The historical backtest table includes `entry_time` (the confirmation bar the trade is priced at) and `confirmation_lag_bars` (how many bars after the pivot that confirmation took), alongside the original `signal_time` (the pivot bar itself).
- This is not trading advice and should be validated with out-of-sample testing before live use.

## Development And Testing

Install dev/test dependencies (adds `pytest` and `playwright` on top of the runtime requirements):

```powershell
pip install -r requirements-dev.txt
```

Run the test suite:

```powershell
pytest
```

Playwright is only needed for manual browser-driven QA (screenshotting the running app), not for `pytest`. One-time browser download after installing the `requirements-dev.txt` deps:

```powershell
python -m playwright install chromium
```

## Benchmarking And Tracking

After running tests, the strict-vs-loose regression test writes:

- `tests/artifacts/strict_mode_metrics.json`

To append each run into a historical JSONL log:

```powershell
python scripts/append_strict_metrics_history.py
```

Optional custom paths:

```powershell
python scripts/append_strict_metrics_history.py --input tests/artifacts/strict_mode_metrics.json --output tests/artifacts/strict_mode_metrics_history.jsonl
```

To summarize drift from history:

```powershell
python scripts/summarize_strict_metrics_history.py
```

Optional rolling window and JSON output:

```powershell
python scripts/summarize_strict_metrics_history.py --window 20 --json
```

Benchmark same-session cache-on vs cache-off scanner runtime using synthetic data:

```powershell
python scripts/benchmark_scan_cache.py --symbols AAPL,MSFT,NVDA,AMZN --timeframes 1h,4h,1d,1w --runs 3
```

Optional fetch-delay tuning for clearer cache deltas:

```powershell
python scripts/benchmark_scan_cache.py --fetch-delay-ms 50
```

The benchmark writes a JSON report to `tests/artifacts/cache_benchmark_session.json`.

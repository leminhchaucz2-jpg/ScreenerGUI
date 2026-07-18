# MACD + RSI Divergence Screener

A Streamlit stock screener for regular and hidden divergence between price and two indicators: RSI and MACD histogram.

It scans multiple symbols and timeframes, ranks the strongest setups, and visualizes each signal with chart overlays and diagnostics.

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
- Optional non-consecutive pivot pairing via per-timeframe pivot lookback
- Session-aware `4h` resampling aligned to US market hours
- Signal deduplication and directional conflict resolution
- Streamlit UI with signal table and chart preview

## Why These Settings

- `soft` MA regime mode is the default because it keeps valid setups when daily and weekly trend context disagree.
- Strict indicator pivots are optional because they reduce false positives, but they also filter more aggressively.
- Non-consecutive pivot pairing is enabled because it can recover valid setups that consecutive-only pairing misses.
- Session-aware `4h` resampling keeps intraday bars aligned to regular US market hours.

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
- MA regime mode:
  - `soft`: reject only signals directly opposing regime
  - `hard`: require exact regime alignment
- The persistent scanner candle cache is session-scoped; use the UI button to clear it when you want a fresh scan session.
- The benchmark script uses synthetic same-session repeats, so it measures scanner cache behavior without network noise.
- This is not trading advice and should be validated with out-of-sample testing before live use.

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

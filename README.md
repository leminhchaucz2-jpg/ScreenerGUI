# MACD + RSI Divergence Screener (MVP)

This app is a Streamlit-based stock screener that looks for regular bullish/bearish divergence between price and two indicators: RSI and MACD histogram.

It is designed to quickly scan multiple symbols and timeframes, show detected signals in a table, and visualize them in an interactive chart.

## Features

- Timeframes: `1h`, `4h`, `1d`, `1w`
- Divergence detection:
  - Regular bullish (price lower low, indicator higher low)
  - Regular bearish (price higher high, indicator lower high)
- Indicators:
  - MACD histogram
  - RSI
- Multi-timeframe scoring model for quick ranking
- Streamlit UI with signal table and chart preview

## Project Structure

- `app.py`: Streamlit application entrypoint
- `src/screener/config.py`: constants and timeframe settings
- `src/screener/data.py`: market data fetching + resampling
- `src/screener/indicators.py`: RSI and MACD calculations
- `src/screener/divergence.py`: pivot and divergence engine
- `src/screener/scanner.py`: scanner orchestration and scoring

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

Alternative command (if `streamlit` is directly available on PATH):

```powershell
streamlit run app.py
```

## Notes

- Data source currently uses Yahoo Finance (`yfinance`) for speed of prototyping.
- `4h` candles are resampled from `1h` data in this MVP.
- This is not trading advice and should be validated with backtesting before live use.

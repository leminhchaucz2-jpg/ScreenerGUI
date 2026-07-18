"""Market data loading and timeframe transformation."""

import pandas as pd
import yfinance as yf


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [c[0] for   c in frame.columns]
    frame = frame.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    needed = ["open", "high", "low", "close", "volume"]
    return frame[[c for c in needed if c in frame.columns]].dropna(subset=["open", "high", "low", "close"])


def fetch_candles(symbol: str, interval: str, period: str) -> pd.DataFrame:
    frame = yf.download(
        tickers=symbol,
        interval=interval,
        period=period,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if frame.empty:
        return frame
    frame = _normalize_columns(frame)
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame.sort_index()


def resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    if frame.empty:
        return frame

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = frame.resample(rule).agg(agg).dropna(subset=["open", "high", "low", "close"])
    return out

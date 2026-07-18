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


def fetch_candles(symbol: str, interval: str, period: str, *, auto_adjust: bool = True) -> pd.DataFrame:
    frame = yf.download(
        tickers=symbol,
        interval=interval,
        period=period,
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,
    )
    if frame.empty:
        return frame
    frame = _normalize_columns(frame)
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame.sort_index()


def _session_aware_resample(
    frame: pd.DataFrame,
    rule: str,
    *,
    session_timezone: str,
    session_start: str,
    session_end: str,
) -> pd.DataFrame:
    local = frame.tz_convert(session_timezone)
    local = local[local.index.dayofweek < 5]

    # Restrict to market session before resampling so bins are aligned to regular-hours structure.
    local = local.between_time(session_start, session_end, inclusive="left")
    if local.empty:
        return local

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = (
        local.resample(rule, origin="start_day", offset="9h30min")
        .agg(agg)
        .dropna(subset=["open", "high", "low", "close"])
    )
    return out.tz_convert("UTC")


def resample_ohlcv(
    frame: pd.DataFrame,
    rule: str,
    *,
    session_aware: bool = False,
    session_timezone: str | None = None,
    session_start: str = "09:30",
    session_end: str = "16:00",
) -> pd.DataFrame:
    if frame.empty:
        return frame

    if session_aware and session_timezone:
        return _session_aware_resample(
            frame,
            rule,
            session_timezone=session_timezone,
            session_start=session_start,
            session_end=session_end,
        )

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = frame.resample(rule).agg(agg).dropna(subset=["open", "high", "low", "close"])
    return out

"""Rule-based entry/exit strategy experiments.

These are separate from the divergence scanner: instead of scoring a one-off
pivot pattern, each strategy here defines a stateful buy rule and a stateful
sell rule and walks a symbol's price history bar by bar, long only.
"""

from __future__ import annotations

import pandas as pd

from .config import MACD_FAST, MACD_SIGNAL, MACD_SLOW, RSI_PERIOD
from .indicators import compute_macd, compute_rsi


def macd_zero_cross_signal_frame(
    candles: pd.DataFrame,
    *,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    """Entry: MACD line above 0. Exit: MACD histogram negative.

    MACD histogram = MACD line - signal line, so "histogram goes negative" and
    "MACD line crosses below its signal line" are the same event - one exit rule.
    """
    macd = compute_macd(candles["close"], fast=fast, slow=slow, signal=signal)
    frame = pd.DataFrame({"close": candles["close"]}).join(macd)
    frame = frame.dropna(subset=["macd", "macd_signal", "macd_hist"])
    frame["entry_signal"] = frame["macd"] > 0
    frame["exit_signal"] = frame["macd_hist"] < 0
    return frame


def _asof_align(target_index: pd.DatetimeIndex, series: pd.Series) -> pd.Series:
    """Align a lower-frequency series onto target_index using the last known value
    at or before each target timestamp (no look-ahead into a still-forming bar)."""
    right = series.dropna().rename("value").reset_index()
    right.columns = ["time", "value"]
    left = pd.DataFrame({"time": target_index})
    merged = pd.merge_asof(left, right, on="time", direction="backward")
    return pd.Series(merged["value"].to_numpy(), index=target_index)


def rsi_mtf_signal_frame(
    exec_candles: pd.DataFrame,
    higher_candles_by_tf: dict[str, pd.DataFrame],
    *,
    exec_rsi_period: int = RSI_PERIOD,
    higher_rsi_period: int = RSI_PERIOD,
) -> pd.DataFrame:
    """Entry: any higher-timeframe RSI overbought while execution-timeframe RSI is
    oversold. Exit: execution-timeframe RSI, having gone overbought, fades back down.
    """
    frame = pd.DataFrame(
        {
            "close": exec_candles["close"],
            "rsi_exec": compute_rsi(exec_candles["close"], period=exec_rsi_period),
        }
    )

    higher_cols: list[str] = []
    for tf, higher_candles in higher_candles_by_tf.items():
        higher_rsi = compute_rsi(higher_candles["close"], period=higher_rsi_period)
        col = f"rsi_{tf}"
        frame[col] = _asof_align(frame.index, higher_rsi)
        higher_cols.append(col)

    frame["rsi_higher"] = frame[higher_cols].max(axis=1)
    return frame.dropna(subset=["rsi_exec", "rsi_higher"])


def _make_trade_row(
    symbol: str,
    strategy: str,
    frame: pd.DataFrame,
    entry_i: int,
    exit_i: int,
    exit_reason: str,
) -> dict[str, object]:
    entry_price = float(frame.iloc[entry_i]["close"])
    exit_price = float(frame.iloc[exit_i]["close"])
    return_pct = ((exit_price / entry_price) - 1.0) * 100.0
    return {
        "symbol": symbol,
        "strategy": strategy,
        "entry_time": frame.index[entry_i],
        "entry_price": entry_price,
        "exit_time": frame.index[exit_i],
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "bars_held": exit_i - entry_i,
        "return_pct": return_pct,
        "win": return_pct > 0,
    }


def backtest_macd_zero_cross(symbol: str, signal_frame: pd.DataFrame) -> pd.DataFrame:
    """Long only. A signal is only known once its bar has closed, so both entry and
    exit fill on the close of the bar *after* the condition first holds true."""
    trades: list[dict[str, object]] = []
    in_position = False
    entry_i = 0
    n = len(signal_frame)
    for i in range(n - 1):
        row = signal_frame.iloc[i]
        if not in_position:
            if bool(row["entry_signal"]):
                entry_i = i + 1
                in_position = True
        elif bool(row["exit_signal"]):
            exit_i = i + 1
            trades.append(
                _make_trade_row(symbol, "macd_zero_cross", signal_frame, entry_i, exit_i, "macd_hist_negative")
            )
            in_position = False
    return pd.DataFrame(trades)


def backtest_rsi_mtf(
    symbol: str,
    signal_frame: pd.DataFrame,
    *,
    higher_entry_thresh: float = 70.0,
    exec_entry_thresh: float = 30.0,
    exec_exit_thresh: float = 70.0,
) -> pd.DataFrame:
    """Long only. Exit only arms once execution-timeframe RSI has actually gone
    overbought post-entry, then fires the first time it fades back below that
    threshold - a raw RSI < 70 check alone would exit on the very next dip."""
    trades: list[dict[str, object]] = []
    in_position = False
    armed = False
    entry_i = 0
    n = len(signal_frame)
    for i in range(n - 1):
        row = signal_frame.iloc[i]
        if not in_position:
            if row["rsi_higher"] > higher_entry_thresh and row["rsi_exec"] < exec_entry_thresh:
                entry_i = i + 1
                in_position = True
                armed = False
        elif row["rsi_exec"] >= exec_exit_thresh:
            armed = True
        elif armed:
            exit_i = i + 1
            trades.append(
                _make_trade_row(
                    symbol, "rsi_mtf_alignment", signal_frame, entry_i, exit_i, "rsi_faded_below_exit_thresh"
                )
            )
            in_position = False
            armed = False
    return pd.DataFrame(trades)


def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            [
                {
                    "trades": 0,
                    "win_rate_pct": None,
                    "avg_return_pct": None,
                    "median_return_pct": None,
                    "avg_bars_held": None,
                }
            ]
        )
    return pd.DataFrame(
        [
            {
                "trades": len(trades),
                "win_rate_pct": round(trades["win"].mean() * 100.0, 2),
                "avg_return_pct": round(trades["return_pct"].mean(), 2),
                "median_return_pct": round(trades["return_pct"].median(), 2),
                "avg_bars_held": round(trades["bars_held"].mean(), 1),
            }
        ]
    )

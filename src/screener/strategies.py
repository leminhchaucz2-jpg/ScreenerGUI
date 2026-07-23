"""Rule-based entry/exit strategy experiments.

These are separate from the divergence scanner: instead of scoring a one-off
pivot pattern, each strategy here defines a buy rule and a sell rule and walks
a symbol's price history bar by bar, long only.

Most strategies reduce to two boolean columns on a per-bar signal frame -
`entry_signal` and `exit_signal` - and share the generic `_run_signal_backtest`
engine below. RSI Multi-Timeframe Alignment needs a bit of memory (the exit
only "arms" once RSI has actually gone overbought since entry), so it gets its
own small state machine.
"""

from __future__ import annotations

import pandas as pd

from .config import (
    ADX_PERIOD,
    ATR_PERIOD,
    BOLLINGER_NUM_STD,
    BOLLINGER_PERIOD,
    CCI_PERIOD,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_PERIOD,
    STOCH_RSI_PERIOD,
    STOCH_RSI_SMOOTH_D,
    STOCH_RSI_SMOOTH_K,
)
from .indicators import (
    compute_adx,
    compute_atr,
    compute_bollinger_bands,
    compute_cci,
    compute_macd,
    compute_obv,
    compute_rsi,
    compute_sma,
    compute_stoch_rsi,
)


# ---------------------------------------------------------------------------
# Signal frame builders - one per strategy. Each returns a DataFrame indexed
# like the candles, with a "close" column plus whatever entry/exit needs.
# ---------------------------------------------------------------------------


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


def bollinger_breakout_signal_frame(
    candles: pd.DataFrame,
    *,
    period: int = BOLLINGER_PERIOD,
    num_std: float = BOLLINGER_NUM_STD,
    exit_percent_b: float = 0.5,
) -> pd.DataFrame:
    """Entry: close breaks above the upper Bollinger band (%B > 1.0) - a momentum
    breakout read, not the usual mean-reversion use of the bands. Exit: %B fades
    back below the midline threshold."""
    bb = compute_bollinger_bands(candles["close"], period=period, num_std=num_std)
    frame = pd.DataFrame({"close": candles["close"]}).join(bb).dropna(subset=["bb_percent_b"])
    frame["entry_signal"] = frame["bb_percent_b"] > 1.0
    frame["exit_signal"] = frame["bb_percent_b"] < exit_percent_b
    return frame


def golden_cross_trend_signal_frame(
    candles: pd.DataFrame,
    *,
    fast_period: int = 50,
    slow_period: int = 200,
    adx_period: int = ADX_PERIOD,
    adx_entry_thresh: float = 25.0,
    adx_exit_thresh: float = 20.0,
) -> pd.DataFrame:
    """Entry: SMA50 crosses above SMA200 while ADX confirms an actual trend (not
    chop). Exit: SMA50 crosses back below SMA200, or ADX fades below its exit
    threshold even without a cross yet (the trend has already lost steam)."""
    close = candles["close"]
    sma_fast = compute_sma(close, period=fast_period)
    sma_slow = compute_sma(close, period=slow_period)
    adx_df = compute_adx(candles["high"], candles["low"], close, period=adx_period)

    frame = pd.DataFrame({"close": close, "sma_fast": sma_fast, "sma_slow": sma_slow, "adx": adx_df["adx"]})
    frame = frame.dropna(subset=["sma_fast", "sma_slow", "adx"])

    diff = frame["sma_fast"] - frame["sma_slow"]
    prev_diff = diff.shift(1)
    cross_up = (prev_diff <= 0) & (diff > 0)
    cross_down = (prev_diff >= 0) & (diff < 0)

    frame["entry_signal"] = cross_up & (frame["adx"] > adx_entry_thresh)
    frame["exit_signal"] = cross_down | (frame["adx"] < adx_exit_thresh)
    return frame


def stoch_rsi_swing_signal_frame(
    candles: pd.DataFrame,
    *,
    rsi_period: int = RSI_PERIOD,
    stoch_period: int = STOCH_RSI_PERIOD,
    smooth_k: int = STOCH_RSI_SMOOTH_K,
    smooth_d: int = STOCH_RSI_SMOOTH_D,
    oversold: float = 20.0,
    overbought: float = 80.0,
) -> pd.DataFrame:
    """Entry: %K crosses above %D while both are oversold. Exit: %K crosses back
    below %D while both are overbought. A faster, noisier cousin of plain RSI,
    better suited to shorter timeframes."""
    stoch = compute_stoch_rsi(
        candles["close"], rsi_period=rsi_period, stoch_period=stoch_period, smooth_k=smooth_k, smooth_d=smooth_d
    )
    frame = pd.DataFrame({"close": candles["close"]}).join(stoch).dropna(subset=["stoch_rsi_k", "stoch_rsi_d"])

    k, d = frame["stoch_rsi_k"], frame["stoch_rsi_d"]
    k_prev, d_prev = k.shift(1), d.shift(1)
    cross_up = (k_prev <= d_prev) & (k > d)
    cross_down = (k_prev >= d_prev) & (k < d)

    frame["entry_signal"] = cross_up & (k < oversold) & (d < oversold)
    frame["exit_signal"] = cross_down & (k > overbought) & (d > overbought)
    return frame


def cci_extreme_reversal_signal_frame(
    candles: pd.DataFrame,
    *,
    period: int = CCI_PERIOD,
    lower_band: float = -100.0,
    upper_band: float = 100.0,
) -> pd.DataFrame:
    """Entry: CCI crosses back above the oversold band from below. Exit: CCI
    crosses back below the overbought band from above. Symmetric mean-reversion
    using CCI's own +-100 bands."""
    cci = compute_cci(candles["high"], candles["low"], candles["close"], period=period)
    frame = pd.DataFrame({"close": candles["close"], "cci": cci}).dropna(subset=["cci"])

    cci_prev = frame["cci"].shift(1)
    frame["entry_signal"] = (cci_prev <= lower_band) & (frame["cci"] > lower_band)
    frame["exit_signal"] = (cci_prev >= upper_band) & (frame["cci"] < upper_band)
    return frame


def di_crossover_signal_frame(
    candles: pd.DataFrame,
    *,
    period: int = ADX_PERIOD,
    adx_entry_thresh: float = 20.0,
) -> pd.DataFrame:
    """Entry: +DI crosses above -DI while ADX confirms a trend is forming. Exit:
    -DI crosses back above +DI. Catches the start of a directional move instead
    of waiting for a lagging moving-average cross."""
    adx_df = compute_adx(candles["high"], candles["low"], candles["close"], period=period)
    frame = pd.DataFrame({"close": candles["close"]}).join(adx_df).dropna(subset=["adx", "plus_di", "minus_di"])

    plus_prev = frame["plus_di"].shift(1)
    minus_prev = frame["minus_di"].shift(1)
    cross_up = (plus_prev <= minus_prev) & (frame["plus_di"] > frame["minus_di"])
    cross_down = (minus_prev <= plus_prev) & (frame["minus_di"] > frame["plus_di"])

    frame["entry_signal"] = cross_up & (frame["adx"] > adx_entry_thresh)
    frame["exit_signal"] = cross_down
    return frame


def volume_confirmed_breakout_signal_frame(
    candles: pd.DataFrame,
    *,
    lookback_bars: int = 20,
    obv_sma_period: int = 20,
) -> pd.DataFrame:
    """Entry: close makes a new N-bar high and OBV also makes a new N-bar high -
    volume backs the breakout. Exit: OBV rolls over below its own short moving
    average while price hasn't broken down yet - an early "smart money is
    leaving" warning."""
    close = candles["close"]
    obv = compute_obv(close, candles["volume"])
    obv_sma = compute_sma(obv, period=obv_sma_period)
    rolling_high_close = close.rolling(window=lookback_bars, min_periods=lookback_bars).max()
    rolling_high_obv = obv.rolling(window=lookback_bars, min_periods=lookback_bars).max()

    frame = pd.DataFrame({"close": close, "obv": obv, "obv_sma": obv_sma})
    frame = frame.dropna(subset=["obv_sma"])
    frame["entry_signal"] = (close >= rolling_high_close) & (obv >= rolling_high_obv)
    frame["exit_signal"] = frame["obv"] < frame["obv_sma"]
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


def add_atr_column(candles: pd.DataFrame, frame: pd.DataFrame, *, atr_period: int = ATR_PERIOD) -> pd.DataFrame:
    """Attach an ATR column to a signal frame, for the optional trailing-stop exit."""
    atr = compute_atr(candles["high"], candles["low"], candles["close"], period=atr_period)
    frame = frame.copy()
    frame["atr"] = atr.reindex(frame.index)
    return frame.dropna(subset=["atr"])


# ---------------------------------------------------------------------------
# Backtest engines
# ---------------------------------------------------------------------------


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


def _run_signal_backtest(
    symbol: str,
    strategy: str,
    frame: pd.DataFrame,
    exit_reason: str,
    *,
    atr_trailing_stop_multiplier: float | None = None,
) -> pd.DataFrame:
    """Long only. A signal is only known once its bar has closed, so both entry
    and exit fill on the close of the bar *after* the condition first holds true.

    If atr_trailing_stop_multiplier is set (and frame has an "atr" column), a
    chandelier-style stop also closes the trade the bar after price closes below
    (highest close since entry - multiplier * ATR), whichever exit fires first.
    """
    use_trailing_stop = atr_trailing_stop_multiplier is not None and "atr" in frame.columns
    trades: list[dict[str, object]] = []
    in_position = False
    entry_i = 0
    running_high_close = float("-inf")
    n = len(frame)

    for i in range(n - 1):
        row = frame.iloc[i]
        if not in_position:
            if bool(row["entry_signal"]):
                entry_i = i + 1
                in_position = True
                running_high_close = float("-inf")
            continue

        running_high_close = max(running_high_close, float(row["close"]))
        stop_hit = use_trailing_stop and float(row["close"]) < (
            running_high_close - atr_trailing_stop_multiplier * float(row["atr"])
        )
        rule_exit = bool(row["exit_signal"])
        if stop_hit or rule_exit:
            exit_i = i + 1
            reason = exit_reason if rule_exit else "atr_trailing_stop"
            trades.append(_make_trade_row(symbol, strategy, frame, entry_i, exit_i, reason))
            in_position = False

    return pd.DataFrame(trades)


def backtest_macd_zero_cross(symbol: str, signal_frame: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return _run_signal_backtest(symbol, "macd_zero_cross", signal_frame, "macd_hist_negative", **kwargs)


def backtest_bollinger_breakout(symbol: str, signal_frame: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return _run_signal_backtest(symbol, "bollinger_breakout", signal_frame, "percent_b_faded", **kwargs)


def backtest_golden_cross_trend(symbol: str, signal_frame: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return _run_signal_backtest(symbol, "golden_cross_trend", signal_frame, "cross_down_or_weak_adx", **kwargs)


def backtest_stoch_rsi_swing(symbol: str, signal_frame: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return _run_signal_backtest(symbol, "stoch_rsi_swing", signal_frame, "overbought_cross_down", **kwargs)


def backtest_cci_extreme_reversal(symbol: str, signal_frame: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return _run_signal_backtest(symbol, "cci_extreme_reversal", signal_frame, "cci_faded_from_overbought", **kwargs)


def backtest_di_crossover(symbol: str, signal_frame: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return _run_signal_backtest(symbol, "di_crossover", signal_frame, "minus_di_cross_up", **kwargs)


def backtest_volume_confirmed_breakout(symbol: str, signal_frame: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return _run_signal_backtest(symbol, "volume_confirmed_breakout", signal_frame, "obv_rolled_over", **kwargs)


def backtest_rsi_mtf(
    symbol: str,
    signal_frame: pd.DataFrame,
    *,
    higher_entry_thresh: float = 70.0,
    exec_entry_thresh: float = 30.0,
    exec_exit_thresh: float = 70.0,
    atr_trailing_stop_multiplier: float | None = None,
) -> pd.DataFrame:
    """Long only. Exit only arms once execution-timeframe RSI has actually gone
    overbought post-entry, then fires the first time it fades back below that
    threshold - a raw RSI < 70 check alone would exit on the very next dip."""
    use_trailing_stop = atr_trailing_stop_multiplier is not None and "atr" in signal_frame.columns
    trades: list[dict[str, object]] = []
    in_position = False
    armed = False
    entry_i = 0
    running_high_close = float("-inf")
    n = len(signal_frame)

    for i in range(n - 1):
        row = signal_frame.iloc[i]
        if not in_position:
            if row["rsi_higher"] > higher_entry_thresh and row["rsi_exec"] < exec_entry_thresh:
                entry_i = i + 1
                in_position = True
                armed = False
                running_high_close = float("-inf")
            continue

        running_high_close = max(running_high_close, float(row["close"]))
        stop_hit = use_trailing_stop and float(row["close"]) < (
            running_high_close - atr_trailing_stop_multiplier * float(row["atr"])
        )
        if row["rsi_exec"] >= exec_exit_thresh:
            armed = True

        rule_exit = armed and row["rsi_exec"] < exec_exit_thresh
        if stop_hit or rule_exit:
            exit_i = i + 1
            reason = "rsi_faded_below_exit_thresh" if rule_exit else "atr_trailing_stop"
            trades.append(
                _make_trade_row(symbol, "rsi_mtf_alignment", signal_frame, entry_i, exit_i, reason)
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

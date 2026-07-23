from __future__ import annotations

import pandas as pd
import pytest

from src.screener.strategies import (
    backtest_macd_zero_cross,
    backtest_rsi_mtf,
    macd_zero_cross_signal_frame,
    rsi_mtf_signal_frame,
    summarize_trades,
)


def _candles(index: pd.DatetimeIndex, close: list[float]) -> pd.DataFrame:
    close_series = pd.Series(close, index=index, dtype=float)
    return pd.DataFrame(
        {
            "open": close_series,
            "high": close_series + 1.0,
            "low": close_series - 1.0,
            "close": close_series,
            "volume": 1_000_000,
        },
        index=index,
    )


def test_macd_zero_cross_signal_frame_flags_expected_columns() -> None:
    index = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
    # A sustained downtrend then uptrend gives MACD a clean negative-to-positive swing.
    close = [100.0 - i for i in range(30)] + [70.0 + i for i in range(30)]
    candles = _candles(index, close)

    frame = macd_zero_cross_signal_frame(candles)

    assert {"macd", "macd_signal", "macd_hist", "entry_signal", "exit_signal"} <= set(frame.columns)
    assert bool(frame["entry_signal"].iloc[-1])  # trend has turned firmly bullish by the end
    assert (frame["exit_signal"] == (frame["macd_hist"] < 0)).all()


def test_backtest_macd_zero_cross_fills_one_bar_after_signal() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    close = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    frame = pd.DataFrame(
        {
            "close": close,
            "entry_signal": [False, True, False, False, False, False],
            "exit_signal": [False, False, False, True, False, False],
        },
        index=index,
    )

    trades = backtest_macd_zero_cross("TEST", frame)

    assert len(trades) == 1
    row = trades.iloc[0]
    # Entry confirmed at bar 1 -> fills at bar 2's close; exit confirmed at bar 3 -> fills at bar 4's close.
    assert row["entry_time"] == index[2]
    assert row["entry_price"] == pytest.approx(12.0)
    assert row["exit_time"] == index[4]
    assert row["exit_price"] == pytest.approx(14.0)
    assert row["bars_held"] == 2
    assert row["return_pct"] == pytest.approx(((14.0 / 12.0) - 1.0) * 100.0)
    assert row["win"]


def test_backtest_macd_zero_cross_drops_position_still_open_at_end() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "close": [10.0, 11.0, 12.0, 13.0],
            "entry_signal": [False, True, False, False],
            "exit_signal": [False, False, False, False],
        },
        index=index,
    )

    trades = backtest_macd_zero_cross("TEST", frame)
    assert trades.empty


def test_rsi_mtf_signal_frame_aligns_higher_timeframe_without_lookahead() -> None:
    exec_index = pd.date_range("2024-01-01", periods=8, freq="4h", tz="UTC")
    exec_candles = _candles(exec_index, [100.0 + i for i in range(8)])

    higher_index = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    higher_candles = _candles(higher_index, [200.0, 210.0])

    frame = rsi_mtf_signal_frame(
        exec_candles,
        {"1d": higher_candles},
        exec_rsi_period=2,
        higher_rsi_period=1,
    )

    assert "rsi_higher" in frame.columns
    # Every exec bar on day 1 should see day 1's higher-tf RSI (the only bar at/before it),
    # never a value that could only be known once day 2 exists.
    day_one_mask = frame.index < higher_index[1]
    assert day_one_mask.any()
    assert frame.loc[day_one_mask, "rsi_higher"].nunique() == 1


def test_backtest_rsi_mtf_requires_arming_before_exit() -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="4h", tz="UTC")
    frame = pd.DataFrame(
        {
            "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            "rsi_higher": [80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0],
            # oversold at bar 0 (entry trigger) -> dips again at bar 3 but should NOT
            # exit there, since RSI never went overbought after entry (not armed yet).
            "rsi_exec": [20.0, 40.0, 60.0, 25.0, 75.0, 72.0, 60.0, 65.0],
        },
        index=index,
    )

    trades = backtest_rsi_mtf("TEST", frame)

    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["entry_time"] == index[1]
    # rsi_exec confirms armed (>=70) at bar 4, confirms the fade back below 70 at bar
    # 6 -> exit fills one bar later, at bar 7's close.
    assert row["exit_time"] == index[7]
    assert row["exit_reason"] == "rsi_faded_below_exit_thresh"


def test_summarize_trades_handles_empty_and_populated() -> None:
    empty_summary = summarize_trades(pd.DataFrame())
    assert empty_summary.iloc[0]["trades"] == 0

    trades = pd.DataFrame(
        [
            {"return_pct": 5.0, "win": True, "bars_held": 2},
            {"return_pct": -1.0, "win": False, "bars_held": 4},
        ]
    )
    summary = summarize_trades(trades)
    row = summary.iloc[0]
    assert row["trades"] == 2
    assert row["win_rate_pct"] == pytest.approx(50.0)
    assert row["avg_return_pct"] == pytest.approx(2.0)
    assert row["avg_bars_held"] == pytest.approx(3.0)

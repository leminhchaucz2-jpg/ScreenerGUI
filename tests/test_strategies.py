from __future__ import annotations

import pandas as pd
import pytest

import src.screener.strategies as strategies
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
    assert (frame["entry_signal"] == (frame["macd"] > 0)).all()


def test_macd_zero_cross_signal_frame_exit_requires_two_consecutive_negative_bars(monkeypatch) -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    candles = _candles(index, [10.0] * 6)
    macd_df = pd.DataFrame(
        {
            "macd": [1.0] * 6,
            "macd_signal": [0.0] * 6,
            # A single-bar dip at index 2 is a normal wobble inside a trend and
            # should NOT exit; only the second of two consecutive negative bars
            # (index 4-5) confirms a real exit.
            "macd_hist": [1.0, 1.0, -1.0, 1.0, -1.0, -1.0],
        },
        index=index,
    )
    monkeypatch.setattr(strategies, "compute_macd", lambda *a, **k: macd_df)

    frame = strategies.macd_zero_cross_signal_frame(candles)

    assert list(frame["exit_signal"]) == [False, False, False, False, False, True]


def test_macd_zero_cross_golden_gated_signal_frame_blocks_entries_when_all_higher_tf_death_cross(
    monkeypatch,
) -> None:
    # 12 4h-bars spanning two days: bars 0-5 fall on day 1, bars 6-11 on day 2.
    exec_index = pd.date_range("2024-01-01", periods=12, freq="4h", tz="UTC")
    exec_candles = _candles(exec_index, [10.0] * 12)

    macd_vals = [1.0, 1.0, 1.0, 1.0, -1.0, 1.0] * 2
    macd_hist_vals = [1.0, 1.0, 1.0, -1.0, -1.0, 1.0] * 2
    macd_df = pd.DataFrame(
        {"macd": macd_vals, "macd_signal": [0.0] * 12, "macd_hist": macd_hist_vals},
        index=exec_index,
    )
    monkeypatch.setattr(strategies, "compute_macd", lambda *a, **k: macd_df)

    # A single higher timeframe (1d): golden cross (fast > slow) on day 1, death
    # cross on day 2. ADX confirms a real trend on both days, so the gate here
    # turns purely on the SMA cross.
    higher_index = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    higher_candles = _candles(higher_index, [10.0, 10.0])
    fast_vals = pd.Series([2.0, 1.0], index=higher_index)
    slow_vals = pd.Series([1.0, 2.0], index=higher_index)

    def fake_sma(close, period):
        return fast_vals if period == 50 else slow_vals

    monkeypatch.setattr(strategies, "compute_sma", fake_sma)
    monkeypatch.setattr(
        strategies, "compute_adx", lambda *a, **k: pd.DataFrame({"adx": [30.0, 30.0]}, index=higher_index)
    )

    frame = strategies.macd_zero_cross_golden_gated_signal_frame(exec_candles, {"1d": higher_candles})

    # Bars 0-5 (day 1) see the golden cross so entry tracks MACD > 0 there; bars
    # 6-11 (day 2) see the death cross, so no entries at all regardless of MACD.
    assert list(frame["entry_signal"]) == [True, True, True, True, False, True, False, False, False, False, False, False]
    # Exit requires 2 consecutive negative-histogram bars: the second bar of each
    # negative pair (indices 4 and 10) confirms, not the first (indices 3 and 9).
    assert list(frame["exit_signal"]) == [False, False, False, False, True, False] * 2


def test_macd_zero_cross_golden_gated_signal_frame_any_higher_tf_golden_cross_is_enough(
    monkeypatch,
) -> None:
    exec_index = pd.date_range("2024-01-01", periods=2, freq="4h", tz="UTC")
    exec_candles = _candles(exec_index, [10.0, 10.0])

    macd_df = pd.DataFrame(
        {"macd": [1.0, 1.0], "macd_signal": [0.0, 0.0], "macd_hist": [1.0, 1.0]},
        index=exec_index,
    )
    monkeypatch.setattr(strategies, "compute_macd", lambda *a, **k: macd_df)

    higher_index = pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC")
    death_cross_tf = _candles(higher_index, [10.0])
    golden_cross_tf = _candles(higher_index, [10.0])

    def fake_sma(close, period):
        # Distinguish the two higher timeframes by identity of the close series.
        if close is death_cross_tf["close"]:
            return pd.Series([1.0], index=higher_index) if period == 50 else pd.Series([2.0], index=higher_index)
        return pd.Series([2.0], index=higher_index) if period == 50 else pd.Series([1.0], index=higher_index)

    monkeypatch.setattr(strategies, "compute_sma", fake_sma)
    # ADX confirms a real trend on both higher timeframes, so this test isolates
    # the "any higher timeframe golden" OR logic from the ADX gate.
    monkeypatch.setattr(strategies, "compute_adx", lambda *a, **k: pd.DataFrame({"adx": [30.0]}, index=higher_index))

    frame = strategies.macd_zero_cross_golden_gated_signal_frame(
        exec_candles, {"1d": death_cross_tf, "1w": golden_cross_tf}
    )

    # 1d is in a death cross but 1w is in a golden cross - one is enough to gate the entry on.
    assert bool(frame["entry_signal"].iloc[0])


def test_macd_zero_cross_golden_gated_signal_frame_requires_adx_confirmation(monkeypatch) -> None:
    # Two exec bars, one per higher-tf day: SMA is golden on both days, but ADX
    # only confirms a real trend on day 2 - day 1's cross is stale/weak and should
    # NOT gate an entry despite technically being a golden cross.
    exec_index = pd.date_range("2024-01-01", periods=2, freq="1D", tz="UTC")
    exec_candles = _candles(exec_index, [10.0, 10.0])

    macd_df = pd.DataFrame(
        {"macd": [1.0, 1.0], "macd_signal": [0.0, 0.0], "macd_hist": [1.0, 1.0]},
        index=exec_index,
    )
    monkeypatch.setattr(strategies, "compute_macd", lambda *a, **k: macd_df)

    higher_index = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    higher_candles = _candles(higher_index, [10.0, 10.0])
    fast_vals = pd.Series([2.0, 2.0], index=higher_index)  # golden (fast > slow) both days
    slow_vals = pd.Series([1.0, 1.0], index=higher_index)
    adx_vals = pd.DataFrame({"adx": [10.0, 30.0]}, index=higher_index)  # weak, then confirmed

    monkeypatch.setattr(strategies, "compute_sma", lambda close, period: fast_vals if period == 50 else slow_vals)
    monkeypatch.setattr(strategies, "compute_adx", lambda *a, **k: adx_vals)

    frame = strategies.macd_zero_cross_golden_gated_signal_frame(exec_candles, {"1d": higher_candles})

    assert list(frame["entry_signal"]) == [False, True]


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
    gross_return_pct = ((14.0 / 12.0) - 1.0) * 100.0
    assert row["gross_return_pct"] == pytest.approx(gross_return_pct)
    assert row["return_pct"] == pytest.approx(gross_return_pct)  # cost_bps defaults to 0
    assert row["win"]


def test_backtest_macd_zero_cross_deducts_round_trip_cost_from_net_return_only() -> None:
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

    trades = backtest_macd_zero_cross("TEST", frame, cost_bps=50.0)

    assert len(trades) == 1
    row = trades.iloc[0]
    gross_return_pct = ((14.0 / 12.0) - 1.0) * 100.0
    # gross_return_pct is untouched by cost; return_pct (net) has 50bps = 0.50
    # percentage points of round-trip cost subtracted from it.
    assert row["gross_return_pct"] == pytest.approx(gross_return_pct)
    assert row["return_pct"] == pytest.approx(gross_return_pct - 0.50)
    assert row["win"]  # still a net win once cost is deducted

    trades_high_cost = backtest_macd_zero_cross("TEST", frame, cost_bps=2000.0)
    row_high_cost = trades_high_cost.iloc[0]
    # A big enough cost can flip a gross winner into a net loser.
    assert row_high_cost["gross_return_pct"] == pytest.approx(gross_return_pct)
    assert not row_high_cost["win"]


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


def test_bollinger_breakout_signal_frame_crossing_logic(monkeypatch) -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    candles = _candles(index, [10.0] * 5)
    bb_df = pd.DataFrame(
        {
            "bb_mid": [0.0] * 5,
            "bb_upper": [0.0] * 5,
            "bb_lower": [0.0] * 5,
            "bb_percent_b": [0.3, 1.1, 0.6, 0.4, 1.2],
        },
        index=index,
    )
    monkeypatch.setattr(strategies, "compute_bollinger_bands", lambda *a, **k: bb_df)

    frame = strategies.bollinger_breakout_signal_frame(candles)

    assert list(frame["entry_signal"]) == [False, True, False, False, True]
    assert list(frame["exit_signal"]) == [True, False, False, True, False]


def test_golden_cross_trend_signal_frame_requires_adx_confirmation(monkeypatch) -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")
    candles = _candles(index, [10.0] * 8)

    fast_vals = pd.Series([1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0], index=index)
    slow_vals = pd.Series([2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0], index=index)

    def fake_sma(close, period):
        return fast_vals if period == 50 else slow_vals

    adx_df = pd.DataFrame(
        {
            "adx": [30.0, 30.0, 10.0, 30.0, 30.0, 30.0, 30.0, 30.0],
            "plus_di": [0.0] * 8,
            "minus_di": [0.0] * 8,
        },
        index=index,
    )

    monkeypatch.setattr(strategies, "compute_sma", fake_sma)
    monkeypatch.setattr(strategies, "compute_adx", lambda *a, **k: adx_df)

    frame = strategies.golden_cross_trend_signal_frame(candles)

    # First golden cross (bar 2) is gated out by weak ADX (10); the second (bar 6)
    # goes through since ADX confirms a real trend there.
    assert list(frame["entry_signal"]) == [False, False, False, False, False, False, True, False]
    # Death cross at bar 4 exits; ADX dipping below 20 at bar 2 also forces an exit
    # there even without a cross, since the trend has already gone weak.
    assert list(frame["exit_signal"]) == [False, False, True, False, True, False, False, False]


def test_stoch_rsi_swing_signal_frame_crossing_logic(monkeypatch) -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    candles = _candles(index, [10.0] * 6)
    stoch_df = pd.DataFrame(
        {
            "stoch_rsi_k": [25.0, 10.0, 15.0, 30.0, 90.0, 65.0],
            "stoch_rsi_d": [20.0, 18.0, 12.0, 25.0, 95.0, 85.0],
        },
        index=index,
    )
    monkeypatch.setattr(strategies, "compute_stoch_rsi", lambda *a, **k: stoch_df)

    frame = strategies.stoch_rsi_swing_signal_frame(candles)

    assert list(frame["entry_signal"]) == [False, False, True, False, False, False]
    assert list(frame["exit_signal"]) == [False, False, False, False, True, False]


def test_cci_extreme_reversal_signal_frame_crossing_logic(monkeypatch) -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    candles = _candles(index, [10.0] * 6)
    cci_values = pd.Series([-150.0, -120.0, -80.0, 50.0, 130.0, 90.0], index=index)
    monkeypatch.setattr(strategies, "compute_cci", lambda high, low, close, period=14: cci_values)

    frame = strategies.cci_extreme_reversal_signal_frame(candles)

    assert list(frame["entry_signal"]) == [False, False, True, False, False, False]
    assert list(frame["exit_signal"]) == [False, False, False, False, False, True]


def test_di_crossover_signal_frame_crossing_logic_and_adx_gating(monkeypatch) -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    candles = _candles(index, [10.0] * 6)
    adx_df = pd.DataFrame(
        {
            "adx": [30.0, 30.0, 30.0, 30.0, 5.0, 30.0],
            "plus_di": [10.0, 15.0, 25.0, 20.0, 25.0, 10.0],
            "minus_di": [20.0, 18.0, 15.0, 30.0, 20.0, 25.0],
        },
        index=index,
    )
    monkeypatch.setattr(strategies, "compute_adx", lambda *a, **k: adx_df)

    frame = strategies.di_crossover_signal_frame(candles)

    # +DI crosses above -DI at bar 2 (ADX confirms) and again at bar 4 (ADX too weak
    # there to confirm, so that one is gated out).
    assert list(frame["entry_signal"]) == [False, False, True, False, False, False]
    # -DI crosses back above +DI at bars 3 and 5 - no ADX gate on the exit side.
    assert list(frame["exit_signal"]) == [False, False, False, True, False, True]


def test_volume_confirmed_breakout_signal_frame(monkeypatch) -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    candles = _candles(index, [10.0, 12.0, 11.0, 13.0, 9.0, 14.0])

    obv_vals = pd.Series([100.0, 200.0, 150.0, 250.0, 50.0, 300.0], index=index)
    obv_sma_vals = pd.Series([90.0, 90.0, 90.0, 90.0, 200.0, 90.0], index=index)

    monkeypatch.setattr(strategies, "compute_obv", lambda close, volume: obv_vals)
    monkeypatch.setattr(strategies, "compute_sma", lambda series, period: obv_sma_vals)

    frame = strategies.volume_confirmed_breakout_signal_frame(candles, lookback_bars=3)

    assert list(frame["entry_signal"]) == [False, False, False, True, False, True]
    assert list(frame["exit_signal"]) == [False, False, False, False, True, False]


def test_atr_trailing_stop_exits_before_a_rule_signal_that_never_fires() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "close": [10.0, 12.0, 20.0, 14.0, 14.0, 14.0],
            "atr": [1.0] * 6,
            "entry_signal": [False, True, False, False, False, False],
            "exit_signal": [False, False, False, False, False, False],
        },
        index=index,
    )

    without_stop = backtest_macd_zero_cross("TEST", frame)
    assert without_stop.empty  # the rule exit never fires, so nothing closes it out

    with_stop = backtest_macd_zero_cross("TEST", frame, atr_trailing_stop_multiplier=2.0)
    assert len(with_stop) == 1
    row = with_stop.iloc[0]
    # Runs up to a high of 20 (bar 2), then gives back more than 2*ATR (2.0) from
    # that high at bar 3 (20 - 2 = 18, close of 14 is well below) -> stop confirmed
    # at bar 3, fills at bar 4's close.
    assert row["entry_price"] == pytest.approx(20.0)
    assert row["exit_time"] == index[4]
    assert row["exit_price"] == pytest.approx(14.0)
    assert row["exit_reason"] == "atr_trailing_stop"


def test_summarize_trades_handles_empty_and_populated() -> None:
    empty_summary = summarize_trades(pd.DataFrame())
    assert empty_summary.iloc[0]["trades"] == 0

    # Without a gross_return_pct column (older/hand-built trade frames), gross falls
    # back to the net return.
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
    assert row["avg_gross_return_pct"] == pytest.approx(2.0)
    assert row["avg_bars_held"] == pytest.approx(3.0)

    # With gross_return_pct present (real backtest output), net and gross diverge
    # by the cost that was deducted per trade.
    trades_with_cost = pd.DataFrame(
        [
            {"gross_return_pct": 5.5, "return_pct": 5.0, "win": True, "bars_held": 2},
            {"gross_return_pct": -0.5, "return_pct": -1.0, "win": False, "bars_held": 4},
        ]
    )
    summary_with_cost = summarize_trades(trades_with_cost)
    row_with_cost = summary_with_cost.iloc[0]
    assert row_with_cost["avg_return_pct"] == pytest.approx(2.0)
    assert row_with_cost["avg_gross_return_pct"] == pytest.approx(2.5)

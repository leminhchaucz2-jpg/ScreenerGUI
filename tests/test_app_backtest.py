import pandas as pd
import pytest

import app


def _candles(index: pd.DatetimeIndex, close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000_000,
        },
        index=index,
    )


def test_backtest_entry_uses_confirmation_bar_not_pivot_bar(monkeypatch) -> None:
    index = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
    close = pd.Series(100.0 + pd.Series(range(40), index=index) * 1.0, index=index)
    candles = _candles(index, close)

    monkeypatch.setattr(app, "load_price", lambda symbol, timeframe: candles)

    pivot_b_time = index[10]
    history_results = pd.DataFrame(
        [
            {
                "symbol": "TEST",
                "timeframe": "1d",
                "divergence_type": "regular_bullish",
                "indicator": "rsi",
                "pivot_b_time": pivot_b_time,
            }
        ]
    )

    backtest_df = app._build_backtest_frame(history_results, horizon_bars=5)

    assert len(backtest_df) == 1
    row = backtest_df.iloc[0]

    confirmation_lag = app._confirmation_lag_bars("1d")
    assert confirmation_lag > 0  # sanity: the 1d rule requires right-side confirmation bars

    entry_idx = 10 + confirmation_lag
    expected_entry_time = index[entry_idx]
    expected_entry_close = float(close.iloc[entry_idx])
    expected_future_close = float(close.iloc[entry_idx + 5])
    expected_return_pct = ((expected_future_close / expected_entry_close) - 1.0) * 100.0

    assert row["entry_time"] == expected_entry_time
    assert row["confirmation_lag_bars"] == confirmation_lag
    assert row["forward_return_pct"] == pytest.approx(expected_return_pct)
    assert row["trade_side"] == "long"

    # The old (buggy) behavior anchored the entry on the pivot bar itself, which is
    # not actually tradable until `confirmation_lag` bars later -- verify the fix
    # actually moved the entry price away from that unconfirmed bar.
    old_entry_close = float(close.loc[pivot_b_time])
    assert expected_entry_close != old_entry_close


def test_direction_by_divergence_type_covers_all_types() -> None:
    assert app._DIRECTION_BY_DIVERGENCE_TYPE == {
        "regular_bullish": 1.0,
        "regular_bearish": -1.0,
        "hidden_bullish": 1.0,
        "hidden_bearish": -1.0,
    }

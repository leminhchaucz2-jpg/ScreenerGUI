from datetime import datetime

import pandas as pd

from src.screener.divergence import find_divergences
from src.screener.indicators import compute_rsi
import src.screener.scanner as scanner
from src.screener.types import DivergenceSignal


def test_rsi_edge_cases_without_lookahead() -> None:
    uptrend = pd.Series(range(1, 80), dtype=float)
    downtrend = pd.Series(range(80, 1, -1), dtype=float)
    flat = pd.Series([100.0] * 80)

    rsi_up = compute_rsi(uptrend, period=14)
    rsi_down = compute_rsi(downtrend, period=14)
    rsi_flat = compute_rsi(flat, period=14)

    assert pd.isna(rsi_up.iloc[0])
    assert float(rsi_up.iloc[-1]) > 99.0
    assert float(rsi_down.iloc[-1]) < 1.0
    assert abs(float(rsi_flat.iloc[-1]) - 50.0) < 1e-9


def test_regular_and_hidden_divergence_detection() -> None:
    index = pd.date_range("2024-01-01", periods=9, freq="D", tz="UTC")

    # Regular bullish: price LL, indicator HL
    price_regular = pd.Series([10.0, 9.0, 11.0, 8.0, 12.0, 7.0, 13.0, 8.5, 14.0], index=index)
    ind_regular = pd.Series([35.0, 20.0, 40.0, 24.0, 45.0, 28.0, 48.0, 30.0, 50.0], index=index)

    regular = find_divergences(
        price_regular,
        ind_regular,
        left=1,
        right=1,
        max_gap_bars=10,
        min_price_move_pct=0.01,
        min_indicator_move=1.0,
        divergence_type="regular_bullish",
        lookaround_bars=0,
        enforce_unique_extreme=False,
        min_pivot_prominence_atr=0.0,
    )
    assert regular

    # Hidden bullish: price HL, indicator LL
    price_hidden = pd.Series([10.0, 7.0, 12.0, 8.0, 13.0, 9.0, 14.0, 10.0, 15.0], index=index)
    ind_hidden = pd.Series([50.0, 35.0, 55.0, 30.0, 58.0, 25.0, 60.0, 20.0, 62.0], index=index)

    hidden = find_divergences(
        price_hidden,
        ind_hidden,
        left=1,
        right=1,
        max_gap_bars=10,
        min_price_move_pct=0.01,
        min_indicator_move=1.0,
        divergence_type="hidden_bullish",
        lookaround_bars=0,
        enforce_unique_extreme=False,
        min_pivot_prominence_atr=0.0,
    )
    assert hidden


def _mk_signal(divergence_type: str, score: int, indicator_move: float) -> DivergenceSignal:
    now = datetime(2024, 1, 1)
    return DivergenceSignal(
        symbol="AAPL",
        timeframe="1d",
        divergence_type=divergence_type,
        indicator="rsi",
        pivot_a_time=now,
        pivot_b_time=now,
        pivot_a_price=100.0,
        pivot_b_price=95.0,
        indicator_a=20.0,
        indicator_b=25.0,
        indicator_a_time=now,
        indicator_b_time=now,
        price_move_pct=0.05,
        indicator_move=indicator_move,
        pivot_gap_bars=12,
        sma_50=120.0,
        sma_200=110.0,
        sma_cross="golden_cross",
        sma_cross_time=now,
        ma_regime="bullish",
        ma_regime_timeframe="1d,1w",
        score_components={"timeframe_weight": 3},
        score=score,
        note="test",
    )


def test_dedup_keeps_stronger_conflict() -> None:
    weak = _mk_signal("regular_bearish", score=4, indicator_move=0.8)
    strong = _mk_signal("regular_bullish", score=7, indicator_move=1.2)

    deduped = scanner._deduplicate_signals([weak, strong])
    assert len(deduped) == 1
    assert deduped[0].divergence_type == "regular_bullish"
    assert deduped[0].score == 7


def test_dedup_can_keep_opposite_conflicts_when_enabled(monkeypatch) -> None:
    weak = _mk_signal("regular_bearish", score=4, indicator_move=0.8)
    strong = _mk_signal("regular_bullish", score=7, indicator_move=1.2)

    monkeypatch.setattr(scanner, "KEEP_STRONGEST_CONFLICT_ONLY", False)
    deduped = scanner._deduplicate_signals([weak, strong])

    assert len(deduped) == 2
    assert {sig.divergence_type for sig in deduped} == {"regular_bearish", "regular_bullish"}


def test_strict_indicator_pivots_reject_non_pivot_extremes() -> None:
    index = pd.date_range("2024-01-01", periods=9, freq="D", tz="UTC")
    price = pd.Series([10.0, 9.0, 11.0, 8.5, 12.0, 7.5, 13.0, 8.0, 14.0], index=index)

    # Monotonic indicator can form higher-lows numerically, but has no local swing lows near price pivots.
    indicator = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0], index=index)

    loose = find_divergences(
        price,
        indicator,
        left=1,
        right=1,
        max_gap_bars=10,
        min_price_move_pct=0.01,
        min_indicator_move=0.5,
        divergence_type="regular_bullish",
        lookaround_bars=1,
        enforce_unique_extreme=False,
        min_pivot_prominence_atr=0.0,
        require_indicator_pivots=False,
    )
    strict = find_divergences(
        price,
        indicator,
        left=1,
        right=1,
        max_gap_bars=10,
        min_price_move_pct=0.01,
        min_indicator_move=0.5,
        divergence_type="regular_bullish",
        lookaround_bars=1,
        enforce_unique_extreme=False,
        min_pivot_prominence_atr=0.0,
        require_indicator_pivots=True,
    )

    assert loose
    assert strict == []


def test_non_consecutive_pivot_pairing_can_find_signal() -> None:
    index = pd.date_range("2024-01-01", periods=9, freq="D", tz="UTC")

    # Price lows at index 1 (10), 3 (9), 5 (8): consecutive pair (9->8) fails threshold,
    # non-consecutive pair (10->8) passes threshold.
    price = pd.Series([12.0, 10.0, 13.0, 9.0, 14.0, 8.0, 15.0, 11.0, 16.0], index=index)
    indicator = pd.Series([40.0, 20.0, 45.0, 18.0, 46.0, 25.0, 47.0, 30.0, 48.0], index=index)

    consecutive_only = find_divergences(
        price,
        indicator,
        left=1,
        right=1,
        max_gap_bars=10,
        min_price_move_pct=0.15,
        min_indicator_move=1.0,
        divergence_type="regular_bullish",
        lookaround_bars=0,
        enforce_unique_extreme=False,
        min_pivot_prominence_atr=0.0,
        pivot_pair_lookback=1,
    )
    lookback_pairs = find_divergences(
        price,
        indicator,
        left=1,
        right=1,
        max_gap_bars=10,
        min_price_move_pct=0.15,
        min_indicator_move=1.0,
        divergence_type="regular_bullish",
        lookaround_bars=0,
        enforce_unique_extreme=False,
        min_pivot_prominence_atr=0.0,
        pivot_pair_lookback=3,
    )

    assert consecutive_only == []
    assert lookback_pairs

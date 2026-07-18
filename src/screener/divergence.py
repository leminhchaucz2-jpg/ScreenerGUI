"""Pivot-based divergence detection engine."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PivotPair:
    a_time: pd.Timestamp
    b_time: pd.Timestamp
    a_price: float
    b_price: float
    a_indicator: float
    b_indicator: float


def _is_valid_divergence(
    *,
    bullish: bool,
    price_a: float,
    price_b: float,
    indicator_a: float,
    indicator_b: float,
    price_move_pct: float,
    min_price_move_pct: float,
    indicator_move: float,
    min_indicator_move: float,
) -> bool:
    if price_move_pct < min_price_move_pct or indicator_move < min_indicator_move:
        return False

    if bullish:
        return price_b < price_a and indicator_b > indicator_a

    return price_b > price_a and indicator_b < indicator_a


def _find_pivots(series: pd.Series, left: int, right: int, is_low: bool) -> pd.Series:
    values = series.values
    pivots = np.full(len(values), False)

    for i in range(left, len(values) - right):
        window = values[i - left : i + right + 1]
        center = values[i]
        if np.isnan(center) or np.isnan(window).any():
            continue
        if is_low and center == np.min(window):
            pivots[i] = True
        if not is_low and center == np.max(window):
            pivots[i] = True

    return pd.Series(pivots, index=series.index)


def _nearest_indicator_value(
    indicator: pd.Series,
    pivot_time: pd.Timestamp,
    lookaround_bars: int,
    is_low: bool,
) -> float | None:
    if pivot_time not in indicator.index:
        return None

    idx = indicator.index.get_loc(pivot_time)
    if isinstance(idx, slice):
        idx = idx.start

    start = max(0, idx - lookaround_bars)
    end = min(len(indicator), idx + lookaround_bars + 1)
    window = indicator.iloc[start:end].dropna()
    if window.empty:
        return None

    return float(window.min() if is_low else window.max())


def detect_regular_divergence(
    price_close: pd.Series,
    indicator: pd.Series,
    *,
    left: int,
    right: int,
    max_gap_bars: int,
    min_price_move_pct: float,
    min_indicator_move: float,
    bullish: bool,
) -> PivotPair | None:
    matches = find_regular_divergences(
        price_close,
        indicator,
        left=left,
        right=right,
        max_gap_bars=max_gap_bars,
        min_price_move_pct=min_price_move_pct,
        min_indicator_move=min_indicator_move,
        bullish=bullish,
    )
    return matches[-1] if matches else None


def find_regular_divergences(
    price_close: pd.Series,
    indicator: pd.Series,
    *,
    left: int,
    right: int,
    max_gap_bars: int,
    min_price_move_pct: float,
    min_indicator_move: float,
    bullish: bool,
) -> list[PivotPair]:
    pivot_mask = _find_pivots(price_close, left=left, right=right, is_low=bullish)
    pivots = price_close[pivot_mask]
    if len(pivots) < 2:
        return []

    matches: list[PivotPair] = []
    for index in range(1, len(pivots)):
        p_a_time = pivots.index[index - 1]
        p_b_time = pivots.index[index]

        price_a = float(pivots.iloc[index - 1])
        price_b = float(pivots.iloc[index])

        i_a = _nearest_indicator_value(indicator, p_a_time, lookaround_bars=2, is_low=bullish)
        i_b = _nearest_indicator_value(indicator, p_b_time, lookaround_bars=2, is_low=bullish)
        if i_a is None or i_b is None:
            continue

        gap = abs(price_close.index.get_loc(p_b_time) - price_close.index.get_loc(p_a_time))
        if gap > max_gap_bars:
            continue

        price_move_pct = abs(price_b - price_a) / max(abs(price_a), 1e-9)
        indicator_move = abs(i_b - i_a)
        if not _is_valid_divergence(
            bullish=bullish,
            price_a=price_a,
            price_b=price_b,
            indicator_a=i_a,
            indicator_b=i_b,
            price_move_pct=price_move_pct,
            min_price_move_pct=min_price_move_pct,
            indicator_move=indicator_move,
            min_indicator_move=min_indicator_move,
        ):
            continue

        matches.append(
            PivotPair(
                a_time=p_a_time,
                b_time=p_b_time,
                a_price=price_a,
                b_price=price_b,
                a_indicator=i_a,
                b_indicator=i_b,
            )
        )

    return matches

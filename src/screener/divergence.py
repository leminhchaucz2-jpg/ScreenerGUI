"""Pivot-based divergence detection engine."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


DivergenceType = Literal[
    "regular_bullish",
    "regular_bearish",
    "hidden_bullish",
    "hidden_bearish",
]


@dataclass(frozen=True)
class PivotPair:
    a_time: pd.Timestamp
    b_time: pd.Timestamp
    a_price: float
    b_price: float
    a_indicator: float
    b_indicator: float
    a_indicator_time: pd.Timestamp
    b_indicator_time: pd.Timestamp
    price_move_pct: float
    indicator_move: float
    gap_bars: int


def _is_valid_divergence(
    *,
    divergence_type: DivergenceType,
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

    if divergence_type == "regular_bullish":
        return price_b < price_a and indicator_b > indicator_a

    if divergence_type == "regular_bearish":
        return price_b > price_a and indicator_b < indicator_a

    if divergence_type == "hidden_bullish":
        return price_b > price_a and indicator_b < indicator_a

    if divergence_type == "hidden_bearish":
        return price_b < price_a and indicator_b > indicator_a

    return False


def _pivot_prominence_ok(series: pd.Series, pivot_idx: int, is_low: bool, min_prominence_atr: float) -> bool:
    if min_prominence_atr <= 0:
        return True

    if pivot_idx <= 0 or pivot_idx >= len(series) - 1:
        return False

    atr_proxy = series.diff().abs().rolling(window=14, min_periods=14).mean()
    atr_value = atr_proxy.iloc[pivot_idx]
    if pd.isna(atr_value) or atr_value <= 0:
        return True

    center = float(series.iloc[pivot_idx])
    left = float(series.iloc[pivot_idx - 1])
    right = float(series.iloc[pivot_idx + 1])

    if is_low:
        local_contrast = min(left - center, right - center)
    else:
        local_contrast = min(center - left, center - right)

    return local_contrast >= (min_prominence_atr * float(atr_value))


def _find_pivots(
    series: pd.Series,
    left: int,
    right: int,
    is_low: bool,
    *,
    enforce_unique_extreme: bool,
    min_prominence_atr: float,
) -> pd.Series:
    values = series.values
    pivots = np.full(len(values), False)

    for i in range(left, len(values) - right):
        window = values[i - left : i + right + 1]
        center = values[i]
        if np.isnan(center) or np.isnan(window).any():
            continue

        if enforce_unique_extreme and np.sum(window == center) > 1:
            continue

        if is_low and center == np.min(window) and _pivot_prominence_ok(series, i, is_low=True, min_prominence_atr=min_prominence_atr):
            pivots[i] = True
        if not is_low and center == np.max(window) and _pivot_prominence_ok(series, i, is_low=False, min_prominence_atr=min_prominence_atr):
            pivots[i] = True

    return pd.Series(pivots, index=series.index)


def _nearest_indicator_value(
    indicator: pd.Series,
    pivot_time: pd.Timestamp,
    lookaround_bars: int,
    is_low: bool,
) -> tuple[pd.Timestamp, float] | None:
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

    extreme_time = window.idxmin() if is_low else window.idxmax()
    return pd.Timestamp(extreme_time), float(window.loc[extreme_time])


def _nearest_indicator_pivot_value(
    indicator: pd.Series,
    indicator_pivot_mask: pd.Series,
    pivot_time: pd.Timestamp,
    lookaround_bars: int,
) -> tuple[pd.Timestamp, float] | None:
    if pivot_time not in indicator.index:
        return None

    idx = indicator.index.get_loc(pivot_time)
    if isinstance(idx, slice):
        idx = idx.start

    start = max(0, idx - lookaround_bars)
    end = min(len(indicator), idx + lookaround_bars + 1)

    pivot_window = indicator_pivot_mask.iloc[start:end]
    if not pivot_window.any():
        return None

    pivot_times = pivot_window[pivot_window].index
    values = indicator.loc[pivot_times].dropna()
    if values.empty:
        return None

    nearest_time = min(
        values.index,
        key=lambda t: abs(indicator.index.get_loc(t) - idx),
    )
    return pd.Timestamp(nearest_time), float(values.loc[nearest_time])


def _is_bullish(divergence_type: DivergenceType) -> bool:
    return divergence_type in {"regular_bullish", "hidden_bullish"}


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


def detect_divergence(
    price_close: pd.Series,
    indicator: pd.Series,
    *,
    left: int,
    right: int,
    max_gap_bars: int,
    min_price_move_pct: float,
    min_indicator_move: float,
    divergence_type: DivergenceType,
    lookaround_bars: int = 2,
    enforce_unique_extreme: bool = True,
    min_pivot_prominence_atr: float = 0.35,
    require_indicator_pivots: bool = False,
    pivot_pair_lookback: int = 1,
) -> PivotPair | None:
    matches = find_divergences(
        price_close,
        indicator,
        left=left,
        right=right,
        max_gap_bars=max_gap_bars,
        min_price_move_pct=min_price_move_pct,
        min_indicator_move=min_indicator_move,
        divergence_type=divergence_type,
        lookaround_bars=lookaround_bars,
        enforce_unique_extreme=enforce_unique_extreme,
        min_pivot_prominence_atr=min_pivot_prominence_atr,
        require_indicator_pivots=require_indicator_pivots,
        pivot_pair_lookback=pivot_pair_lookback,
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
    require_indicator_pivots: bool = False,
    pivot_pair_lookback: int = 1,
) -> list[PivotPair]:
    divergence_type: DivergenceType = "regular_bullish" if bullish else "regular_bearish"
    return find_divergences(
        price_close,
        indicator,
        left=left,
        right=right,
        max_gap_bars=max_gap_bars,
        min_price_move_pct=min_price_move_pct,
        min_indicator_move=min_indicator_move,
        divergence_type=divergence_type,
        require_indicator_pivots=require_indicator_pivots,
        pivot_pair_lookback=pivot_pair_lookback,
    )


def find_divergences(
    price_close: pd.Series,
    indicator: pd.Series,
    *,
    left: int,
    right: int,
    max_gap_bars: int,
    min_price_move_pct: float,
    min_indicator_move: float,
    divergence_type: DivergenceType,
    lookaround_bars: int = 2,
    enforce_unique_extreme: bool = True,
    min_pivot_prominence_atr: float = 0.35,
    require_indicator_pivots: bool = False,
    pivot_pair_lookback: int = 1,
) -> list[PivotPair]:
    bullish = _is_bullish(divergence_type)
    pivot_mask = _find_pivots(
        price_close,
        left=left,
        right=right,
        is_low=bullish,
        enforce_unique_extreme=enforce_unique_extreme,
        min_prominence_atr=min_pivot_prominence_atr,
    )

    indicator_pivot_mask = _find_pivots(
        indicator,
        left=left,
        right=right,
        is_low=bullish,
        enforce_unique_extreme=enforce_unique_extreme,
        min_prominence_atr=min_pivot_prominence_atr,
    )
    pivots = price_close[pivot_mask]
    if len(pivots) < 2:
        return []

    lookback = max(1, int(pivot_pair_lookback))

    matches: list[PivotPair] = []
    for index in range(1, len(pivots)):
        start_prev = max(0, index - lookback)
        for prev_index in range(start_prev, index):
            p_a_time = pivots.index[prev_index]
            p_b_time = pivots.index[index]

            price_a = float(pivots.iloc[prev_index])
            price_b = float(pivots.iloc[index])

            if require_indicator_pivots:
                i_a = _nearest_indicator_pivot_value(
                    indicator,
                    indicator_pivot_mask,
                    p_a_time,
                    lookaround_bars=lookaround_bars,
                )
                i_b = _nearest_indicator_pivot_value(
                    indicator,
                    indicator_pivot_mask,
                    p_b_time,
                    lookaround_bars=lookaround_bars,
                )
            else:
                i_a = _nearest_indicator_value(indicator, p_a_time, lookaround_bars=lookaround_bars, is_low=bullish)
                i_b = _nearest_indicator_value(indicator, p_b_time, lookaround_bars=lookaround_bars, is_low=bullish)
            if i_a is None or i_b is None:
                continue

            i_a_time, i_a_value = i_a
            i_b_time, i_b_value = i_b

            gap = abs(price_close.index.get_loc(p_b_time) - price_close.index.get_loc(p_a_time))
            if gap > max_gap_bars:
                continue

            price_move_pct = abs(price_b - price_a) / max(abs(price_a), 1e-9)
            indicator_move = abs(i_b_value - i_a_value)
            if not _is_valid_divergence(
                divergence_type=divergence_type,
                price_a=price_a,
                price_b=price_b,
                indicator_a=i_a_value,
                indicator_b=i_b_value,
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
                    a_indicator=i_a_value,
                    b_indicator=i_b_value,
                    a_indicator_time=i_a_time,
                    b_indicator_time=i_b_time,
                    price_move_pct=price_move_pct,
                    indicator_move=indicator_move,
                    gap_bars=gap,
                )
            )

    return matches

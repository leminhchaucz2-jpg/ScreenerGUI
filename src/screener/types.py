"""Shared types for scanner outputs."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class DivergenceSignal:
    symbol: str
    timeframe: str
    divergence_type: str
    indicator: str
    pivot_a_time: datetime
    pivot_b_time: datetime
    pivot_a_price: float
    pivot_b_price: float
    indicator_a: float
    indicator_b: float
    indicator_a_time: datetime
    indicator_b_time: datetime
    price_move_pct: float
    indicator_move: float
    pivot_gap_bars: int
    sma_50: float | None
    sma_200: float | None
    sma_cross: str
    sma_cross_time: datetime | None
    ma_regime: str
    ma_regime_timeframe: str | None
    score_components: dict[str, int]
    score: int
    note: str
    atr: float | None = None
    atr_pct: float | None = None
    bb_percent_b: float | None = None
    adx: float | None = None
    plus_di: float | None = None
    minus_di: float | None = None


@dataclass(frozen=True)
class TimeframeData:
    symbol: str
    timeframe: str
    candles_count: int

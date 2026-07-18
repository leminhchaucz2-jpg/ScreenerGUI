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
    sma_50: float | None
    sma_200: float | None
    sma_cross: str
    sma_cross_time: datetime | None
    score: int
    note: str


@dataclass(frozen=True)
class TimeframeData:
    symbol: str
    timeframe: str
    candles_count: int

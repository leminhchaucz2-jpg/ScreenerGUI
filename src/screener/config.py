"""Static configuration for the screener MVP."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeframeConfig:
    label: str
    source_interval: str
    period: str
    resample_rule: str | None = None


TIMEFRAMES: dict[str, TimeframeConfig] = {
    "1h": TimeframeConfig(label="1h", source_interval="1h", period="730d"),
    "4h": TimeframeConfig(label="4h", source_interval="1h", period="730d", resample_rule="4h"),
    "1d": TimeframeConfig(label="1d", source_interval="1d", period="max"),
    "1w": TimeframeConfig(label="1w", source_interval="1wk", period="max"),
}

TIMEFRAME_WEIGHTS: dict[str, int] = {
    "1h": 1,
    "4h": 2,
    "1d": 3,
    "1w": 4,
}

MIN_BARS_REQUIRED = 120
PIVOT_LEFT_BARS = 3
PIVOT_RIGHT_BARS = 3
PIVOT_MAX_GAP_BARS = 60
MIN_PRICE_MOVE_PCT = 0.005
MIN_INDICATOR_MOVE = 0.5
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BACKTEST_FORWARD_BARS = 10

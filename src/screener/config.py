"""Static configuration for the screener MVP."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeframeConfig:
    label: str
    source_interval: str
    period: str
    resample_rule: str | None = None
    session_aware_resample: bool = False
    session_timezone: str | None = None
    session_start: str = "09:30"
    session_end: str = "16:00"


@dataclass(frozen=True)
class ScanRule:
    min_bars_required: int
    pivot_left_bars: int
    pivot_right_bars: int
    pivot_max_gap_bars: int
    min_price_move_pct: float
    min_indicator_move: float
    pivot_lookaround_bars: int = 2
    min_pivot_prominence_atr: float = 0.35
    pivot_pair_lookback: int = 1


TIMEFRAMES: dict[str, TimeframeConfig] = {
    "1h": TimeframeConfig(label="1h", source_interval="1h", period="730d"),
    "4h": TimeframeConfig(
        label="4h",
        source_interval="1h",
        period="730d",
        resample_rule="4h",
        session_aware_resample=True,
        session_timezone="America/New_York",
        session_start="09:30",
        session_end="16:00",
    ),
    "1d": TimeframeConfig(label="1d", source_interval="1d", period="max"),
    "1w": TimeframeConfig(label="1w", source_interval="1wk", period="max"),
}

TIMEFRAME_WEIGHTS: dict[str, int] = {
    "1h": 1,
    "4h": 2,
    "1d": 3,
    "1w": 4,
}

TIMEFRAME_SCAN_RULES: dict[str, ScanRule] = {
    "1h": ScanRule(
        min_bars_required=260,
        pivot_left_bars=4,
        pivot_right_bars=4,
        pivot_max_gap_bars=40,
        min_price_move_pct=0.0075,
        min_indicator_move=0.65,
        pivot_lookaround_bars=2,
        min_pivot_prominence_atr=0.45,
        pivot_pair_lookback=2,
    ),
    "4h": ScanRule(
        min_bars_required=240,
        pivot_left_bars=4,
        pivot_right_bars=4,
        pivot_max_gap_bars=50,
        min_price_move_pct=0.006,
        min_indicator_move=0.55,
        pivot_lookaround_bars=2,
        min_pivot_prominence_atr=0.4,
        pivot_pair_lookback=2,
    ),
    "1d": ScanRule(
        min_bars_required=260,
        pivot_left_bars=3,
        pivot_right_bars=3,
        pivot_max_gap_bars=75,
        min_price_move_pct=0.008,
        min_indicator_move=0.45,
        pivot_lookaround_bars=2,
        min_pivot_prominence_atr=0.35,
        pivot_pair_lookback=3,
    ),
    "1w": ScanRule(
        min_bars_required=220,
        pivot_left_bars=2,
        pivot_right_bars=2,
        pivot_max_gap_bars=60,
        min_price_move_pct=0.02,
        min_indicator_move=0.35,
        pivot_lookaround_bars=1,
        min_pivot_prominence_atr=0.3,
        pivot_pair_lookback=3,
    ),
}

ENABLE_REGULAR_DIVERGENCE = True
ENABLE_HIDDEN_DIVERGENCE = True
DEFAULT_ENABLED_DIVERGENCE_TYPES = [
    "regular_bullish",
    "regular_bearish",
    "hidden_bullish",
    "hidden_bearish",
]

DEFAULT_ENABLED_INDICATORS = ["rsi", "macd_hist"]

USE_ADJUSTED_PRICES = True

# Blue-chip oriented trend context: require daily/weekly regime alignment unless disabled.
USE_MA_REGIME_FILTER = True
MA_REGIME_TIMEFRAMES = ["1d", "1w"]
MA_REGIME_FILTER_MODE = "soft"
USE_STRICT_INDICATOR_PIVOTS = False

DEDUP_BY_SIGNAL_TIME = True
KEEP_STRONGEST_CONFLICT_ONLY = True

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

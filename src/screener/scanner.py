"""Scanner orchestration for multi-timeframe divergence detection."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Callable

import pandas as pd

from .config import (
    ADX_PERIOD,
    ATR_PERIOD,
    BOLLINGER_NUM_STD,
    BOLLINGER_PERIOD,
    CCI_MIN_INDICATOR_MOVE,
    CCI_PERIOD,
    DEFAULT_ENABLED_DIVERGENCE_TYPES,
    DEFAULT_ENABLED_INDICATORS,
    DEDUP_BY_SIGNAL_TIME,
    ENABLE_HIDDEN_DIVERGENCE,
    ENABLE_REGULAR_DIVERGENCE,
    KEEP_STRONGEST_CONFLICT_ONLY,
    MACD_FAST,
    MACD_HIST_MIN_MOVE_PCT,
    MA_REGIME_FILTER_MODE,
    MA_REGIME_TIMEFRAMES,
    MACD_SIGNAL,
    MACD_SLOW,
    MIN_BARS_REQUIRED,
    MIN_INDICATOR_MOVE,
    MIN_PRICE_MOVE_PCT,
    OBV_MIN_MOVE_PCT_OF_AVG_VOLUME,
    PIVOT_LEFT_BARS,
    PIVOT_MAX_GAP_BARS,
    PIVOT_RIGHT_BARS,
    RSI_PERIOD,
    STOCH_RSI_PERIOD,
    STOCH_RSI_SMOOTH_D,
    STOCH_RSI_SMOOTH_K,
    TIMEFRAMES,
    TIMEFRAME_SCAN_RULES,
    TIMEFRAME_WEIGHTS,
    USE_ADJUSTED_PRICES,
    USE_MA_REGIME_FILTER,
    USE_STRICT_INDICATOR_PIVOTS,
)
from .data import fetch_candles, resample_ohlcv
from .divergence import DivergenceType, PivotPair, find_divergences
from . import indicators as indicator_lib
from .types import DivergenceSignal

compute_macd = indicator_lib.compute_macd
compute_rsi = indicator_lib.compute_rsi
compute_atr = indicator_lib.compute_atr
compute_stoch_rsi = indicator_lib.compute_stoch_rsi
compute_obv = indicator_lib.compute_obv
compute_cci = indicator_lib.compute_cci
compute_bollinger_bands = indicator_lib.compute_bollinger_bands
compute_adx = indicator_lib.compute_adx
compute_sma = getattr(
    indicator_lib,
    "compute_sma",
    lambda close, period: close.rolling(window=period, min_periods=period).mean(),
)

_BOUNDED_OSCILLATORS = {"rsi", "stoch_rsi"}
_ZERO_CROSS_OSCILLATORS = {"macd_hist", "cci"}

_CANDLE_CACHE: dict[tuple[object, ...], pd.DataFrame] = {}

_BULLISH_TYPES = {"regular_bullish", "hidden_bullish"}
_BEARISH_TYPES = {"regular_bearish", "hidden_bearish"}


def _direction_from_type(divergence_type: str) -> str:
    if divergence_type in _BULLISH_TYPES:
        return "bullish"
    if divergence_type in _BEARISH_TYPES:
        return "bearish"
    return "unknown"


def _safe_float_at(series: pd.Series, time: pd.Timestamp) -> float | None:
    if time not in series.index:
        return None
    value = series.loc[time]
    return None if pd.isna(value) else float(value)


def clear_candle_cache() -> None:
    _CANDLE_CACHE.clear()


def _candle_cache_key(
    symbol: str,
    tf: str,
    tf_conf: object,
) -> tuple[object, ...]:
    return (
        symbol,
        tf,
        getattr(tf_conf, "source_interval", None),
        getattr(tf_conf, "period", None),
        getattr(tf_conf, "resample_rule", None),
        getattr(tf_conf, "session_aware_resample", False),
        getattr(tf_conf, "session_timezone", None),
        getattr(tf_conf, "session_start", None),
        getattr(tf_conf, "session_end", None),
        USE_ADJUSTED_PRICES,
    )


def _ma_state_from_series(sma_50: pd.Series, sma_200: pd.Series, at_time: pd.Timestamp) -> str:
    s50 = sma_50.loc[:at_time].dropna()
    s200 = sma_200.loc[:at_time].dropna()
    if s50.empty or s200.empty:
        return "neutral"

    value_50 = float(s50.iloc[-1])
    value_200 = float(s200.iloc[-1])
    if value_50 > value_200:
        return "bullish"
    if value_50 < value_200:
        return "bearish"
    return "neutral"


def _combined_ma_regime(regime_by_tf: dict[str, str]) -> str:
    known = [state for state in regime_by_tf.values() if state in {"bullish", "bearish"}]
    if not known:
        return "neutral"
    if all(state == "bullish" for state in known):
        return "bullish"
    if all(state == "bearish" for state in known):
        return "bearish"
    return "mixed"


def _confirmation_bonus(
    indicator_name: str,
    bullish: bool,
    indicator_slice: pd.Series,
) -> int:
    if len(indicator_slice) < 2:
        return 0
    prev = float(indicator_slice.iloc[-2])
    curr = float(indicator_slice.iloc[-1])

    if indicator_name in _BOUNDED_OSCILLATORS:
        if bullish and prev <= 30 < curr:
            return 1
        if (not bullish) and prev >= 70 > curr:
            return 1
        return 0

    if indicator_name in _ZERO_CROSS_OSCILLATORS:
        if bullish and prev < 0 <= curr:
            return 1
        if (not bullish) and prev > 0 >= curr:
            return 1
        return 0

    # OBV and other unbounded, non-oscillating indicators have no natural
    # overbought/zero-cross confirmation level.
    return 0


def _latest_return_bonus(price_slice: pd.Series, bullish: bool) -> int:
    if len(price_slice) < 2:
        return 0
    ret = (float(price_slice.iloc[-1]) / max(float(price_slice.iloc[-2]), 1e-9)) - 1.0
    if bullish and ret > 0.008:
        return 1
    if (not bullish) and ret < -0.008:
        return 1
    return 0


def _score_signal(
    timeframe: str,
    indicator_name: str,
    pair: PivotPair,
    bullish: bool,
    price_slice: pd.Series,
    indicator_slice: pd.Series,
    regime_component: int,
    min_price_move_pct: float,
    min_indicator_move: float,
) -> tuple[int, dict[str, int]]:
    components: dict[str, int] = {
        "timeframe_weight": TIMEFRAME_WEIGHTS[timeframe],
        "indicator_confirmation": _confirmation_bonus(indicator_name, bullish, indicator_slice),
        "price_impulse": _latest_return_bonus(price_slice, bullish),
        "price_strength": 1 if pair.price_move_pct >= (min_price_move_pct * 1.75) else 0,
        "indicator_strength": 1 if pair.indicator_move >= (min_indicator_move * 1.5) else 0,
        "compact_swing": 1 if pair.gap_bars <= 30 else 0,
        "regime_alignment": regime_component,
    }

    score = int(sum(components.values()))
    if score < 0:
        score = 0
    return score, components


def _macd_hist_effective_min_move(
    last_price: float,
    macd_hist_min_move_pct: float,
    fallback_min_indicator_move: float,
) -> float:
    """Scale the MACD histogram move threshold by price.

    MACD histogram is priced in raw dollars (EMA-fast - EMA-slow), unlike RSI's
    bounded 0-100 scale, so a fixed absolute min_indicator_move is not comparable
    across symbols at different price levels: it under-filters expensive names and
    over-filters cheap ones. Falls back to the absolute threshold if price is
    unusable (e.g. no candles yet).
    """
    if last_price > 0:
        return macd_hist_min_move_pct * last_price
    return fallback_min_indicator_move


def _obv_effective_min_move(
    avg_volume: float,
    obv_min_move_pct_of_avg_volume: float,
    fallback_min_indicator_move: float,
) -> float:
    """Scale the OBV move threshold by the symbol's own recent average bar volume.

    OBV is a running sum of signed volume with no natural bound, unlike RSI's
    0-100 scale, so a fixed absolute threshold would under-filter high-volume
    names and over-filter thinly traded ones. Falls back to the absolute
    threshold if volume data is unusable.
    """
    if avg_volume > 0:
        return (obv_min_move_pct_of_avg_volume / 100.0) * avg_volume
    return fallback_min_indicator_move


def _filter_divergence_types(divergence_types: list[str] | None) -> list[DivergenceType]:
    supported = {
        "regular_bullish",
        "regular_bearish",
        "hidden_bullish",
        "hidden_bearish",
    }

    requested = DEFAULT_ENABLED_DIVERGENCE_TYPES if divergence_types is None else divergence_types
    output = [d for d in requested if d in supported]
    if not ENABLE_REGULAR_DIVERGENCE:
        output = [d for d in output if not d.startswith("regular_")]
    if not ENABLE_HIDDEN_DIVERGENCE:
        output = [d for d in output if not d.startswith("hidden_")]
    return [d for d in output if d in supported]


_SUPPORTED_DIVERGENCE_INDICATORS = {"rsi", "macd_hist", "stoch_rsi", "obv", "cci"}


def _filter_indicators(indicators: list[str] | None) -> list[str]:
    requested = DEFAULT_ENABLED_INDICATORS if indicators is None else indicators
    return [name for name in requested if name in _SUPPORTED_DIVERGENCE_INDICATORS]


def _pick_stronger_signal(a: DivergenceSignal, b: DivergenceSignal) -> DivergenceSignal:
    if a.score != b.score:
        return a if a.score > b.score else b
    if a.indicator_move != b.indicator_move:
        return a if a.indicator_move > b.indicator_move else b
    if a.price_move_pct != b.price_move_pct:
        return a if a.price_move_pct > b.price_move_pct else b
    return a if a.pivot_b_time >= b.pivot_b_time else b


def _deduplicate_signals(signals: list[DivergenceSignal]) -> list[DivergenceSignal]:
    if not signals:
        return signals

    if not KEEP_STRONGEST_CONFLICT_ONLY:
        by_key_keep_direction: dict[tuple[str, str, str, datetime, str], DivergenceSignal] = {}
        for sig in signals:
            key = (sig.symbol, sig.timeframe, sig.indicator, sig.pivot_b_time, sig.divergence_type)
            existing = by_key_keep_direction.get(key)
            by_key_keep_direction[key] = sig if existing is None else _pick_stronger_signal(existing, sig)
        return list(by_key_keep_direction.values())

    by_key: dict[tuple[str, str, str, datetime], DivergenceSignal] = {}
    for sig in signals:
        key = (sig.symbol, sig.timeframe, sig.indicator, sig.pivot_b_time)
        existing = by_key.get(key)
        by_key[key] = sig if existing is None else _pick_stronger_signal(existing, sig)

    collapsed = list(by_key.values())

    # Resolve directional conflicts at the same symbol/timeframe/indicator/time.
    final_map: dict[tuple[str, str, str, datetime], DivergenceSignal] = {}
    for sig in collapsed:
        key = (sig.symbol, sig.timeframe, sig.indicator, sig.pivot_b_time)
        existing = final_map.get(key)
        if existing is None:
            final_map[key] = sig
            continue

        if _direction_from_type(existing.divergence_type) == _direction_from_type(sig.divergence_type):
            final_map[key] = _pick_stronger_signal(existing, sig)
        else:
            final_map[key] = _pick_stronger_signal(existing, sig)

    return list(final_map.values())


def _latest_sma_cross(sma_50: pd.Series, sma_200: pd.Series) -> tuple[str, pd.Timestamp | None]:
    valid = pd.DataFrame({"sma_50": sma_50, "sma_200": sma_200}).dropna()
    if len(valid) < 2:
        return "none", None

    diff = valid["sma_50"] - valid["sma_200"]
    prev = diff.shift(1)
    events = valid[(prev <= 0) & (diff > 0) | (prev >= 0) & (diff < 0)]
    if events.empty:
        return "none", None

    cross_time = events.index[-1]
    cross_value = diff.loc[cross_time]
    cross_type = "golden_cross" if cross_value > 0 else "death_cross"
    return cross_type, cross_time


def scan_symbol(symbol: str, timeframes: list[str]) -> list[DivergenceSignal]:
    return _scan_symbol(symbol, timeframes, historical=False)


def scan_symbol_history(symbol: str, timeframes: list[str]) -> list[DivergenceSignal]:
    return _scan_symbol(symbol, timeframes, historical=True)


def _build_regime_context(
    get_candles_for_tf: Callable[[str], pd.DataFrame],
) -> dict[str, tuple[pd.Series, pd.Series]]:
    regime_context: dict[str, tuple[pd.Series, pd.Series]] = {}
    for regime_tf in MA_REGIME_TIMEFRAMES:
        candles = get_candles_for_tf(regime_tf)
        if candles.empty:
            continue
        close = candles["close"]
        regime_context[regime_tf] = (compute_sma(close, period=50), compute_sma(close, period=200))
    return regime_context


def _regime_filter_decision(bullish_signal: bool, regime_state: str, mode: str) -> tuple[bool, int]:
    normalized_mode = mode.lower().strip()

    if normalized_mode == "hard":
        if bullish_signal:
            aligned = regime_state == "bullish"
        else:
            aligned = regime_state == "bearish"
        return aligned, (2 if aligned else -2)

    # Soft mode: reject only direct opposition; keep mixed/neutral with no bonus.
    if bullish_signal and regime_state == "bearish":
        return False, -2
    if (not bullish_signal) and regime_state == "bullish":
        return False, -2
    if bullish_signal and regime_state == "bullish":
        return True, 2
    if (not bullish_signal) and regime_state == "bearish":
        return True, 2
    return True, 0


def _regime_at_signal_time(
    regime_context: dict[str, tuple[pd.Series, pd.Series]],
    signal_time: pd.Timestamp,
) -> tuple[str, str | None]:
    states: dict[str, str] = {}
    for tf, (sma_50, sma_200) in regime_context.items():
        states[tf] = _ma_state_from_series(sma_50, sma_200, signal_time)
    regime = _combined_ma_regime(states)
    timeframe_label = ",".join(sorted(states.keys())) if states else None
    return regime, timeframe_label


def _scan_symbol(
    symbol: str,
    timeframes: list[str],
    *,
    historical: bool,
    divergence_types: list[str] | None = None,
    indicators: list[str] | None = None,
    use_ma_regime_filter: bool = USE_MA_REGIME_FILTER,
    ma_regime_filter_mode: str = MA_REGIME_FILTER_MODE,
    strict_indicator_pivots: bool = USE_STRICT_INDICATOR_PIVOTS,
    use_candle_cache: bool = True,
) -> list[DivergenceSignal]:
    signals: list[DivergenceSignal] = []
    selected_divergence_types = _filter_divergence_types(divergence_types)
    selected_indicators = _filter_indicators(indicators)

    def get_candles_for_tf(tf: str) -> pd.DataFrame:
        if tf not in TIMEFRAMES:
            return pd.DataFrame()

        tf_conf = TIMEFRAMES[tf]
        cache_key = _candle_cache_key(symbol, tf, tf_conf)
        if use_candle_cache and cache_key in _CANDLE_CACHE:
            return _CANDLE_CACHE[cache_key]

        candles = fetch_candles(
            symbol,
            interval=tf_conf.source_interval,
            period=tf_conf.period,
            auto_adjust=USE_ADJUSTED_PRICES,
        )
        if tf_conf.resample_rule:
            candles = resample_ohlcv(
                candles,
                tf_conf.resample_rule,
                session_aware=tf_conf.session_aware_resample,
                session_timezone=tf_conf.session_timezone,
                session_start=tf_conf.session_start,
                session_end=tf_conf.session_end,
            )
        if use_candle_cache:
            _CANDLE_CACHE[cache_key] = candles
        return candles

    regime_context = _build_regime_context(get_candles_for_tf) if use_ma_regime_filter else {}

    if not selected_divergence_types or not selected_indicators:
        return signals

    for tf in timeframes:
        if tf not in TIMEFRAMES:
            continue

        rule = TIMEFRAME_SCAN_RULES.get(tf)
        min_bars_required = MIN_BARS_REQUIRED if rule is None else rule.min_bars_required
        pivot_left = PIVOT_LEFT_BARS if rule is None else rule.pivot_left_bars
        pivot_right = PIVOT_RIGHT_BARS if rule is None else rule.pivot_right_bars
        pivot_gap = PIVOT_MAX_GAP_BARS if rule is None else rule.pivot_max_gap_bars
        min_price_move = MIN_PRICE_MOVE_PCT if rule is None else rule.min_price_move_pct
        min_indicator_move = MIN_INDICATOR_MOVE if rule is None else rule.min_indicator_move
        macd_hist_min_move_pct = MACD_HIST_MIN_MOVE_PCT if rule is None else rule.macd_hist_min_move_pct
        lookaround = 2 if rule is None else rule.pivot_lookaround_bars
        min_prominence = 0.35 if rule is None else rule.min_pivot_prominence_atr
        pivot_pair_lookback = 1 if rule is None else rule.pivot_pair_lookback

        candles = get_candles_for_tf(tf)
        if candles.empty or len(candles) < min_bars_required:
            continue

        close = candles["close"]
        rsi = compute_rsi(close, period=RSI_PERIOD)
        macd_df = compute_macd(close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
        macd_hist = macd_df["macd_hist"]
        sma_50 = compute_sma(close, period=50)
        sma_200 = compute_sma(close, period=200)
        atr = compute_atr(candles["high"], candles["low"], close, period=ATR_PERIOD)
        stoch_rsi_k = compute_stoch_rsi(
            close,
            rsi_period=STOCH_RSI_PERIOD,
            stoch_period=STOCH_RSI_PERIOD,
            smooth_k=STOCH_RSI_SMOOTH_K,
            smooth_d=STOCH_RSI_SMOOTH_D,
        )["stoch_rsi_k"]
        obv = compute_obv(close, candles["volume"])
        cci = compute_cci(candles["high"], candles["low"], close, period=CCI_PERIOD)
        bollinger = compute_bollinger_bands(close, period=BOLLINGER_PERIOD, num_std=BOLLINGER_NUM_STD)
        adx_df = compute_adx(candles["high"], candles["low"], close, period=ADX_PERIOD)

        indicator_map = {
            "rsi": rsi,
            "macd_hist": macd_hist,
            "stoch_rsi": stoch_rsi_k,
            "obv": obv,
            "cci": cci,
        }

        last_price = float(close.iloc[-1]) if not close.empty and pd.notna(close.iloc[-1]) else 0.0
        avg_volume = candles["volume"].rolling(window=50, min_periods=10).mean()
        last_avg_volume = float(avg_volume.iloc[-1]) if not avg_volume.empty and pd.notna(avg_volume.iloc[-1]) else 0.0
        indicator_min_move_map = {
            "rsi": min_indicator_move,
            "macd_hist": _macd_hist_effective_min_move(last_price, macd_hist_min_move_pct, min_indicator_move),
            "stoch_rsi": min_indicator_move,
            "cci": CCI_MIN_INDICATOR_MOVE,
            "obv": _obv_effective_min_move(last_avg_volume, OBV_MIN_MOVE_PCT_OF_AVG_VOLUME, min_indicator_move),
        }

        for div_type in selected_divergence_types:
            indicator_direction_bullish = _direction_from_type(div_type) == "bullish"
            for indicator_name in selected_indicators:
                indicator_series = indicator_map[indicator_name]
                indicator_min_move = indicator_min_move_map[indicator_name]
                pairs = find_divergences(
                close,
                indicator_series,
                    left=pivot_left,
                    right=pivot_right,
                    max_gap_bars=pivot_gap,
                    min_price_move_pct=min_price_move,
                    min_indicator_move=indicator_min_move,
                    divergence_type=div_type,  # type: ignore[arg-type]
                    lookaround_bars=lookaround,
                    enforce_unique_extreme=True,
                    min_pivot_prominence_atr=min_prominence,
                    require_indicator_pivots=strict_indicator_pivots,
                    pivot_pair_lookback=pivot_pair_lookback,
                )
                if not pairs:
                    continue

                selected_pairs = pairs if historical else [pairs[-1]]

                for pair in selected_pairs:
                    signal_time = pair.b_time
                    signal_idx = close.index.get_loc(signal_time)
                    if isinstance(signal_idx, slice):
                        signal_idx = signal_idx.stop - 1
                    if int(signal_idx) < 1:
                        continue

                    close_slice = close.iloc[: int(signal_idx) + 1]
                    indicator_slice = indicator_series.iloc[: int(signal_idx) + 1]

                    sma_50_slice = sma_50.iloc[: int(signal_idx) + 1]
                    sma_200_slice = sma_200.iloc[: int(signal_idx) + 1]
                    sma_cross, sma_cross_time = _latest_sma_cross(sma_50_slice, sma_200_slice)

                    latest_sma_50 = _safe_float_at(sma_50, signal_time)
                    latest_sma_200 = _safe_float_at(sma_200, signal_time)

                    latest_atr = _safe_float_at(atr, signal_time)
                    signal_price = _safe_float_at(close, signal_time)
                    latest_atr_pct = (
                        (latest_atr / signal_price) * 100.0
                        if latest_atr is not None and signal_price
                        else None
                    )

                    latest_bb_percent_b = _safe_float_at(bollinger["bb_percent_b"], signal_time)
                    latest_adx = _safe_float_at(adx_df["adx"], signal_time)
                    latest_plus_di = _safe_float_at(adx_df["plus_di"], signal_time)
                    latest_minus_di = _safe_float_at(adx_df["minus_di"], signal_time)

                    regime_state, regime_source = _regime_at_signal_time(regime_context, signal_time)
                    if use_ma_regime_filter:
                        allowed, regime_component = _regime_filter_decision(
                            indicator_direction_bullish,
                            regime_state,
                            ma_regime_filter_mode,
                        )
                        if not allowed:
                            continue
                    else:
                        regime_component = 0

                    score, score_components = _score_signal(
                        tf,
                        indicator_name,
                        pair,
                        indicator_direction_bullish,
                        close_slice,
                        indicator_slice,
                        regime_component=regime_component,
                        min_price_move_pct=min_price_move,
                        min_indicator_move=indicator_min_move,
                    )

                    signals.append(
                        DivergenceSignal(
                            symbol=symbol,
                            timeframe=tf,
                            divergence_type=div_type,
                            indicator=indicator_name,
                            pivot_a_time=pair.a_time.to_pydatetime(),
                            pivot_b_time=pair.b_time.to_pydatetime(),
                            pivot_a_price=pair.a_price,
                            pivot_b_price=pair.b_price,
                            indicator_a=pair.a_indicator,
                            indicator_b=pair.b_indicator,
                            indicator_a_time=pair.a_indicator_time.to_pydatetime(),
                            indicator_b_time=pair.b_indicator_time.to_pydatetime(),
                            price_move_pct=pair.price_move_pct,
                            indicator_move=pair.indicator_move,
                            pivot_gap_bars=pair.gap_bars,
                            sma_50=latest_sma_50,
                            sma_200=latest_sma_200,
                            sma_cross=sma_cross,
                            sma_cross_time=sma_cross_time.to_pydatetime() if sma_cross_time is not None else None,
                            ma_regime=regime_state,
                            ma_regime_timeframe=regime_source,
                            atr=latest_atr,
                            atr_pct=latest_atr_pct,
                            bb_percent_b=latest_bb_percent_b,
                            adx=latest_adx,
                            plus_di=latest_plus_di,
                            minus_di=latest_minus_di,
                            score_components=score_components,
                            score=score,
                            note="Pivot divergence with causal score",
                        )
                    )

    if DEDUP_BY_SIGNAL_TIME:
        signals = _deduplicate_signals(signals)

    return sorted(signals, key=lambda s: (s.score, s.pivot_b_time), reverse=True)


def scan_universe(
    symbols: list[str],
    timeframes: list[str],
    *,
    divergence_types: list[str] | None = None,
    indicators: list[str] | None = None,
    use_ma_regime_filter: bool = USE_MA_REGIME_FILTER,
    ma_regime_filter_mode: str = MA_REGIME_FILTER_MODE,
    strict_indicator_pivots: bool = USE_STRICT_INDICATOR_PIVOTS,
    use_candle_cache: bool = True,
) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        symbol = symbol.strip().upper()
        if not symbol:
            continue
        for signal in _scan_symbol(
            symbol,
            timeframes,
            historical=False,
            divergence_types=divergence_types,
            indicators=indicators,
            use_ma_regime_filter=use_ma_regime_filter,
            ma_regime_filter_mode=ma_regime_filter_mode,
            strict_indicator_pivots=strict_indicator_pivots,
            use_candle_cache=use_candle_cache,
        ):
            rows.append(asdict(signal))

    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "timeframe",
                "divergence_type",
                "indicator",
                "pivot_a_time",
                "pivot_b_time",
                "pivot_a_price",
                "pivot_b_price",
                "indicator_a",
                "indicator_b",
                "indicator_a_time",
                "indicator_b_time",
                "price_move_pct",
                "indicator_move",
                "pivot_gap_bars",
                "sma_50",
                "sma_200",
                "sma_cross",
                "sma_cross_time",
                "ma_regime",
                "ma_regime_timeframe",
                "atr",
                "atr_pct",
                "bb_percent_b",
                "adx",
                "plus_di",
                "minus_di",
                "score_components",
                "score",
                "note",
            ]
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values(by=["score", "timeframe", "symbol"], ascending=[False, True, True]).reset_index(drop=True)


def scan_universe_history(
    symbols: list[str],
    timeframes: list[str],
    *,
    divergence_types: list[str] | None = None,
    indicators: list[str] | None = None,
    use_ma_regime_filter: bool = USE_MA_REGIME_FILTER,
    ma_regime_filter_mode: str = MA_REGIME_FILTER_MODE,
    strict_indicator_pivots: bool = USE_STRICT_INDICATOR_PIVOTS,
    use_candle_cache: bool = True,
) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        symbol = symbol.strip().upper()
        if not symbol:
            continue
        for signal in _scan_symbol(
            symbol,
            timeframes,
            historical=True,
            divergence_types=divergence_types,
            indicators=indicators,
            use_ma_regime_filter=use_ma_regime_filter,
            ma_regime_filter_mode=ma_regime_filter_mode,
            strict_indicator_pivots=strict_indicator_pivots,
            use_candle_cache=use_candle_cache,
        ):
            rows.append(asdict(signal))

    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "timeframe",
                "divergence_type",
                "indicator",
                "pivot_a_time",
                "pivot_b_time",
                "pivot_a_price",
                "pivot_b_price",
                "indicator_a",
                "indicator_b",
                "indicator_a_time",
                "indicator_b_time",
                "price_move_pct",
                "indicator_move",
                "pivot_gap_bars",
                "sma_50",
                "sma_200",
                "sma_cross",
                "sma_cross_time",
                "ma_regime",
                "ma_regime_timeframe",
                "atr",
                "atr_pct",
                "bb_percent_b",
                "adx",
                "plus_di",
                "minus_di",
                "score_components",
                "score",
                "note",
            ]
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values(by=["pivot_b_time", "score", "timeframe", "symbol"], ascending=[False, False, True, True]).reset_index(drop=True)

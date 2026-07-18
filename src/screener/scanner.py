"""Scanner orchestration for multi-timeframe divergence detection."""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from .config import (
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MIN_BARS_REQUIRED,
    MIN_INDICATOR_MOVE,
    MIN_PRICE_MOVE_PCT,
    PIVOT_LEFT_BARS,
    PIVOT_MAX_GAP_BARS,
    PIVOT_RIGHT_BARS,
    RSI_PERIOD,
    TIMEFRAMES,
    TIMEFRAME_WEIGHTS,
)
from .data import fetch_candles, resample_ohlcv
from .divergence import detect_regular_divergence, find_regular_divergences
from . import indicators as indicator_lib
from .types import DivergenceSignal

compute_macd = indicator_lib.compute_macd
compute_rsi = indicator_lib.compute_rsi
compute_sma = getattr(
    indicator_lib,
    "compute_sma",
    lambda close, period: close.rolling(window=period, min_periods=period).mean(),
)


def _score_signal(timeframe: str, indicator_name: str, price: pd.Series, rsi: pd.Series, macd_hist: pd.Series) -> int:
    score = TIMEFRAME_WEIGHTS[timeframe]

    if indicator_name == "rsi":
        if (rsi.iloc[-2] <= 30 < rsi.iloc[-1]) or (rsi.iloc[-2] >= 70 > rsi.iloc[-1]):
            score += 1
    else:
        if (macd_hist.iloc[-2] < 0 <= macd_hist.iloc[-1]) or (macd_hist.iloc[-2] > 0 >= macd_hist.iloc[-1]):
            score += 1

    if len(price) >= 2 and abs((price.iloc[-1] / price.iloc[-2]) - 1.0) > 0.01:
        score += 1

    return score


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


def _scan_symbol(symbol: str, timeframes: list[str], *, historical: bool) -> list[DivergenceSignal]:
    signals: list[DivergenceSignal] = []

    for tf in timeframes:
        tf_conf = TIMEFRAMES[tf]
        candles = fetch_candles(symbol, interval=tf_conf.source_interval, period=tf_conf.period)
        if tf_conf.resample_rule:
            candles = resample_ohlcv(candles, tf_conf.resample_rule)
        if candles.empty or len(candles) < MIN_BARS_REQUIRED:
            continue

        close = candles["close"]
        rsi = compute_rsi(close, period=RSI_PERIOD)
        macd_df = compute_macd(close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
        macd_hist = macd_df["macd_hist"]
        sma_50 = compute_sma(close, period=50)
        sma_200 = compute_sma(close, period=200)
        sma_cross, sma_cross_time = _latest_sma_cross(sma_50, sma_200)

        latest_sma_50 = None if pd.isna(sma_50.iloc[-1]) else float(sma_50.iloc[-1])
        latest_sma_200 = None if pd.isna(sma_200.iloc[-1]) else float(sma_200.iloc[-1])

        checks = [
            ("regular_bullish", True, "rsi", rsi),
            ("regular_bearish", False, "rsi", rsi),
            ("regular_bullish", True, "macd_hist", macd_hist),
            ("regular_bearish", False, "macd_hist", macd_hist),
        ]

        for div_type, bullish, indicator_name, indicator_series in checks:
            pairs = find_regular_divergences(
                close,
                indicator_series,
                left=PIVOT_LEFT_BARS,
                right=PIVOT_RIGHT_BARS,
                max_gap_bars=PIVOT_MAX_GAP_BARS,
                min_price_move_pct=MIN_PRICE_MOVE_PCT,
                min_indicator_move=MIN_INDICATOR_MOVE,
                bullish=bullish,
            )
            if not pairs:
                continue

            selected_pairs = pairs if historical else [pairs[-1]]

            for pair in selected_pairs:
                score = _score_signal(tf, indicator_name, close, rsi, macd_hist)
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
                        sma_50=latest_sma_50,
                        sma_200=latest_sma_200,
                        sma_cross=sma_cross,
                        sma_cross_time=sma_cross_time.to_pydatetime() if sma_cross_time is not None else None,
                        score=score,
                        note="MVP pivot-based divergence",
                    )
                )

    return sorted(signals, key=lambda s: s.score, reverse=True)


def scan_universe(symbols: list[str], timeframes: list[str]) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        symbol = symbol.strip().upper()
        if not symbol:
            continue
        for signal in scan_symbol(symbol, timeframes):
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
                "sma_50",
                "sma_200",
                "sma_cross",
                "sma_cross_time",
                "score",
                "note",
            ]
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values(by=["score", "timeframe", "symbol"], ascending=[False, True, True]).reset_index(drop=True)


def scan_universe_history(symbols: list[str], timeframes: list[str]) -> pd.DataFrame:
    rows = []
    for symbol in symbols:
        symbol = symbol.strip().upper()
        if not symbol:
            continue
        for signal in scan_symbol_history(symbol, timeframes):
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
                "sma_50",
                "sma_200",
                "sma_cross",
                "sma_cross_time",
                "score",
                "note",
            ]
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values(by=["pivot_b_time", "score", "timeframe", "symbol"], ascending=[False, False, True, True]).reset_index(drop=True)

"""Benchmark scanner runtime with candle cache enabled vs disabled.

This benchmark uses synthetic OHLCV data and a controllable fetch delay so the
measurement reflects same-session scanner caching rather than network noise.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import screener.scanner as scanner


def _make_synthetic_frame(symbol: str, interval: str) -> pd.DataFrame:
    if interval in {"1h", "4h"}:
        periods = 900
        freq = "h"
    elif interval == "1d":
        periods = 700
        freq = "D"
    else:
        periods = 520
        freq = "W-MON"

    index = pd.date_range("2020-01-01", periods=periods, freq=freq, tz="UTC")
    base = pd.Series(range(periods), index=index).astype(float)
    symbol_bias = (sum(ord(c) for c in symbol) % 17) * 0.13
    wave = (base.mod(19) - 9.0) * 0.35 + (base.mod(41) - 20.0) * 0.12
    trend = base * 0.03
    close = 100.0 + symbol_bias + trend + wave

    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "volume": 1_000_000,
        },
        index=index,
    )
    return frame


def _install_synthetic_fetch(fetch_delay_ms: int) -> callable:
    original_fetch = scanner.fetch_candles

    def fake_fetch_candles(symbol: str, interval: str, period: str, auto_adjust: bool = True) -> pd.DataFrame:
        if fetch_delay_ms > 0:
            time.sleep(fetch_delay_ms / 1000.0)
        return _make_synthetic_frame(symbol, interval)

    scanner.fetch_candles = fake_fetch_candles
    return original_fetch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark scanner with and without candle cache.")
    parser.add_argument(
        "--symbols",
        default="AAPL,MSFT,NVDA,AMZN,GOOGL,META",
        help="Comma-separated symbols.",
    )
    parser.add_argument(
        "--timeframes",
        default="1h,4h,1d,1w",
        help="Comma-separated timeframes.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Measured runs per mode after warm-up (default: 3).",
    )
    parser.add_argument(
        "--fetch-delay-ms",
        type=int,
        default=25,
        help="Artificial delay applied on cache misses to make cache impact visible (default: 25).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/artifacts/cache_benchmark_session.json"),
        help="Output JSON path for benchmark report.",
    )
    return parser.parse_args()


def _single_run(symbols: list[str], timeframes: list[str], use_candle_cache: bool) -> float:
    start = time.perf_counter()
    _ = scanner.scan_universe(
        symbols=symbols,
        timeframes=timeframes,
        use_candle_cache=use_candle_cache,
    )
    end = time.perf_counter()
    return end - start


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def main() -> int:
    args = _parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    timeframes = [tf.strip() for tf in args.timeframes.split(",") if tf.strip()]

    original_fetch = _install_synthetic_fetch(args.fetch_delay_ms)
    try:
        if hasattr(scanner, "clear_candle_cache"):
            scanner.clear_candle_cache()

        # Warm-up once per mode to measure steady-state repeated scans.
        _single_run(symbols, timeframes, use_candle_cache=True)
        _single_run(symbols, timeframes, use_candle_cache=False)

        cache_on_times: list[float] = []
        cache_off_times: list[float] = []

        for _ in range(max(1, args.runs)):
            cache_on_times.append(_single_run(symbols, timeframes, use_candle_cache=True))
            cache_off_times.append(_single_run(symbols, timeframes, use_candle_cache=False))

    finally:
        scanner.fetch_candles = original_fetch

    mean_on = _mean(cache_on_times)
    mean_off = _mean(cache_off_times)
    improvement_pct = 0.0 if mean_off <= 0 else ((mean_off - mean_on) / mean_off) * 100.0

    report = {
        "symbols": symbols,
        "timeframes": timeframes,
        "runs": max(1, args.runs),
        "fetch_delay_ms": args.fetch_delay_ms,
        "benchmark_type": "same_session_synthetic_repeat",
        "cache_on_seconds": cache_on_times,
        "cache_off_seconds": cache_off_times,
        "cache_on_mean_seconds": round(mean_on, 4),
        "cache_off_mean_seconds": round(mean_off, 4),
        "cache_improvement_pct": round(improvement_pct, 2),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Report written to: {args.output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

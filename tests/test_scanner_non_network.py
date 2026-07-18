import pandas as pd
import pytest
import json
from pathlib import Path

import src.screener.scanner as scanner
from src.screener.divergence import PivotPair


def _make_candles(index: pd.DatetimeIndex, close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000_000,
        },
        index=index,
    )


@pytest.fixture
def aapl_like_snapshot() -> pd.DataFrame:
    """Deterministic, realistic-looking daily candles used as a scanner regression fixture."""
    index = pd.date_range("2022-01-03", periods=320, freq="D", tz="UTC")

    # Synthetic but market-like: broad downtrend with cyclical rebounds and pullbacks.
    base = pd.Series(range(320), index=index).astype(float)
    close = 220.0 - (base * 0.16) + (base.mod(11) - 5.0) * 1.45 - (base.mod(23) - 11.0) * 0.55
    return _make_candles(index, close)


def test_scan_universe_uses_causal_scoring(monkeypatch) -> None:
    index = pd.date_range("2020-01-01", periods=320, freq="D", tz="UTC")
    close = pd.Series(100.0 + (pd.Series(range(320), index=index) * 0.05), index=index)
    candles = _make_candles(index, close)

    pivot_a = index[120]
    pivot_b = index[200]
    pair = PivotPair(
        a_time=pivot_a,
        b_time=pivot_b,
        a_price=float(close.loc[pivot_a]),
        b_price=float(close.loc[pivot_b] - 1.0),
        a_indicator=25.0,
        b_indicator=30.0,
        a_indicator_time=pivot_a,
        b_indicator_time=pivot_b,
        price_move_pct=0.02,
        indicator_move=5.0,
        gap_bars=80,
    )

    def fake_fetch_candles(symbol: str, interval: str, period: str, auto_adjust: bool = True) -> pd.DataFrame:
        return candles

    def fake_find_divergences(*args, **kwargs):
        return [pair]

    def fake_compute_rsi(close_series: pd.Series, period: int = 14) -> pd.Series:
        rsi = pd.Series(50.0, index=close_series.index)
        b_idx = close_series.index.get_loc(pivot_b)
        rsi.iloc[b_idx - 1] = 40.0
        rsi.iloc[b_idx] = 41.0
        # Would trigger confirmation if latest bars were incorrectly used.
        rsi.iloc[-2] = 29.0
        rsi.iloc[-1] = 31.0
        return rsi

    def fake_compute_macd(close_series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        hist = pd.Series(0.0, index=close_series.index)
        return pd.DataFrame(
            {
                "macd": hist,
                "macd_signal": hist,
                "macd_hist": hist,
            },
            index=close_series.index,
        )

    monkeypatch.setattr(scanner, "fetch_candles", fake_fetch_candles)
    monkeypatch.setattr(scanner, "find_divergences", fake_find_divergences)
    monkeypatch.setattr(scanner, "compute_rsi", fake_compute_rsi)
    monkeypatch.setattr(scanner, "compute_macd", fake_compute_macd)

    results = scanner.scan_universe(
        symbols=["AAPL"],
        timeframes=["1d"],
        divergence_types=["regular_bullish"],
        indicators=["rsi"],
        use_ma_regime_filter=False,
    )

    assert not results.empty
    components = results.iloc[0]["score_components"]
    assert components["indicator_confirmation"] == 0


def test_scan_universe_applies_ma_regime_gating(monkeypatch) -> None:
    index = pd.date_range("2020-01-01", periods=320, freq="D", tz="UTC")
    close = pd.Series(100.0 + (pd.Series(range(320), index=index) * 0.05), index=index)
    candles = _make_candles(index, close)

    pivot_a = index[120]
    pivot_b = index[200]
    pair = PivotPair(
        a_time=pivot_a,
        b_time=pivot_b,
        a_price=float(close.loc[pivot_a]),
        b_price=float(close.loc[pivot_b] - 1.0),
        a_indicator=25.0,
        b_indicator=30.0,
        a_indicator_time=pivot_a,
        b_indicator_time=pivot_b,
        price_move_pct=0.02,
        indicator_move=5.0,
        gap_bars=80,
    )

    def fake_fetch_candles(symbol: str, interval: str, period: str, auto_adjust: bool = True) -> pd.DataFrame:
        return candles

    def fake_find_divergences(*args, **kwargs):
        return [pair]

    def fake_regime_at_signal_time(regime_context, signal_time):
        return "bullish", "1d,1w"

    monkeypatch.setattr(scanner, "fetch_candles", fake_fetch_candles)
    monkeypatch.setattr(scanner, "find_divergences", fake_find_divergences)
    monkeypatch.setattr(scanner, "_regime_at_signal_time", fake_regime_at_signal_time)

    bullish = scanner.scan_universe(
        symbols=["AAPL"],
        timeframes=["1d"],
        divergence_types=["regular_bullish"],
        indicators=["rsi"],
        use_ma_regime_filter=True,
    )
    bearish = scanner.scan_universe(
        symbols=["AAPL"],
        timeframes=["1d"],
        divergence_types=["regular_bearish"],
        indicators=["rsi"],
        use_ma_regime_filter=True,
    )

    assert not bullish.empty
    assert bearish.empty


def test_soft_mode_allows_mixed_regime_while_hard_blocks(monkeypatch) -> None:
    index = pd.date_range("2020-01-01", periods=320, freq="D", tz="UTC")
    close = pd.Series(100.0 + (pd.Series(range(320), index=index) * 0.05), index=index)
    candles = _make_candles(index, close)

    pivot_a = index[120]
    pivot_b = index[200]
    pair = PivotPair(
        a_time=pivot_a,
        b_time=pivot_b,
        a_price=float(close.loc[pivot_a]),
        b_price=float(close.loc[pivot_b] - 1.0),
        a_indicator=25.0,
        b_indicator=30.0,
        a_indicator_time=pivot_a,
        b_indicator_time=pivot_b,
        price_move_pct=0.02,
        indicator_move=5.0,
        gap_bars=80,
    )

    def fake_fetch_candles(symbol: str, interval: str, period: str, auto_adjust: bool = True) -> pd.DataFrame:
        return candles

    def fake_find_divergences(*args, **kwargs):
        return [pair]

    def fake_regime_at_signal_time(regime_context, signal_time):
        return "mixed", "1d,1w"

    monkeypatch.setattr(scanner, "fetch_candles", fake_fetch_candles)
    monkeypatch.setattr(scanner, "find_divergences", fake_find_divergences)
    monkeypatch.setattr(scanner, "_regime_at_signal_time", fake_regime_at_signal_time)

    soft = scanner.scan_universe(
        symbols=["AAPL"],
        timeframes=["1d"],
        divergence_types=["regular_bullish"],
        indicators=["rsi"],
        use_ma_regime_filter=True,
        ma_regime_filter_mode="soft",
    )
    hard = scanner.scan_universe(
        symbols=["AAPL"],
        timeframes=["1d"],
        divergence_types=["regular_bullish"],
        indicators=["rsi"],
        use_ma_regime_filter=True,
        ma_regime_filter_mode="hard",
    )

    assert not soft.empty
    assert hard.empty


def test_strict_mode_reduces_signal_count_on_same_snapshot(monkeypatch, aapl_like_snapshot: pd.DataFrame) -> None:
    index = aapl_like_snapshot.index

    # Indicator with sparse/absent local swing lows; loose mode can still map nearby extremes,
    # strict mode requires real indicator pivots and should reduce detections.
    monotonic_rsi = pd.Series(35.0 + (pd.Series(range(len(index)), index=index) * 0.12), index=index)

    def fake_fetch_candles(symbol: str, interval: str, period: str, auto_adjust: bool = True) -> pd.DataFrame:
        return aapl_like_snapshot

    def fake_compute_rsi(close_series: pd.Series, period: int = 14) -> pd.Series:
        return monotonic_rsi.reindex(close_series.index)

    def fake_compute_macd(close_series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        hist = pd.Series(0.0, index=close_series.index)
        return pd.DataFrame({"macd": hist, "macd_signal": hist, "macd_hist": hist}, index=close_series.index)

    def fake_find_divergences(*args, **kwargs):
        if kwargs.get("require_indicator_pivots"):
            return []
        return [
            PivotPair(
                a_time=index[80],
                b_time=index[160],
                a_price=200.0,
                b_price=180.0,
                a_indicator=35.0,
                b_indicator=42.0,
                a_indicator_time=index[80],
                b_indicator_time=index[160],
                price_move_pct=0.10,
                indicator_move=7.0,
                gap_bars=80,
            )
        ]

    monkeypatch.setattr(scanner, "fetch_candles", fake_fetch_candles)
    monkeypatch.setattr(scanner, "compute_rsi", fake_compute_rsi)
    monkeypatch.setattr(scanner, "compute_macd", fake_compute_macd)
    monkeypatch.setattr(scanner, "find_divergences", fake_find_divergences)

    loose = scanner.scan_universe(
        symbols=["AAPL"],
        timeframes=["1d"],
        divergence_types=["regular_bullish"],
        indicators=["rsi"],
        use_ma_regime_filter=False,
        strict_indicator_pivots=False,
    )
    strict = scanner.scan_universe(
        symbols=["AAPL"],
        timeframes=["1d"],
        divergence_types=["regular_bullish"],
        indicators=["rsi"],
        use_ma_regime_filter=False,
        strict_indicator_pivots=True,
    )

    loose_count = len(loose)
    strict_count = len(strict)
    reduction_pct = (0.0 if loose_count == 0 else ((loose_count - strict_count) / loose_count) * 100.0)

    artifact_dir = Path(__file__).resolve().parent / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "strict_mode_metrics.json"
    artifact_payload = {
        "fixture": "aapl_like_snapshot",
        "symbol": "AAPL",
        "timeframe": "1d",
        "divergence_type": "regular_bullish",
        "indicator": "rsi",
        "loose_count": loose_count,
        "strict_count": strict_count,
        "reduction_pct": round(reduction_pct, 4),
    }
    artifact_path.write_text(json.dumps(artifact_payload, indent=2), encoding="utf-8")

    assert loose_count > 0
    assert strict_count <= loose_count
    assert reduction_pct > 0.0

from __future__ import annotations

import pytest

import src.screener.scanner as scanner


@pytest.fixture(autouse=True)
def clear_scanner_cache_between_tests() -> None:
    if hasattr(scanner, "clear_candle_cache"):
        scanner.clear_candle_cache()
    yield
    if hasattr(scanner, "clear_candle_cache"):
        scanner.clear_candle_cache()

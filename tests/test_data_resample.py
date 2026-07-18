import pandas as pd

from src.screener.data import resample_ohlcv


def test_session_aware_resample_aligns_4h_to_market_session() -> None:
    # Jan 3, 2022 is in EST (UTC-5). 9:30 and 13:30 ET map to 14:30 and 18:30 UTC.
    index = pd.to_datetime(
        [
            "2022-01-03 14:30:00+00:00",
            "2022-01-03 15:30:00+00:00",
            "2022-01-03 16:30:00+00:00",
            "2022-01-03 17:30:00+00:00",
            "2022-01-03 18:30:00+00:00",
            "2022-01-03 19:30:00+00:00",
            "2022-01-03 20:30:00+00:00",
        ]
    )
    frame = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104, 105, 106],
            "high": [101, 102, 103, 104, 105, 106, 107],
            "low": [99, 100, 101, 102, 103, 104, 105],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5],
            "volume": [10, 11, 12, 13, 14, 15, 16],
        },
        index=index,
    )

    out = resample_ohlcv(
        frame,
        "4h",
        session_aware=True,
        session_timezone="America/New_York",
        session_start="09:30",
        session_end="16:00",
    )

    assert len(out) == 2
    assert out.index[0] == pd.Timestamp("2022-01-03 14:30:00+00:00")
    assert out.index[1] == pd.Timestamp("2022-01-03 18:30:00+00:00")
    assert float(out.iloc[0]["open"]) == 100.0
    assert float(out.iloc[1]["close"]) == 106.5

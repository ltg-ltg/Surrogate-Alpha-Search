import pandas as pd
import pytest

from src.data.download import dedupe_share_classes
from src.data.universe import jp_to_yf_ticker, us_to_yf_ticker


def test_us_ticker_conversion():
    assert us_to_yf_ticker("BRK.B") == "BRK-B"
    assert us_to_yf_ticker(" bf.b ") == "BF-B"
    assert us_to_yf_ticker("AAPL") == "AAPL"


def test_jp_ticker_conversion():
    assert jp_to_yf_ticker("7203", ".T") == "7203.T"
    assert jp_to_yf_ticker("285A", ".T") == "285A.T"
    with pytest.raises(ValueError):
        jp_to_yf_ticker("72O3x", ".T")


def test_dedupe_keeps_higher_is_volume_class():
    universe = pd.DataFrame(
        {
            "ticker": ["GOOGL", "GOOG", "AAPL"],
            "group_id": ["1652044", "1652044", "320193"],
        }
    )
    dates = pd.date_range("2016-01-01", periods=4, freq="D").append(
        pd.date_range("2024-01-01", periods=2, freq="D")
    )
    # GOOG has higher volume only outside IS; IS decides.
    volume = pd.DataFrame(
        {
            "GOOGL": [10, 10, 10, 10, 1, 1],
            "GOOG": [5, 5, 5, 5, 1000, 1000],
            "AAPL": [1, 1, 1, 1, 1, 1],
        },
        index=dates,
        dtype=float,
    )
    kept, excluded = dedupe_share_classes(universe, volume, ("2016-01-01", "2021-12-31"))
    assert sorted(kept["ticker"]) == ["AAPL", "GOOGL"]
    assert [e["ticker"] for e in excluded] == ["GOOG"]
    assert excluded[0]["reason"] == "duplicate_share_class"


def test_clean_panels_drops_holidays_and_masks_zero_volume():
    from src.data.download import clean_panels

    dates = pd.to_datetime(["2018-12-21", "2018-12-24", "2018-12-25"])
    close = pd.DataFrame({"A": [1.0, 1.0, 1.1], "B": [2.0, 2.0, 2.1], "C": [3.0, 3.0, 3.0]}, dates)
    # 12-24: all zero volume -> holiday. 12-25: only C zero -> masked cell.
    volume = pd.DataFrame({"A": [5.0, 0, 5], "B": [5.0, 0, 5], "C": [5.0, 0, 0]}, dates)
    cleaned, stats = clean_panels({"close": close, "volume": volume}, 0.5)
    assert list(cleaned["close"].index) == [dates[0], dates[2]]
    assert stats["holiday_dates_dropped"] == ["2018-12-24"]
    assert pd.isna(cleaned["close"].at[dates[2], "C"])
    assert pd.isna(cleaned["volume"].at[dates[2], "C"])
    assert cleaned["close"].at[dates[2], "A"] == 1.1
    assert stats["zero_volume_cells_masked"] == 1

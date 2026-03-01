"""
Tests for InformationBarGenerator.

Validates that dollar bars, volume bars, and tick bars are correctly
generated from OHLCV source data with proper aggregation, auto-calibration,
column normalisation, and edge-case handling.
"""

import numpy as np
import pandas as pd
import pytest

from src.phase_02_preprocessing.information_bars import InformationBarGenerator


# ---- Fixtures ---------------------------------------------------------------


@pytest.fixture
def minute_df():
    """
    Simulate ~5 trading days of 1-min OHLCV data (390 bars/day = 1,950 rows).

    Prices hover around 450 with small random moves.
    Volume varies between 50k and 200k per bar.
    """
    np.random.seed(42)
    n_days = 5
    bars_per_day = 390
    n_rows = n_days * bars_per_day

    # Build a DatetimeIndex with intraday minutes (9:30-16:00 each day)
    dates = pd.bdate_range("2024-06-03", periods=n_days)
    timestamps = []
    for d in dates:
        market_open = d + pd.Timedelta(hours=9, minutes=30)
        for m in range(bars_per_day):
            timestamps.append(market_open + pd.Timedelta(minutes=m))
    index = pd.DatetimeIndex(timestamps)

    close = 450.0 + np.cumsum(np.random.normal(0, 0.05, n_rows))
    opens = close + np.random.normal(0, 0.02, n_rows)
    highs = np.maximum(opens, close) + np.abs(np.random.normal(0.03, 0.01, n_rows))
    lows = np.minimum(opens, close) - np.abs(np.random.normal(0.03, 0.01, n_rows))
    volume = np.random.randint(50_000, 200_000, n_rows).astype(float)

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


@pytest.fixture
def small_df():
    """A tiny 10-row DataFrame for precise aggregation checks."""
    index = pd.date_range("2024-06-03 09:30", periods=10, freq="min")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0,
                     105.0, 106.0, 107.0, 108.0, 109.0],
            "high": [100.5, 101.5, 102.5, 103.5, 104.5,
                     105.5, 106.5, 107.5, 108.5, 109.5],
            "low": [99.5, 100.5, 101.5, 102.5, 103.5,
                    104.5, 105.5, 106.5, 107.5, 108.5],
            "close": [100.2, 101.2, 102.2, 103.2, 104.2,
                      105.2, 106.2, 107.2, 108.2, 109.2],
            "volume": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0,
                       6000.0, 7000.0, 8000.0, 9000.0, 10000.0],
        },
        index=index,
    )


# ---- Test: Dollar bars produce correct OHLCV aggregation -------------------


def test_dollar_bars_ohlcv_aggregation(small_df):
    """Dollar bars should aggregate OHLCV correctly: O=first, H=max, L=min, C=last."""
    # Rows 0-4 have dollar_vol ~= 100*1000 + 101*2000 + ... compute exact
    # Use a threshold that groups roughly the first 5 rows together
    dollar_vols = small_df["close"].values * small_df["volume"].values
    threshold = sum(dollar_vols[:5]) + 1.0  # first 5 rows just under threshold

    # Actually set threshold so first 5 rows exactly trip it
    threshold = sum(dollar_vols[:5])

    gen = InformationBarGenerator(bar_type="dollar", threshold=threshold)
    bars = gen.generate(small_df)

    assert len(bars) >= 1
    first_bar = bars.iloc[0]

    # First bar covers rows 0..4
    assert first_bar["open"] == pytest.approx(100.0)
    assert first_bar["high"] == pytest.approx(104.5)
    assert first_bar["low"] == pytest.approx(99.5)
    assert first_bar["close"] == pytest.approx(104.2)
    assert first_bar["volume"] == pytest.approx(15000.0)


# ---- Test: Volume bars produce correct output --------------------------------


def test_volume_bars_correct_output(small_df):
    """Volume bars should create new bar when cumulative volume >= threshold."""
    # Total volume = 55000; threshold = 15000 should give ~3-4 bars
    gen = InformationBarGenerator(bar_type="volume", threshold=15000.0)
    bars = gen.generate(small_df)

    assert len(bars) >= 2
    assert "open" in bars.columns
    assert "high" in bars.columns
    assert "low" in bars.columns
    assert "close" in bars.columns
    assert "volume" in bars.columns

    # Each bar's volume should be >= threshold (except possibly the last)
    for i in range(len(bars) - 1):
        assert bars.iloc[i]["volume"] >= 15000.0


# ---- Test: Tick bars with fixed threshold -----------------------------------


def test_tick_bars_fixed_threshold(small_df):
    """Tick bars with threshold=3 should produce ceil(10/3) = 4 bars."""
    gen = InformationBarGenerator(bar_type="tick", threshold=3)
    bars = gen.generate(small_df)

    # 10 rows / 3 per bar = 3 full bars + 1 partial bar = 4 bars
    assert len(bars) == 4

    # First bar should span rows 0-2
    first_bar = bars.iloc[0]
    assert first_bar["open"] == pytest.approx(100.0)
    assert first_bar["high"] == pytest.approx(102.5)
    assert first_bar["low"] == pytest.approx(99.5)
    assert first_bar["close"] == pytest.approx(102.2)
    assert first_bar["bar_duration"] == 3

    # Last bar should span rows 9 only (1 row remaining)
    last_bar = bars.iloc[-1]
    assert last_bar["bar_duration"] == 1
    assert last_bar["open"] == pytest.approx(109.0)
    assert last_bar["close"] == pytest.approx(109.2)


# ---- Test: Auto-calibration produces reasonable number of bars ---------------


def test_auto_calibration_reasonable_count(minute_df):
    """Auto-calibrated bars should produce roughly 1 bar per trading day (~5 bars)."""
    gen = InformationBarGenerator(bar_type="dollar", threshold=None, auto_calibrate=True)
    bars = gen.generate(minute_df)

    # With 5 days of data, auto-calibration targets ~5 bars
    # Allow a range of 3 to 8 (boundaries can shift by +/- a few)
    assert 3 <= len(bars) <= 8, f"Expected ~5 bars, got {len(bars)}"
    assert gen.calibrated_threshold is not None
    assert gen.calibrated_threshold > 0

    # Also test volume bars auto-calibration
    gen_vol = InformationBarGenerator(bar_type="volume", threshold=None, auto_calibrate=True)
    bars_vol = gen_vol.generate(minute_df)
    assert 3 <= len(bars_vol) <= 8, f"Expected ~5 volume bars, got {len(bars_vol)}"

    # And tick bars
    gen_tick = InformationBarGenerator(bar_type="tick", threshold=None, auto_calibrate=True)
    bars_tick = gen_tick.generate(minute_df)
    assert 3 <= len(bars_tick) <= 8, f"Expected ~5 tick bars, got {len(bars_tick)}"


# ---- Test: Empty DataFrame --------------------------------------------------


def test_empty_dataframe():
    """Empty input should return empty DataFrame with correct columns."""
    gen = InformationBarGenerator(bar_type="dollar", threshold=1_000_000.0)
    empty = pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
    )
    empty.index = pd.DatetimeIndex([])

    result = gen.generate(empty)

    assert result.empty
    assert "open" in result.columns
    assert "high" in result.columns
    assert "close" in result.columns
    assert "volume" in result.columns
    assert "bar_duration" in result.columns
    assert "bar_dollar_volume" in result.columns


# ---- Test: Column name normalisation (uppercase/lowercase) ------------------


def test_column_name_normalisation():
    """Generator should accept both uppercase and mixed-case column names."""
    index = pd.date_range("2024-06-03 09:30", periods=5, freq="min")
    df = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "HIGH": [100.5, 101.5, 102.5, 103.5, 104.5],
            "Low": [99.5, 100.5, 101.5, 102.5, 103.5],
            "CLOSE": [100.2, 101.2, 102.2, 103.2, 104.2],
            "Volume": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        },
        index=index,
    )

    gen = InformationBarGenerator(bar_type="tick", threshold=3)
    bars = gen.generate(df)

    # Should succeed and produce bars with lowercase columns
    assert len(bars) == 2  # 5 rows / 3 = 1 full + 1 partial
    assert "open" in bars.columns
    assert "high" in bars.columns
    assert "low" in bars.columns
    assert "close" in bars.columns
    assert "volume" in bars.columns


# ---- Test: bar_duration is correct ------------------------------------------


def test_bar_duration_correct(small_df):
    """bar_duration should equal the number of source rows aggregated into each bar."""
    gen = InformationBarGenerator(bar_type="tick", threshold=4)
    bars = gen.generate(small_df)

    # 10 rows / 4 = 2 full bars (4 rows each) + 1 partial (2 rows)
    assert len(bars) == 3
    assert bars.iloc[0]["bar_duration"] == 4
    assert bars.iloc[1]["bar_duration"] == 4
    assert bars.iloc[2]["bar_duration"] == 2

    # Total duration should equal total source rows
    assert bars["bar_duration"].sum() == len(small_df)


# ---- Test: Dollar volume is correctly computed --------------------------------


def test_dollar_volume_correct(small_df):
    """bar_dollar_volume should equal sum(close * volume) for each bar's rows."""
    gen = InformationBarGenerator(bar_type="tick", threshold=5)
    bars = gen.generate(small_df)

    # First bar covers rows 0-4
    expected_dv = sum(
        small_df["close"].values[i] * small_df["volume"].values[i]
        for i in range(5)
    )
    assert bars.iloc[0]["bar_dollar_volume"] == pytest.approx(expected_dv)

    # Second bar covers rows 5-9
    expected_dv2 = sum(
        small_df["close"].values[i] * small_df["volume"].values[i]
        for i in range(5, 10)
    )
    assert bars.iloc[1]["bar_dollar_volume"] == pytest.approx(expected_dv2)


# ---- Test: Output index is datetime ----------------------------------------


def test_output_index_is_datetime(minute_df):
    """When input has DatetimeIndex, output should also have DatetimeIndex."""
    gen = InformationBarGenerator(bar_type="dollar", threshold=None, auto_calibrate=True)
    bars = gen.generate(minute_df)

    assert isinstance(bars.index, pd.DatetimeIndex), (
        f"Expected DatetimeIndex, got {type(bars.index)}"
    )
    # Index values should be from the source data
    assert bars.index[0] >= minute_df.index[0]
    assert bars.index[-1] <= minute_df.index[-1]


# ---- Test: Single row input ------------------------------------------------


def test_single_row_input():
    """A single-row DataFrame should produce exactly one bar."""
    index = pd.DatetimeIndex([pd.Timestamp("2024-06-03 09:30")])
    df = pd.DataFrame(
        {
            "open": [450.0],
            "high": [451.0],
            "low": [449.0],
            "close": [450.5],
            "volume": [100000.0],
        },
        index=index,
    )

    for bar_type in ("dollar", "volume", "tick"):
        gen = InformationBarGenerator(bar_type=bar_type, threshold=1e18)
        bars = gen.generate(df)

        assert len(bars) == 1, f"{bar_type}: expected 1 bar, got {len(bars)}"
        assert bars.iloc[0]["open"] == pytest.approx(450.0)
        assert bars.iloc[0]["high"] == pytest.approx(451.0)
        assert bars.iloc[0]["low"] == pytest.approx(449.0)
        assert bars.iloc[0]["close"] == pytest.approx(450.5)
        assert bars.iloc[0]["volume"] == pytest.approx(100000.0)
        assert bars.iloc[0]["bar_duration"] == 1


# ---- Test: Invalid bar type raises ValueError --------------------------------


def test_invalid_bar_type_raises():
    """An invalid bar_type should raise ValueError."""
    with pytest.raises(ValueError, match="Invalid bar_type"):
        InformationBarGenerator(bar_type="invalid")


# ---- Test: Missing columns raises ValueError ----------------------------------


def test_missing_columns_raises():
    """DataFrame without required OHLCV columns should raise ValueError."""
    df = pd.DataFrame({"open": [100], "high": [101], "close": [100.5]})
    gen = InformationBarGenerator(bar_type="tick", threshold=1)

    with pytest.raises(ValueError, match="Missing required OHLCV columns"):
        gen.generate(df)

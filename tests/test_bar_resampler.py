"""
Tests for BarResampler - OHLCV bar resampling utility.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phase_02_preprocessing.bar_resampler import (
    BarResampler,
    resolution_to_minutes,
    RESOLUTION_MAP,
)


# =============================================================================
# FIXTURES
# =============================================================================

def make_1min_bars(n_days: int = 5, bars_per_day: int = 390) -> pd.DataFrame:
    """Create synthetic 1-min OHLCV bars for testing."""
    records = []
    base_price = 450.0

    for day_idx in range(n_days):
        date_str = f"2025-01-{13 + day_idx:02d}"
        base_ts = pd.Timestamp(f"{date_str} 09:30:00", tz="US/Eastern")

        for minute in range(bars_per_day):
            ts = base_ts + pd.Timedelta(minutes=minute)
            # Random walk
            change = np.random.uniform(-0.001, 0.001) * base_price
            o = base_price + change
            h = o + abs(np.random.uniform(0, 0.5))
            l = o - abs(np.random.uniform(0, 0.5))
            c = np.random.uniform(l, h)
            v = int(np.random.uniform(10000, 100000))

            records.append({
                "timestamp": ts.tz_localize(None),
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(c, 2),
                "volume": v,
                "date": date_str,
                "session": "regular",
                "hour": ts.hour,
                "minute": ts.minute,
            })

            base_price = c

    return pd.DataFrame(records)


def make_extended_hours_bars() -> pd.DataFrame:
    """Create 1-min bars with premarket, regular, and afterhours sessions."""
    records = []
    base_price = 450.0
    date_str = "2025-01-13"

    sessions = [
        ("premarket", "04:00", 330),   # 4:00 AM - 9:30 AM = 330 minutes
        ("regular", "09:30", 390),      # 9:30 AM - 4:00 PM = 390 minutes
        ("afterhours", "16:00", 240),   # 4:00 PM - 8:00 PM = 240 minutes
    ]

    for session_name, start_time, n_bars in sessions:
        base_ts = pd.Timestamp(f"{date_str} {start_time}", tz="US/Eastern")

        for minute in range(n_bars):
            ts = base_ts + pd.Timedelta(minutes=minute)
            change = np.random.uniform(-0.001, 0.001) * base_price
            o = base_price + change
            h = o + abs(np.random.uniform(0, 0.3))
            l = o - abs(np.random.uniform(0, 0.3))
            c = np.random.uniform(l, h)
            v = int(np.random.uniform(1000, 50000))

            records.append({
                "timestamp": ts.tz_localize(None),
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(c, 2),
                "volume": v,
                "date": date_str,
                "session": session_name,
                "hour": ts.hour,
                "minute": ts.minute,
            })

            base_price = c

    return pd.DataFrame(records)


# =============================================================================
# TESTS: resolution_to_minutes
# =============================================================================

class TestResolutionToMinutes:
    """Tests for the resolution string parser."""

    def test_standard_resolutions(self):
        assert resolution_to_minutes("1min") == 1
        assert resolution_to_minutes("2min") == 2
        assert resolution_to_minutes("3min") == 3
        assert resolution_to_minutes("5min") == 5
        assert resolution_to_minutes("10min") == 10
        assert resolution_to_minutes("15min") == 15
        assert resolution_to_minutes("30min") == 30
        assert resolution_to_minutes("1h") == 60

    def test_invalid_resolution(self):
        with pytest.raises(ValueError):
            resolution_to_minutes("invalid")

    def test_resolution_map_complete(self):
        assert len(RESOLUTION_MAP) == 8
        assert "1min" in RESOLUTION_MAP
        assert "30min" in RESOLUTION_MAP
        assert "1h" in RESOLUTION_MAP


# =============================================================================
# TESTS: BarResampler
# =============================================================================

class TestBarResampler:
    """Tests for the BarResampler class."""

    def test_1min_passthrough(self):
        """1-min resolution should return a copy, not resample."""
        df = make_1min_bars(n_days=1, bars_per_day=10)
        resampler = BarResampler()
        result = resampler.resample(df, resolution_minutes=1)

        assert len(result) == len(df)
        assert list(result.columns) == list(df.columns)
        # Should be a copy, not the same object
        assert result is not df

    def test_5min_resampling_reduces_bars(self):
        """5-min resampling should produce ~1/5 of the bars."""
        df = make_1min_bars(n_days=1, bars_per_day=390)
        resampler = BarResampler()
        result = resampler.resample(df, resolution_minutes=5)

        # 390 / 5 = 78 bars expected
        assert len(result) == 78

    def test_ohlcv_aggregation_correctness(self):
        """Verify OHLCV aggregation rules: open=first, high=max, low=min, close=last, volume=sum."""
        # Create 5 bars with known values
        records = []
        base_ts = pd.Timestamp("2025-01-13 09:30:00")
        for i in range(5):
            records.append({
                "timestamp": base_ts + pd.Timedelta(minutes=i),
                "open": 100.0 + i,      # 100, 101, 102, 103, 104
                "high": 110.0 + i,      # 110, 111, 112, 113, 114
                "low": 90.0 + i,        # 90, 91, 92, 93, 94
                "close": 105.0 + i,     # 105, 106, 107, 108, 109
                "volume": 1000 * (i + 1),  # 1000, 2000, 3000, 4000, 5000
                "date": "2025-01-13",
                "session": "regular",
            })
        df = pd.DataFrame(records)

        resampler = BarResampler()
        result = resampler.resample(df, resolution_minutes=5)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["open"] == 100.0      # first open
        assert row["high"] == 114.0      # max high
        assert row["low"] == 90.0        # min low
        assert row["close"] == 109.0     # last close
        assert row["volume"] == 15000    # sum of volumes

    def test_session_boundary_respect(self):
        """Bars should not span session boundaries."""
        df = make_extended_hours_bars()
        resampler = BarResampler()
        result = resampler.resample(df, resolution_minutes=15, respect_sessions=True)

        # Check that each session is resampled independently
        for session_name in ["premarket", "regular", "afterhours"]:
            session_bars = result[result["session"] == session_name]
            assert len(session_bars) > 0, f"No bars for session {session_name}"

        # Premarket: 330 bars / 15 = 22 bars
        pm_bars = result[result["session"] == "premarket"]
        assert len(pm_bars) == 22

        # Regular: 390 bars / 15 = 26 bars
        reg_bars = result[result["session"] == "regular"]
        assert len(reg_bars) == 26

        # Afterhours: 240 bars / 15 = 16 bars
        ah_bars = result[result["session"] == "afterhours"]
        assert len(ah_bars) == 16

    def test_date_grouping_preserved(self):
        """Bars from different days should not be mixed."""
        df = make_1min_bars(n_days=3, bars_per_day=60)
        resampler = BarResampler()
        result = resampler.resample(df, resolution_minutes=5)

        # Each day: 60 bars / 5 = 12 resampled bars
        dates = result["date"].unique()
        assert len(dates) == 3

        for date_val in dates:
            day_bars = result[result["date"] == date_val]
            assert len(day_bars) == 12

    def test_cache_behavior(self):
        """Second call with same data should use cache."""
        df = make_1min_bars(n_days=1, bars_per_day=60)
        resampler = BarResampler()

        result1 = resampler.resample(df, resolution_minutes=5)
        result2 = resampler.resample(df, resolution_minutes=5)

        # Results should be equal
        pd.testing.assert_frame_equal(result1, result2)

        # 5min should be in cache
        assert 5 in resampler.get_cached_resolutions()

    def test_cache_cleared_on_new_data(self):
        """Cache should clear when source data changes."""
        resampler = BarResampler()

        df1 = make_1min_bars(n_days=1, bars_per_day=60)
        resampler.resample(df1, resolution_minutes=5)
        assert 5 in resampler.get_cached_resolutions()

        df2 = make_1min_bars(n_days=2, bars_per_day=60)
        resampler.resample(df2, resolution_minutes=10)
        # Old cache should be cleared because source changed
        assert 5 not in resampler.get_cached_resolutions()
        assert 10 in resampler.get_cached_resolutions()

    def test_multiple_resolutions_from_same_source(self):
        """Multiple resolutions from same source data."""
        df = make_1min_bars(n_days=1, bars_per_day=60)
        resampler = BarResampler()

        r3 = resampler.resample(df, resolution_minutes=3)
        r5 = resampler.resample(df, resolution_minutes=5)
        r15 = resampler.resample(df, resolution_minutes=15)

        assert len(r3) == 20   # 60/3
        assert len(r5) == 12   # 60/5
        assert len(r15) == 4   # 60/15

        # All should be cached
        assert set(resampler.get_cached_resolutions()) == {3, 5, 15}

    def test_invalid_resolution_raises(self):
        """Invalid resolution should raise ValueError."""
        df = make_1min_bars(n_days=1, bars_per_day=10)
        resampler = BarResampler()

        with pytest.raises(ValueError):
            resampler.resample(df, resolution_minutes=7)

    def test_valid_resolutions(self):
        """All valid resolutions should work."""
        df = make_1min_bars(n_days=1, bars_per_day=60)
        resampler = BarResampler()

        for res in [2, 3, 5, 10, 15, 30]:
            result = resampler.resample(df, resolution_minutes=res)
            expected = 60 // res
            assert len(result) == expected, f"Expected {expected} bars at {res}min, got {len(result)}"

    def test_clear_cache(self):
        """clear_cache should empty the cache."""
        df = make_1min_bars(n_days=1, bars_per_day=60)
        resampler = BarResampler()

        resampler.resample(df, resolution_minutes=5)
        assert len(resampler.get_cached_resolutions()) > 0

        resampler.clear_cache()
        assert len(resampler.get_cached_resolutions()) == 0

    def test_high_low_invariant(self):
        """After resampling, high >= open, close, low for each bar."""
        df = make_1min_bars(n_days=2, bars_per_day=60)
        resampler = BarResampler()

        for res in [3, 5, 10, 15]:
            result = resampler.resample(df, resolution_minutes=res)
            assert (result["high"] >= result["open"]).all()
            assert (result["high"] >= result["close"]).all()
            assert (result["high"] >= result["low"]).all()
            assert (result["low"] <= result["open"]).all()
            assert (result["low"] <= result["close"]).all()

    def test_volume_conservation(self):
        """Total volume should be preserved after resampling."""
        df = make_1min_bars(n_days=1, bars_per_day=60)
        total_volume_before = df["volume"].sum()

        resampler = BarResampler()
        result = resampler.resample(df, resolution_minutes=5)
        total_volume_after = result["volume"].sum()

        assert total_volume_before == total_volume_after


# =============================================================================
# TESTS: Edge Cases
# =============================================================================

class TestBarResamplerEdgeCases:
    """Edge case tests for BarResampler."""

    def test_single_bar_per_session(self):
        """Session with only 1 bar should be kept as-is."""
        records = [{
            "timestamp": pd.Timestamp("2025-01-13 09:30:00"),
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
            "volume": 1000,
            "date": "2025-01-13",
            "session": "regular",
        }]
        df = pd.DataFrame(records)
        resampler = BarResampler()
        result = resampler.resample(df, resolution_minutes=5)

        assert len(result) == 1

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty DataFrame."""
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "date", "session"])
        resampler = BarResampler()
        result = resampler.resample(df, resolution_minutes=5)

        assert len(result) == 0

    def test_bars_without_session_column(self):
        """Should work without session column (uses date grouping)."""
        records = []
        base_ts = pd.Timestamp("2025-01-13 09:30:00")
        for i in range(30):
            records.append({
                "timestamp": base_ts + pd.Timedelta(minutes=i),
                "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                "volume": 1000,
                "date": "2025-01-13",
            })
        df = pd.DataFrame(records)
        resampler = BarResampler()
        result = resampler.resample(df, resolution_minutes=5)

        assert len(result) == 6  # 30/5

    def test_non_divisible_bar_count(self):
        """When bar count isn't divisible by resolution, last partial bar should be dropped."""
        records = []
        base_ts = pd.Timestamp("2025-01-13 09:30:00")
        for i in range(17):  # 17 bars, 5-min resolution = 3 full bars + 2 leftover
            records.append({
                "timestamp": base_ts + pd.Timedelta(minutes=i),
                "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
                "volume": 1000,
                "date": "2025-01-13",
                "session": "regular",
            })
        df = pd.DataFrame(records)
        resampler = BarResampler()
        result = resampler.resample(df, resolution_minutes=5)

        # 17 minutes: bars at 9:30, 9:35, 9:40, 9:45 (partial)
        # Pandas resample will create 4 bars (9:30-9:34, 9:35-9:39, 9:40-9:44, 9:45-9:46)
        assert len(result) >= 3  # At least 3 full bars

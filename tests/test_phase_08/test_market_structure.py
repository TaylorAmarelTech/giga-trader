"""
Tests for MarketStructureFeatures (mstr_ prefix, 18 features).
"""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.market_structure_features import MarketStructureFeatures
from src.phase_10_feature_processing.group_aware_processor import FEATURE_GROUPS


# ------------------------------------------------------------------ #
#  Fixtures                                                           #
# ------------------------------------------------------------------ #

@pytest.fixture
def ohlcv_df():
    """Create a realistic OHLCV DataFrame with 300 rows."""
    np.random.seed(42)
    n = 300
    close = 400.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.8)
    low = close - np.abs(np.random.randn(n) * 0.8)
    open_ = close + np.random.randn(n) * 0.3
    volume = np.random.randint(50_000_000, 200_000_000, size=n).astype(float)
    dates = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def close_only_df():
    """DataFrame with only a close column (minimal case)."""
    np.random.seed(99)
    n = 200
    close = 450.0 + np.cumsum(np.random.randn(n) * 0.4)
    return pd.DataFrame({"close": close})


@pytest.fixture
def short_df():
    """DataFrame with only 10 rows — too short."""
    np.random.seed(7)
    n = 10
    close = 400.0 + np.cumsum(np.random.randn(n) * 0.3)
    return pd.DataFrame({
        "close": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "open": close + 0.1,
        "volume": np.full(n, 1e8),
    })


@pytest.fixture
def mstr():
    return MarketStructureFeatures()


# ------------------------------------------------------------------ #
#  Invariant Tests                                                    #
# ------------------------------------------------------------------ #

class TestInvariants:
    def test_all_18_features_created(self, mstr, ohlcv_df):
        result = mstr.create_market_structure_features(ohlcv_df)
        mstr_cols = [c for c in result.columns if c.startswith("mstr_")]
        assert len(mstr_cols) == 18, f"Expected 18 mstr_ cols, got {len(mstr_cols)}: {mstr_cols}"

    def test_no_nan(self, mstr, ohlcv_df):
        result = mstr.create_market_structure_features(ohlcv_df)
        mstr_cols = [c for c in result.columns if c.startswith("mstr_")]
        for col in mstr_cols:
            assert not result[col].isna().any(), f"NaN found in {col}"

    def test_no_inf(self, mstr, ohlcv_df):
        result = mstr.create_market_structure_features(ohlcv_df)
        mstr_cols = [c for c in result.columns if c.startswith("mstr_")]
        for col in mstr_cols:
            assert not np.isinf(result[col].values).any(), f"Inf found in {col}"

    def test_preserves_rows(self, mstr, ohlcv_df):
        result = mstr.create_market_structure_features(ohlcv_df)
        assert len(result) == len(ohlcv_df)

    def test_preserves_original_columns(self, mstr, ohlcv_df):
        orig_cols = set(ohlcv_df.columns)
        result = mstr.create_market_structure_features(ohlcv_df)
        assert orig_cols.issubset(set(result.columns))

    def test_close_only_works(self, mstr, close_only_df):
        """Should work with only a close column (degraded mode)."""
        result = mstr.create_market_structure_features(close_only_df)
        mstr_cols = [c for c in result.columns if c.startswith("mstr_")]
        assert len(mstr_cols) >= 10  # At least close-based features

    def test_short_data_no_crash(self, mstr, short_df):
        """Should not crash on very short data."""
        result = mstr.create_market_structure_features(short_df)
        assert len(result) == len(short_df)


# ------------------------------------------------------------------ #
#  Compression Tests                                                  #
# ------------------------------------------------------------------ #

class TestCompression:
    def test_atr_ratio_positive(self, mstr, ohlcv_df):
        result = mstr.create_market_structure_features(ohlcv_df)
        vals = result["mstr_atr_ratio_5_40"].values
        # ATR ratio should be non-negative
        assert np.all(vals[~np.isnan(vals)] >= 0.0)

    def test_bbw_percentile_bounded(self, mstr, ohlcv_df):
        result = mstr.create_market_structure_features(ohlcv_df)
        vals = result["mstr_bbw_percentile_120"].values
        assert np.all(vals >= 0.0) and np.all(vals <= 1.0)

    def test_rv_percentile_bounded(self, mstr, ohlcv_df):
        result = mstr.create_market_structure_features(ohlcv_df)
        vals = result["mstr_rv_percentile_252"].values
        assert np.all(vals >= 0.0) and np.all(vals <= 1.0)

    def test_range_percentile_bounded(self, mstr, ohlcv_df):
        result = mstr.create_market_structure_features(ohlcv_df)
        vals = result["mstr_range_percentile_60"].values
        assert np.all(vals >= 0.0) and np.all(vals <= 1.0)

    def test_squeeze_binary(self, mstr, ohlcv_df):
        result = mstr.create_market_structure_features(ohlcv_df)
        vals = result["mstr_squeeze_on"].values
        unique = set(np.unique(vals))
        assert unique.issubset({0.0, 1.0})

    def test_squeeze_duration_non_negative(self, mstr, ohlcv_df):
        result = mstr.create_market_structure_features(ohlcv_df)
        vals = result["mstr_squeeze_duration"].values
        assert np.all(vals >= 0.0)

    def test_nr_count_non_negative(self, mstr, ohlcv_df):
        result = mstr.create_market_structure_features(ohlcv_df)
        vals = result["mstr_nr_count_21"].values
        assert np.all(vals >= 0.0)


# ------------------------------------------------------------------ #
#  Attractor Tests                                                    #
# ------------------------------------------------------------------ #

class TestAttractors:
    def test_vwap_deviation_reasonable(self, mstr, ohlcv_df):
        result = mstr.create_market_structure_features(ohlcv_df)
        vals = result["mstr_vwap_deviation"].values
        # Clipped to [-10, 10]
        assert np.all(vals >= -10.0) and np.all(vals <= 10.0)

    def test_ma_ribbon_non_negative(self, mstr, ohlcv_df):
        result = mstr.create_market_structure_features(ohlcv_df)
        vals = result["mstr_ma_ribbon_width"].values
        assert np.all(vals >= 0.0)


# ------------------------------------------------------------------ #
#  Inflection Tests                                                   #
# ------------------------------------------------------------------ #

class TestInflection:
    def test_hurst_distance_bounded(self, mstr, ohlcv_df):
        result = mstr.create_market_structure_features(ohlcv_df)
        vals = result["mstr_hurst_distance_50"].values
        # |Hurst - 0.5| is bounded [0, 0.5]
        assert np.all(vals >= 0.0) and np.all(vals <= 0.5)

    def test_cusum_non_negative(self, mstr, ohlcv_df):
        result = mstr.create_market_structure_features(ohlcv_df)
        vals = result["mstr_cusum_buildup"].values
        assert np.all(vals >= 0.0)

    def test_confluence_bounded(self, mstr, ohlcv_df):
        result = mstr.create_market_structure_features(ohlcv_df)
        vals = result["mstr_confluence_score"].values
        assert np.all(vals >= 0.0) and np.all(vals <= 5.0)


# ------------------------------------------------------------------ #
#  Directional Bias Tests                                             #
# ------------------------------------------------------------------ #

class TestDirectionalBias:
    def test_close_in_range_bounded(self, mstr, ohlcv_df):
        result = mstr.create_market_structure_features(ohlcv_df)
        vals = result["mstr_close_in_range"].values
        assert np.all(vals >= 0.0) and np.all(vals <= 1.0)

    def test_volume_skew_bounded(self, mstr, ohlcv_df):
        result = mstr.create_market_structure_features(ohlcv_df)
        vals = result["mstr_volume_skew"].values
        assert np.all(vals >= 0.0) and np.all(vals <= 1.0)

    def test_compression_energy_bounded(self, mstr, ohlcv_df):
        result = mstr.create_market_structure_features(ohlcv_df)
        vals = result["mstr_compression_energy"].values
        assert np.all(vals >= 0.0) and np.all(vals <= 1.0)


# ------------------------------------------------------------------ #
#  Analyze Tests                                                      #
# ------------------------------------------------------------------ #

class TestAnalyze:
    def test_analyze_returns_dict(self, mstr, ohlcv_df):
        result = mstr.create_market_structure_features(ohlcv_df)
        analysis = mstr.analyze_current_structure(result)
        assert isinstance(analysis, dict)

    def test_analyze_has_required_keys(self, mstr, ohlcv_df):
        result = mstr.create_market_structure_features(ohlcv_df)
        analysis = mstr.analyze_current_structure(result)
        for key in ["squeeze_on", "squeeze_duration", "compression_energy",
                     "close_in_range", "compression_regime"]:
            assert key in analysis, f"Missing key: {key}"

    def test_analyze_none_on_empty(self, mstr):
        df = pd.DataFrame({"close": []})
        analysis = mstr.analyze_current_structure(df)
        assert analysis is None


# ------------------------------------------------------------------ #
#  Feature Group Integration                                          #
# ------------------------------------------------------------------ #

class TestFeatureGroup:
    def test_market_structure_group_exists(self):
        assert "market_structure" in FEATURE_GROUPS

    def test_market_structure_group_prefix(self):
        assert FEATURE_GROUPS["market_structure"] == ["mstr_"]

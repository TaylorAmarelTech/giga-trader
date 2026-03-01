"""
Tests for TimeSeriesModelFeatures (tsm_ prefix, 10-15 features).
"""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.time_series_model_features import TimeSeriesModelFeatures
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
    """DataFrame with only a close column."""
    np.random.seed(99)
    n = 200
    close = 450.0 + np.cumsum(np.random.randn(n) * 0.4)
    return pd.DataFrame({"close": close})


@pytest.fixture
def short_df():
    """DataFrame with only 20 rows — minimal."""
    np.random.seed(7)
    n = 20
    close = 400.0 + np.cumsum(np.random.randn(n) * 0.3)
    return pd.DataFrame({"close": close})


@pytest.fixture
def tsm():
    """TSM with optional deps disabled."""
    return TimeSeriesModelFeatures(use_chronos=False, use_catch22=False)


# ------------------------------------------------------------------ #
#  Invariant Tests                                                    #
# ------------------------------------------------------------------ #

class TestInvariants:
    def test_arima_features_always_created(self, tsm, ohlcv_df):
        """At minimum, ARIMA features (3) + cross-model (2) + stubs should exist."""
        result = tsm.create_time_series_model_features(ohlcv_df)
        tsm_cols = [c for c in result.columns if c.startswith("tsm_")]
        # 3 ARIMA + 5 Chronos stubs + 2 cross-model + 5 catch22 stubs = 15
        assert len(tsm_cols) >= 10, f"Expected >=10 tsm_ cols, got {len(tsm_cols)}: {tsm_cols}"

    def test_no_nan(self, tsm, ohlcv_df):
        result = tsm.create_time_series_model_features(ohlcv_df)
        tsm_cols = [c for c in result.columns if c.startswith("tsm_")]
        for col in tsm_cols:
            assert not result[col].isna().any(), f"NaN found in {col}"

    def test_no_inf(self, tsm, ohlcv_df):
        result = tsm.create_time_series_model_features(ohlcv_df)
        tsm_cols = [c for c in result.columns if c.startswith("tsm_")]
        for col in tsm_cols:
            assert not np.isinf(result[col].values).any(), f"Inf found in {col}"

    def test_preserves_rows(self, tsm, ohlcv_df):
        result = tsm.create_time_series_model_features(ohlcv_df)
        assert len(result) == len(ohlcv_df)

    def test_preserves_original_columns(self, tsm, ohlcv_df):
        orig_cols = set(ohlcv_df.columns)
        result = tsm.create_time_series_model_features(ohlcv_df)
        assert orig_cols.issubset(set(result.columns))


# ------------------------------------------------------------------ #
#  ARIMA Tests                                                        #
# ------------------------------------------------------------------ #

class TestARIMA:
    def test_residual_reasonable_magnitude(self, tsm, ohlcv_df):
        result = tsm.create_time_series_model_features(ohlcv_df)
        vals = result["tsm_arima_residual"].values
        # Residuals should be small (daily return scale)
        assert np.max(np.abs(vals)) < 1.0, "ARIMA residuals too large"

    def test_residual_vol_non_negative(self, tsm, ohlcv_df):
        result = tsm.create_time_series_model_features(ohlcv_df)
        vals = result["tsm_arima_residual_vol"].values
        assert np.all(vals >= 0.0)

    def test_trend_clipped(self, tsm, ohlcv_df):
        result = tsm.create_time_series_model_features(ohlcv_df)
        vals = result["tsm_arima_trend"].values
        assert np.all(vals >= -5.0) and np.all(vals <= 5.0)


# ------------------------------------------------------------------ #
#  Chronos Tests                                                      #
# ------------------------------------------------------------------ #

class TestChronos:
    def test_chronos_zeroed_when_unavailable(self, tsm, ohlcv_df):
        """When Chronos is not installed, features should be 0.0."""
        result = tsm.create_time_series_model_features(ohlcv_df)
        for col in ["tsm_chronos_residual_1d", "tsm_chronos_residual_5d",
                     "tsm_chronos_interval_width", "tsm_chronos_surprise"]:
            if col in result.columns:
                assert (result[col] == 0.0).all(), f"{col} should be 0.0 when Chronos unavailable"

    def test_chronos_pctile_default(self, tsm, ohlcv_df):
        result = tsm.create_time_series_model_features(ohlcv_df)
        if "tsm_chronos_interval_pctile" in result.columns:
            assert (result["tsm_chronos_interval_pctile"] == 0.5).all()


# ------------------------------------------------------------------ #
#  Cross-Model Tests                                                  #
# ------------------------------------------------------------------ #

class TestCrossModel:
    def test_disagreement_non_negative(self, tsm, ohlcv_df):
        result = tsm.create_time_series_model_features(ohlcv_df)
        vals = result["tsm_model_disagreement"].values
        assert np.all(vals >= 0.0)

    def test_agreement_single_model_neutral(self, tsm, ohlcv_df):
        """With only ARIMA (no Chronos), agreement should be 0.5 (neutral)."""
        result = tsm.create_time_series_model_features(ohlcv_df)
        vals = result["tsm_directional_agreement"].values
        assert np.all(vals == 0.5)


# ------------------------------------------------------------------ #
#  catch22 Tests                                                      #
# ------------------------------------------------------------------ #

class TestCatch22:
    def test_catch22_zeroed_when_unavailable(self, ohlcv_df):
        """When catch22 is not installed, c22 features should be 0.0."""
        tsm = TimeSeriesModelFeatures(use_chronos=False, use_catch22=True)
        result = tsm.create_time_series_model_features(ohlcv_df)
        c22_cols = [c for c in result.columns if c.startswith("tsm_c22_")]
        # If pycatch22 is not installed, all should be 0.0
        if not tsm._catch22_available:
            for col in c22_cols:
                assert (result[col] == 0.0).all(), f"{col} should be 0.0"

    def test_catch22_columns_exist(self, ohlcv_df):
        tsm = TimeSeriesModelFeatures(use_chronos=False, use_catch22=True)
        result = tsm.create_time_series_model_features(ohlcv_df)
        c22_cols = [c for c in result.columns if c.startswith("tsm_c22_")]
        assert len(c22_cols) == 5


# ------------------------------------------------------------------ #
#  Graceful Degradation Tests                                         #
# ------------------------------------------------------------------ #

class TestGraceful:
    def test_short_data_no_crash(self, tsm, short_df):
        """Should handle < 30 rows gracefully."""
        result = tsm.create_time_series_model_features(short_df)
        assert len(result) == len(short_df)

    def test_missing_close_no_crash(self, tsm):
        df = pd.DataFrame({"volume": [1e8] * 50})
        result = tsm.create_time_series_model_features(df)
        assert len(result) == 50

    def test_close_only_works(self, tsm, close_only_df):
        result = tsm.create_time_series_model_features(close_only_df)
        tsm_cols = [c for c in result.columns if c.startswith("tsm_")]
        assert len(tsm_cols) >= 10


# ------------------------------------------------------------------ #
#  Analyze Tests                                                      #
# ------------------------------------------------------------------ #

class TestAnalyze:
    def test_analyze_returns_dict(self, tsm, ohlcv_df):
        result = tsm.create_time_series_model_features(ohlcv_df)
        analysis = tsm.analyze_current_ts(result)
        assert isinstance(analysis, dict)

    def test_analyze_has_required_keys(self, tsm, ohlcv_df):
        result = tsm.create_time_series_model_features(ohlcv_df)
        analysis = tsm.analyze_current_ts(result)
        for key in ["arima_available", "chronos_available", "arima_residual"]:
            assert key in analysis, f"Missing key: {key}"

    def test_analyze_none_on_empty(self, tsm):
        df = pd.DataFrame({"close": []})
        analysis = tsm.analyze_current_ts(df)
        assert analysis is None


# ------------------------------------------------------------------ #
#  Feature Group Integration                                          #
# ------------------------------------------------------------------ #

class TestFeatureGroup:
    def test_time_series_model_group_exists(self):
        assert "time_series_model" in FEATURE_GROUPS

    def test_time_series_model_group_prefix(self):
        assert FEATURE_GROUPS["time_series_model"] == ["tsm_"]

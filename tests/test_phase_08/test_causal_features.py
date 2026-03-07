"""Tests for CausalFeatureSelector."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.causal_features import CausalFeatureSelector


def _make_daily_data(n=300, seed=42):
    """Generate synthetic daily data with known causal feature."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    daily_return = np.diff(close, prepend=close[0]) / close
    daily_return[0] = 0

    # Add some columns that might be causal
    volume = rng.randint(1_000_000, 10_000_000, n).astype(float)
    rsi = 50 + rng.randn(n) * 15  # Random RSI-like column

    df = pd.DataFrame({
        "date": dates,
        "close": close,
        "daily_return": daily_return,
        "volume": volume,
        "rsi_14": rsi,
        "macd": rng.randn(n) * 0.5,
        "bb_width": rng.uniform(0.01, 0.05, n),
    })
    return df


class TestCausalFeatureSelectorInit:
    def test_default_construction(self):
        sel = CausalFeatureSelector()
        assert sel.max_lag == 5
        assert sel.significance_level == 0.05

    def test_custom_params(self):
        sel = CausalFeatureSelector(max_lag=3, significance_level=0.10)
        assert sel.max_lag == 3
        assert sel.significance_level == 0.10

    def test_feature_names_defined(self):
        assert len(CausalFeatureSelector.FEATURE_NAMES) == 6
        for name in CausalFeatureSelector.FEATURE_NAMES:
            assert name.startswith("causal_")


class TestCausalFeatureCreation:
    def test_creates_all_features(self):
        df = _make_daily_data()
        sel = CausalFeatureSelector(rolling_window=100)
        result = sel.create_causal_features(df)
        for col in CausalFeatureSelector.FEATURE_NAMES:
            assert col in result.columns, f"Missing: {col}"

    def test_no_nans_in_output(self):
        df = _make_daily_data()
        sel = CausalFeatureSelector(rolling_window=100)
        result = sel.create_causal_features(df)
        for col in CausalFeatureSelector.FEATURE_NAMES:
            assert not result[col].isna().any(), f"NaN in {col}"

    def test_no_infs_in_output(self):
        df = _make_daily_data()
        sel = CausalFeatureSelector(rolling_window=100)
        result = sel.create_causal_features(df)
        for col in CausalFeatureSelector.FEATURE_NAMES:
            assert not np.isinf(result[col]).any(), f"Inf in {col}"

    def test_n_causes_is_nonnegative(self):
        df = _make_daily_data()
        sel = CausalFeatureSelector(rolling_window=100)
        result = sel.create_causal_features(df)
        assert (result["causal_n_causes"] >= 0).all()


class TestCausalEdgeCases:
    def test_missing_columns_zero_fills(self):
        df = pd.DataFrame({"date": pd.bdate_range("2023-01-01", periods=50)})
        sel = CausalFeatureSelector()
        result = sel.create_causal_features(df)
        for col in CausalFeatureSelector.FEATURE_NAMES:
            assert col in result.columns
            assert (result[col] == 0.0).all()

    def test_short_data_zero_fills(self):
        df = _make_daily_data(n=5)
        sel = CausalFeatureSelector(rolling_window=100)
        result = sel.create_causal_features(df)
        for col in CausalFeatureSelector.FEATURE_NAMES:
            assert col in result.columns

    def test_preserves_original_columns(self):
        df = _make_daily_data()
        original_cols = set(df.columns)
        sel = CausalFeatureSelector(rolling_window=100)
        result = sel.create_causal_features(df)
        for col in original_cols:
            assert col in result.columns

    def test_large_max_lag(self):
        """max_lag larger than available data should not crash."""
        df = _make_daily_data(n=50)
        sel = CausalFeatureSelector(max_lag=20, rolling_window=30)
        result = sel.create_causal_features(df)
        assert "causal_n_causes" in result.columns


class TestCausalWithKnownSignal:
    def test_detects_lagged_signal(self):
        """If feature X at t-1 perfectly predicts returns at t, causal strength > 0."""
        rng = np.random.RandomState(123)
        n = 500
        dates = pd.bdate_range("2022-01-01", periods=n)
        signal = rng.randn(n)
        # returns = signal shifted by 1 lag + noise
        daily_return = np.zeros(n)
        daily_return[1:] = signal[:-1] * 0.8 + rng.randn(n - 1) * 0.2
        close = 100 + np.cumsum(daily_return)

        df = pd.DataFrame({
            "date": dates,
            "close": close,
            "daily_return": daily_return,
            "causal_signal": signal,
            "noise": rng.randn(n),
        })
        sel = CausalFeatureSelector(
            max_lag=3, rolling_window=200, n_features_to_test=5,
        )
        result = sel.create_causal_features(df)
        # The last rows should show at least some causal detection
        last_100 = result.iloc[-100:]
        assert last_100["causal_max_strength"].max() > 0

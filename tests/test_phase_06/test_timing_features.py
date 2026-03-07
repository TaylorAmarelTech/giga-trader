"""Tests for TimingFeatureEngineer (Phase 6 Intraday Features)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_06_features_intraday.timing_features import TimingFeatureEngineer


def _make_daily(n=100, seed=42):
    """Generate synthetic daily OHLCV data."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2025-01-01", periods=n)
    close = 450.0 + np.cumsum(rng.randn(n) * 0.5)
    high = close + rng.uniform(0.2, 1.5, n)
    low = close - rng.uniform(0.2, 1.5, n)
    opn = close + rng.randn(n) * 0.3
    vol = rng.randint(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame({
        "date": dates,
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    })


class TestTimingFeatureInit:
    def test_construction(self):
        eng = TimingFeatureEngineer()
        assert eng.scaler is not None
        assert eng.feature_names == []


class TestTimingFeatureCreation:
    def test_creates_features(self):
        df = _make_daily()
        eng = TimingFeatureEngineer()
        features = eng.create_features(df)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(df)

    def test_expected_columns(self):
        df = _make_daily()
        eng = TimingFeatureEngineer()
        features = eng.create_features(df)
        expected = [
            "prev_return", "prev_range", "prev_body",
            "volatility_5d", "volatility_20d", "volatility_ratio",
            "momentum_3d", "momentum_5d", "momentum_10d",
            "close_vs_ma5", "close_vs_ma20",
            "volume_ratio", "gap",
        ]
        for col in expected:
            assert col in features.columns, f"Missing: {col}"

    def test_no_nans(self):
        df = _make_daily()
        eng = TimingFeatureEngineer()
        features = eng.create_features(df)
        assert not features.isna().any().any()

    def test_no_infs(self):
        df = _make_daily()
        eng = TimingFeatureEngineer()
        features = eng.create_features(df)
        for col in features.columns:
            assert not np.isinf(features[col]).any(), f"Inf in {col}"

    def test_calendar_features(self):
        df = _make_daily()
        eng = TimingFeatureEngineer()
        features = eng.create_features(df)
        assert "is_monday" in features.columns
        assert "is_friday" in features.columns
        assert set(features["is_monday"].unique()).issubset({0, 1})

    def test_feature_names_populated(self):
        df = _make_daily()
        eng = TimingFeatureEngineer()
        eng.create_features(df)
        assert len(eng.feature_names) > 0

    def test_short_data(self):
        df = _make_daily(n=5)
        eng = TimingFeatureEngineer()
        features = eng.create_features(df)
        assert len(features) == 5
        assert not features.isna().any().any()


class TestTimingFeatureTransform:
    def test_fit_transform(self):
        df = _make_daily()
        eng = TimingFeatureEngineer()
        features = eng.create_features(df)
        X = eng.fit_transform(features)
        assert X.shape == features.shape
        # Should be roughly standardized (mean near 0, std near 1)
        assert abs(X.mean()) < 1.0

    def test_transform_after_fit(self):
        df = _make_daily()
        eng = TimingFeatureEngineer()
        features = eng.create_features(df)
        eng.fit_transform(features)
        X = eng.transform(features)
        assert X.shape == features.shape

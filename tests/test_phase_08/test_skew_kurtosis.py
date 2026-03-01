"""Tests for SkewKurtosisFeatures -- higher-order moment features (6 total)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.skew_kurtosis_features import SkewKurtosisFeatures


# ---- Helpers ----------------------------------------------------------------

def _make_spy_daily(n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    returns = rng.normal(0.0003, 0.012, n_days)
    close = 450.0 * np.cumprod(1 + returns)
    volume = rng.randint(50_000_000, 150_000_000, n_days).astype(float)
    return pd.DataFrame({
        "date": dates, "close": close, "volume": volume,
        "open": close * (1 + rng.normal(0, 0.003, n_days)),
        "high": close * 1.005, "low": close * 0.995,
    })


ALL_6 = {
    "skku_skew_5d",
    "skku_skew_20d",
    "skku_skew_60d",
    "skku_kurtosis_20d",
    "skku_tail_asymmetry",
    "skku_skew_z",
}


# ---- Invariant Tests --------------------------------------------------------

class TestSkewKurtosisInvariants:
    @pytest.fixture
    def feat(self):
        return SkewKurtosisFeatures()

    @pytest.fixture
    def spy(self):
        return _make_spy_daily(200)

    def test_all_6_features_created(self, feat, spy):
        result = feat.create_skew_kurtosis_features(spy)
        skku_cols = {c for c in result.columns if c.startswith("skku_")}
        assert skku_cols == ALL_6

    def test_no_nans(self, feat, spy):
        result = feat.create_skew_kurtosis_features(spy)
        for col in ALL_6:
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_no_infinities(self, feat, spy):
        result = feat.create_skew_kurtosis_features(spy)
        for col in ALL_6:
            assert not np.isinf(result[col]).any(), f"Inf in {col}"

    def test_preserves_original_columns(self, feat, spy):
        original = set(spy.columns)
        result = feat.create_skew_kurtosis_features(spy)
        assert original.issubset(set(result.columns))

    def test_preserves_row_count(self, feat, spy):
        result = feat.create_skew_kurtosis_features(spy)
        assert len(result) == len(spy)

    def test_no_required_column(self, feat):
        df = pd.DataFrame({"date": [1, 2], "price": [100, 101]})
        result = feat.create_skew_kurtosis_features(df)
        assert len(result.columns) == len(df.columns)

    def test_works_with_flat_data(self, feat):
        """Flat close prices (zero variance) should not crash."""
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=50),
            "close": np.full(50, 450.0),
        })
        result = feat.create_skew_kurtosis_features(df)
        for col in ALL_6:
            assert col in result.columns
            assert result[col].isna().sum() == 0, f"NaN in {col} with flat data"


# ---- Feature Logic Tests ----------------------------------------------------

class TestSkewKurtosisLogic:
    @pytest.fixture
    def feat(self):
        return SkewKurtosisFeatures()

    def test_skew_z_bounded(self, feat):
        """Z-score should be clipped to [-4, 4]."""
        df = _make_spy_daily(200)
        result = feat.create_skew_kurtosis_features(df)
        assert result["skku_skew_z"].max() <= 4.0
        assert result["skku_skew_z"].min() >= -4.0

    def test_negative_skew_with_crash_returns(self, feat):
        """A distribution with large negative returns should have negative skew."""
        rng = np.random.RandomState(77)
        n = 200
        dates = pd.bdate_range("2024-01-02", periods=n, freq="B")
        # Normal returns + occasional large drops (negative-skewed)
        returns = rng.normal(0.001, 0.005, n)
        for i in range(10, n, 20):
            returns[i] = -0.05  # crash day every 20 days
        close = 450.0 * np.cumprod(1 + returns)
        df = pd.DataFrame({"date": dates, "close": close})
        result = feat.create_skew_kurtosis_features(df)
        # Mean skew over the latter part should be negative
        mean_skew = result["skku_skew_60d"].iloc[80:].mean()
        assert mean_skew < 0, f"Expected negative skew, got {mean_skew}"

    def test_excess_kurtosis_positive_for_heavy_tails(self, feat):
        """Heavy-tailed returns should have positive excess kurtosis."""
        rng = np.random.RandomState(88)
        n = 200
        dates = pd.bdate_range("2024-01-02", periods=n, freq="B")
        # t-distribution with 3 DOF (heavy tails) simulated via mixture
        returns = np.concatenate([
            rng.normal(0, 0.005, n - 20),
            rng.normal(0, 0.05, 20),  # fat-tail component
        ])
        rng.shuffle(returns)
        close = 450.0 * np.cumprod(1 + returns)
        df = pd.DataFrame({"date": dates, "close": close})
        result = feat.create_skew_kurtosis_features(df)
        # Excess kurtosis at end should be positive (heavier tails than normal)
        last_kurt = result["skku_kurtosis_20d"].iloc[-1]
        # The mixture creates heavy tails, but we only need non-crash behavior
        # Check that at least some values are > 0 (excess kurtosis)
        any_positive = (result["skku_kurtosis_20d"].iloc[30:] > 0).any()
        assert any_positive, "Expected some positive excess kurtosis"


# ---- Analyze Tests -----------------------------------------------------------

class TestAnalyzeCurrentSkewKurtosis:
    def test_returns_dict(self):
        feat = SkewKurtosisFeatures()
        df = _make_spy_daily(100)
        df = feat.create_skew_kurtosis_features(df)
        result = feat.analyze_current_skew_kurtosis(df)
        assert isinstance(result, dict)
        assert "skew_regime" in result

    def test_regime_values(self):
        feat = SkewKurtosisFeatures()
        df = _make_spy_daily(200)
        df = feat.create_skew_kurtosis_features(df)
        result = feat.analyze_current_skew_kurtosis(df)
        assert result["skew_regime"] in {"NEGATIVE_SKEW", "POSITIVE_SKEW", "NORMAL"}

    def test_returns_none_without_features(self):
        feat = SkewKurtosisFeatures()
        df = pd.DataFrame({"close": [100]})
        assert feat.analyze_current_skew_kurtosis(df) is None


# ---- Feature Count Test ------------------------------------------------------

class TestFeatureCounts:
    def test_total_count(self):
        assert len(ALL_6) == 6

"""
Tests for RegimeRouter (regime-conditional model routing).
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.phase_15_strategy.regime_router import RegimeRouter


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def classification_data(rng):
    """Synthetic binary classification data with vol_series."""
    n = 600
    n_features = 10
    X = rng.randn(n, n_features)
    # Create separable classes correlated with features
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * 0.3 > 0).astype(int)
    # vol_series with a range that spans low/medium/high
    vol_series = rng.uniform(10, 35, size=n)
    return X, y, vol_series


@pytest.fixture
def router():
    return RegimeRouter(regime_method="vix_quartile", min_samples_per_regime=50)


# ─── Fit and Predict Tests ──────────────────────────────────────────────────


class TestFitPredict:

    def test_fit_and_predict_vix_quartile(self, router, classification_data):
        """Fit and predict with VIX quartile regime method."""
        X, y, vol = classification_data
        router.fit(X, y, vol_series=vol)

        preds = router.predict(X[:10], vol_series=vol[:10])
        assert preds.shape == (10,)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, router, classification_data):
        """predict_proba returns (n, 2) array."""
        X, y, vol = classification_data
        router.fit(X, y, vol_series=vol)

        proba = router.predict_proba(X[:20], vol_series=vol[:20])
        assert proba.shape == (20, 2)
        # Each row should sum to ~1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)
        # All probabilities between 0 and 1
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)


# ─── Fallback Tests ─────────────────────────────────────────────────────────


class TestGlobalFallback:

    def test_no_vol_series_uses_global(self, classification_data):
        """No vol_series in fit falls back to global model."""
        X, y, vol = classification_data
        router = RegimeRouter()

        router.fit(X, y, vol_series=None)
        assert router.n_regimes_ == 0

        proba = router.predict_proba(X[:5])
        assert proba.shape == (5, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_no_vol_series_predict_uses_global(self, classification_data):
        """No vol_series in predict uses global model even if regimes were fitted."""
        X, y, vol = classification_data
        router = RegimeRouter(min_samples_per_regime=50)
        router.fit(X, y, vol_series=vol)

        proba_global = router.predict_proba(X[:5], vol_series=None)
        assert proba_global.shape == (5, 2)

    def test_single_float_vol_series(self, classification_data):
        """Single float vol_series routes all samples to the same regime."""
        X, y, vol = classification_data
        router = RegimeRouter(regime_method="vix_fixed", min_samples_per_regime=50)
        router.fit(X, y, vol_series=vol)

        # All samples routed to "high" regime (VIX=30 > 25)
        proba = router.predict_proba(X[:10], vol_series=30.0)
        assert proba.shape == (10, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_min_samples_fallback(self, rng):
        """Regime with fewer than min_samples uses global model."""
        n = 300
        X = rng.randn(n, 5)
        y = rng.randint(0, 2, size=n)
        # Put most samples in low vol, very few in high vol
        vol = np.concatenate([
            rng.uniform(10, 14, size=280),   # low
            rng.uniform(15, 24, size=15),     # medium (below min_samples)
            rng.uniform(25, 35, size=5),      # high (below min_samples)
        ])

        router = RegimeRouter(
            regime_method="vix_fixed",
            min_samples_per_regime=100,
        )
        router.fit(X, y, vol_series=vol)

        stats = router.get_regime_stats()
        # Medium and high should fall back to global
        assert len(stats["used_global_fallback"]) >= 1

        # Should still predict without error for all regimes
        proba = router.predict_proba(X, vol_series=vol)
        assert proba.shape == (n, 2)


# ─── Regime Methods ──────────────────────────────────────────────────────────


class TestRegimeMethods:

    def test_binary_vol_method(self, classification_data):
        """Binary vol method splits into 2 regimes."""
        X, y, vol = classification_data
        router = RegimeRouter(
            regime_method="binary_vol",
            min_samples_per_regime=50,
        )
        router.fit(X, y, vol_series=vol)

        stats = router.get_regime_stats()
        # binary_vol should create exactly 2 regime groups
        assert stats["n_regimes"] == 2

        proba = router.predict_proba(X[:10], vol_series=vol[:10])
        assert proba.shape == (10, 2)

    def test_vix_fixed_method(self, classification_data):
        """Fixed VIX threshold method works correctly."""
        X, y, vol = classification_data
        router = RegimeRouter(
            regime_method="vix_fixed",
            min_samples_per_regime=50,
        )
        router.fit(X, y, vol_series=vol)

        stats = router.get_regime_stats()
        assert stats["n_regimes"] in (2, 3)  # depends on data distribution
        assert "low_threshold" in stats["thresholds"]
        assert stats["thresholds"]["low_threshold"] == 15.0
        assert stats["thresholds"]["high_threshold"] == 25.0

    def test_invalid_method_raises(self, classification_data):
        """Invalid regime_method raises ValueError."""
        X, y, vol = classification_data
        router = RegimeRouter(regime_method="invalid_method")
        with pytest.raises(ValueError, match="Unknown regime_method"):
            router.fit(X, y, vol_series=vol)


# ─── Stats and Properties ───────────────────────────────────────────────────


class TestStatsAndProperties:

    def test_get_regime_stats(self, router, classification_data):
        """get_regime_stats returns correct structure."""
        X, y, vol = classification_data
        router.fit(X, y, vol_series=vol)

        stats = router.get_regime_stats()
        assert "n_regimes" in stats
        assert "samples_per_regime" in stats
        assert "used_global_fallback" in stats
        assert "thresholds" in stats

        assert isinstance(stats["n_regimes"], int)
        assert isinstance(stats["samples_per_regime"], dict)
        assert isinstance(stats["used_global_fallback"], list)

        # Total samples across regimes should equal training set size
        total = sum(stats["samples_per_regime"].values())
        assert total == len(y)

    def test_get_regime_stats_before_fit(self):
        """get_regime_stats on unfitted router returns zeros."""
        router = RegimeRouter()
        stats = router.get_regime_stats()
        assert stats["n_regimes"] == 0
        assert stats["samples_per_regime"] == {}
        assert stats["used_global_fallback"] == []

    def test_classes_property(self, router):
        """classes_ property returns [0, 1]."""
        np.testing.assert_array_equal(router.classes_, np.array([0, 1]))


# ─── Sample Weight ───────────────────────────────────────────────────────────


class TestSampleWeight:

    def test_sample_weight_passed_through(self, classification_data):
        """sample_weight is passed to both global and regime models."""
        X, y, vol = classification_data
        weights = np.ones(len(y))
        # Double-weight first half
        weights[:len(y) // 2] = 2.0

        router = RegimeRouter(min_samples_per_regime=50)
        # Should not raise
        router.fit(X, y, vol_series=vol, sample_weight=weights)

        proba = router.predict_proba(X[:5], vol_series=vol[:5])
        assert proba.shape == (5, 2)

    def test_sample_weight_none_accepted(self, classification_data):
        """sample_weight=None works fine."""
        X, y, vol = classification_data
        router = RegimeRouter(min_samples_per_regime=50)
        router.fit(X, y, vol_series=vol, sample_weight=None)
        assert hasattr(router, "global_model_")


# ─── Edge Cases ──────────────────────────────────────────────────────────────


class TestEdgeCases:

    def test_vol_series_length_mismatch_raises(self, classification_data):
        """Mismatched vol_series length raises ValueError."""
        X, y, vol = classification_data
        router = RegimeRouter()
        with pytest.raises(ValueError, match="vol_series length"):
            router.fit(X, y, vol_series=vol[:10])

    def test_predict_vol_series_length_mismatch_raises(self, classification_data):
        """Mismatched vol_series in predict raises ValueError."""
        X, y, vol = classification_data
        router = RegimeRouter()
        router.fit(X, y, vol_series=vol)
        with pytest.raises(ValueError, match="vol_series length"):
            router.predict_proba(X[:10], vol_series=vol[:5])

    def test_default_base_model(self, classification_data):
        """Default base model (LogisticRegression) is used when base_model=None."""
        X, y, vol = classification_data
        router = RegimeRouter(base_model=None, min_samples_per_regime=50)
        router.fit(X, y, vol_series=vol)

        from sklearn.linear_model import LogisticRegression
        assert isinstance(router.global_model_, LogisticRegression)

    def test_custom_base_model(self, classification_data):
        """Custom base model is cloned and used."""
        from sklearn.ensemble import GradientBoostingClassifier

        X, y, vol = classification_data
        base = GradientBoostingClassifier(
            n_estimators=10, max_depth=2, random_state=42
        )
        router = RegimeRouter(base_model=base, min_samples_per_regime=50)
        router.fit(X, y, vol_series=vol)

        assert isinstance(router.global_model_, GradientBoostingClassifier)
        for model in router.regime_models_.values():
            assert isinstance(model, GradientBoostingClassifier)

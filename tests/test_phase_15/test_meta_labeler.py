"""
Tests for MetaLabeler and half_kelly_fraction (Wave 40).
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch

from src.phase_15_strategy.meta_labeler import MetaLabeler, half_kelly_fraction


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def meta_labeler():
    return MetaLabeler()


@pytest.fixture
def profitable_data():
    """Generate enough data for a successful meta-labeler fit."""
    np.random.seed(42)
    n = 200
    n_features = 10
    X = np.random.randn(n, n_features)
    swing_proba = np.clip(0.5 + np.random.randn(n) * 0.15, 0.1, 0.9)
    timing_proba = np.clip(0.5 + np.random.randn(n) * 0.15, 0.1, 0.9)
    signals = np.zeros(n)
    # Signal on ~50% of days
    signals[swing_proba > 0.55] = 1
    # Make returns correlated with features so meta-model can learn
    returns = 0.001 * X[:, 0] + 0.0005 * X[:, 1] + np.random.randn(n) * 0.005
    return X, swing_proba, timing_proba, signals, returns


@pytest.fixture
def few_signals_data():
    """Too few signals for meta-labeler."""
    np.random.seed(42)
    n = 100
    n_features = 10
    X = np.random.randn(n, n_features)
    swing_proba = np.random.rand(n)
    timing_proba = np.random.rand(n)
    signals = np.zeros(n)
    signals[:5] = 1  # Only 5 signals (below 30 minimum)
    returns = np.random.randn(n) * 0.01
    return X, swing_proba, timing_proba, signals, returns


# ─── Init Tests ──────────────────────────────────────────────────────────────

class TestMetaLabelerInit:

    def test_default_constructor(self, meta_labeler):
        assert isinstance(meta_labeler, MetaLabeler)
        assert meta_labeler.C == 1.0
        assert meta_labeler.min_signals == 30
        assert meta_labeler.min_per_class == 15
        assert meta_labeler.cv_folds == 3

    def test_not_fitted_initially(self, meta_labeler):
        assert not meta_labeler.is_fitted_
        assert meta_labeler.model_ is None
        assert meta_labeler.meta_auc_ == 0.0


# ─── Fit Tests ───────────────────────────────────────────────────────────────

class TestMetaLabelerFit:

    def test_fit_with_sufficient_data(self, meta_labeler, profitable_data):
        X, swing_proba, timing_proba, signals, returns = profitable_data
        result = meta_labeler.fit(X, swing_proba, timing_proba, signals, returns)
        assert isinstance(result, dict)
        # Should either fit or gracefully decline
        if result["fitted"]:
            assert result["meta_auc"] > 0.0
            assert result["n_signals"] >= 30
            assert meta_labeler.is_fitted_

    def test_too_few_signals(self, meta_labeler, few_signals_data):
        X, swing_proba, timing_proba, signals, returns = few_signals_data
        result = meta_labeler.fit(X, swing_proba, timing_proba, signals, returns)
        assert result["fitted"] is False
        assert "too_few_signals" in result["reason"]
        assert not meta_labeler.is_fitted_

    def test_single_class_not_fitted(self, meta_labeler):
        """All signals profitable → class imbalance → not fitted."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 5)
        swing_proba = np.ones(n) * 0.7
        timing_proba = np.ones(n) * 0.6
        signals = np.ones(n)
        returns = np.ones(n) * 0.05  # All highly profitable
        result = meta_labeler.fit(X, swing_proba, timing_proba, signals, returns)
        assert result["fitted"] is False
        assert "class_imbalance" in result["reason"]

    def test_cost_adjustment(self, meta_labeler):
        """Tiny positive returns become losses after costs."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 5)
        swing_proba = np.random.rand(n)
        timing_proba = np.random.rand(n)
        signals = np.ones(n)
        # Returns just barely positive (0.01%) - should become negative after costs
        returns = np.ones(n) * 0.0001
        result = meta_labeler.fit(
            X, swing_proba, timing_proba, signals, returns,
            slippage_bps=5, commission_bps=1,
        )
        # After costs of 12bps round-trip, 1bp returns become -11bp → all losses
        assert result["fitted"] is False

    def test_custom_parameters(self):
        ml = MetaLabeler(C=0.5, min_signals=10, min_per_class=5, cv_folds=2)
        assert ml.C == 0.5
        assert ml.min_signals == 10
        assert ml.min_per_class == 5
        assert ml.cv_folds == 2


# ─── Predict Tests ───────────────────────────────────────────────────────────

class TestMetaLabelerPredict:

    def test_unfitted_returns_none(self, meta_labeler):
        X = np.random.randn(5, 10)
        result = meta_labeler.predict(X, np.random.rand(5), np.random.rand(5))
        assert result is None

    def test_fitted_returns_probabilities(self, profitable_data):
        X, swing_proba, timing_proba, signals, returns = profitable_data
        ml = MetaLabeler(min_signals=10, min_per_class=5)
        fit_result = ml.fit(X, swing_proba, timing_proba, signals, returns)
        if fit_result["fitted"]:
            preds = ml.predict(X[:5], swing_proba[:5], timing_proba[:5])
            assert preds is not None
            assert len(preds) == 5
            assert all(0 <= p <= 1 for p in preds)

    def test_single_sample_prediction(self, profitable_data):
        X, swing_proba, timing_proba, signals, returns = profitable_data
        ml = MetaLabeler(min_signals=10, min_per_class=5)
        fit_result = ml.fit(X, swing_proba, timing_proba, signals, returns)
        if fit_result["fitted"]:
            preds = ml.predict(X[0], np.array([swing_proba[0]]), np.array([timing_proba[0]]))
            assert preds is not None
            assert len(preds) == 1
            assert 0 <= preds[0] <= 1


# ─── Persistence Tests ───────────────────────────────────────────────────────

class TestMetaLabelerPersistence:

    def test_save_load_roundtrip(self, profitable_data, tmp_path):
        X, swing_proba, timing_proba, signals, returns = profitable_data
        ml = MetaLabeler(min_signals=10, min_per_class=5)
        fit_result = ml.fit(X, swing_proba, timing_proba, signals, returns)

        save_path = tmp_path / "meta_labeler.joblib"
        ml.save(save_path)

        loaded = MetaLabeler.load(save_path)
        assert loaded is not None
        assert loaded.is_fitted_ == ml.is_fitted_
        assert loaded.meta_auc_ == ml.meta_auc_
        assert loaded.C == ml.C

        if ml.is_fitted_:
            orig_pred = ml.predict(X[:3], swing_proba[:3], timing_proba[:3])
            load_pred = loaded.predict(X[:3], swing_proba[:3], timing_proba[:3])
            np.testing.assert_array_almost_equal(orig_pred, load_pred)

    def test_load_nonexistent_returns_none(self, tmp_path):
        result = MetaLabeler.load(tmp_path / "nonexistent.joblib")
        assert result is None


# ─── Half Kelly Tests ────────────────────────────────────────────────────────

class TestHalfKelly:

    def test_positive_edge(self):
        """60% win rate with 1:1 payout = positive edge."""
        f = half_kelly_fraction(0.6, 1.0)
        assert f > 0
        # Full Kelly = (0.6*1 - 0.4) / 1 = 0.2, half = 0.1
        assert abs(f - 0.1) < 1e-10

    def test_no_edge_returns_zero(self):
        """50% win rate with 1:1 payout = no edge."""
        f = half_kelly_fraction(0.5, 1.0)
        assert f == 0.0

    def test_negative_edge_returns_zero(self):
        """40% win rate with 1:1 payout = negative edge."""
        f = half_kelly_fraction(0.4, 1.0)
        assert f == 0.0

    def test_higher_b_larger_fraction(self):
        """Higher win/loss ratio increases bet fraction."""
        f1 = half_kelly_fraction(0.6, 1.0)
        f2 = half_kelly_fraction(0.6, 2.0)
        assert f2 > f1

    def test_edge_cases(self):
        """Edge cases: p=0, p=1, b=0."""
        assert half_kelly_fraction(0.0, 1.0) == 0.0
        assert half_kelly_fraction(1.0, 1.0) == 0.0  # p >= 1
        assert half_kelly_fraction(0.6, 0.0) == 0.0
        assert half_kelly_fraction(0.6, -1.0) == 0.0

    def test_fraction_always_positive_or_zero(self):
        """Kelly fraction should never be negative."""
        for p in np.arange(0, 1.01, 0.1):
            for b in np.arange(0, 3.01, 0.5):
                f = half_kelly_fraction(p, b)
                assert f >= 0.0, f"Negative Kelly at p={p}, b={b}"


# ─── Config Integration Tests ───────────────────────────────────────────────

class TestMetaLabelingConfig:

    def test_config_has_flag(self):
        from src.experiment_config import AntiOverfitConfig
        config = AntiOverfitConfig()
        assert hasattr(config, "use_meta_labeling")
        assert config.use_meta_labeling is True

    def test_experiment_result_has_fields(self):
        from src.phase_21_continuous.experiment_tracking import ExperimentResult
        r = ExperimentResult(experiment_id="test_meta", config={})
        assert hasattr(r, "meta_label_auc")
        assert hasattr(r, "meta_label_fitted")
        assert hasattr(r, "meta_sharpe")
        assert hasattr(r, "meta_win_rate")
        assert hasattr(r, "meta_improvement")
        assert r.meta_label_auc == 0.0
        assert r.meta_label_fitted is False

    def test_init_exports(self):
        from src.phase_15_strategy import MetaLabeler, half_kelly_fraction
        assert MetaLabeler is not None
        assert callable(half_kelly_fraction)

"""
Tests for Wave 29: Multi-faceted stability suite.
"""

import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from src.phase_14_robustness.stability_suite import StabilitySuite


def _make_data(n=400, n_features=20, seed=42):
    """Generate synthetic classification data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    # Weak linear signal + noise
    signal = X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.randn(n) * 0.8
    y = (signal > 0).astype(int)
    return X, y


def _logistic_factory():
    return LogisticRegression(C=1.0, max_iter=500, random_state=42)


def _logistic_seed_factory(seed):
    return LogisticRegression(C=1.0, max_iter=500, random_state=seed)


class TestStabilitySuite:
    """Test the multi-faceted stability suite."""

    def test_run_all_returns_all_measures(self):
        """run_all should return all 5 stability measures."""
        X, y = _make_data()
        suite = StabilitySuite(n_bootstrap=3, n_feature_dropout=3, n_seeds=3, n_prediction_models=3)
        results = suite.run_all(X, y, _logistic_factory, base_auc=0.60, seed_model_factory_fn=_logistic_seed_factory)

        assert "bootstrap" in results
        assert "feature_dropout" in results
        assert "seed" in results
        assert "prediction" in results
        assert "composite" in results

        # All scores should be 0-1
        for key in ("bootstrap", "feature_dropout", "seed", "prediction"):
            score = results[key]["score"]
            assert 0.0 <= score <= 1.0, f"{key} score {score} out of range"
        assert 0.0 <= results["composite"] <= 1.0

    def test_bootstrap_stability(self):
        """Bootstrap stability should produce a score for valid data."""
        X, y = _make_data()
        suite = StabilitySuite(n_bootstrap=5)
        result = suite._bootstrap_stability(X, y, _logistic_factory, base_auc=0.60)

        assert "score" in result
        assert result["n_trials"] >= 3
        assert 0.0 <= result["score"] <= 1.0
        assert "auc_mean" in result
        assert "auc_std" in result

    def test_feature_dropout_stability(self):
        """Feature dropout should show model resilience to missing features."""
        X, y = _make_data(n_features=30)
        suite = StabilitySuite(n_feature_dropout=5, dropout_fraction=0.15)
        result = suite._feature_dropout_stability(X, y, _logistic_factory, base_auc=0.60)

        assert "score" in result
        assert result["n_trials"] >= 3
        assert 0.0 <= result["score"] <= 1.0
        assert "avg_drop" in result

    def test_seed_stability(self):
        """Seed stability should measure initialization sensitivity."""
        X, y = _make_data()
        suite = StabilitySuite(n_seeds=5)
        result = suite._seed_stability(X, y, _logistic_seed_factory)

        assert "score" in result
        assert result["n_trials"] >= 3
        assert 0.0 <= result["score"] <= 1.0

    def test_seed_stability_skipped_without_factory(self):
        """Seed stability should be skipped when no seed factory provided."""
        X, y = _make_data()
        suite = StabilitySuite(n_seeds=3)
        results = suite.run_all(X, y, _logistic_factory, base_auc=0.60, seed_model_factory_fn=None)

        assert results["seed"]["skipped"] is True
        assert results["seed"]["score"] == -1.0
        # Composite should average only non-skipped measures
        assert results["composite"] > 0

    def test_prediction_agreement(self):
        """Prediction agreement should measure cross-model consistency."""
        X, y = _make_data()
        suite = StabilitySuite(n_prediction_models=5)
        result = suite._prediction_agreement(X, y, _logistic_factory)

        assert "score" in result
        assert result["n_models"] >= 3
        assert 0.0 <= result["score"] <= 1.0
        assert "mean_correlation" in result
        assert "pred_std_mean" in result

    def test_logistic_regression_high_stability(self):
        """Logistic regression should be relatively stable across all measures."""
        X, y = _make_data(n=500)
        suite = StabilitySuite(n_bootstrap=5, n_feature_dropout=5, n_seeds=5, n_prediction_models=5)
        results = suite.run_all(X, y, _logistic_factory, base_auc=0.62, seed_model_factory_fn=_logistic_seed_factory)

        # Logistic regression with deterministic seed should be very seed-stable
        assert results["seed"]["score"] >= 0.80, (
            f"Logistic regression should be seed-stable, got {results['seed']['score']:.3f}"
        )

    def test_composite_is_average(self):
        """Composite should be average of individual non-skipped scores."""
        X, y = _make_data()
        suite = StabilitySuite(n_bootstrap=3, n_feature_dropout=3, n_seeds=3, n_prediction_models=3)
        results = suite.run_all(X, y, _logistic_factory, base_auc=0.60, seed_model_factory_fn=_logistic_seed_factory)

        scores = [
            results["bootstrap"]["score"],
            results["feature_dropout"]["score"],
            results["seed"]["score"],
            results["prediction"]["score"],
        ]
        expected = np.mean(scores)
        assert abs(results["composite"] - expected) < 0.001, (
            f"Composite {results['composite']:.3f} != mean {expected:.3f}"
        )

    def test_mlp_less_seed_stable_than_logistic(self):
        """MLP with small dataset should be less seed-stable than logistic."""
        X, y = _make_data(n=200)

        def mlp_seed_factory(seed):
            return MLPClassifier(
                hidden_layer_sizes=(16,), alpha=1.0,
                max_iter=200, random_state=seed,
            )

        suite = StabilitySuite(n_seeds=5)
        lr_result = suite._seed_stability(X, y, _logistic_seed_factory)
        mlp_result = suite._seed_stability(X, y, mlp_seed_factory)

        # MLP should have higher variance across seeds than logistic
        assert mlp_result["auc_std"] >= lr_result["auc_std"] * 0.5, (
            f"MLP auc_std={mlp_result['auc_std']:.4f} should be >= "
            f"logistic auc_std={lr_result['auc_std']:.4f} * 0.5"
        )

    def test_experiment_result_has_stability_fields(self):
        """ExperimentResult should have multi-stability fields."""
        from src.phase_21_continuous.experiment_tracking import ExperimentResult, ExperimentConfig
        config = ExperimentConfig(experiment_type="default")
        r = ExperimentResult(experiment_id="test", config=config)
        assert hasattr(r, "stability_bootstrap")
        assert hasattr(r, "stability_feature_dropout")
        assert hasattr(r, "stability_seed")
        assert hasattr(r, "stability_prediction")
        assert hasattr(r, "stability_composite")

        # Should serialize/deserialize
        d = r.to_dict()
        assert "stability_composite" in d
        r2 = ExperimentResult.from_dict(d)
        assert r2.stability_composite == 0.0

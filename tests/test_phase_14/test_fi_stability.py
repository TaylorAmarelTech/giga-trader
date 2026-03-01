"""
Tests for Feature Importance Stability Gate.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

from src.phase_14_robustness.feature_importance_stability import (
    FeatureImportanceStabilityGate,
)


def _make_structured_data(n=500, n_features=20, n_informative=5, seed=42):
    """Generate well-structured classification data with clear signal."""
    X, y = make_classification(
        n_samples=n,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=3,
        n_clusters_per_class=2,
        flip_y=0.05,
        random_state=seed,
    )
    return X, y


def _make_random_data(n=500, n_features=20, seed=99):
    """Generate random data with no signal (pure noise)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    y = rng.randint(0, 2, size=n)
    return X, y


class TestFeatureImportanceStabilityGate:
    """Tests for FeatureImportanceStabilityGate."""

    def test_logistic_high_stability_on_structured_data(self):
        """Logistic regression should have high FI stability on well-structured data."""
        X, y = _make_structured_data(n=600, n_features=20, n_informative=5)
        model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        gate = FeatureImportanceStabilityGate(n_folds=5, threshold=0.5)

        result = gate.run(model, X, y)

        assert not result["skipped"]
        assert result["score"] >= 0.5, (
            f"Logistic on structured data should be stable, got score={result['score']:.3f}"
        )
        assert result["passed"] is True

    def test_decision_tree_lower_stability_on_random_data(self):
        """Decision tree on random data should have lower stability than logistic on structured."""
        X_struct, y_struct = _make_structured_data(n=500, n_features=20, n_informative=5)
        X_rand, y_rand = _make_random_data(n=500, n_features=20)

        gate = FeatureImportanceStabilityGate(n_folds=5, threshold=0.5)

        lr_model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        result_lr = gate.run(lr_model, X_struct, y_struct)

        dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
        result_dt = gate.run(dt_model, X_rand, y_rand)

        # Tree on noise should be less stable than logistic on signal
        assert result_dt["score"] < result_lr["score"], (
            f"DT on noise ({result_dt['score']:.3f}) should be less stable "
            f"than LR on signal ({result_lr['score']:.3f})"
        )

    def test_score_between_zero_and_one(self):
        """Score should always be clamped to [0, 1]."""
        X, y = _make_structured_data(n=300)
        model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        gate = FeatureImportanceStabilityGate(n_folds=5, threshold=0.5)

        result = gate.run(model, X, y)

        assert 0.0 <= result["score"] <= 1.0

    def test_passed_flag_matches_threshold(self):
        """passed flag should be True iff score >= threshold."""
        X, y = _make_structured_data(n=400)
        model = LogisticRegression(C=1.0, max_iter=500, random_state=42)

        # Very low threshold -> should pass
        gate_low = FeatureImportanceStabilityGate(n_folds=5, threshold=0.01)
        result_low = gate_low.run(model, X, y)
        assert result_low["passed"] is True
        assert result_low["score"] >= 0.01

        # Very high threshold -> likely fails
        gate_high = FeatureImportanceStabilityGate(n_folds=5, threshold=0.999)
        result_high = gate_high.run(model, X, y)
        assert result_high["passed"] == (result_high["score"] >= 0.999)

    def test_stable_and_unstable_features_shape(self):
        """most_stable_features and least_stable_features should have up to 5 items."""
        X, y = _make_structured_data(n=400, n_features=15)
        model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        gate = FeatureImportanceStabilityGate(n_folds=5, threshold=0.3)

        result = gate.run(model, X, y)

        assert not result["skipped"]
        assert len(result["most_stable_features"]) == 5
        assert len(result["least_stable_features"]) == 5

        # All should be valid feature indices
        for idx in result["most_stable_features"]:
            assert 0 <= idx < 15
        for idx in result["least_stable_features"]:
            assert 0 <= idx < 15

        # They should not be identical (unless all features are equally stable)
        # At least check they're lists of ints
        assert all(isinstance(i, int) for i in result["most_stable_features"])
        assert all(isinstance(i, int) for i in result["least_stable_features"])

    def test_pairwise_correlations_count(self):
        """pairwise_correlations should have n_folds choose 2 entries."""
        X, y = _make_structured_data(n=500, n_features=20)
        model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        n_folds = 5
        gate = FeatureImportanceStabilityGate(n_folds=n_folds, threshold=0.3)

        result = gate.run(model, X, y)

        assert not result["skipped"]
        n_successful = result["n_folds"]
        expected_pairs = n_successful * (n_successful - 1) // 2
        assert len(result["pairwise_correlations"]) == expected_pairs, (
            f"Expected {expected_pairs} pairs for {n_successful} folds, "
            f"got {len(result['pairwise_correlations'])}"
        )

    def test_tree_based_native_importance(self):
        """Tree-based model should use native feature_importances_."""
        X, y = _make_structured_data(n=400, n_features=15)
        model = GradientBoostingClassifier(
            n_estimators=30, max_depth=3, random_state=42
        )
        gate = FeatureImportanceStabilityGate(
            n_folds=4, threshold=0.3, importance_method="native"
        )

        result = gate.run(model, X, y)

        assert not result["skipped"]
        assert result["n_folds"] >= 3
        assert 0.0 <= result["score"] <= 1.0

    def test_linear_model_coef_importance(self):
        """Linear model should use coef_-based importance."""
        X, y = _make_structured_data(n=400, n_features=15)
        model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        gate = FeatureImportanceStabilityGate(
            n_folds=4, threshold=0.3, importance_method="native"
        )

        result = gate.run(model, X, y)

        assert not result["skipped"]
        assert result["n_folds"] >= 3
        assert 0.0 <= result["score"] <= 1.0

    def test_too_few_samples_skips(self):
        """Should skip gracefully when too few samples for folding."""
        X = np.random.randn(8, 5)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        gate = FeatureImportanceStabilityGate(n_folds=5, threshold=0.5)

        result = gate.run(model, X, y)

        assert result["skipped"] is True
        assert result["passed"] is False
        assert "too_few_samples" in result["reason"]

    def test_single_class_skips(self):
        """Should skip when only one class present."""
        X = np.random.randn(100, 10)
        y = np.ones(100, dtype=int)
        model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        gate = FeatureImportanceStabilityGate(n_folds=5, threshold=0.5)

        result = gate.run(model, X, y)

        assert result["skipped"] is True
        assert "single_class" in result["reason"]

"""
Tests for QuantileForestClassifier.
====================================

Comprehensive tests covering fit/predict, prediction intervals,
quantile estimation, regularization enforcement, sklearn compatibility,
and edge cases.

Wave E3: Quantile Random Forest classifier wrapper.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase_12_model_training.quantile_forest_wrapper import (
    QuantileForestClassifier,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def clf():
    """Default QuantileForestClassifier with small forest for fast tests."""
    return QuantileForestClassifier(
        n_estimators=30,
        max_depth=4,
        min_samples_leaf=5,
        random_state=42,
    )


@pytest.fixture
def separable_data():
    """Linearly separable binary classification dataset."""
    np.random.seed(42)
    n = 200
    X_pos = np.random.randn(n // 2, 5) + 2.0
    X_neg = np.random.randn(n // 2, 5) - 2.0
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * (n // 2) + [0] * (n // 2))
    # Shuffle
    idx = np.random.permutation(n)
    return X[idx], y[idx]


@pytest.fixture
def noisy_data():
    """Noisy classification dataset with overlap (ambiguous region)."""
    np.random.seed(123)
    n = 300
    X = np.random.randn(n, 8)
    # Target is weakly correlated with first feature
    logits = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n) * 1.5
    y = (logits > 0).astype(int)
    return X, y


@pytest.fixture
def tiny_data():
    """Very small dataset (n=20)."""
    np.random.seed(99)
    X = np.random.randn(20, 3)
    y = (X[:, 0] > 0).astype(int)
    return X, y


@pytest.fixture
def single_feature_data():
    """Dataset with a single feature."""
    np.random.seed(77)
    n = 100
    X = np.random.randn(n, 1)
    y = (X[:, 0] > 0.0).astype(int)
    return X, y


# ─── TestFitPredict ──────────────────────────────────────────────────────────


class TestFitPredict:
    """Test basic fit/predict cycle."""

    def test_fit_returns_self(self, clf, separable_data):
        X, y = separable_data
        result = clf.fit(X, y)
        assert result is clf

    def test_predict_returns_binary(self, clf, separable_data):
        X, y = separable_data
        clf.fit(X, y)
        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_shape(self, clf, separable_data):
        X, y = separable_data
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_predict_proba_returns_two_columns(self, clf, separable_data):
        X, y = separable_data
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)

    def test_predict_proba_sums_to_one(self, clf, separable_data):
        X, y = separable_data
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_predict_proba_bounded(self, clf, separable_data):
        X, y = separable_data
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert np.all(proba >= 0.01)
        assert np.all(proba <= 0.99)

    def test_reasonable_accuracy_separable(self, clf, separable_data):
        X, y = separable_data
        clf.fit(X, y)
        preds = clf.predict(X)
        acc = np.mean(preds == y)
        # Separable data should achieve high accuracy
        assert acc > 0.85, f"Accuracy {acc:.3f} too low on separable data"

    def test_fit_with_sample_weights(self, clf, separable_data):
        X, y = separable_data
        weights = np.random.rand(X.shape[0]) + 0.1
        clf.fit(X, y, sample_weight=weights)
        preds = clf.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_predict_consistent_with_proba(self, clf, separable_data):
        """predict() should agree with predict_proba threshold at 0.5."""
        X, y = separable_data
        clf.fit(X, y)
        preds = clf.predict(X)
        proba = clf.predict_proba(X)
        from_proba = (proba[:, 1] >= 0.5).astype(int)
        np.testing.assert_array_equal(preds, from_proba)

    def test_classes_attribute_set(self, clf, separable_data):
        X, y = separable_data
        clf.fit(X, y)
        np.testing.assert_array_equal(clf.classes_, np.array([0, 1]))


# ─── TestPredictionIntervals ─────────────────────────────────────────────────


class TestPredictionIntervals:
    """Test predict_with_intervals functionality."""

    def test_returns_four_arrays(self, clf, separable_data):
        X, y = separable_data
        clf.fit(X, y)
        result = clf.predict_with_intervals(X)
        assert len(result) == 4
        median, lower, upper, width = result
        assert median.shape == (X.shape[0],)
        assert lower.shape == (X.shape[0],)
        assert upper.shape == (X.shape[0],)
        assert width.shape == (X.shape[0],)

    def test_lower_le_median_le_upper(self, clf, separable_data):
        X, y = separable_data
        clf.fit(X, y)
        median, lower, upper, width = clf.predict_with_intervals(X)
        assert np.all(lower <= median + 1e-10), "lower > median found"
        assert np.all(median <= upper + 1e-10), "median > upper found"

    def test_width_equals_upper_minus_lower(self, clf, separable_data):
        X, y = separable_data
        clf.fit(X, y)
        median, lower, upper, width = clf.predict_with_intervals(X)
        np.testing.assert_allclose(width, upper - lower, atol=1e-10)

    def test_width_nonnegative(self, clf, separable_data):
        X, y = separable_data
        clf.fit(X, y)
        _, _, _, width = clf.predict_with_intervals(X)
        assert np.all(width >= -1e-10), "Negative interval width found"

    def test_wider_intervals_in_ambiguous_region(self, clf, noisy_data):
        """Samples near decision boundary should have wider intervals."""
        X, y = noisy_data
        clf.fit(X, y)
        median, lower, upper, width = clf.predict_with_intervals(X)

        # Ambiguous: median near 0.5; confident: median near 0 or 1
        ambiguous_mask = (median > 0.3) & (median < 0.7)
        confident_mask = (median < 0.2) | (median > 0.8)

        if ambiguous_mask.sum() > 5 and confident_mask.sum() > 5:
            avg_ambiguous_width = width[ambiguous_mask].mean()
            avg_confident_width = width[confident_mask].mean()
            # Ambiguous region should generally have wider intervals
            assert avg_ambiguous_width >= avg_confident_width * 0.8, (
                f"Ambiguous width {avg_ambiguous_width:.4f} not wider than "
                f"confident width {avg_confident_width:.4f}"
            )

    def test_intervals_consistent_with_proba(self, clf, separable_data):
        """Median from intervals should match proba[:,1] approximately."""
        X, y = separable_data
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        median, _, _, _ = clf.predict_with_intervals(X)
        # proba[:,1] is clipped median, so they should be close
        clipped_median = np.clip(median, 0.01, 0.99)
        np.testing.assert_allclose(proba[:, 1], clipped_median, atol=1e-10)


# ─── TestQuantileEstimation ──────────────────────────────────────────────────


class TestQuantileEstimation:
    """Test quantile ordering and custom quantile configurations."""

    def test_default_quantiles(self, clf):
        assert clf.quantiles == (0.10, 0.50, 0.90)

    def test_custom_quantiles(self, separable_data):
        X, y = separable_data
        clf = QuantileForestClassifier(
            n_estimators=30,
            max_depth=4,
            min_samples_leaf=5,
            quantiles=(0.25, 0.50, 0.75),
            random_state=42,
        )
        clf.fit(X, y)
        median, lower, upper, width = clf.predict_with_intervals(X)
        assert np.all(lower <= median + 1e-10)
        assert np.all(median <= upper + 1e-10)

    def test_narrow_quantiles_give_narrower_intervals(self, noisy_data):
        """(0.25, 0.50, 0.75) intervals should be narrower than (0.10, 0.50, 0.90)."""
        X, y = noisy_data

        clf_wide = QuantileForestClassifier(
            n_estimators=50, max_depth=4, min_samples_leaf=5,
            quantiles=(0.10, 0.50, 0.90), random_state=42,
        )
        clf_narrow = QuantileForestClassifier(
            n_estimators=50, max_depth=4, min_samples_leaf=5,
            quantiles=(0.25, 0.50, 0.75), random_state=42,
        )

        clf_wide.fit(X, y)
        clf_narrow.fit(X, y)

        _, _, _, width_wide = clf_wide.predict_with_intervals(X)
        _, _, _, width_narrow = clf_narrow.predict_with_intervals(X)

        assert width_narrow.mean() <= width_wide.mean() + 1e-6, (
            f"Narrow quantile width {width_narrow.mean():.4f} should be <= "
            f"wide quantile width {width_wide.mean():.4f}"
        )

    def test_invalid_quantiles_wrong_count(self, separable_data):
        X, y = separable_data
        clf = QuantileForestClassifier(quantiles=(0.25, 0.75))
        with pytest.raises(ValueError, match="exactly 3 elements"):
            clf.fit(X, y)

    def test_invalid_quantiles_wrong_order(self, separable_data):
        X, y = separable_data
        clf = QuantileForestClassifier(quantiles=(0.75, 0.50, 0.25))
        with pytest.raises(ValueError, match="0 < lower < median < upper < 1"):
            clf.fit(X, y)

    def test_invalid_quantiles_out_of_range(self, separable_data):
        X, y = separable_data
        clf = QuantileForestClassifier(quantiles=(0.0, 0.5, 1.0))
        with pytest.raises(ValueError, match="0 < lower < median < upper < 1"):
            clf.fit(X, y)

    def test_tree_predictions_shape(self, clf, separable_data):
        """Internal _get_tree_predictions should return (n_samples, n_trees)."""
        X, y = separable_data
        clf.fit(X, y)
        tree_preds = clf._get_tree_predictions(X)
        assert tree_preds.shape == (X.shape[0], clf.n_estimators)

    def test_tree_predictions_in_valid_range(self, clf, separable_data):
        """Each tree prediction should be in [0, 1] for binary targets."""
        X, y = separable_data
        clf.fit(X, y)
        tree_preds = clf._get_tree_predictions(X)
        assert np.all(tree_preds >= -0.01), "Tree prediction below 0"
        assert np.all(tree_preds <= 1.01), "Tree prediction above 1"


# ─── TestRegularization ─────────────────────────────────────────────────────


class TestRegularization:
    """Test that EDGE 1 regularization constraints are enforced."""

    def test_max_depth_capped_at_5(self, separable_data):
        """Even if max_depth=10 is requested, effective depth should be 5."""
        X, y = separable_data
        clf = QuantileForestClassifier(
            n_estimators=20, max_depth=10, min_samples_leaf=5, random_state=42,
        )
        assert clf._effective_max_depth() == 5

    def test_max_depth_none_capped(self):
        """max_depth=None should be capped to 5."""
        clf = QuantileForestClassifier(max_depth=None)
        assert clf._effective_max_depth() == 5

    def test_max_depth_3_unchanged(self):
        """max_depth=3 (within limit) should be unchanged."""
        clf = QuantileForestClassifier(max_depth=3)
        assert clf._effective_max_depth() == 3

    def test_max_depth_5_unchanged(self):
        """max_depth=5 (at limit) should be unchanged."""
        clf = QuantileForestClassifier(max_depth=5)
        assert clf._effective_max_depth() == 5

    def test_actual_tree_depth_respects_cap(self, separable_data):
        """Verify the internal forest's actual max_depth is capped."""
        X, y = separable_data
        clf = QuantileForestClassifier(
            n_estimators=10, max_depth=20, min_samples_leaf=5, random_state=42,
        )
        clf.fit(X, y)
        assert clf._forest.max_depth == 5

    def test_high_min_samples_leaf_changes_interval_width(self, noisy_data):
        """Different min_samples_leaf values should produce different interval patterns."""
        X, y = noisy_data

        clf_low_leaf = QuantileForestClassifier(
            n_estimators=50, max_depth=5, min_samples_leaf=2, random_state=42,
        )
        clf_high_leaf = QuantileForestClassifier(
            n_estimators=50, max_depth=5, min_samples_leaf=50, random_state=42,
        )

        clf_low_leaf.fit(X, y)
        clf_high_leaf.fit(X, y)

        _, _, _, width_low = clf_low_leaf.predict_with_intervals(X)
        _, _, _, width_high = clf_high_leaf.predict_with_intervals(X)

        # Both should produce valid non-negative intervals
        assert np.all(width_low >= -1e-10), "Low leaf widths should be non-negative"
        assert np.all(width_high >= -1e-10), "High leaf widths should be non-negative"

        # Different regularization should produce different interval distributions
        # (exact relationship depends on data, so just verify they differ)
        assert not np.allclose(width_low, width_high, atol=1e-4), (
            "Different min_samples_leaf should produce different interval widths"
        )


# ─── TestSklearnCompat ───────────────────────────────────────────────────────


class TestSklearnCompat:
    """Test sklearn API compatibility."""

    def test_get_params(self, clf):
        params = clf.get_params()
        assert params["n_estimators"] == 30
        assert params["max_depth"] == 4
        assert params["min_samples_leaf"] == 5
        assert params["max_features"] == "sqrt"
        assert params["quantiles"] == (0.10, 0.50, 0.90)
        assert params["random_state"] == 42

    def test_set_params(self, clf):
        clf.set_params(n_estimators=200, max_depth=3)
        assert clf.n_estimators == 200
        assert clf.max_depth == 3

    def test_set_params_returns_self(self, clf):
        result = clf.set_params(n_estimators=50)
        assert result is clf

    def test_set_invalid_param_raises(self, clf):
        with pytest.raises(ValueError, match="Invalid parameter"):
            clf.set_params(nonexistent_param=42)

    def test_classes_attribute_after_fit(self, clf, separable_data):
        X, y = separable_data
        clf.fit(X, y)
        assert hasattr(clf, "classes_")
        np.testing.assert_array_equal(clf.classes_, [0, 1])

    def test_n_features_in_after_fit(self, clf, separable_data):
        X, y = separable_data
        clf.fit(X, y)
        assert hasattr(clf, "n_features_in_")
        assert clf.n_features_in_ == X.shape[1]

    def test_predict_before_fit_raises(self):
        clf = QuantileForestClassifier()
        X = np.random.randn(10, 5)
        with pytest.raises(AttributeError, match="not fitted"):
            clf.predict(X)

    def test_predict_proba_before_fit_raises(self):
        clf = QuantileForestClassifier()
        X = np.random.randn(10, 5)
        with pytest.raises(AttributeError, match="not fitted"):
            clf.predict_proba(X)

    def test_predict_with_intervals_before_fit_raises(self):
        clf = QuantileForestClassifier()
        X = np.random.randn(10, 5)
        with pytest.raises(AttributeError, match="not fitted"):
            clf.predict_with_intervals(X)

    def test_sklearn_cross_val_score(self, noisy_data):
        """Should work with sklearn's cross_val_score."""
        from sklearn.model_selection import cross_val_score

        X, y = noisy_data
        clf = QuantileForestClassifier(
            n_estimators=20, max_depth=3, min_samples_leaf=10, random_state=42,
        )
        scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_sklearn_clone(self, clf):
        """Should work with sklearn's clone."""
        from sklearn.base import clone

        clf2 = clone(clf)
        assert clf2 is not clf
        assert clf2.get_params() == clf.get_params()

    def test_decision_function(self, clf, separable_data):
        X, y = separable_data
        clf.fit(X, y)
        scores = clf.decision_function(X)
        assert scores.shape == (X.shape[0],)
        # Decision function should be continuous, roughly in [0, 1]
        assert np.all(scores >= -0.01)
        assert np.all(scores <= 1.01)

    def test_repr(self, clf):
        r = repr(clf)
        assert "QuantileForestClassifier" in r
        assert "n_estimators=30" in r
        assert "max_depth=4" in r


# ─── TestEdgeCases ───────────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_tiny_dataset(self, tiny_data):
        """Should handle very small datasets (n=20)."""
        X, y = tiny_data
        clf = QuantileForestClassifier(
            n_estimators=10, max_depth=3, min_samples_leaf=2, random_state=42,
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (20,)
        proba = clf.predict_proba(X)
        assert proba.shape == (20, 2)

    def test_single_feature(self, single_feature_data):
        """Should work with a single feature column."""
        X, y = single_feature_data
        clf = QuantileForestClassifier(
            n_estimators=20, max_depth=3, min_samples_leaf=5, random_state=42,
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (X.shape[0],)
        median, lower, upper, width = clf.predict_with_intervals(X)
        assert np.all(lower <= median + 1e-10)

    def test_all_same_class_target(self):
        """All-ones target should predict all 1s."""
        np.random.seed(55)
        X = np.random.randn(50, 4)
        y = np.ones(50, dtype=int)
        clf = QuantileForestClassifier(
            n_estimators=20, max_depth=3, min_samples_leaf=2, random_state=42,
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        assert np.all(preds == 1)

    def test_all_same_class_zero(self):
        """All-zeros target should predict all 0s."""
        np.random.seed(55)
        X = np.random.randn(50, 4)
        y = np.zeros(50, dtype=int)
        clf = QuantileForestClassifier(
            n_estimators=20, max_depth=3, min_samples_leaf=2, random_state=42,
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        assert np.all(preds == 0)

    def test_all_same_class_narrow_intervals(self):
        """All-same-class should produce very narrow intervals."""
        np.random.seed(55)
        X = np.random.randn(50, 4)
        y = np.ones(50, dtype=int)
        clf = QuantileForestClassifier(
            n_estimators=30, max_depth=3, min_samples_leaf=2, random_state=42,
        )
        clf.fit(X, y)
        _, _, _, width = clf.predict_with_intervals(X)
        # All trees should predict ~1.0, so intervals should be very narrow
        assert width.mean() < 0.1, f"Width {width.mean():.4f} too wide for constant target"

    def test_pandas_dataframe_input(self, separable_data):
        """Should accept pandas DataFrame as input."""
        import pandas as pd
        X, y = separable_data
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)
        clf = QuantileForestClassifier(
            n_estimators=20, max_depth=3, min_samples_leaf=5, random_state=42,
        )
        clf.fit(X_df, y_series)
        preds = clf.predict(X_df)
        assert preds.shape == (X.shape[0],)

    def test_float_target_values(self, separable_data):
        """Should handle float targets that are 0.0/1.0."""
        X, y = separable_data
        y_float = y.astype(np.float64)
        clf = QuantileForestClassifier(
            n_estimators=20, max_depth=3, min_samples_leaf=5, random_state=42,
        )
        clf.fit(X, y_float)
        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_reproducibility(self, separable_data):
        """Two classifiers with same random_state should give identical results."""
        X, y = separable_data

        clf1 = QuantileForestClassifier(
            n_estimators=20, max_depth=3, min_samples_leaf=5, random_state=42,
        )
        clf2 = QuantileForestClassifier(
            n_estimators=20, max_depth=3, min_samples_leaf=5, random_state=42,
        )

        clf1.fit(X, y)
        clf2.fit(X, y)

        proba1 = clf1.predict_proba(X)
        proba2 = clf2.predict_proba(X)
        np.testing.assert_array_equal(proba1, proba2)

    def test_different_random_state_gives_different_results(self, noisy_data):
        """Different random_states should produce different (but valid) results."""
        X, y = noisy_data

        clf1 = QuantileForestClassifier(
            n_estimators=20, max_depth=3, min_samples_leaf=5, random_state=42,
        )
        clf2 = QuantileForestClassifier(
            n_estimators=20, max_depth=3, min_samples_leaf=5, random_state=99,
        )

        clf1.fit(X, y)
        clf2.fit(X, y)

        proba1 = clf1.predict_proba(X)
        proba2 = clf2.predict_proba(X)
        # They should not be exactly equal
        assert not np.allclose(proba1, proba2, atol=1e-8)

    def test_many_features(self):
        """Should handle high-dimensional data."""
        np.random.seed(42)
        X = np.random.randn(100, 200)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        clf = QuantileForestClassifier(
            n_estimators=20, max_depth=3, min_samples_leaf=10, random_state=42,
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (100,)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_sample_weight_zeros(self, separable_data):
        """Samples with zero weight should effectively be ignored."""
        X, y = separable_data
        weights = np.ones(X.shape[0])
        # Zero out weight for half the samples
        weights[:X.shape[0] // 2] = 0.0
        clf = QuantileForestClassifier(
            n_estimators=20, max_depth=3, min_samples_leaf=5, random_state=42,
        )
        # Should not crash
        clf.fit(X, y, sample_weight=weights)
        preds = clf.predict(X)
        assert preds.shape == (X.shape[0],)

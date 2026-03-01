"""
Tests for StackingEnsembleClassifier.
======================================

Comprehensive tests covering fit/predict, OOF generation, purge-aware CV,
meta-learner training, sample_weight passthrough, sklearn compatibility,
and edge cases.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.linear_model import LogisticRegression

from src.phase_12_model_training.stacking_ensemble import (
    StackingEnsembleClassifier,
)


# --- Fixtures ----------------------------------------------------------------


@pytest.fixture
def clf():
    """Default StackingEnsembleClassifier with small models for fast tests."""
    return StackingEnsembleClassifier(
        meta_C=1.0,
        n_cv_folds=3,
        purge_days=2,
        embargo_days=1,
        random_state=42,
    )


@pytest.fixture
def binary_data():
    """Binary classification dataset with moderate signal (200 samples)."""
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 10)
    logits = 0.8 * X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2]
    y = (logits > 0).astype(int)
    return X, y


@pytest.fixture
def separable_data():
    """Well-separated binary classification dataset."""
    np.random.seed(99)
    n = 250
    X_pos = np.random.randn(n // 2, 8) + 2.0
    X_neg = np.random.randn(n - n // 2, 8) - 2.0
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * (n // 2) + [0] * (n - n // 2))
    return X, y


@pytest.fixture
def small_data():
    """Small dataset (50 samples) for edge case testing."""
    np.random.seed(77)
    n = 50
    X = np.random.randn(n, 5)
    y = (X[:, 0] > 0).astype(int)
    return X, y


# --- TestFitPredict -----------------------------------------------------------


class TestFitPredict:
    """Test basic fit/predict cycle."""

    def test_fit_returns_self(self, clf, binary_data):
        X, y = binary_data
        result = clf.fit(X, y)
        assert result is clf

    def test_predict_proba_shape(self, clf, binary_data):
        """predict_proba should return (n_samples, 2) array."""
        X, y = binary_data
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)

    def test_predict_proba_sums_to_one(self, clf, binary_data):
        X, y = binary_data
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_predict_proba_bounded(self, clf, binary_data):
        X, y = binary_data
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_predict_returns_binary_labels(self, clf, binary_data):
        """predict should return 0/1 labels."""
        X, y = binary_data
        clf.fit(X, y)
        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})
        assert preds.shape == (X.shape[0],)

    def test_predict_consistent_with_proba(self, clf, binary_data):
        """predict should agree with predict_proba threshold at 0.5."""
        X, y = binary_data
        clf.fit(X, y)
        preds = clf.predict(X)
        proba = clf.predict_proba(X)
        from_proba = (proba[:, 1] >= 0.5).astype(int)
        np.testing.assert_array_equal(preds, from_proba)

    def test_reasonable_accuracy_on_separable(self, separable_data):
        """On well-separated data, accuracy should be decent."""
        X, y = separable_data
        clf = StackingEnsembleClassifier(
            n_cv_folds=3, purge_days=2, embargo_days=1, random_state=42,
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        acc = np.mean(preds == y)
        assert acc > 0.80, f"Accuracy {acc:.3f} too low on separable data"


# --- TestOOFPredictions -------------------------------------------------------


class TestOOFPredictions:
    """Test that OOF predictions are generated correctly."""

    def test_oof_no_leakage(self, binary_data):
        """OOF prediction for sample i should NOT be trained on sample i.

        Verify this by checking that OOF generation produces a mask and that
        the stacking doesn't simply memorize on the training set.
        """
        X, y = binary_data
        clf = StackingEnsembleClassifier(
            n_cv_folds=5, purge_days=2, embargo_days=1, random_state=42,
        )

        # Access internal method to inspect OOF
        oof_matrix, oof_mask = clf._generate_oof_predictions(X, y)

        # All samples used as test in some fold should have OOF predictions
        # With 5 folds, most samples should be covered
        coverage = oof_mask.sum() / len(y)
        assert coverage > 0.8, (
            f"OOF coverage {coverage:.2%} is too low; expected >80%"
        )

        # Check that OOF matrix has the right shape (n_samples x n_models)
        configs = clf._get_base_configs()
        assert oof_matrix.shape == (X.shape[0], len(configs))

        # OOF predictions should be valid probabilities
        valid_oof = oof_matrix[oof_mask]
        assert np.all(valid_oof >= 0.0)
        assert np.all(valid_oof <= 1.0)

    def test_oof_matrix_not_all_same(self, binary_data):
        """OOF predictions from different base models should differ."""
        X, y = binary_data
        clf = StackingEnsembleClassifier(
            n_cv_folds=3, purge_days=2, embargo_days=1, random_state=42,
        )
        oof_matrix, oof_mask = clf._generate_oof_predictions(X, y)
        valid_oof = oof_matrix[oof_mask]

        # At least two columns should differ
        if valid_oof.shape[1] >= 2:
            assert not np.allclose(valid_oof[:, 0], valid_oof[:, 1], atol=0.01), (
                "OOF predictions from different base models should not be identical"
            )


# --- TestMetaLearner ----------------------------------------------------------


class TestMetaLearner:
    """Test that the meta-learner is fitted on OOF predictions."""

    def test_meta_learner_exists_after_fit(self, clf, binary_data):
        X, y = binary_data
        clf.fit(X, y)
        assert hasattr(clf, "meta_learner_")
        assert isinstance(clf.meta_learner_, LogisticRegression)

    def test_meta_learner_input_dim_matches_n_models(self, clf, binary_data):
        """Meta-learner should take n_base_models features as input."""
        X, y = binary_data
        clf.fit(X, y)
        configs = clf._get_base_configs()
        n_models = len(configs)
        # meta_learner coef shape: (1, n_models) for binary
        assert clf.meta_learner_.coef_.shape[1] == n_models


# --- TestBaseModelRefit -------------------------------------------------------


class TestBaseModelRefit:
    """Test that base models are refit on full data after OOF generation."""

    def test_base_models_stored_after_fit(self, clf, binary_data):
        X, y = binary_data
        clf.fit(X, y)
        assert hasattr(clf, "base_models_")
        configs = clf._get_base_configs()
        assert len(clf.base_models_) == len(configs)

    def test_base_models_are_fitted(self, clf, binary_data):
        """Each base model should be fitted (have predict_proba or similar)."""
        X, y = binary_data
        clf.fit(X, y)
        for model in clf.base_models_:
            # Fitted models should be able to predict
            preds = model.predict(X[:5])
            assert preds.shape == (5,)


# --- TestClassesProperty ------------------------------------------------------


class TestClassesProperty:
    """Test the classes_ property."""

    def test_classes_after_fit(self, clf, binary_data):
        X, y = binary_data
        clf.fit(X, y)
        np.testing.assert_array_equal(clf.classes_, np.array([0, 1]))

    def test_classes_before_fit_raises(self):
        clf = StackingEnsembleClassifier()
        with pytest.raises(AttributeError, match="not fitted"):
            _ = clf.classes_


# --- TestCustomBaseModels -----------------------------------------------------


class TestCustomBaseModels:
    """Test that custom base models can be provided."""

    def test_custom_single_model(self, binary_data):
        """Should work with a single custom base model."""
        from sklearn.linear_model import LogisticRegression

        X, y = binary_data
        custom_configs = [
            {"name": "lr_custom", "model": LogisticRegression(C=0.5, max_iter=500)},
        ]
        clf = StackingEnsembleClassifier(
            base_model_configs=custom_configs,
            n_cv_folds=3,
            purge_days=2,
            embargo_days=1,
            random_state=42,
        )
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)
        assert len(clf.base_models_) == 1

    def test_custom_multiple_models(self, binary_data):
        """Should work with multiple custom base models."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier

        X, y = binary_data
        custom_configs = [
            {"name": "lr", "model": LogisticRegression(C=1.0, max_iter=500)},
            {"name": "dt", "model": DecisionTreeClassifier(max_depth=3)},
        ]
        clf = StackingEnsembleClassifier(
            base_model_configs=custom_configs,
            n_cv_folds=3,
            purge_days=2,
            embargo_days=1,
            random_state=42,
        )
        clf.fit(X, y)
        assert len(clf.base_models_) == 2
        preds = clf.predict(X)
        assert preds.shape == (X.shape[0],)


# --- TestSampleWeight ---------------------------------------------------------


class TestSampleWeight:
    """Test that sample_weight is passed through."""

    def test_fit_with_sample_weight(self, clf, binary_data):
        """Should accept and use sample_weight without error."""
        X, y = binary_data
        weights = np.random.rand(X.shape[0]) + 0.1
        clf.fit(X, y, sample_weight=weights)
        proba = clf.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)

    def test_sample_weight_affects_predictions(self, binary_data):
        """Different sample weights should produce different models."""
        X, y = binary_data

        clf1 = StackingEnsembleClassifier(
            n_cv_folds=3, purge_days=2, embargo_days=1, random_state=42,
        )
        clf2 = StackingEnsembleClassifier(
            n_cv_folds=3, purge_days=2, embargo_days=1, random_state=42,
        )

        # Uniform weights
        w1 = np.ones(X.shape[0])
        # Heavily skewed weights
        w2 = np.ones(X.shape[0])
        w2[:X.shape[0] // 2] = 10.0
        w2[X.shape[0] // 2:] = 0.1

        clf1.fit(X, y, sample_weight=w1)
        clf2.fit(X, y, sample_weight=w2)

        proba1 = clf1.predict_proba(X)
        proba2 = clf2.predict_proba(X)

        # Predictions should differ due to different weights
        assert not np.allclose(proba1, proba2, atol=0.01), (
            "Different sample weights should produce different predictions"
        )


# --- TestSmallDataset ---------------------------------------------------------


class TestSmallDataset:
    """Test with a small dataset (50 samples)."""

    def test_works_with_50_samples(self, small_data):
        X, y = small_data
        clf = StackingEnsembleClassifier(
            n_cv_folds=3,
            purge_days=1,
            embargo_days=1,
            random_state=42,
        )
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)
        preds = clf.predict(X)
        assert preds.shape == (X.shape[0],)


# --- TestPurgeEmbargo ---------------------------------------------------------


class TestPurgeEmbargo:
    """Test that purge and embargo reduce training indices near test boundaries."""

    def test_purge_removes_boundary_samples(self):
        """Training indices should exclude rows near test fold boundaries."""
        clf = StackingEnsembleClassifier(
            n_cv_folds=5,
            purge_days=5,
            embargo_days=3,
            random_state=42,
        )
        n_samples = 200
        splits = clf._create_time_series_splits(n_samples)

        for train_idx, test_idx in splits:
            test_start = test_idx[0]
            test_end = test_idx[-1] + 1

            # No training sample should be within purge_days before test
            purge_before = max(0, test_start - clf.purge_days)
            for idx in train_idx:
                if idx < test_start:
                    assert idx < purge_before, (
                        f"Train idx {idx} is within purge zone "
                        f"[{purge_before}, {test_start}) before test"
                    )

            # No training sample should be within purge_days after test end
            purge_after = min(n_samples, test_end + clf.purge_days)
            for idx in train_idx:
                if idx >= test_end:
                    assert idx >= purge_after, (
                        f"Train idx {idx} is within purge zone "
                        f"[{test_end}, {purge_after}) after test"
                    )

    def test_embargo_removes_post_test_samples(self):
        """Embargo should remove samples immediately after test block."""
        clf = StackingEnsembleClassifier(
            n_cv_folds=4,
            purge_days=3,
            embargo_days=5,
            random_state=42,
        )
        n_samples = 200
        splits = clf._create_time_series_splits(n_samples)

        for train_idx, test_idx in splits:
            test_end = test_idx[-1] + 1
            embargo_end = min(n_samples, test_end + clf.embargo_days)

            # No training sample should be in the embargo zone
            for idx in train_idx:
                if test_end <= idx < embargo_end:
                    # This should not happen; embargo zone should be excluded
                    # But since purge_days >= embargo_days in this test,
                    # the purge zone is wider and covers embargo too.
                    pass

            # At minimum, the embargo zone should be excluded
            embargo_zone = set(range(test_end, min(n_samples, test_end + clf.embargo_days)))
            train_set = set(train_idx.tolist())
            overlap = train_set & embargo_zone
            assert len(overlap) == 0, (
                f"Embargo zone {embargo_zone} overlaps with train set: {overlap}"
            )

    def test_no_overlap_between_train_and_test(self):
        """Train and test indices should never overlap."""
        clf = StackingEnsembleClassifier(
            n_cv_folds=5,
            purge_days=3,
            embargo_days=2,
            random_state=42,
        )
        n_samples = 200
        splits = clf._create_time_series_splits(n_samples)

        for train_idx, test_idx in splits:
            overlap = set(train_idx.tolist()) & set(test_idx.tolist())
            assert len(overlap) == 0, f"Train/test overlap: {overlap}"

    def test_purge_reduces_train_size(self):
        """Larger purge_days should result in fewer training samples."""
        n_samples = 200

        clf_no_purge = StackingEnsembleClassifier(
            n_cv_folds=5, purge_days=0, embargo_days=0,
        )
        clf_big_purge = StackingEnsembleClassifier(
            n_cv_folds=5, purge_days=10, embargo_days=5,
        )

        splits_no = clf_no_purge._create_time_series_splits(n_samples)
        splits_big = clf_big_purge._create_time_series_splits(n_samples)

        # Average training size should be smaller with bigger purge
        avg_no = np.mean([len(tr) for tr, _ in splits_no])
        avg_big = np.mean([len(tr) for tr, _ in splits_big])
        assert avg_big < avg_no, (
            f"Purge should reduce train size: {avg_big} >= {avg_no}"
        )


# --- TestSklearnCompat --------------------------------------------------------


class TestSklearnCompat:
    """Test sklearn API compatibility."""

    def test_get_params(self, clf):
        params = clf.get_params()
        assert params["meta_C"] == 1.0
        assert params["n_cv_folds"] == 3
        assert params["purge_days"] == 2
        assert params["embargo_days"] == 1
        assert params["random_state"] == 42
        assert params["base_model_configs"] is None

    def test_set_params(self, clf):
        clf.set_params(meta_C=0.5, purge_days=5)
        assert clf.meta_C == 0.5
        assert clf.purge_days == 5

    def test_set_params_returns_self(self, clf):
        result = clf.set_params(meta_C=2.0)
        assert result is clf

    def test_set_invalid_param_raises(self, clf):
        with pytest.raises(ValueError, match="Invalid parameter"):
            clf.set_params(nonexistent_param=42)

    def test_predict_before_fit_raises(self):
        clf = StackingEnsembleClassifier()
        X = np.random.randn(10, 5)
        with pytest.raises(AttributeError, match="not fitted"):
            clf.predict(X)

    def test_predict_proba_before_fit_raises(self):
        clf = StackingEnsembleClassifier()
        X = np.random.randn(10, 5)
        with pytest.raises(AttributeError, match="not fitted"):
            clf.predict_proba(X)

    def test_n_features_in_after_fit(self, clf, binary_data):
        X, y = binary_data
        clf.fit(X, y)
        assert hasattr(clf, "n_features_in_")
        assert clf.n_features_in_ == X.shape[1]

    def test_repr(self, clf):
        r = repr(clf)
        assert "StackingEnsembleClassifier" in r
        assert "meta_C=1.0" in r
        assert "n_cv_folds=3" in r

    def test_sklearn_clone(self, clf):
        """Should work with sklearn's clone."""
        from sklearn.base import clone

        clf2 = clone(clf)
        assert clf2 is not clf
        assert clf2.get_params() == clf.get_params()


# --- TestDecisionFunctionFallback ---------------------------------------------


class TestDecisionFunctionFallback:
    """Test handling of base models without predict_proba."""

    def test_svc_fallback_to_decision_function(self, binary_data):
        """SVC without probability=True should use decision_function fallback."""
        from sklearn.svm import SVC

        X, y = binary_data
        custom_configs = [
            {"name": "svc", "model": SVC(kernel="linear", C=1.0)},
        ]
        clf = StackingEnsembleClassifier(
            base_model_configs=custom_configs,
            n_cv_folds=3,
            purge_days=2,
            embargo_days=1,
            random_state=42,
        )
        # Should not raise even though SVC has no predict_proba by default
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (X.shape[0], 2)

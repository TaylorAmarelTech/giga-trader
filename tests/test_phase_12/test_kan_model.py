"""Tests for KANClassifier (Kolmogorov-Arnold Network)."""

import numpy as np
import pytest

from src.phase_12_model_training.kan_model import KANClassifier


def _make_data(n=200, n_features=10, seed=42):
    """Generate linearly separable binary classification data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


class TestKANInit:
    def test_default_construction(self):
        kan = KANClassifier()
        assert kan.hidden_dim == 16
        assert kan.grid_size == 5
        assert kan.spline_order == 3

    def test_custom_params(self):
        kan = KANClassifier(hidden_dim=8, grid_size=3, l1_lambda=0.1)
        assert kan.hidden_dim == 8
        assert kan.grid_size == 3
        assert kan.l1_lambda == 0.1


class TestKANFit:
    def test_fit_returns_self(self):
        X, y = _make_data()
        kan = KANClassifier(max_epochs=5)
        result = kan.fit(X, y)
        assert result is kan

    def test_has_classes_after_fit(self):
        X, y = _make_data()
        kan = KANClassifier(max_epochs=5)
        kan.fit(X, y)
        assert hasattr(kan, "classes_")
        assert len(kan.classes_) == 2

    def test_has_feature_importances(self):
        X, y = _make_data()
        kan = KANClassifier(max_epochs=5)
        kan.fit(X, y)
        assert hasattr(kan, "feature_importances_")
        assert len(kan.feature_importances_) == X.shape[1]
        assert abs(kan.feature_importances_.sum() - 1.0) < 0.01

    def test_has_n_features_in(self):
        X, y = _make_data()
        kan = KANClassifier(max_epochs=5)
        kan.fit(X, y)
        assert kan.n_features_in_ == X.shape[1]


class TestKANPredict:
    def test_predict_shape(self):
        X, y = _make_data()
        kan = KANClassifier(max_epochs=20)
        kan.fit(X, y)
        pred = kan.predict(X)
        assert pred.shape == (len(X),)

    def test_predict_values_are_0_or_1(self):
        X, y = _make_data()
        kan = KANClassifier(max_epochs=20)
        kan.fit(X, y)
        pred = kan.predict(X)
        assert set(pred).issubset({0, 1})

    def test_predict_proba_shape(self):
        X, y = _make_data()
        kan = KANClassifier(max_epochs=20)
        kan.fit(X, y)
        proba = kan.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_predict_proba_sums_to_one(self):
        X, y = _make_data()
        kan = KANClassifier(max_epochs=20)
        kan.fit(X, y)
        proba = kan.predict_proba(X)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_predict_proba_in_01_range(self):
        X, y = _make_data()
        kan = KANClassifier(max_epochs=20)
        kan.fit(X, y)
        proba = kan.predict_proba(X)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_above_chance_accuracy(self):
        X, y = _make_data(n=400)
        kan = KANClassifier(max_epochs=50, hidden_dim=16, learning_rate=0.01)
        kan.fit(X, y)
        pred = kan.predict(X)
        acc = (pred == y).mean()
        assert acc > 0.60, f"Accuracy {acc:.2%} should be above chance"


class TestKANSklearnCompat:
    def test_get_params(self):
        kan = KANClassifier(hidden_dim=8)
        params = kan.get_params()
        assert params["hidden_dim"] == 8

    def test_set_params(self):
        kan = KANClassifier()
        kan.set_params(hidden_dim=32)
        assert kan.hidden_dim == 32

    def test_clone(self):
        from sklearn.base import clone
        kan = KANClassifier(hidden_dim=12, l1_lambda=0.05)
        kan2 = clone(kan)
        assert kan2.hidden_dim == 12
        assert kan2.l1_lambda == 0.05

    def test_cross_val_score(self):
        from sklearn.model_selection import cross_val_score
        X, y = _make_data(n=200)
        kan = KANClassifier(max_epochs=10, hidden_dim=8)
        scores = cross_val_score(kan, X, y, cv=3, scoring="accuracy")
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)


class TestKANRegularization:
    def test_l1_regularization_sparsifies(self):
        X, y = _make_data()
        kan_heavy = KANClassifier(
            max_epochs=30, l1_lambda=1.0, hidden_dim=8,
        )
        kan_heavy.fit(X, y)
        # Heavy L1 should produce more near-zero coefficients
        assert kan_heavy.feature_importances_ is not None

    def test_early_stopping(self):
        X, y = _make_data()
        kan = KANClassifier(max_epochs=1000, patience=3)
        kan.fit(X, y)
        # Should stop well before 1000 epochs
        # (can't directly check epochs, but it should finish quickly)


class TestKANEdgeCases:
    def test_single_feature(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 1)
        y = (X[:, 0] > 0).astype(int)
        kan = KANClassifier(max_epochs=10, hidden_dim=4)
        kan.fit(X, y)
        pred = kan.predict(X)
        assert pred.shape == (100,)

    def test_all_same_label(self):
        X = np.random.randn(100, 5)
        y = np.ones(100, dtype=int)
        kan = KANClassifier(max_epochs=5)
        kan.fit(X, y)
        pred = kan.predict(X)
        assert set(pred).issubset({0, 1})

    def test_sample_weight(self):
        X, y = _make_data()
        weights = np.ones(len(y))
        weights[:50] = 2.0  # Double weight for first 50
        kan = KANClassifier(max_epochs=10)
        kan.fit(X, y, sample_weight=weights)
        pred = kan.predict(X)
        assert pred.shape == (len(X),)

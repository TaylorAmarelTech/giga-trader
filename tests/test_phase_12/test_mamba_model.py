"""Tests for MambaClassifier (Selective State-Space Model)."""

import numpy as np
import pytest

from src.phase_12_model_training.mamba_model import MambaClassifier


def _make_data(n=200, n_features=10, seed=42):
    """Generate linearly separable binary classification data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


class TestMambaInit:
    def test_default_construction(self):
        clf = MambaClassifier()
        assert clf.hidden_dim == 16
        assert clf.n_layers == 2
        assert clf.d_state == 8
        assert clf.l2_lambda == 0.01
        assert clf.patience == 5

    def test_custom_params(self):
        clf = MambaClassifier(hidden_dim=8, n_layers=1, l2_lambda=0.1, d_state=4)
        assert clf.hidden_dim == 8
        assert clf.n_layers == 1
        assert clf.l2_lambda == 0.1
        assert clf.d_state == 4


class TestMambaFit:
    def test_fit_returns_self(self):
        X, y = _make_data()
        clf = MambaClassifier(max_epochs=2, hidden_dim=4, d_state=2, n_layers=1)
        result = clf.fit(X, y)
        assert result is clf

    def test_has_classes_after_fit(self):
        X, y = _make_data()
        clf = MambaClassifier(max_epochs=2, hidden_dim=4, d_state=2, n_layers=1)
        clf.fit(X, y)
        assert hasattr(clf, "classes_")
        assert len(clf.classes_) == 2

    def test_has_feature_importances(self):
        X, y = _make_data()
        clf = MambaClassifier(max_epochs=2, hidden_dim=4, d_state=2, n_layers=1)
        clf.fit(X, y)
        assert hasattr(clf, "feature_importances_")
        assert len(clf.feature_importances_) == X.shape[1]
        assert abs(clf.feature_importances_.sum() - 1.0) < 0.01

    def test_has_n_features_in(self):
        X, y = _make_data()
        clf = MambaClassifier(max_epochs=2, hidden_dim=4, d_state=2, n_layers=1)
        clf.fit(X, y)
        assert clf.n_features_in_ == X.shape[1]


class TestMambaPredict:
    def test_predict_shape(self):
        X, y = _make_data()
        clf = MambaClassifier(max_epochs=3, hidden_dim=4, d_state=2, n_layers=1)
        clf.fit(X, y)
        pred = clf.predict(X)
        assert pred.shape == (len(X),)

    def test_predict_values_are_0_or_1(self):
        X, y = _make_data()
        clf = MambaClassifier(max_epochs=3, hidden_dim=4, d_state=2, n_layers=1)
        clf.fit(X, y)
        pred = clf.predict(X)
        assert set(pred).issubset({0, 1})

    def test_predict_proba_shape(self):
        X, y = _make_data()
        clf = MambaClassifier(max_epochs=3, hidden_dim=4, d_state=2, n_layers=1)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_predict_proba_sums_to_one(self):
        X, y = _make_data()
        clf = MambaClassifier(max_epochs=3, hidden_dim=4, d_state=2, n_layers=1)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_predict_proba_in_01_range(self):
        X, y = _make_data()
        clf = MambaClassifier(max_epochs=3, hidden_dim=4, d_state=2, n_layers=1)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_above_chance_accuracy(self):
        X, y = _make_data(n=400)
        clf = MambaClassifier(
            max_epochs=30, hidden_dim=8, d_state=4,
            n_layers=1, learning_rate=0.01,
        )
        clf.fit(X, y)
        pred = clf.predict(X)
        acc = (pred == y).mean()
        assert acc > 0.55, f"Accuracy {acc:.2%} should be above chance"


class TestMambaSklearnCompat:
    def test_get_params(self):
        clf = MambaClassifier(hidden_dim=8)
        params = clf.get_params()
        assert params["hidden_dim"] == 8

    def test_set_params(self):
        clf = MambaClassifier()
        clf.set_params(hidden_dim=32)
        assert clf.hidden_dim == 32

    def test_clone(self):
        from sklearn.base import clone
        clf = MambaClassifier(hidden_dim=12, l2_lambda=0.05)
        clf2 = clone(clf)
        assert clf2.hidden_dim == 12
        assert clf2.l2_lambda == 0.05

    def test_cross_val_score(self):
        from sklearn.model_selection import cross_val_score
        X, y = _make_data(n=200)
        clf = MambaClassifier(
            max_epochs=3, hidden_dim=4, d_state=2, n_layers=1,
        )
        scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)


class TestMambaEdgeCases:
    def test_single_feature(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 1)
        y = (X[:, 0] > 0).astype(int)
        clf = MambaClassifier(
            max_epochs=3, hidden_dim=4, d_state=2, n_layers=1,
        )
        clf.fit(X, y)
        pred = clf.predict(X)
        assert pred.shape == (100,)

    def test_all_same_label(self):
        X = np.random.randn(100, 5)
        y = np.ones(100, dtype=int)
        clf = MambaClassifier(
            max_epochs=3, hidden_dim=4, d_state=2, n_layers=1,
        )
        clf.fit(X, y)
        pred = clf.predict(X)
        assert set(pred).issubset({0, 1})

    def test_sample_weight(self):
        X, y = _make_data()
        weights = np.ones(len(y))
        weights[:50] = 2.0  # Double weight for first 50
        clf = MambaClassifier(
            max_epochs=3, hidden_dim=4, d_state=2, n_layers=1,
        )
        clf.fit(X, y, sample_weight=weights)
        pred = clf.predict(X)
        assert pred.shape == (len(X),)

    def test_not_fitted_raises(self):
        clf = MambaClassifier()
        X = np.random.randn(10, 5)
        with pytest.raises(AttributeError, match="not fitted"):
            clf.predict(X)

    def test_decision_function(self):
        X, y = _make_data()
        clf = MambaClassifier(
            max_epochs=3, hidden_dim=4, d_state=2, n_layers=1,
        )
        clf.fit(X, y)
        scores = clf.decision_function(X)
        assert scores.shape == (len(X),)
        # Logits should be finite real numbers
        assert np.all(np.isfinite(scores))

    def test_repr(self):
        clf = MambaClassifier(hidden_dim=8, n_layers=1)
        r = repr(clf)
        assert "MambaClassifier" in r
        assert "hidden_dim=8" in r
        assert "n_layers=1" in r

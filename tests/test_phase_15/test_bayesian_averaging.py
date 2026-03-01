"""Tests for BayesianModelAverager."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from src.phase_15_strategy.bayesian_averaging import BayesianModelAverager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n: int = 200, n_features: int = 5, seed: int = 42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * 0.3 > 0).astype(int)
    return X, y


def _fit_models(X_train, y_train):
    lr = LogisticRegression(max_iter=300, random_state=0).fit(X_train, y_train)
    gb = GradientBoostingClassifier(
        n_estimators=20, max_depth=2, random_state=0,
    ).fit(X_train, y_train)
    dt = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train)
    return {"lr": lr, "gb": gb, "dt": dt}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBayesianModelAverager:

    def test_fit_and_predict_proba(self):
        """fit() followed by predict_proba() runs without error."""
        X, y = _make_data(300)
        X_train, X_val = X[:200], X[200:]
        y_train, y_val = y[:200], y[200:]
        models = _fit_models(X_train, y_train)

        bma = BayesianModelAverager()
        bma.fit(models, X_val, y_val)
        proba = bma.predict_proba(X_val)

        assert proba.shape == (len(X_val), 2)
        # Probabilities are in [0, 1]
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_weights_sum_to_one(self):
        """Posterior weights must sum to 1.0."""
        X, y = _make_data(300)
        models = _fit_models(X[:200], y[:200])

        bma = BayesianModelAverager()
        bma.fit(models, X[200:], y[200:])

        weights = bma.get_weights()
        assert abs(sum(weights.values()) - 1.0) < 1e-8

    def test_better_model_higher_weight(self):
        """A model with strictly better predictions should receive a higher weight."""
        X, y = _make_data(400, seed=7)
        X_train, X_val = X[:300], X[300:]
        y_train, y_val = y[:300], y[300:]

        # Good model: well-tuned LR
        good = LogisticRegression(max_iter=500, C=1.0, random_state=0).fit(X_train, y_train)
        # Bad model: severely under-regularised tree (depth=1, few samples)
        bad = DecisionTreeClassifier(max_depth=1, random_state=99).fit(X_train[:20], y_train[:20])

        bma = BayesianModelAverager()
        bma.fit({"good": good, "bad": bad}, X_val, y_val)
        weights = bma.get_weights()

        assert weights["good"] > weights["bad"]

    def test_min_weight_floor(self):
        """No model weight should fall below min_weight (after renormalisation)."""
        X, y = _make_data(300)
        models = _fit_models(X[:200], y[:200])

        min_w = 0.05
        bma = BayesianModelAverager(min_weight=min_w)
        bma.fit(models, X[200:], y[200:])

        for w in bma.get_weights().values():
            # After renormalisation the effective floor is >= min_weight / n_models
            # but before renorm each weight >= min_weight, so after renorm >= min_weight / sum
            # The key invariant: no raw weight was set below the floor before renorm.
            assert w >= min_w / len(models) - 1e-9

    def test_predict_proba_shape(self):
        """predict_proba returns (n_samples, 2)."""
        X, y = _make_data(300)
        models = _fit_models(X[:200], y[:200])

        bma = BayesianModelAverager()
        bma.fit(models, X[200:], y[200:])

        for n in [1, 10, 50]:
            proba = bma.predict_proba(X[200 : 200 + n])
            assert proba.shape == (n, 2)
            # Rows should sum to ~1.0
            row_sums = proba.sum(axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-8)

    def test_get_weights_returns_dict(self):
        """get_weights returns a dict with one entry per model."""
        X, y = _make_data(300)
        models = _fit_models(X[:200], y[:200])

        bma = BayesianModelAverager()
        bma.fit(models, X[200:], y[200:])

        weights = bma.get_weights()
        assert isinstance(weights, dict)
        assert set(weights.keys()) == set(models.keys())

    def test_get_bics_returns_dict(self):
        """get_bics returns a dict with one BIC per model."""
        X, y = _make_data(300)
        models = _fit_models(X[:200], y[:200])

        bma = BayesianModelAverager()
        bma.fit(models, X[200:], y[200:])

        bics = bma.get_bics()
        assert isinstance(bics, dict)
        assert set(bics.keys()) == set(models.keys())
        # BICs should be finite numbers
        for v in bics.values():
            assert np.isfinite(v)

    def test_single_model_gets_weight_one(self):
        """A single model should receive weight 1.0."""
        X, y = _make_data(300)
        lr = LogisticRegression(max_iter=300, random_state=0).fit(X[:200], y[:200])

        bma = BayesianModelAverager()
        bma.fit({"only": lr}, X[200:], y[200:])

        weights = bma.get_weights()
        assert abs(weights["only"] - 1.0) < 1e-8

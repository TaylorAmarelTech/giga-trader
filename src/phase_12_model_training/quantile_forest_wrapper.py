"""
Quantile Random Forest Classifier Wrapper
==========================================

A classifier wrapper around sklearn's RandomForestRegressor that provides
quantile predictions and prediction intervals. Uses the forest's individual
tree predictions to estimate quantiles of the conditional distribution.

Why not use `quantile-forest` package?
    To avoid external dependencies. We get quantile estimates from a standard
    RandomForestRegressor by accessing individual tree predictions.

How it works:
    1. Fit a RandomForestRegressor on float targets (0.0 / 1.0).
    2. At prediction time, collect each tree's individual prediction.
    3. Compute percentiles across trees to get median, lower, and upper bounds.
    4. The median serves as the probability estimate for classification.
    5. The interval width (upper - lower) is a natural uncertainty metric.

Overfit models produce artificially narrow intervals in-sample that blow up
out-of-sample. The interval width therefore serves as a diagnostic for
overfitting.

Follows EDGE 1 (Regularization-First): max_depth capped at 5, heavy
min_samples_leaf default.

Usage:
    from src.phase_12_model_training.quantile_forest_wrapper import (
        QuantileForestClassifier,
    )

    clf = QuantileForestClassifier(n_estimators=200, max_depth=4)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    median, lower, upper, width = clf.predict_with_intervals(X_test)
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class QuantileForestClassifier(ClassifierMixin, BaseEstimator):
    """Random Forest with quantile prediction intervals for classification.

    Uses individual tree predictions to estimate the full conditional
    distribution, providing prediction intervals alongside point estimates.
    Wide intervals indicate uncertainty; narrow intervals indicate confidence.

    Overfit models produce artificially narrow intervals that blow up OOS.
    The interval width serves as a natural uncertainty metric.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int, default=5
        Maximum tree depth. Capped at 5 per EDGE 1 (regularization-first).
    min_samples_leaf : int, default=50
        Minimum samples per leaf. High value for heavy regularization.
    max_features : str or float, default="sqrt"
        Number of features to consider for best split.
    quantiles : tuple of float, default=(0.10, 0.50, 0.90)
        (lower, median, upper) quantile levels. Values must be in (0, 1).
    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        Class labels [0, 1]. Set after fit().
    _forest : RandomForestRegressor
        The underlying fitted regressor.

    Examples
    --------
    >>> clf = QuantileForestClassifier(n_estimators=50, max_depth=3)
    >>> clf.fit(X_train, y_train)
    >>> clf.predict_proba(X_test)[:5]
    array([[0.62, 0.38],
           [0.31, 0.69],
           ...])
    >>> median, lower, upper, width = clf.predict_with_intervals(X_test)
    """

    # Maximum allowed depth per EDGE 1 (regularization-first philosophy)
    _MAX_ALLOWED_DEPTH = 5

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_leaf: int = 50,
        max_features: str = "sqrt",
        quantiles: tuple = (0.10, 0.50, 0.90),
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.quantiles = quantiles
        self.random_state = random_state

    def _validate_quantiles(self) -> None:
        """Validate the quantiles tuple."""
        if len(self.quantiles) != 3:
            raise ValueError(
                f"quantiles must have exactly 3 elements (lower, median, upper), "
                f"got {len(self.quantiles)}"
            )
        lo, med, hi = self.quantiles
        if not (0 < lo < med < hi < 1):
            raise ValueError(
                f"quantiles must satisfy 0 < lower < median < upper < 1, "
                f"got ({lo}, {med}, {hi})"
            )

    def _effective_max_depth(self) -> int:
        """Return max_depth capped at _MAX_ALLOWED_DEPTH per EDGE 1."""
        if self.max_depth is None:
            return self._MAX_ALLOWED_DEPTH
        return min(self.max_depth, self._MAX_ALLOWED_DEPTH)

    def fit(self, X, y, sample_weight=None):
        """Fit the quantile forest on binary classification targets.

        Internally fits a RandomForestRegressor on float targets (0.0/1.0).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Binary target labels (0 or 1).
        sample_weight : array-like of shape (n_samples,), optional
            Per-sample weights for training.

        Returns
        -------
        self
            Fitted classifier.
        """
        from sklearn.ensemble import RandomForestRegressor

        self._validate_quantiles()

        effective_depth = self._effective_max_depth()

        self._forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=effective_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
        )

        y_float = np.asarray(y, dtype=float)
        self._forest.fit(X, y_float, sample_weight=sample_weight)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else len(X[0])
        return self

    def _check_fitted(self) -> None:
        """Raise if the model has not been fitted yet."""
        if not hasattr(self, "_forest"):
            raise AttributeError(
                "QuantileForestClassifier is not fitted yet. "
                "Call fit() before predict/predict_proba."
            )

    def _get_tree_predictions(self, X) -> np.ndarray:
        """Get individual tree predictions for quantile estimation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        tree_preds : ndarray of shape (n_samples, n_trees)
            Each column is one tree's prediction for all samples.
        """
        self._check_fitted()
        preds = np.array([
            tree.predict(X) for tree in self._forest.estimators_
        ])
        # preds shape: (n_trees, n_samples) -> transpose to (n_samples, n_trees)
        return preds.T

    def predict(self, X) -> np.ndarray:
        """Return binary predictions (median >= 0.5 -> class 1).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        """Return class probability estimates using median tree prediction.

        The median across all trees is used as P(class=1). This is more
        robust than the mean because it is less sensitive to outlier trees.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Column 0 = P(class=0), Column 1 = P(class=1).
            Columns sum to 1.0 for each row.
        """
        tree_preds = self._get_tree_predictions(X)
        median_q = self.quantiles[1] * 100
        median_p = np.percentile(tree_preds, median_q, axis=1)
        median_p = np.clip(median_p, 0.01, 0.99)
        return np.column_stack([1 - median_p, median_p])

    def predict_with_intervals(self, X):
        """Return median prediction and quantile-based prediction intervals.

        For each sample, collects predictions from all trees and computes
        the configured quantiles to produce lower, median, and upper bounds.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        median : ndarray of shape (n_samples,)
            Median tree prediction (point estimate for P(class=1)).
        lower : ndarray of shape (n_samples,)
            Lower quantile bound.
        upper : ndarray of shape (n_samples,)
            Upper quantile bound.
        width : ndarray of shape (n_samples,)
            Interval width (upper - lower). Wider = more uncertain.
        """
        tree_preds = self._get_tree_predictions(X)

        lo_q = self.quantiles[0] * 100
        med_q = self.quantiles[1] * 100
        hi_q = self.quantiles[2] * 100

        lower = np.percentile(tree_preds, lo_q, axis=1)
        median = np.percentile(tree_preds, med_q, axis=1)
        upper = np.percentile(tree_preds, hi_q, axis=1)
        width = upper - lower

        return median, lower, upper, width

    def decision_function(self, X) -> np.ndarray:
        """Return continuous prediction scores (median tree prediction).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Median tree predictions, continuous in [0, 1].
        """
        tree_preds = self._get_tree_predictions(X)
        median_q = self.quantiles[1] * 100
        return np.percentile(tree_preds, median_q, axis=1)

    def get_params(self, deep=True):
        """Get parameters for this estimator (sklearn-compatible).

        Parameters
        ----------
        deep : bool, default=True
            Not used; included for sklearn API compatibility.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "quantiles": self.quantiles,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        """Set the parameters of this estimator (sklearn-compatible).

        Parameters
        ----------
        **params : dict
            Estimator parameters to set.

        Returns
        -------
        self
            The estimator instance.
        """
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(
                    f"Invalid parameter '{key}' for QuantileForestClassifier."
                )
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        depth = self._effective_max_depth()
        return (
            f"QuantileForestClassifier("
            f"n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth} (effective={depth}), "
            f"min_samples_leaf={self.min_samples_leaf}, "
            f"quantiles={self.quantiles})"
        )

"""
Purge-Aware Stacking Ensemble Classifier
==========================================

A meta-learner (LogisticRegression) trained on out-of-fold base model
predictions with purge-aware OOF generation.

How it works:
    1. Divide training data into sequential blocks (time-series folds).
    2. For each fold, train each base model on the other folds (with purge
       and embargo buffers removed from training) and collect OOF predictions
       on the held-out fold.
    3. Stack OOF predictions into a meta-feature matrix (n_samples x n_models).
    4. Fit a regularized LogisticRegression (meta-learner) on the stacked OOF
       predictions.
    5. Refit all base models on the full training data so that predict_proba
       uses models trained on as much data as possible.

Purge-aware CV:
    For each fold i as test, all other folds form the training set, but rows
    within ``purge_days`` of any test fold boundary are removed from training,
    and ``embargo_days`` rows immediately after the test block are also removed.
    This prevents information leakage through temporal proximity.

Follows EDGE 1 (Regularization-First): meta-learner uses L2-regularized
logistic regression by default.

Follows project standards:
    - ClassifierMixin before BaseEstimator (sklearn 1.6+ tags requirement)
    - Supports sample_weight passthrough to base and meta models
    - Handles base models that only have decision_function (sigmoid fallback)
    - max_depth never exceeds 5 for tree-based base models

Usage:
    from src.phase_12_model_training.stacking_ensemble import (
        StackingEnsembleClassifier,
    )

    clf = StackingEnsembleClassifier()
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
"""

import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


class StackingEnsembleClassifier(ClassifierMixin, BaseEstimator):
    """Purge-aware stacking ensemble classifier.

    Trains base models with out-of-fold (OOF) predictions using purged
    time-series CV, then fits a meta-learner (logistic regression) on those
    OOF predictions.

    Note: ClassifierMixin MUST come before BaseEstimator (sklearn 1.6+ tags
    requirement).

    Parameters
    ----------
    base_model_configs : list of dict, optional
        Each dict must have "name" (str) and "model" (estimator) keys.
        Default: logistic regression, gradient boosting, hist gradient boosting.
    meta_C : float, default=1.0
        Regularization strength for the meta-learner LogisticRegression.
    n_cv_folds : int, default=5
        Number of CV folds for OOF generation.
    purge_days : int, default=10
        Number of rows to purge from training that are within this distance
        of the test fold boundaries.
    embargo_days : int, default=3
        Number of rows to remove from training immediately after the test
        fold.
    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    base_models_ : list of fitted estimators
        Base models refit on full training data.
    meta_learner_ : LogisticRegression
        Fitted meta-learner.
    n_features_in_ : int
        Number of features seen during fit.
    """

    def __init__(
        self,
        base_model_configs: Optional[List[Dict[str, Any]]] = None,
        meta_C: float = 1.0,
        n_cv_folds: int = 5,
        purge_days: int = 10,
        embargo_days: int = 3,
        random_state: int = 42,
    ):
        self.base_model_configs = base_model_configs
        self.meta_C = meta_C
        self.n_cv_folds = n_cv_folds
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.random_state = random_state

    def _get_base_configs(self) -> List[Dict[str, Any]]:
        """Return base model configs, using defaults if none provided."""
        if self.base_model_configs is not None:
            return self.base_model_configs
        return [
            {
                "name": "lr",
                "model": LogisticRegression(
                    C=0.1, max_iter=1000, random_state=self.random_state,
                ),
            },
            {
                "name": "gb",
                "model": GradientBoostingClassifier(
                    n_estimators=75, max_depth=3, random_state=self.random_state,
                ),
            },
            {
                "name": "hgb",
                "model": HistGradientBoostingClassifier(
                    max_iter=75, max_depth=3, random_state=self.random_state,
                ),
            },
        ]

    def fit(self, X, y, sample_weight=None):
        """Fit the stacking ensemble.

        Steps:
            1. Generate OOF predictions from each base model using purged
               time-series CV.
            2. Stack OOF predictions into meta-features matrix
               (n_samples x n_base_models).
            3. Fit meta-learner on stacked OOF predictions.
            4. Refit all base models on full training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Binary target labels (0 or 1).
        sample_weight : array-like of shape (n_samples,), optional
            Per-sample weights passed to both base models and meta-learner.

        Returns
        -------
        self
            Fitted stacking ensemble.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n_samples = X.shape[0]
        self.n_features_in_ = X.shape[1]

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)

        configs = self._get_base_configs()
        n_models = len(configs)

        # Step 1: Generate OOF predictions
        logger.info(
            "Generating OOF predictions with %d folds, purge=%d, embargo=%d",
            self.n_cv_folds, self.purge_days, self.embargo_days,
        )
        oof_matrix, oof_mask = self._generate_oof_predictions(
            X, y, sample_weight=sample_weight,
        )

        # Step 2: Fit meta-learner on OOF predictions (only where mask is True)
        valid_idx = np.where(oof_mask)[0]
        if len(valid_idx) < 10:
            logger.warning(
                "Only %d samples have OOF predictions; meta-learner may be "
                "unreliable. Consider reducing purge_days/embargo_days or "
                "increasing data size.",
                len(valid_idx),
            )

        meta_X = oof_matrix[valid_idx]
        meta_y = y[valid_idx]
        meta_sw = sample_weight[valid_idx] if sample_weight is not None else None

        logger.info(
            "Fitting meta-learner on %d/%d samples (%d base models)",
            len(valid_idx), n_samples, n_models,
        )
        self.meta_learner_ = LogisticRegression(
            C=self.meta_C,
            max_iter=1000,
            random_state=self.random_state,
        )
        self.meta_learner_.fit(meta_X, meta_y, sample_weight=meta_sw)

        # Step 3: Refit all base models on full training data
        logger.info("Refitting %d base models on full training data", n_models)
        self.base_models_ = []
        for cfg in configs:
            model = deepcopy(cfg["model"])
            name = cfg["name"]
            try:
                if sample_weight is not None and self._model_supports_sample_weight(model):
                    model.fit(X, y, sample_weight=sample_weight)
                else:
                    model.fit(X, y)
            except Exception:
                logger.exception("Failed to fit base model '%s'", name)
                raise
            self.base_models_.append(model)

        logger.info("StackingEnsembleClassifier fit complete")
        return self

    def predict_proba(self, X):
        """Predict class probabilities.

        Steps:
            1. Get predictions from each base model.
            2. Stack into meta-features.
            3. Pass through meta-learner.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Class probability estimates. Column 0 = P(class=0),
            Column 1 = P(class=1).
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float)

        meta_features = self._get_base_predictions(X)
        return self.meta_learner_.predict_proba(meta_features)

    def predict(self, X):
        """Predict class labels.

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

    @property
    def classes_(self):
        """Return classes from the meta-learner."""
        self._check_fitted()
        return self.meta_learner_.classes_

    def _check_fitted(self) -> None:
        """Raise if the model has not been fitted yet."""
        if not hasattr(self, "meta_learner_"):
            raise AttributeError(
                "StackingEnsembleClassifier is not fitted yet. "
                "Call fit() before predict/predict_proba."
            )

    def _generate_oof_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate out-of-fold predictions using purged time-series splits.

        For each fold, removes ``purge_days`` rows from training data that
        are within ``purge_days`` of the test fold boundaries, and removes
        ``embargo_days`` rows from immediately after the test fold.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training features.
        y : ndarray of shape (n_samples,)
            Target labels.
        sample_weight : ndarray of shape (n_samples,), optional
            Per-sample weights.

        Returns
        -------
        oof_matrix : ndarray of shape (n_samples, n_base_models)
            OOF predicted probabilities for class 1.
        oof_mask : ndarray of shape (n_samples,)
            Boolean mask; True where the sample received an OOF prediction.
        """
        n_samples = X.shape[0]
        configs = self._get_base_configs()
        n_models = len(configs)

        oof_matrix = np.full((n_samples, n_models), np.nan)
        oof_mask = np.zeros(n_samples, dtype=bool)

        splits = self._create_time_series_splits(n_samples)

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test = X[test_idx]
            sw_train = sample_weight[train_idx] if sample_weight is not None else None

            for model_idx, cfg in enumerate(configs):
                model = deepcopy(cfg["model"])
                name = cfg["name"]

                try:
                    if sw_train is not None and self._model_supports_sample_weight(model):
                        model.fit(X_train, y_train, sample_weight=sw_train)
                    else:
                        model.fit(X_train, y_train)
                except Exception:
                    logger.warning(
                        "Base model '%s' failed on fold %d; filling OOF with 0.5",
                        name, fold_idx,
                    )
                    oof_matrix[test_idx, model_idx] = 0.5
                    continue

                # Get probability predictions for class 1
                preds = self._get_model_proba(model, X_test)
                oof_matrix[test_idx, model_idx] = preds

            oof_mask[test_idx] = True

        # Any remaining NaN values (should be rare) get filled with 0.5
        nan_mask = np.isnan(oof_matrix)
        if nan_mask.any():
            logger.debug(
                "Filling %d NaN OOF entries with 0.5", nan_mask.sum(),
            )
            oof_matrix[nan_mask] = 0.5

        return oof_matrix, oof_mask

    def _create_time_series_splits(
        self, n_samples: int,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create time-series aware CV splits with purging and embargo.

        Divides data into ``n_cv_folds`` sequential blocks. For each fold i
        as test, uses all other blocks as training, but removes:
            - ``purge_days`` rows from training that are within ``purge_days``
              of the test block boundaries (before and after).
            - ``embargo_days`` rows from immediately after the test block.

        Parameters
        ----------
        n_samples : int
            Total number of samples.

        Returns
        -------
        splits : list of (train_idx, test_idx)
            Each tuple contains arrays of training and test indices.
        """
        n_folds = self.n_cv_folds
        fold_size = n_samples // n_folds
        splits = []

        for fold_idx in range(n_folds):
            test_start = fold_idx * fold_size
            test_end = (fold_idx + 1) * fold_size if fold_idx < n_folds - 1 else n_samples

            test_idx = np.arange(test_start, test_end)

            # Start with all indices as potential training
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_idx] = False

            # Purge: remove rows within purge_days of test boundaries
            purge_before_start = max(0, test_start - self.purge_days)
            train_mask[purge_before_start:test_start] = False

            # Embargo: remove rows immediately after test block
            embargo_end = min(n_samples, test_end + self.embargo_days)
            train_mask[test_end:embargo_end] = False

            # Also purge rows before the test block end
            # (symmetric purge on the after-boundary side)
            purge_after_end = min(n_samples, test_end + self.purge_days)
            train_mask[test_end:purge_after_end] = False

            train_idx = np.where(train_mask)[0]

            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))

        return splits

    def _get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from all fitted base models.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        meta_features : ndarray of shape (n_samples, n_base_models)
            Predicted probability for class 1 from each base model.
        """
        n_samples = X.shape[0]
        n_models = len(self.base_models_)
        meta_features = np.zeros((n_samples, n_models))

        for i, model in enumerate(self.base_models_):
            meta_features[:, i] = self._get_model_proba(model, X)

        return meta_features

    @staticmethod
    def _get_model_proba(model: Any, X: np.ndarray) -> np.ndarray:
        """Get P(class=1) from a model, with fallback to decision_function.

        Parameters
        ----------
        model : estimator
            A fitted sklearn-compatible estimator.
        X : ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        proba : ndarray of shape (n_samples,)
            Estimated probability for class 1.
        """
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            return _sigmoid(scores)
        else:
            # Last resort: use predict and treat as probability
            return model.predict(X).astype(float)

    @staticmethod
    def _model_supports_sample_weight(model: Any) -> bool:
        """Check if a model's fit method accepts sample_weight.

        Parameters
        ----------
        model : estimator
            An sklearn-compatible estimator.

        Returns
        -------
        supports : bool
            True if the model's fit method signature includes sample_weight.
        """
        import inspect
        try:
            sig = inspect.signature(model.fit)
            return "sample_weight" in sig.parameters
        except (ValueError, TypeError):
            return False

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
            "base_model_configs": self.base_model_configs,
            "meta_C": self.meta_C,
            "n_cv_folds": self.n_cv_folds,
            "purge_days": self.purge_days,
            "embargo_days": self.embargo_days,
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
        valid_keys = {
            "base_model_configs", "meta_C", "n_cv_folds",
            "purge_days", "embargo_days", "random_state",
        }
        for key, value in params.items():
            if key not in valid_keys:
                raise ValueError(
                    f"Invalid parameter '{key}' for StackingEnsembleClassifier. "
                    f"Valid parameters: {sorted(valid_keys)}"
                )
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        configs = self._get_base_configs()
        names = [c["name"] for c in configs]
        return (
            f"StackingEnsembleClassifier("
            f"base_models={names}, "
            f"meta_C={self.meta_C}, "
            f"n_cv_folds={self.n_cv_folds}, "
            f"purge_days={self.purge_days}, "
            f"embargo_days={self.embargo_days})"
        )

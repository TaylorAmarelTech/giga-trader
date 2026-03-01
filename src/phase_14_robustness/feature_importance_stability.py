"""
GIGA TRADER - Feature Importance Stability Gate
================================================
Measures how stable feature importance rankings are across CV folds.

Trains the model on multiple CV folds, extracts feature importance per fold,
then computes pairwise Spearman rank correlation between folds.

If importance rankings shuffle wildly across folds, the model is likely
memorizing noise rather than learning genuine structure.

Score = mean pairwise Spearman correlation of importance ranks.
Gate rejects if score < threshold.
"""

import logging
import warnings
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


class FeatureImportanceStabilityGate:
    """
    Feature importance stability gate. Trains model on multiple CV folds,
    computes feature importance per fold, then measures rank correlation
    (Spearman) between folds. Unstable importance -> model is likely overfit.

    Score = mean pairwise Spearman correlation across folds.
    Gate rejects if score < threshold.

    Parameters
    ----------
    n_folds : int
        Number of CV folds (default 5).
    threshold : float
        Minimum acceptable stability score (default 0.5).
    importance_method : str
        How to extract feature importance:
        - "auto": Try feature_importances_ first, then coef_, then permutation
        - "permutation": Always use permutation importance
        - "native": Only use model's native importance (feature_importances_ or coef_)
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_folds: int = 5,
        threshold: float = 0.5,
        importance_method: str = "auto",
        random_state: int = 42,
    ):
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")
        if importance_method not in ("auto", "permutation", "native"):
            raise ValueError(
                f"importance_method must be 'auto', 'permutation', or 'native', "
                f"got '{importance_method}'"
            )
        self.n_folds = n_folds
        self.threshold = threshold
        self.importance_method = importance_method
        self.random_state = random_state

    def run(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict:
        """
        Run feature importance stability analysis.

        Args:
            model: Unfitted sklearn-compatible model (will be cloned per fold).
            X: Feature matrix of shape (n_samples, n_features).
            y: Labels of shape (n_samples,).

        Returns:
            Dict with keys:
              - "score": float 0-1 (mean pairwise Spearman rank correlation)
              - "passed": bool (score >= threshold)
              - "n_folds": int
              - "n_features": int
              - "pairwise_correlations": List[float]
              - "most_stable_features": List[int] (indices of top-5 most stable)
              - "least_stable_features": List[int] (indices of bottom-5 least stable)
              - "skipped": bool
              - "reason": str (if skipped)
        """
        n_samples, n_features = X.shape

        # Minimum samples check: need enough for at least 3 folds
        min_per_fold = 5
        effective_folds = min(self.n_folds, n_samples // min_per_fold)
        if effective_folds < 3:
            logger.warning(
                "Too few samples (%d) for %d folds, skipping FI stability",
                n_samples, self.n_folds,
            )
            return self._skipped_result(
                n_features=n_features,
                reason=f"too_few_samples ({n_samples} samples, need >= {min_per_fold * 3})",
            )

        # Ensure at least 2 classes
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            return self._skipped_result(
                n_features=n_features,
                reason="single_class",
            )

        # Collect importance vectors per fold
        kf = KFold(
            n_splits=effective_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        importance_per_fold: List[np.ndarray] = []
        folds_trained = 0

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]

            # Need both classes in training fold
            if len(np.unique(y_train)) < 2:
                logger.debug("Fold %d has single class, skipping", fold_idx)
                continue

            try:
                fold_model = clone(model)
                fold_model.fit(X_train, y_train)
                imp = self._get_feature_importance(
                    fold_model, X_train, y_train
                )
                if imp is not None and len(imp) == n_features:
                    importance_per_fold.append(imp)
                    folds_trained += 1
            except Exception as e:
                logger.debug("Fold %d failed: %s", fold_idx, e)
                continue

        if len(importance_per_fold) < 3:
            return self._skipped_result(
                n_features=n_features,
                reason=f"too_few_successful_folds ({len(importance_per_fold)} of {effective_folds})",
            )

        # Compute Spearman rank correlations between all fold pairs
        # We correlate RANKS of importance, not raw values
        n_successful = len(importance_per_fold)
        pairwise_correlations: List[float] = []

        for i, j in combinations(range(n_successful), 2):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, _ = spearmanr(importance_per_fold[i], importance_per_fold[j])

            # Handle NaN (can occur if all importances are identical)
            if np.isnan(rho):
                rho = 0.0
            pairwise_correlations.append(float(rho))

        score = float(np.mean(pairwise_correlations))
        # Clamp to [0, 1]
        score = float(max(0.0, min(1.0, score)))

        # Identify most/least stable features by rank variance across folds
        rank_matrix = np.zeros((n_successful, n_features))
        for i, imp in enumerate(importance_per_fold):
            # rankdata: higher importance -> higher rank
            rank_matrix[i] = _rankdata(imp)

        rank_variance = np.var(rank_matrix, axis=0)  # shape (n_features,)

        # Most stable = lowest rank variance
        n_report = min(5, n_features)
        most_stable = np.argsort(rank_variance)[:n_report].tolist()
        least_stable = np.argsort(rank_variance)[-n_report:][::-1].tolist()

        passed = score >= self.threshold

        logger.info(
            "FI Stability: score=%.3f, passed=%s, folds=%d/%d, features=%d",
            score, passed, n_successful, effective_folds, n_features,
        )

        return {
            "score": score,
            "passed": passed,
            "n_folds": n_successful,
            "n_features": n_features,
            "pairwise_correlations": pairwise_correlations,
            "most_stable_features": most_stable,
            "least_stable_features": least_stable,
            "skipped": False,
            "reason": "",
        }

    def _get_feature_importance(
        self, model: Any, X: np.ndarray, y: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Extract feature importance from a fitted model.

        Returns array of shape (n_features,) with importance values,
        or None if importance cannot be extracted.

        Priority (if importance_method == "auto"):
        1. model.feature_importances_ (tree-based models)
        2. np.abs(model.coef_).ravel() (linear models)
        3. Permutation importance on training data (fallback)

        For models with a .estimator attribute (e.g. CalibratedClassifierCV),
        we attempt to get importance from the inner estimator first.
        """
        # Unwrap inner estimator if present (e.g. CalibratedClassifierCV)
        inner = model
        if hasattr(model, "estimator") and model.estimator is not None:
            inner = model.estimator
        # Also check calibrated_classifiers_ (fitted CalibratedClassifierCV)
        if hasattr(model, "calibrated_classifiers_"):
            try:
                inner = model.calibrated_classifiers_[0].estimator
            except (IndexError, AttributeError):
                pass

        if self.importance_method in ("auto", "native"):
            # Try native feature_importances_ (tree-based)
            if hasattr(inner, "feature_importances_"):
                imp = np.array(inner.feature_importances_).ravel()
                if len(imp) == X.shape[1]:
                    return imp

            # Try coef_ (linear models)
            if hasattr(inner, "coef_"):
                imp = np.abs(np.array(inner.coef_)).ravel()
                # Multi-class: coef_ shape is (n_classes, n_features)
                if imp.shape[0] != X.shape[1] and hasattr(inner.coef_, "shape"):
                    if len(inner.coef_.shape) == 2:
                        imp = np.abs(inner.coef_).mean(axis=0).ravel()
                if len(imp) == X.shape[1]:
                    return imp

            if self.importance_method == "native":
                logger.debug(
                    "Model %s has no native importance, returning None",
                    type(inner).__name__,
                )
                return None

        # Permutation importance fallback (auto or explicit permutation)
        if self.importance_method in ("auto", "permutation"):
            try:
                from sklearn.inspection import permutation_importance

                result = permutation_importance(
                    model,
                    X,
                    y,
                    n_repeats=5,
                    random_state=self.random_state,
                    scoring="accuracy",
                    n_jobs=1,
                )
                imp = result.importances_mean
                if len(imp) == X.shape[1]:
                    return imp
            except Exception as e:
                logger.debug("Permutation importance failed: %s", e)

        return None

    def _skipped_result(self, n_features: int, reason: str) -> Dict:
        """Return a result dict for skipped analysis."""
        logger.info("FI Stability skipped: %s", reason)
        return {
            "score": 0.0,
            "passed": False,
            "n_folds": 0,
            "n_features": n_features,
            "pairwise_correlations": [],
            "most_stable_features": [],
            "least_stable_features": [],
            "skipped": True,
            "reason": reason,
        }


def _rankdata(arr: np.ndarray) -> np.ndarray:
    """
    Simple rank assignment (average method).

    Avoids importing scipy.stats.rankdata for a lightweight operation.
    Ties get the average of their would-be ranks.
    """
    n = len(arr)
    order = np.argsort(arr)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)

    # Average ranks for ties
    sorted_arr = arr[order]
    i = 0
    while i < n:
        j = i
        while j < n and sorted_arr[j] == sorted_arr[i]:
            j += 1
        if j > i + 1:
            avg_rank = np.mean(ranks[order[i:j]])
            ranks[order[i:j]] = avg_rank
        i = j

    return ranks

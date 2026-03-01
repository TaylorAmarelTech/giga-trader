"""
GIGA TRADER - Knockoff Feature Gate
====================================
FDR-controlled feature discovery using knockoff features (permuted copies).

Generates permuted copies (knockoffs) of each feature, trains a model with
real + knockoff features combined, and checks if any knockoff appears in
the top-K most important features. If knockoffs rank highly, the model is
likely memorizing noise rather than learning true signal.

Score = fraction of knockoffs that DON'T appear in top-K importance.
A score of 1.0 means no knockoffs leaked into top features (good).

Reference: Barber & Candes (2015), "Controlling the false discovery rate
via knockoffs."
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.base import clone

logger = logging.getLogger(__name__)

# Optional import for permutation importance fallback
try:
    from sklearn.inspection import permutation_importance as sklearn_perm_imp
    _HAS_PERM_IMP = True
except ImportError:
    _HAS_PERM_IMP = False


class KnockoffGate:
    """
    Knockoff feature gate. Generates permuted copies (knockoffs) of each feature,
    trains a model with real + knockoff features, and checks if any knockoff
    appears in the top-K most important features. If knockoffs rank highly,
    the model is likely memorizing noise.

    Score = fraction of knockoffs that DON'T appear in top-K importance.
    A score of 1.0 means no knockoffs leaked into top features (good).

    Parameters
    ----------
    top_k_fraction : float
        Fraction of features considered "top-K" (0.3 = top 30%).
    n_repeats : int
        Number of knockoff+train repeats for averaging.
    random_state : int
        Random seed.
    """

    def __init__(
        self,
        top_k_fraction: float = 0.3,
        n_repeats: int = 3,
        random_state: int = 42,
    ):
        self.top_k_fraction = top_k_fraction
        self.n_repeats = n_repeats
        self.random_state = random_state

    def run(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict:
        """
        Run knockoff gate analysis.

        Args:
            model: Unfitted sklearn-compatible model (will be cloned per repeat)
            X: Feature matrix of shape (n_samples, n_features)
            y: Labels

        Returns:
            Dict with keys:
              - "score": float 0-1 (1 = no knockoffs in top-K, good)
              - "passed": bool (score >= 0.7 by default)
              - "n_knockoffs_in_top_k": float (average across repeats)
              - "n_features": int (original feature count)
              - "top_k": int (number of features considered "top")
              - "knockoff_importance_ratios": List[float] (per-repeat ratio
                of knockoff importance to real importance)
              - "skipped": bool
              - "reason": str (if skipped)

        Algorithm per repeat:
        1. Generate knockoff features: For each feature column, create a
           permuted copy
        2. Concatenate: X_augmented = [X_real | X_knockoff]
           (n_samples x 2*n_features)
        3. Train model on X_augmented, y
        4. Extract feature importance (native or permutation-based)
        5. Count how many knockoff features appear in top-K importance
        6. Score = 1 - (avg knockoffs in top-K / top-K)
        """
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape

        # Guard: too few samples
        if n_samples < 100:
            return {
                "score": 0.0,
                "passed": False,
                "n_knockoffs_in_top_k": 0.0,
                "n_features": n_features,
                "top_k": 0,
                "knockoff_importance_ratios": [],
                "skipped": True,
                "reason": f"Too few samples ({n_samples} < 100)",
            }

        # Guard: too few features
        if n_features < 5:
            return {
                "score": 0.0,
                "passed": False,
                "n_knockoffs_in_top_k": 0.0,
                "n_features": n_features,
                "top_k": 0,
                "knockoff_importance_ratios": [],
                "skipped": True,
                "reason": f"Too few features ({n_features} < 5)",
            }

        # Total features in augmented space = 2 * n_features
        # top_k is computed over the full augmented feature set
        total_augmented_features = 2 * n_features
        top_k = max(1, int(total_augmented_features * self.top_k_fraction))

        rng = np.random.RandomState(self.random_state)

        knockoffs_in_top_k_per_repeat: List[int] = []
        importance_ratios: List[float] = []

        for repeat_idx in range(self.n_repeats):
            # Derive a child RNG per repeat for reproducibility
            repeat_seed = rng.randint(0, 2**31)
            repeat_rng = np.random.RandomState(repeat_seed)

            try:
                # Step 1: Generate knockoff features
                X_knockoff = self._generate_knockoffs(X, repeat_rng)

                # Step 2: Concatenate [real | knockoff]
                X_augmented = np.hstack([X, X_knockoff])

                # Step 3: Clone and train model
                model_clone = clone(model)
                model_clone.fit(X_augmented, y)

                # Step 4: Extract feature importance
                importances = self._get_importance(
                    model_clone, X_augmented, y
                )

                if importances is None:
                    logger.warning(
                        f"  Knockoff repeat {repeat_idx}: could not extract "
                        f"importance, skipping"
                    )
                    continue

                if len(importances) != total_augmented_features:
                    logger.warning(
                        f"  Knockoff repeat {repeat_idx}: importance shape "
                        f"mismatch ({len(importances)} != "
                        f"{total_augmented_features}), skipping"
                    )
                    continue

                # Step 5: Count knockoff features in top-K
                top_k_indices = np.argsort(importances)[::-1][:top_k]
                # Knockoff features are indices [n_features, 2*n_features)
                n_knockoffs_in_top = int(
                    sum(1 for idx in top_k_indices if idx >= n_features)
                )
                knockoffs_in_top_k_per_repeat.append(n_knockoffs_in_top)

                # Compute importance ratio: mean(knockoff) / mean(real)
                real_imp = importances[:n_features]
                knock_imp = importances[n_features:]
                mean_real = float(np.mean(real_imp))
                mean_knock = float(np.mean(knock_imp))
                ratio = mean_knock / (mean_real + 1e-10)
                importance_ratios.append(float(ratio))

            except Exception as e:
                logger.warning(
                    f"  Knockoff repeat {repeat_idx} failed: {e}"
                )
                continue

        if len(knockoffs_in_top_k_per_repeat) == 0:
            return {
                "score": 0.0,
                "passed": False,
                "n_knockoffs_in_top_k": 0.0,
                "n_features": n_features,
                "top_k": top_k,
                "knockoff_importance_ratios": [],
                "skipped": True,
                "reason": "All repeats failed",
            }

        # Step 6: Score = 1 - (avg knockoffs in top-K / top-K)
        avg_knockoffs_in_top_k = float(
            np.mean(knockoffs_in_top_k_per_repeat)
        )
        score = float(
            max(0.0, min(1.0, 1.0 - avg_knockoffs_in_top_k / top_k))
        )
        passed = score >= 0.7

        logger.info(
            f"  [KnockoffGate] score={score:.3f}, "
            f"avg_knockoffs_in_top_{top_k}={avg_knockoffs_in_top_k:.1f}, "
            f"passed={passed}"
        )

        return {
            "score": score,
            "passed": passed,
            "n_knockoffs_in_top_k": avg_knockoffs_in_top_k,
            "n_features": n_features,
            "top_k": top_k,
            "knockoff_importance_ratios": importance_ratios,
            "skipped": False,
            "reason": "",
        }

    def _generate_knockoffs(
        self, X: np.ndarray, rng: np.random.RandomState
    ) -> np.ndarray:
        """
        Generate knockoff features by independently permuting each column.

        Returns X_knockoff of same shape as X, where each column is a
        random permutation of the corresponding column in X.
        """
        n_samples, n_features = X.shape
        X_knockoff = np.empty_like(X)
        for j in range(n_features):
            perm = rng.permutation(n_samples)
            X_knockoff[:, j] = X[perm, j]
        return X_knockoff

    def _get_importance(
        self, model: Any, X: np.ndarray, y: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Extract feature importance from fitted model.

        Tries in order:
        1. feature_importances_ (tree-based models)
        2. abs(coef_) (linear models)
        3. Unwrap CalibratedClassifierCV to access .estimator
        4. Skip VotingClassifier (no meaningful per-feature importance)
        5. Permutation importance fallback

        Returns:
            np.ndarray of shape (n_features,) or None if extraction fails.
        """
        actual_model = model

        # Unwrap CalibratedClassifierCV
        if hasattr(model, "calibrated_classifiers_"):
            # CalibratedClassifierCV stores fitted calibrated classifiers
            # Try to get the underlying estimator
            if hasattr(model, "estimator") and model.estimator is not None:
                actual_model = model.estimator
            elif (
                hasattr(model, "calibrated_classifiers_")
                and len(model.calibrated_classifiers_) > 0
            ):
                inner = model.calibrated_classifiers_[0]
                if hasattr(inner, "estimator"):
                    actual_model = inner.estimator

        # Skip VotingClassifier - no meaningful per-feature importance
        from sklearn.ensemble import VotingClassifier
        if isinstance(actual_model, VotingClassifier):
            return None

        # Try feature_importances_ (tree-based)
        if hasattr(actual_model, "feature_importances_"):
            return np.asarray(actual_model.feature_importances_)

        # Try coef_ (linear models)
        if hasattr(actual_model, "coef_"):
            coef = np.asarray(actual_model.coef_)
            if coef.ndim > 1:
                return np.abs(coef).mean(axis=0)
            return np.abs(coef.ravel())

        # Permutation importance fallback
        if _HAS_PERM_IMP:
            try:
                perm_result = sklearn_perm_imp(
                    model, X, y,
                    n_repeats=5,
                    scoring="accuracy",
                    random_state=42,
                    n_jobs=1,
                )
                return perm_result.importances_mean
            except Exception as e:
                logger.warning(
                    f"  Permutation importance fallback failed: {e}"
                )
                return None

        return None

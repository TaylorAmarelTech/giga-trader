"""Bayesian Model Averaging (BMA) for ensemble model combination.

Computes BIC-based posterior weights so that models with lower BIC
(better fit relative to complexity) receive higher combination weight.
"""

import logging
import numpy as np
from typing import Any, Dict

logger = logging.getLogger(__name__)


class BayesianModelAverager:
    """
    Bayesian Model Averaging (BMA). Computes BIC-based posterior weights
    for model combination. Models with lower BIC get higher weight.

    BIC = n * ln(MSE) + k * ln(n)
    where n = sample size, k = effective parameters, MSE = mean squared error

    Posterior weight: w_i = exp(-BIC_i/2) / sum(exp(-BIC_j/2))
    """

    def __init__(self, min_weight: float = 0.01):
        """
        Args:
            min_weight: Minimum posterior weight per model (floor).
        """
        self.min_weight = min_weight
        self._weights: Dict[str, float] = {}
        self._models: Dict[str, Any] = {}
        self._bics: Dict[str, float] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_weights(
        cls,
        weights: Dict[str, float],
        models: Dict[str, Any],
    ) -> "BayesianModelAverager":
        """Reconstruct a fitted BMA from pre-computed weights + model objects.

        This avoids re-fitting when weights were already computed during training
        and serialized alongside the model bundle.

        Args:
            weights: Model name → posterior weight mapping (must sum to ~1.0).
            models: Model name → fitted sklearn estimator (same keys as weights).

        Returns:
            A fitted BayesianModelAverager ready for ``predict_proba()``.
        """
        bma = cls()
        bma._weights = dict(weights)
        bma._models = dict(models)
        bma._fitted = bool(weights and models)
        return bma

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        models: Dict[str, Any],
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "BayesianModelAverager":
        """Compute BMA weights from validation data.

        Args:
            models: Dict mapping model_name to FITTED sklearn classifier.
            X_val: Validation features, shape (n_samples, n_features).
            y_val: Validation labels (binary 0/1), shape (n_samples,).

        Returns:
            self (for chaining).

        For each model:
          1. Get predicted probabilities (positive class).
          2. Compute MSE = mean((y - p)^2)  (Brier score).
          3. Estimate k (effective parameters).
          4. BIC = n * ln(MSE + 1e-10) + k * ln(n).
          5. Posterior: w_i = exp(-BIC_i / 2) / sum(exp(-BIC_j / 2)).
        """
        if not models:
            raise ValueError("models dict must not be empty")

        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val).ravel()
        n = len(y_val)
        n_features = X_val.shape[1] if X_val.ndim == 2 else 1

        self._models = dict(models)
        raw_bics: Dict[str, float] = {}

        for name, model in models.items():
            # Step 1 -- predicted probabilities
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_val)
                if proba.ndim == 2:
                    p = proba[:, 1]
                else:
                    p = proba
            else:
                # Fall back to decision_function -> sigmoid
                df = model.decision_function(X_val)
                p = 1.0 / (1.0 + np.exp(-df))

            # Step 2 -- Brier score (MSE)
            mse = np.mean((y_val - p) ** 2)

            # Step 3 -- effective parameter count
            k = self._estimate_effective_params(model, n_features)

            # Step 4 -- BIC
            bic = n * np.log(mse + 1e-10) + k * np.log(n)
            raw_bics[name] = bic
            logger.debug("BMA %s: MSE=%.4f  k=%d  BIC=%.2f", name, mse, k, bic)

        self._bics = raw_bics

        # Step 5 -- posterior weights via log-sum-exp for numerical stability
        bic_arr = np.array(list(raw_bics.values()))
        log_weights = -0.5 * bic_arr
        log_weights -= np.max(log_weights)  # shift for stability
        weights = np.exp(log_weights)
        weights /= weights.sum()

        # Apply min_weight floor and renormalise
        names = list(raw_bics.keys())
        weight_dict: Dict[str, float] = {}
        for i, name in enumerate(names):
            weight_dict[name] = float(max(weights[i], self.min_weight))

        total = sum(weight_dict.values())
        self._weights = {k: v / total for k, v in weight_dict.items()}

        self._fitted = True
        logger.info(
            "BMA fitted on %d models. Weights: %s",
            len(models),
            {k: round(v, 4) for k, v in self._weights.items()},
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """BMA-weighted prediction.

        Returns:
            Array of shape (n_samples, 2) with [P(class=0), P(class=1)].
        """
        if not self._fitted:
            raise RuntimeError("BayesianModelAverager has not been fitted yet.")

        X = np.asarray(X)
        n = X.shape[0]
        combined = np.zeros(n, dtype=np.float64)

        for name, model in self._models.items():
            w = self._weights[name]
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                if proba.ndim == 2:
                    p = proba[:, 1]
                else:
                    p = proba
            else:
                df = model.decision_function(X)
                p = 1.0 / (1.0 + np.exp(-df))
            combined += w * p

        out = np.column_stack([1.0 - combined, combined])
        return out

    def get_weights(self) -> Dict[str, float]:
        """Return posterior model weights."""
        return self._weights.copy()

    def get_bics(self) -> Dict[str, float]:
        """Return BIC values per model."""
        return self._bics.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_effective_params(model: Any, n_features: int) -> int:
        """Estimate effective number of parameters for a model.

        Heuristics:
        - LogisticRegression / SGD / linear: n_features + 1 (coefficients + intercept)
        - Tree-based ensembles: n_estimators * max_depth (rough)
        - Single tree: max_depth * 2
        - Default: n_features
        """
        cls_name = type(model).__name__.lower()

        # Linear models
        if any(tok in cls_name for tok in ("logistic", "sgd", "ridge", "lasso", "elastic", "svc", "linear")):
            return n_features + 1

        # Naive Bayes, LDA -- very few effective params
        if any(tok in cls_name for tok in ("nb", "naivebayes", "gaussiannb", "lda", "discriminant")):
            return max(n_features // 2, 2)

        # Ensemble tree-based
        n_estimators = getattr(model, "n_estimators", None)
        max_depth = getattr(model, "max_depth", None)
        if n_estimators is not None and max_depth is not None:
            return int(n_estimators * max_depth)
        if n_estimators is not None:
            return int(n_estimators * 3)  # assume depth ~3

        # Single tree
        if max_depth is not None:
            return int(max_depth * 2)

        return n_features

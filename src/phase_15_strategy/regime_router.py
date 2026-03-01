"""
GIGA TRADER - Regime Router
==============================
Train separate models for different volatility regimes, route at inference time.

Concept:
  - Financial markets exhibit distinct regimes (low/medium/high volatility)
  - Models trained on regime-specific data can capture regime-specific patterns
  - At inference time, current regime determines which model generates predictions

Usage:
  router = RegimeRouter(base_model=LogisticRegression(C=0.1), regime_method="vix_quartile")
  router.fit(X_train, y_train, vol_series=vix_train)
  proba = router.predict_proba(X_test, vol_series=vix_test)

Graceful degradation:
  - If vol_series is None (fit or predict), falls back to a single global model
  - If a regime has too few samples (<min_samples_per_regime), uses global model
  - Global model is always trained as a fallback
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

logger = logging.getLogger(__name__)


class RegimeRouter(ClassifierMixin, BaseEstimator):
    """
    Regime-conditional model router. Trains separate models for different
    volatility regimes (e.g., low/medium/high VIX) and routes predictions
    to the appropriate regime model at inference time.

    Note: ClassifierMixin before BaseEstimator for sklearn 1.6+ compatibility.
    """

    # Supported regime methods
    _VALID_METHODS = ("vix_quartile", "vix_fixed", "binary_vol")

    def __init__(
        self,
        base_model: Any = None,
        regime_method: str = "vix_quartile",
        min_samples_per_regime: int = 100,
        random_state: int = 42,
    ):
        """
        Args:
            base_model: Unfitted sklearn-compatible model to clone per regime.
                       If None, uses LogisticRegression(C=0.1).
            regime_method: How to split regimes:
                "vix_quartile": Low (<25th pct), Med (25-75th), High (>75th pct)
                "vix_fixed": Low (<15), Med (15-25), High (>25) based on VIX levels
                "binary_vol": Low (<median), High (>=median)
            min_samples_per_regime: Minimum samples required per regime. If below
                                   this, fall back to global model for that regime.
            random_state: Random seed.
        """
        self.base_model = base_model
        self.regime_method = regime_method
        self.min_samples_per_regime = min_samples_per_regime
        self.random_state = random_state

    def fit(self, X, y, vol_series=None, sample_weight=None):
        """
        Fit regime-specific models.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Labels (n_samples,).
            vol_series: Volatility series (same length as X). Used to determine
                       regime. If None, falls back to a single global model.
            sample_weight: Optional sample weights (n_samples,).

        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight).ravel()

        # Resolve base model
        base = self._resolve_base_model()

        # Always train global model as fallback
        self.global_model_ = clone(base)
        if sample_weight is not None:
            self.global_model_.fit(X, y, sample_weight=sample_weight)
        else:
            self.global_model_.fit(X, y)

        # Initialize regime tracking
        self.regime_models_: Dict[int, Any] = {}
        self.regime_samples_: Dict[int, int] = {}
        self.fallback_regimes_: List[int] = []
        self.thresholds_: Dict[str, float] = {}

        # If no vol_series, we only have the global model
        if vol_series is None:
            logger.info("No vol_series provided — using global model only.")
            self.n_regimes_ = 0
            return self

        vol_series = np.asarray(vol_series).ravel()
        if len(vol_series) != len(y):
            raise ValueError(
                f"vol_series length ({len(vol_series)}) must match X rows ({len(y)})"
            )

        # Compute and store thresholds from training data
        self.thresholds_ = self._compute_thresholds(vol_series)

        # Assign regimes
        regimes = self._assign_regimes(vol_series)
        unique_regimes = sorted(set(regimes))
        self.n_regimes_ = len(unique_regimes)

        logger.info(
            "Regime split (%s): %d regimes found",
            self.regime_method,
            self.n_regimes_,
        )

        # Train per-regime models
        for regime_id in unique_regimes:
            mask = regimes == regime_id
            n_samples = int(mask.sum())
            self.regime_samples_[regime_id] = n_samples

            if n_samples < self.min_samples_per_regime:
                logger.warning(
                    "Regime %d has %d samples (< %d minimum) — using global fallback",
                    regime_id,
                    n_samples,
                    self.min_samples_per_regime,
                )
                self.fallback_regimes_.append(regime_id)
                continue

            # Check that both classes are present
            y_regime = y[mask]
            if len(np.unique(y_regime)) < 2:
                logger.warning(
                    "Regime %d has only one class — using global fallback",
                    regime_id,
                )
                self.fallback_regimes_.append(regime_id)
                continue

            model = clone(base)
            if sample_weight is not None:
                model.fit(X[mask], y_regime, sample_weight=sample_weight[mask])
            else:
                model.fit(X[mask], y_regime)

            self.regime_models_[regime_id] = model
            logger.info(
                "Regime %d: trained on %d samples", regime_id, n_samples
            )

        return self

    def predict_proba(self, X, vol_series=None):
        """
        Predict using regime-specific model.

        Args:
            X: Feature matrix (n_samples, n_features).
            vol_series: Volatility values for routing. Can be:
                - Array of same length as X (per-sample routing)
                - Single float (all samples use same regime)
                - None (use global model)

        Returns:
            Array of shape (n_samples, 2).
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]

        # No vol_series or no regime models — use global
        if vol_series is None or self.n_regimes_ == 0:
            return self.global_model_.predict_proba(X)

        # Handle single float: broadcast to all samples
        vol_series = np.asarray(vol_series)
        if vol_series.ndim == 0:
            vol_series = np.full(n_samples, float(vol_series))
        vol_series = vol_series.ravel()

        if len(vol_series) != n_samples:
            raise ValueError(
                f"vol_series length ({len(vol_series)}) must match X rows ({n_samples})"
            )

        regimes = self._assign_regimes(vol_series)

        # Allocate output
        proba = np.empty((n_samples, 2), dtype=np.float64)

        for regime_id in np.unique(regimes):
            mask = regimes == regime_id
            model = self.regime_models_.get(regime_id, self.global_model_)
            proba[mask] = model.predict_proba(X[mask])

        return proba

    def predict(self, X, vol_series=None):
        """Predict class labels."""
        proba = self.predict_proba(X, vol_series)
        return (proba[:, 1] >= 0.5).astype(int)

    @property
    def classes_(self):
        """Class labels."""
        return np.array([0, 1])

    def _resolve_base_model(self):
        """Return the base model, defaulting to LogisticRegression(C=0.1)."""
        if self.base_model is not None:
            return self.base_model
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            C=0.1, max_iter=500, random_state=self.random_state
        )

    def _compute_thresholds(self, vol_series: np.ndarray) -> Dict[str, float]:
        """
        Compute regime thresholds from the training vol_series.

        Returns:
            Dict of threshold values used for regime assignment.
        """
        if self.regime_method == "vix_quartile":
            return {
                "q25": float(np.percentile(vol_series, 25)),
                "q75": float(np.percentile(vol_series, 75)),
            }
        elif self.regime_method == "binary_vol":
            return {
                "median": float(np.median(vol_series)),
            }
        elif self.regime_method == "vix_fixed":
            return {
                "low_threshold": 15.0,
                "high_threshold": 25.0,
            }
        else:
            raise ValueError(
                f"Unknown regime_method: {self.regime_method!r}. "
                f"Must be one of {self._VALID_METHODS}"
            )

    def _assign_regimes(self, vol_series: np.ndarray) -> np.ndarray:
        """
        Assign regime labels to each sample based on vol_series.

        Returns:
            Array of regime labels:
              - vix_quartile: 0=low, 1=medium, 2=high
              - vix_fixed: 0=low (<15), 1=medium (15-25), 2=high (>25)
              - binary_vol: 0=low, 1=high
        """
        vol_series = np.asarray(vol_series).ravel()
        regimes = np.zeros(len(vol_series), dtype=int)

        if self.regime_method == "vix_quartile":
            q25 = self.thresholds_["q25"]
            q75 = self.thresholds_["q75"]
            regimes[vol_series >= q75] = 2
            regimes[(vol_series >= q25) & (vol_series < q75)] = 1
            # regime 0 is the default (< q25)

        elif self.regime_method == "vix_fixed":
            low_t = self.thresholds_["low_threshold"]
            high_t = self.thresholds_["high_threshold"]
            regimes[vol_series >= high_t] = 2
            regimes[(vol_series >= low_t) & (vol_series < high_t)] = 1

        elif self.regime_method == "binary_vol":
            median = self.thresholds_["median"]
            regimes[vol_series >= median] = 1

        else:
            raise ValueError(
                f"Unknown regime_method: {self.regime_method!r}. "
                f"Must be one of {self._VALID_METHODS}"
            )

        return regimes

    def get_regime_stats(self) -> Dict:
        """
        Return statistics about fitted regime models.

        Returns:
            Dict with keys:
              - n_regimes: Number of distinct regimes found
              - samples_per_regime: Dict[int, int] mapping regime_id to sample count
              - used_global_fallback: List of regime IDs that fell back to global
              - thresholds: Dict of threshold values used for splitting
        """
        return {
            "n_regimes": getattr(self, "n_regimes_", 0),
            "samples_per_regime": dict(getattr(self, "regime_samples_", {})),
            "used_global_fallback": list(getattr(self, "fallback_regimes_", [])),
            "thresholds": dict(getattr(self, "thresholds_", {})),
        }

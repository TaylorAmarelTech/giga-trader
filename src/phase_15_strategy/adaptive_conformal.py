"""
Adaptive Conformal Inference (ACI) Position Sizer — dynamically adjusts
coverage levels based on recent prediction errors, addressing the
non-stationarity of financial data.

Standard conformal prediction assumes exchangeability (IID), which fails for
financial time series.  ACI (Gibbs & Candes, 2021) tracks online miscoverage
and adapts alpha so that empirical coverage stays near the target even when
the data distribution drifts.

Drop-in replacement for ``ConformalPositionSizer`` — exposes the same
``size_single(X, base_size)`` interface.
"""

import logging
from collections import deque
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Optional dependency ----------------------------------------------------------
try:
    from mapie.classification import SplitConformalClassifier

    _HAS_MAPIE = True
except ImportError:
    _HAS_MAPIE = False


class AdaptiveConformalSizer:
    """Position sizer that adapts conformal alpha online.

    After each trading day the caller invokes :meth:`update` with the
    predicted and true labels.  The class adjusts *alpha* so that the
    running empirical miscoverage rate converges to ``target_alpha``, and
    feeds the resulting prediction-set width into position sizing.

    Parameters
    ----------
    target_alpha : float
        Desired miscoverage rate (default 0.10 = 90 % coverage).
    gamma : float
        Step size for the ACI alpha update rule.
    min_alpha, max_alpha : float
        Hard bounds for the dynamic alpha to prevent degenerate sets.
    min_size_fraction : float
        Floor for position scaling when the prediction set is wide.
    decay_rate : float
        Exponential decay weight for recent errors (0 < decay < 1).
    lookback_window : int
        Number of recent observations kept in the error buffer.
    """

    def __init__(
        self,
        target_alpha: float = 0.10,
        gamma: float = 0.01,
        min_alpha: float = 0.01,
        max_alpha: float = 0.30,
        min_size_fraction: float = 0.25,
        decay_rate: float = 0.95,
        lookback_window: int = 252,
    ) -> None:
        if not 0.0 < target_alpha < 1.0:
            raise ValueError(f"target_alpha must be in (0, 1), got {target_alpha}")
        if not 0.0 < min_size_fraction <= 1.0:
            raise ValueError(
                f"min_size_fraction must be in (0, 1], got {min_size_fraction}"
            )
        if not min_alpha < max_alpha:
            raise ValueError("min_alpha must be less than max_alpha")

        self.target_alpha = target_alpha
        self.gamma = gamma
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.min_size_fraction = min_size_fraction
        self.decay_rate = decay_rate
        self.lookback_window = lookback_window

        # Running state
        self._alpha_t: float = target_alpha
        self._errors: deque = deque(maxlen=lookback_window)
        self._interval_widths: deque = deque(maxlen=lookback_window)
        self._fitted: bool = False
        self._mapie_model: Optional[Any] = None
        self._base_model: Optional[Any] = None
        self._cal_scores: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def fit(
        self,
        model: Any,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> "AdaptiveConformalSizer":
        """Calibrate on held-out data.

        Args:
            model: Fitted sklearn classifier with ``predict_proba``.
            X_cal: Calibration features (2-D array).
            y_cal: Calibration labels (1-D array, 0/1).

        Returns:
            self (for method chaining).
        """
        if not _HAS_MAPIE:
            logger.warning(
                "mapie not installed — AdaptiveConformalSizer will return "
                "base_size for all predictions (no conformal scaling)."
            )
            self._fitted = False
            return self

        X_cal = np.asarray(X_cal)
        y_cal = np.asarray(y_cal)

        if X_cal.ndim == 1:
            X_cal = X_cal.reshape(1, -1)

        if len(np.unique(y_cal)) < 2:
            logger.warning(
                "Calibration data has fewer than 2 classes — "
                "cannot fit conformal predictor."
            )
            self._fitted = False
            return self

        if not hasattr(model, "classes_"):
            model.classes_ = np.array([0, 1])

        self._base_model = model

        # Store nonconformity scores for later adaptive re-quantiling
        try:
            probas = model.predict_proba(X_cal)
            # LAC score: 1 - p(true class)
            self._cal_scores = 1.0 - probas[np.arange(len(y_cal)), y_cal.astype(int)]
        except Exception:
            self._cal_scores = None

        # Initial MAPIE fit at starting alpha
        confidence = 1.0 - self._alpha_t
        try:
            scc = SplitConformalClassifier(
                estimator=model,
                confidence_level=confidence,
                conformity_score="lac",
                prefit=True,
                random_state=42,
            )
            scc.conformalize(X_cal, y_cal)
            self._mapie_model = scc
            self._fitted = True
            logger.info(
                "AdaptiveConformalSizer fitted: target_alpha=%.2f, "
                "gamma=%.3f, cal_samples=%d",
                self.target_alpha,
                self.gamma,
                len(y_cal),
            )
        except Exception as exc:
            logger.warning("Failed to fit adaptive conformal predictor: %s", exc)
            self._fitted = False

        return self

    # ------------------------------------------------------------------
    # Online update (ACI core)
    # ------------------------------------------------------------------

    def update(self, y_pred: int, y_true: int) -> None:
        """Record one observation and adapt alpha.

        Args:
            y_pred: Predicted class label (0 or 1).
            y_true: Observed true label (0 or 1).
        """
        err = 1 if y_pred != y_true else 0
        self._errors.append(err)
        self._adapt_alpha()

    def _adapt_alpha(self) -> None:
        """ACI update rule with exponential weighting.

        Standard ACI:  alpha_{t+1} = alpha_t + gamma * (err_t - target_alpha)

        Enhancement: use exponentially weighted recent error rate instead
        of the single latest error for smoother adaptation.
        """
        if not self._errors:
            return

        # Exponentially weighted error rate
        n = len(self._errors)
        weights = np.array([self.decay_rate ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()
        ew_error_rate = float(np.dot(weights, list(self._errors)))

        # ACI step using smoothed error
        self._alpha_t += self.gamma * (ew_error_rate - self.target_alpha)
        self._alpha_t = float(np.clip(self._alpha_t, self.min_alpha, self.max_alpha))

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def size_single(self, X: np.ndarray, base_size: float) -> float:
        """Size a single observation using the adaptive prediction set.

        Compatible interface with ``ConformalPositionSizer.size_single``.

        Args:
            X: Feature vector (1-D or 2-D with one row).
            base_size: Base position size (e.g. 0.10 for 10 %).

        Returns:
            Scaled position size in ``[min_size_fraction * base_size, base_size]``.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if not self._fitted or self._mapie_model is None:
            return float(base_size)

        try:
            # Predict at the *current* adaptive alpha
            confidence = 1.0 - self._alpha_t
            confidence = float(np.clip(confidence, 0.70, 0.99))
            self._mapie_model.confidence_level = confidence

            _, y_sets = self._mapie_model.predict_set(X)
            if y_sets.ndim == 3:
                y_sets = y_sets[:, :, 0]

            set_width = int(y_sets[0].sum())

            # Nonconformity-score-based scaling
            score_scale = self._score_based_scale(X)

            if set_width == 0:
                # Empty set (very rare) — half position
                scale = 0.5
            elif set_width == 1:
                # Certain — use score-based fine-tuning
                scale = score_scale
            else:
                # Uncertain (2 classes in set) — minimum position
                scale = self.min_size_fraction

            self._interval_widths.append(set_width)
            return float(np.clip(scale * base_size, self.min_size_fraction * base_size, base_size))

        except Exception as exc:
            logger.warning("Adaptive conformal sizing failed: %s", exc)
            return float(base_size)

    def _score_based_scale(self, X: np.ndarray) -> float:
        """Map the nonconformity score to a [min_size_fraction, 1.0] scale.

        Lower score = higher conformity = larger position.
        """
        if self._base_model is None or self._cal_scores is None:
            return 1.0

        try:
            probas = self._base_model.predict_proba(X)
            pred_class = int(np.argmax(probas[0]))
            score = 1.0 - probas[0, pred_class]

            # Percentile of this score among calibration scores
            pct = float(np.mean(self._cal_scores <= score))

            # Linear map: pct=0 (most conforming) -> 1.0, pct=1 -> min_size_fraction
            return 1.0 - pct * (1.0 - self.min_size_fraction)
        except Exception:
            return 1.0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return coverage stats, current alpha, and interval widths."""
        errors = list(self._errors)
        n = len(errors)
        if n == 0:
            empirical_coverage = None
            ew_error_rate = None
        else:
            empirical_coverage = 1.0 - float(np.mean(errors))
            weights = np.array([self.decay_rate ** (n - 1 - i) for i in range(n)])
            weights /= weights.sum()
            ew_error_rate = float(np.dot(weights, errors))

        widths = list(self._interval_widths)
        return {
            "current_alpha": self._alpha_t,
            "target_alpha": self.target_alpha,
            "n_observations": n,
            "empirical_coverage": empirical_coverage,
            "ew_error_rate": ew_error_rate,
            "mean_interval_width": float(np.mean(widths)) if widths else None,
            "fitted": self._fitted,
            "mapie_available": _HAS_MAPIE,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """Check if MAPIE is installed."""
        return _HAS_MAPIE

    @property
    def current_alpha(self) -> float:
        """Current adaptive alpha value."""
        return self._alpha_t

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"AdaptiveConformalSizer(target_alpha={self.target_alpha}, "
            f"gamma={self.gamma}, current_alpha={self._alpha_t:.4f}, {status})"
        )

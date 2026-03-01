"""
Conformal Position Sizer — uses MAPIE conformal prediction to scale
position sizes based on prediction certainty.

Narrow prediction sets (high confidence) get full position;
wide sets (uncertain) get reduced position.
"""

import logging
import numpy as np
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Optional dependency
try:
    from mapie.classification import SplitConformalClassifier
    _HAS_MAPIE = True
except ImportError:
    _HAS_MAPIE = False


class ConformalPositionSizer:
    """
    Uses MAPIE conformal prediction to scale position sizes based on
    prediction certainty. Narrow prediction sets (high confidence) get
    full position; wide sets get reduced position.

    If MAPIE is not installed, gracefully degrades by returning base_size
    for all predictions (no scaling applied).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        min_size_fraction: float = 0.25,
    ):
        """
        Args:
            alpha: Significance level for prediction sets (0.1 = 90% coverage).
            min_size_fraction: Minimum position size as fraction of base_size (floor).
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not 0.0 < min_size_fraction <= 1.0:
            raise ValueError(
                f"min_size_fraction must be in (0, 1], got {min_size_fraction}"
            )

        self.alpha = alpha
        self.min_size_fraction = min_size_fraction
        self._fitted = False
        self._mapie_model: Optional[Any] = None

    def fit(self, model: Any, X_cal: np.ndarray, y_cal: np.ndarray) -> "ConformalPositionSizer":
        """
        Calibrate conformal predictor on calibration data.

        Args:
            model: FITTED sklearn classifier (must have predict_proba and classes_).
            X_cal: Calibration features (2D array).
            y_cal: Calibration labels (1D array of 0/1).

        Returns:
            self (for chaining).
        """
        if not _HAS_MAPIE:
            logger.warning(
                "mapie not installed — ConformalPositionSizer will return "
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

        # Ensure model has classes_ attribute (required by MAPIE)
        if not hasattr(model, "classes_"):
            model.classes_ = np.array([0, 1])

        confidence_level = 1.0 - self.alpha  # e.g., alpha=0.1 -> 90% coverage

        try:
            scc = SplitConformalClassifier(
                estimator=model,
                confidence_level=confidence_level,
                conformity_score="lac",
                prefit=True,
                random_state=42,
            )
            scc.conformalize(X_cal, y_cal)
            self._mapie_model = scc
            self._fitted = True
            logger.info(
                "ConformalPositionSizer fitted: alpha=%.2f, "
                "calibration_samples=%d",
                self.alpha, len(y_cal),
            )
        except Exception as e:
            logger.warning("Failed to fit conformal predictor: %s", e)
            self._fitted = False

        return self

    def size(self, X: np.ndarray, base_size: float) -> np.ndarray:
        """
        Compute position sizes based on prediction set width.

        Args:
            X: Feature matrix (2D array, n_samples x n_features).
            base_size: Base position size (e.g., 0.10 = 10% of portfolio).

        Returns:
            Array of position sizes, each in
            [min_size_fraction * base_size, base_size].

        Algorithm:
            1. Get prediction sets from MAPIE at the configured alpha level.
            2. For each sample, set_width = number of classes in prediction set.
            3. If set_width == 1 (certain): full base_size.
            4. If set_width == 2 (uncertain): min_size_fraction * base_size.
            5. If set_width == 0 (empty set, rare): base_size * 0.5.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]

        # Graceful degradation: no conformal model -> return base_size for all
        if not self._fitted or self._mapie_model is None:
            return np.full(n_samples, base_size)

        try:
            _, y_sets = self._mapie_model.predict_set(X)
            # y_sets shape: (n_samples, n_classes) for single confidence level
            # or (n_samples, n_classes, n_levels) for multiple levels.
            # We use a single confidence level, so squeeze if needed.
            if y_sets.ndim == 3:
                y_sets = y_sets[:, :, 0]

            # set_width = number of True values per row
            set_widths = y_sets.sum(axis=1)  # (n_samples,)

            sizes = np.empty(n_samples, dtype=float)
            for i in range(n_samples):
                w = int(set_widths[i])
                if w == 1:
                    # Certain: full position
                    sizes[i] = base_size
                elif w == 0:
                    # Empty set (very rare): half position
                    sizes[i] = base_size * 0.5
                else:
                    # w >= 2 (uncertain): minimum position
                    sizes[i] = self.min_size_fraction * base_size

            return sizes

        except Exception as e:
            logger.warning(
                "Conformal sizing failed, returning base_size: %s", e
            )
            return np.full(n_samples, base_size)

    def size_single(self, X_single: np.ndarray, base_size: float) -> float:
        """
        Size a single sample. Convenience wrapper for size().

        Args:
            X_single: Feature vector (1D or 2D with shape (1, n_features)).
            base_size: Base position size.

        Returns:
            Scalar position size.
        """
        X_single = np.asarray(X_single)
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)
        return float(self.size(X_single, base_size)[0])

    @property
    def is_available(self) -> bool:
        """Check if MAPIE is installed."""
        return _HAS_MAPIE

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"ConformalPositionSizer(alpha={self.alpha}, "
            f"min_size_fraction={self.min_size_fraction}, {status})"
        )

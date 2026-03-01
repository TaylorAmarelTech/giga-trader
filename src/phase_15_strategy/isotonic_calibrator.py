"""
Isotonic Calibrator — post-hoc probability calibration using isotonic regression.

Maps raw model probabilities to calibrated probabilities using a monotonic,
non-parametric function fitted on held-out data. Improves probability
calibration without changing the model's ranking ability.
"""

import logging
import numpy as np
from typing import Dict, Optional

from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class IsotonicCalibrator:
    """
    Post-hoc probability calibration using isotonic regression.
    Maps raw model probabilities to calibrated probabilities using
    a monotonic, non-parametric function fitted on held-out data.
    """

    def __init__(self, out_of_bounds: str = "clip"):
        """
        Args:
            out_of_bounds: How to handle values outside training range.
                          "clip": Clip to nearest training value (default, safest).
                          "nan": Return NaN for out-of-range.
        """
        if out_of_bounds not in ("clip", "nan"):
            raise ValueError(
                f"out_of_bounds must be 'clip' or 'nan', got '{out_of_bounds}'"
            )

        self.out_of_bounds = out_of_bounds
        self._calibrator = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds=out_of_bounds,
        )
        self._fitted = False

    def fit(
        self, y_pred_proba: np.ndarray, y_true: np.ndarray
    ) -> "IsotonicCalibrator":
        """
        Fit isotonic regression on held-out predictions and true labels.

        Args:
            y_pred_proba: Raw predicted probabilities (1D array,
                         probability of class 1).
            y_true: True binary labels (0/1).

        Returns:
            self (for chaining).
        """
        y_pred_proba = np.asarray(y_pred_proba, dtype=float).ravel()
        y_true = np.asarray(y_true, dtype=float).ravel()

        if len(y_pred_proba) != len(y_true):
            raise ValueError(
                f"Length mismatch: y_pred_proba has {len(y_pred_proba)} "
                f"elements, y_true has {len(y_true)}"
            )

        if len(y_pred_proba) < 2:
            logger.warning(
                "Need at least 2 samples for isotonic calibration, got %d",
                len(y_pred_proba),
            )
            self._fitted = False
            return self

        self._calibrator.fit(y_pred_proba, y_true)
        self._fitted = True
        logger.info(
            "IsotonicCalibrator fitted on %d samples", len(y_pred_proba)
        )
        return self

    def calibrate(self, y_pred_proba: np.ndarray) -> np.ndarray:
        """
        Calibrate raw probabilities.

        Args:
            y_pred_proba: Raw predicted probabilities (1D array).

        Returns:
            Calibrated probabilities, clipped to [0.01, 0.99] to avoid
            extreme values that could cause numerical issues downstream.

        Raises:
            RuntimeError: If calibrate() is called before fit().
        """
        if not self._fitted:
            raise RuntimeError(
                "IsotonicCalibrator has not been fitted. Call fit() first."
            )

        y_pred_proba = np.asarray(y_pred_proba, dtype=float).ravel()
        calibrated = self._calibrator.transform(y_pred_proba)
        # Clip to avoid extremes (0.0 or 1.0 cause log(0) issues)
        calibrated = np.clip(calibrated, 0.01, 0.99)
        return calibrated

    def evaluate(
        self, y_pred_proba: np.ndarray, y_true: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate calibration quality before and after.

        Args:
            y_pred_proba: Raw predicted probabilities (1D array).
            y_true: True binary labels (0/1).

        Returns:
            Dict with:
                "ece_before": ECE of raw probabilities.
                "ece_after": ECE of calibrated probabilities.
                "improvement": ece_before - ece_after (positive = better).

        Raises:
            RuntimeError: If evaluate() is called before fit().
        """
        if not self._fitted:
            raise RuntimeError(
                "IsotonicCalibrator has not been fitted. Call fit() first."
            )

        y_pred_proba = np.asarray(y_pred_proba, dtype=float).ravel()
        y_true = np.asarray(y_true, dtype=float).ravel()

        ece_before = self._compute_ece(y_pred_proba, y_true)
        calibrated = self.calibrate(y_pred_proba)
        ece_after = self._compute_ece(calibrated, y_true)

        return {
            "ece_before": round(ece_before, 6),
            "ece_after": round(ece_after, 6),
            "improvement": round(ece_before - ece_after, 6),
        }

    @staticmethod
    def _compute_ece(
        y_pred_proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error with equal-width bins.

        ECE = sum over bins of (|bin_size / total| * |avg_confidence - avg_accuracy|)

        Args:
            y_pred_proba: Predicted probabilities.
            y_true: True binary labels.
            n_bins: Number of equal-width bins (default 10).

        Returns:
            ECE value (lower is better, 0.0 = perfectly calibrated).
        """
        y_pred_proba = np.asarray(y_pred_proba, dtype=float).ravel()
        y_true = np.asarray(y_true, dtype=float).ravel()

        if len(y_pred_proba) == 0:
            return 0.0

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        total = len(y_pred_proba)

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if i == n_bins - 1:
                # Last bin includes the right edge
                mask = (y_pred_proba >= lo) & (y_pred_proba <= hi)
            else:
                mask = (y_pred_proba >= lo) & (y_pred_proba < hi)

            bin_count = mask.sum()
            if bin_count == 0:
                continue

            avg_confidence = y_pred_proba[mask].mean()
            avg_accuracy = y_true[mask].mean()
            ece += (bin_count / total) * abs(avg_confidence - avg_accuracy)

        return float(ece)

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"IsotonicCalibrator(out_of_bounds='{self.out_of_bounds}', "
            f"{status})"
        )

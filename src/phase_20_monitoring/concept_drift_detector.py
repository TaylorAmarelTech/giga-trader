"""
GIGA TRADER - Concept Drift Detector
======================================
Monitors feature-to-target relationship shifts using Spearman rank
correlation and Fisher z-tests. Unlike feature distribution drift,
this detects when the predictive relationship changes.
"""

import logging
import threading
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ConceptDriftDetector:
    """Detect concept drift via feature-target correlation shifts.

    Compares Spearman correlations between features and target from
    training data (baseline) vs new data, using Fisher z-transformation
    for significance testing.

    Parameters
    ----------
    window_size : int
        Number of recent samples for comparison (default 100).
    significance_threshold : float
        P-value threshold for significant correlation change (default 0.05).
    min_samples : int
        Minimum samples needed before drift detection activates (default 50).
    """

    def __init__(
        self,
        window_size: int = 100,
        significance_threshold: float = 0.05,
        min_samples: int = 50,
    ):
        self.window_size = window_size
        self.significance_threshold = significance_threshold
        self.min_samples = min_samples
        self._lock = threading.Lock()
        self._baseline_correlations: Optional[Dict[str, float]] = None
        self._feature_names: Optional[List[str]] = None
        self._baseline_n: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "ConceptDriftDetector":
        """Store baseline feature-target correlations from training data.

        Parameters
        ----------
        X : np.ndarray
            Training features, shape (n_samples, n_features).
        y : np.ndarray
            Training target, shape (n_samples,).
        feature_names : list of str, optional
            Names for each feature column.
        """
        with self._lock:
            n_features = X.shape[1] if X.ndim > 1 else 1
            self._feature_names = feature_names or [f"feature_{i}" for i in range(n_features)]
            self._baseline_n = len(y)

            self._baseline_correlations = {}
            y_arr = np.asarray(y, dtype=float).ravel()

            for i, name in enumerate(self._feature_names):
                x_col = X[:, i] if X.ndim > 1 else X
                try:
                    corr = self._spearman_corr(x_col, y_arr)
                    self._baseline_correlations[name] = corr
                except Exception:
                    self._baseline_correlations[name] = 0.0

            logger.info(
                f"ConceptDriftDetector: fitted baseline from {self._baseline_n} samples, "
                f"{len(self._feature_names)} features"
            )
        return self

    def detect(self, X_new: np.ndarray, y_new: np.ndarray) -> Dict:
        """Detect concept drift in new data.

        Parameters
        ----------
        X_new : np.ndarray
            New features, shape (n_samples, n_features).
        y_new : np.ndarray
            New targets, shape (n_samples,).

        Returns
        -------
        dict with drift_detected, drift_score, drifted_features,
        correlation_changes, severity, recommendation.
        """
        with self._lock:
            if self._baseline_correlations is None:
                return {
                    "drift_detected": False,
                    "drift_score": 0.0,
                    "drifted_features": [],
                    "correlation_changes": {},
                    "severity": "NONE",
                    "recommendation": "FIT_BASELINE_FIRST",
                }

            n_new = len(y_new)
            if n_new < self.min_samples:
                return {
                    "drift_detected": False,
                    "drift_score": 0.0,
                    "drifted_features": [],
                    "correlation_changes": {},
                    "severity": "NONE",
                    "recommendation": f"INSUFFICIENT_DATA ({n_new}/{self.min_samples})",
                }

            y_arr = np.asarray(y_new, dtype=float).ravel()
            drifted = []
            changes = {}

            for i, name in enumerate(self._feature_names):
                if i >= X_new.shape[1]:
                    continue

                x_col = X_new[:, i] if X_new.ndim > 1 else X_new
                baseline_r = self._baseline_correlations.get(name, 0.0)

                try:
                    current_r = self._spearman_corr(x_col, y_arr)
                except Exception:
                    current_r = 0.0

                # Fisher z-test for correlation difference
                z_stat = self._fisher_z_test(
                    baseline_r, current_r, self._baseline_n, n_new
                )
                significant = abs(z_stat) > 1.96  # ~0.05 two-tailed

                change = current_r - baseline_r
                changes[name] = {
                    "baseline": round(baseline_r, 4),
                    "current": round(current_r, 4),
                    "change": round(change, 4),
                    "z_stat": round(z_stat, 3),
                    "significant": significant,
                }

                if significant:
                    drifted.append(name)

            # Compute drift score
            n_total = len(self._feature_names)
            drift_score = len(drifted) / n_total if n_total > 0 else 0.0

            # Severity
            if drift_score < 0.05:
                severity = "NONE"
                recommendation = "NO_ACTION"
            elif drift_score < 0.20:
                severity = "MILD"
                recommendation = "MONITOR"
            elif drift_score < 0.40:
                severity = "MODERATE"
                recommendation = "RETRAIN_SOON"
            else:
                severity = "SEVERE"
                recommendation = "RETRAIN_NOW"

            drift_detected = severity in ("MODERATE", "SEVERE")

            if drift_detected:
                logger.warning(
                    f"ConceptDrift [{severity}]: {len(drifted)}/{n_total} features "
                    f"drifted (score={drift_score:.2f}). Recommendation: {recommendation}"
                )

            return {
                "drift_detected": drift_detected,
                "drift_score": round(drift_score, 4),
                "drifted_features": drifted,
                "correlation_changes": changes,
                "severity": severity,
                "recommendation": recommendation,
            }

    @staticmethod
    def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
        """Compute Spearman rank correlation."""
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()

        # Remove NaN pairs
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

        if len(x) < 3:
            return 0.0

        # Rank
        x_rank = np.argsort(np.argsort(x)).astype(float)
        y_rank = np.argsort(np.argsort(y)).astype(float)

        # Pearson on ranks
        x_rank -= x_rank.mean()
        y_rank -= y_rank.mean()

        denom = np.sqrt(np.sum(x_rank ** 2) * np.sum(y_rank ** 2))
        if denom == 0:
            return 0.0

        return float(np.sum(x_rank * y_rank) / denom)

    @staticmethod
    def _fisher_z_test(r1: float, r2: float, n1: int, n2: int) -> float:
        """Fisher z-test for comparing two correlations."""
        # Clamp to avoid atanh domain error
        r1 = max(-0.999, min(0.999, r1))
        r2 = max(-0.999, min(0.999, r2))

        z1 = np.arctanh(r1)
        z2 = np.arctanh(r2)

        se = np.sqrt(1.0 / max(n1 - 3, 1) + 1.0 / max(n2 - 3, 1))
        if se == 0:
            return 0.0

        return (z1 - z2) / se

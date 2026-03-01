"""
Feature Drift Monitor
=====================
Detects when live feature distributions diverge from training distributions
using Population Stability Index (PSI).

PSI interpretation:
  PSI < 0.1  : No significant drift
  PSI 0.1-0.2: Moderate drift (warning)
  PSI > 0.2  : Significant drift (alert)

Uses quantile-based binning for robustness to outliers.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("FEATURE_DRIFT")


class FeatureDriftMonitor:
    """
    Monitors feature distribution drift between training and live data
    using Population Stability Index (PSI).

    PSI < 0.1: No significant drift
    PSI 0.1-0.2: Moderate drift (warning)
    PSI > 0.2: Significant drift (alert)
    """

    def __init__(
        self,
        psi_threshold: float = 0.2,
        n_bins: int = 10,
        alert_fraction: float = 0.2,
    ):
        """
        Args:
            psi_threshold: PSI threshold for flagging individual features.
            n_bins: Number of bins for PSI computation.
            alert_fraction: Alert if this fraction of features exceed threshold.
        """
        self.psi_threshold = psi_threshold
        self.n_bins = n_bins
        self.alert_fraction = alert_fraction
        self._fitted = False
        self._train_stats: Dict[str, Dict] = {}  # feature_name -> {bin_edges, expected_proportions, mean, std}
        self._feature_names: List[str] = []

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "FeatureDriftMonitor":
        """
        Store training distribution statistics.

        Args:
            X_train: Training feature matrix (n_samples, n_features).
            feature_names: Optional feature names (auto-generated if None).

        Returns:
            self (for method chaining).

        Stores per feature:
          - Bin edges (quantile-based for robustness)
          - Mean and std
          - Expected bin proportions
        """
        X = np.asarray(X_train, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        if feature_names is not None:
            self._feature_names = list(feature_names)
        else:
            self._feature_names = [f"feature_{i}" for i in range(n_features)]

        self._train_stats = {}

        for i, name in enumerate(self._feature_names):
            col = X[:, i]
            # Remove NaN for statistics
            valid = col[~np.isnan(col)]

            mean = float(np.mean(valid)) if len(valid) > 0 else 0.0
            std = float(np.std(valid)) if len(valid) > 0 else 0.0

            # Check for constant feature
            if len(valid) == 0 or np.all(valid == valid[0]):
                self._train_stats[name] = {
                    "bin_edges": None,
                    "expected_proportions": None,
                    "mean": mean,
                    "std": std,
                    "constant": True,
                    "constant_value": float(valid[0]) if len(valid) > 0 else 0.0,
                }
                continue

            # Quantile-based bin edges for robustness to outliers
            quantiles = np.linspace(0, 100, self.n_bins + 1)
            bin_edges = np.percentile(valid, quantiles)

            # Deduplicate edges (can happen with discrete or skewed data)
            bin_edges = np.unique(bin_edges)

            # Compute expected proportions using these edges
            counts, _ = np.histogram(valid, bins=bin_edges)
            expected_proportions = counts / counts.sum()

            self._train_stats[name] = {
                "bin_edges": bin_edges,
                "expected_proportions": expected_proportions,
                "mean": mean,
                "std": std,
                "constant": False,
            }

        self._fitted = True
        logger.info(
            "FeatureDriftMonitor fitted on %d features, %d samples",
            n_features,
            n_samples,
        )
        return self

    # ------------------------------------------------------------------
    # Checking
    # ------------------------------------------------------------------

    def check(self, X_live: np.ndarray) -> Dict:
        """
        Check live data for feature drift.

        Args:
            X_live: Live feature matrix (same columns as training).

        Returns:
            Dict with:
              - "has_drift": bool (True if drift detected)
              - "n_drifted": int (number of features exceeding PSI threshold)
              - "n_features": int
              - "drift_fraction": float (n_drifted / n_features)
              - "psi_scores": Dict[str, float] (feature_name -> PSI)
              - "drifted_features": List[str] (names exceeding threshold)
              - "severity": str ("none", "moderate", "significant")
        """
        if not self._fitted:
            raise RuntimeError("FeatureDriftMonitor has not been fitted. Call fit() first.")

        X = np.asarray(X_live, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_live_features = X.shape[1]
        n_train_features = len(self._feature_names)

        # Determine which features to check (overlap only)
        if n_live_features != n_train_features:
            logger.warning(
                "Live data has %d features but training had %d. Computing PSI on first %d.",
                n_live_features,
                n_train_features,
                min(n_live_features, n_train_features),
            )

        n_check = min(n_live_features, n_train_features)
        features_to_check = self._feature_names[:n_check]

        psi_scores: Dict[str, float] = {}
        drifted_features: List[str] = []

        for i, name in enumerate(features_to_check):
            stats = self._train_stats[name]
            col = X[:, i]
            valid = col[~np.isnan(col)]

            # Handle constant features
            if stats["constant"]:
                if len(valid) == 0 or np.all(valid == stats["constant_value"]):
                    psi_scores[name] = 0.0
                else:
                    # Training was constant but live is not (or different constant)
                    psi_scores[name] = float("inf")
                    drifted_features.append(name)
                continue

            if len(valid) == 0:
                psi_scores[name] = float("inf")
                drifted_features.append(name)
                continue

            # Compute actual proportions using training bin edges
            bin_edges = stats["bin_edges"]
            actual_counts, _ = np.histogram(valid, bins=bin_edges)
            actual_proportions = actual_counts / actual_counts.sum() if actual_counts.sum() > 0 else actual_counts.astype(float)

            expected_proportions = stats["expected_proportions"]

            # Align lengths (may differ if bin edges were deduplicated differently)
            min_len = min(len(expected_proportions), len(actual_proportions))
            exp = expected_proportions[:min_len]
            act = actual_proportions[:min_len]

            psi = self._compute_psi(exp, act)
            psi_scores[name] = psi

            if psi > self.psi_threshold:
                drifted_features.append(name)

        n_features = len(features_to_check)
        n_drifted = len(drifted_features)
        drift_fraction = n_drifted / n_features if n_features > 0 else 0.0

        has_drift = drift_fraction >= self.alert_fraction

        # Severity classification
        if drift_fraction > 0.2:
            severity = "significant"
        elif drift_fraction > 0.1:
            severity = "moderate"
        else:
            severity = "none"

        result = {
            "has_drift": has_drift,
            "n_drifted": n_drifted,
            "n_features": n_features,
            "drift_fraction": drift_fraction,
            "psi_scores": psi_scores,
            "drifted_features": drifted_features,
            "severity": severity,
        }

        if has_drift:
            logger.warning(
                "Feature drift detected: %d/%d features drifted (%.1f%%), severity=%s",
                n_drifted,
                n_features,
                drift_fraction * 100,
                severity,
            )
        else:
            logger.info(
                "Drift check passed: %d/%d features drifted (%.1f%%)",
                n_drifted,
                n_features,
                drift_fraction * 100,
            )

        return result

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_drift_report(self, X_live: np.ndarray) -> pd.DataFrame:
        """
        Generate detailed drift report.

        Returns:
            DataFrame with columns: feature, psi, mean_train, mean_live,
            std_train, std_live, status ("ok", "warning", "drift").
        """
        if not self._fitted:
            raise RuntimeError("FeatureDriftMonitor has not been fitted. Call fit() first.")

        X = np.asarray(X_live, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_check = min(X.shape[1], len(self._feature_names))
        features_to_check = self._feature_names[:n_check]

        # Get PSI scores
        check_result = self.check(X)
        psi_scores = check_result["psi_scores"]

        rows = []
        for i, name in enumerate(features_to_check):
            stats = self._train_stats[name]
            col = X[:, i]
            valid = col[~np.isnan(col)]

            mean_live = float(np.mean(valid)) if len(valid) > 0 else 0.0
            std_live = float(np.std(valid)) if len(valid) > 0 else 0.0

            psi = psi_scores.get(name, 0.0)

            if psi > self.psi_threshold:
                status = "drift"
            elif psi > 0.1:
                status = "warning"
            else:
                status = "ok"

            rows.append({
                "feature": name,
                "psi": psi,
                "mean_train": stats["mean"],
                "mean_live": mean_live,
                "std_train": stats["std"],
                "std_live": std_live,
                "status": status,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("psi", ascending=False).reset_index(drop=True)

        return df

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_psi(expected: np.ndarray, actual: np.ndarray) -> float:
        """
        Compute Population Stability Index between expected and actual bin proportions.

        PSI = sum((actual_i - expected_i) * ln(actual_i / expected_i))

        Uses small epsilon (1e-6) to avoid log(0).
        """
        eps = 1e-6
        expected = np.clip(expected, eps, None)
        actual = np.clip(actual, eps, None)

        psi = float(np.sum((actual - expected) * np.log(actual / expected)))
        return psi

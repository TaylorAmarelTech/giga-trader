"""
Test FeatureDriftMonitor: PSI-based feature distribution drift detection.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase_20_monitoring.feature_drift_monitor import FeatureDriftMonitor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n_samples: int = 500, n_features: int = 5, seed: int = 42) -> np.ndarray:
    """Generate random normal data for testing."""
    rng = np.random.RandomState(seed)
    return rng.randn(n_samples, n_features)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNoFitRaises:
    """Calling check/report before fit should raise."""

    def test_check_before_fit_raises(self):
        monitor = FeatureDriftMonitor()
        with pytest.raises(RuntimeError, match="not been fitted"):
            monitor.check(np.zeros((10, 3)))

    def test_report_before_fit_raises(self):
        monitor = FeatureDriftMonitor()
        with pytest.raises(RuntimeError, match="not been fitted"):
            monitor.get_drift_report(np.zeros((10, 3)))


class TestNoDrift:
    """When live data comes from the same distribution, no drift should be detected."""

    def test_same_distribution_no_drift(self):
        rng = np.random.RandomState(0)
        X_train = rng.randn(1000, 5)
        X_live = rng.randn(500, 5)

        monitor = FeatureDriftMonitor(psi_threshold=0.2)
        monitor.fit(X_train)
        result = monitor.check(X_live)

        assert result["has_drift"] is False
        assert result["n_drifted"] == 0
        assert result["drift_fraction"] == 0.0
        assert result["severity"] == "none"
        assert len(result["drifted_features"]) == 0


class TestDriftDetected:
    """When live data has a shifted mean, drift should be detected."""

    def test_shifted_mean_triggers_drift(self):
        rng = np.random.RandomState(1)
        X_train = rng.randn(1000, 5)
        # Shift all features by 5 standard deviations -- clear drift
        X_live = rng.randn(500, 5) + 5.0

        monitor = FeatureDriftMonitor(psi_threshold=0.2, alert_fraction=0.2)
        monitor.fit(X_train)
        result = monitor.check(X_live)

        assert result["has_drift"] is True
        assert result["n_drifted"] > 0
        assert result["drift_fraction"] > 0.0
        assert len(result["drifted_features"]) > 0

    def test_partial_drift(self):
        """Only some features drifted."""
        rng = np.random.RandomState(2)
        X_train = rng.randn(1000, 10)
        X_live = rng.randn(500, 10).copy()
        # Shift first 3 features heavily
        X_live[:, 0] += 10.0
        X_live[:, 1] += 10.0
        X_live[:, 2] += 10.0

        monitor = FeatureDriftMonitor(psi_threshold=0.2, alert_fraction=0.5)
        monitor.fit(X_train)
        result = monitor.check(X_live)

        # At least those 3 should be drifted
        assert result["n_drifted"] >= 3
        # But has_drift depends on alert_fraction (50% = 5 out of 10)
        # 3/10 = 30% < 50% -> no system-level alert
        assert result["n_drifted"] >= 3


class TestPSIScores:
    """PSI scores should be non-negative for all features."""

    def test_psi_scores_non_negative(self):
        X_train = _make_data(1000, 8, seed=10)
        X_live = _make_data(500, 8, seed=20)

        monitor = FeatureDriftMonitor()
        monitor.fit(X_train)
        result = monitor.check(X_live)

        for name, psi in result["psi_scores"].items():
            assert psi >= 0.0, f"PSI for {name} is negative: {psi}"


class TestDriftedFeaturesMatchThreshold:
    """drifted_features list should exactly match features exceeding PSI threshold."""

    def test_drifted_features_consistent(self):
        rng = np.random.RandomState(3)
        X_train = rng.randn(1000, 6)
        X_live = rng.randn(500, 6).copy()
        X_live[:, 0] += 8.0  # Heavy drift on feature 0

        threshold = 0.15
        monitor = FeatureDriftMonitor(psi_threshold=threshold)
        monitor.fit(X_train)
        result = monitor.check(X_live)

        for name, psi in result["psi_scores"].items():
            if psi > threshold:
                assert name in result["drifted_features"], (
                    f"{name} has PSI {psi:.4f} > {threshold} but not in drifted_features"
                )
            else:
                assert name not in result["drifted_features"], (
                    f"{name} has PSI {psi:.4f} <= {threshold} but IS in drifted_features"
                )


class TestGetDriftReport:
    """get_drift_report should return a DataFrame with correct columns."""

    def test_report_columns(self):
        X_train = _make_data(500, 4, seed=5)
        X_live = _make_data(200, 4, seed=6)

        monitor = FeatureDriftMonitor()
        monitor.fit(X_train)
        report = monitor.get_drift_report(X_live)

        assert isinstance(report, pd.DataFrame)
        expected_cols = {"feature", "psi", "mean_train", "mean_live", "std_train", "std_live", "status"}
        assert expected_cols == set(report.columns), f"Columns mismatch: {set(report.columns)}"
        assert len(report) == 4

    def test_report_status_values(self):
        X_train = _make_data(500, 3, seed=7)
        X_live = _make_data(200, 3, seed=8)

        monitor = FeatureDriftMonitor()
        monitor.fit(X_train)
        report = monitor.get_drift_report(X_live)

        for status in report["status"]:
            assert status in ("ok", "warning", "drift"), f"Unexpected status: {status}"

    def test_report_sorted_by_psi_descending(self):
        X_train = _make_data(500, 5, seed=9)
        X_live = _make_data(200, 5, seed=10)

        monitor = FeatureDriftMonitor()
        monitor.fit(X_train)
        report = monitor.get_drift_report(X_live)

        psi_values = report["psi"].tolist()
        assert psi_values == sorted(psi_values, reverse=True), "Report not sorted by PSI descending"


class TestConstantFeatures:
    """Constant features should be handled gracefully."""

    def test_both_constant_same_value(self):
        """Both train and live are constant at the same value -> PSI = 0."""
        X_train = np.ones((100, 3))
        X_live = np.ones((50, 3))

        monitor = FeatureDriftMonitor()
        monitor.fit(X_train)
        result = monitor.check(X_live)

        for name, psi in result["psi_scores"].items():
            assert psi == 0.0, f"Constant feature {name} should have PSI=0, got {psi}"
        assert result["has_drift"] is False

    def test_train_constant_live_varies(self):
        """Train is constant but live varies -> detected as drift."""
        X_train = np.ones((100, 2))
        rng = np.random.RandomState(11)
        X_live = rng.randn(50, 2) + 5.0

        monitor = FeatureDriftMonitor(psi_threshold=0.2)
        monitor.fit(X_train)
        result = monitor.check(X_live)

        # Should flag drift for constant-to-varying features
        assert result["n_drifted"] == 2

    def test_mixed_constant_and_varying(self):
        """One constant feature, rest normal."""
        rng = np.random.RandomState(12)
        X_train = rng.randn(200, 3)
        X_train[:, 2] = 7.0  # Make third feature constant

        X_live = rng.randn(100, 3)
        X_live[:, 2] = 7.0  # Also constant at same value

        monitor = FeatureDriftMonitor()
        monitor.fit(X_train)
        result = monitor.check(X_live)

        assert result["psi_scores"]["feature_2"] == 0.0


class TestNumpyArrays:
    """fit() and check() should work with plain numpy arrays."""

    def test_fit_and_check_with_numpy(self):
        X_train = np.random.RandomState(20).randn(300, 4)
        X_live = np.random.RandomState(21).randn(100, 4)

        monitor = FeatureDriftMonitor()
        returned = monitor.fit(X_train)

        # fit returns self
        assert returned is monitor
        assert monitor._fitted is True

        result = monitor.check(X_live)
        assert "has_drift" in result
        assert "n_drifted" in result
        assert "psi_scores" in result
        assert isinstance(result["psi_scores"], dict)
        assert len(result["psi_scores"]) == 4

    def test_1d_array(self):
        """1D array should be treated as single feature."""
        X_train = np.random.RandomState(30).randn(200)
        X_live = np.random.RandomState(31).randn(100)

        monitor = FeatureDriftMonitor()
        monitor.fit(X_train)
        result = monitor.check(X_live)

        assert result["n_features"] == 1
        assert len(result["psi_scores"]) == 1


class TestSeverityLevels:
    """Severity should be 'none', 'moderate', or 'significant' based on drift fraction."""

    def test_severity_none(self):
        rng = np.random.RandomState(40)
        X_train = rng.randn(500, 10)
        X_live = rng.randn(200, 10)  # Same distribution

        monitor = FeatureDriftMonitor(psi_threshold=0.2)
        monitor.fit(X_train)
        result = monitor.check(X_live)

        assert result["severity"] == "none"

    def test_severity_significant(self):
        rng = np.random.RandomState(41)
        X_train = rng.randn(500, 5)
        # Shift ALL features massively -> drift_fraction should be > 0.2
        X_live = rng.randn(200, 5) + 10.0

        monitor = FeatureDriftMonitor(psi_threshold=0.2)
        monitor.fit(X_train)
        result = monitor.check(X_live)

        assert result["severity"] == "significant"
        assert result["drift_fraction"] > 0.2


class TestFeatureNames:
    """Custom feature names should be used in output."""

    def test_custom_feature_names(self):
        X_train = _make_data(200, 3, seed=50)
        X_live = _make_data(100, 3, seed=51)
        names = ["alpha", "beta", "gamma"]

        monitor = FeatureDriftMonitor()
        monitor.fit(X_train, feature_names=names)
        result = monitor.check(X_live)

        assert set(result["psi_scores"].keys()) == set(names)

    def test_auto_generated_names(self):
        X_train = _make_data(200, 3, seed=52)
        X_live = _make_data(100, 3, seed=53)

        monitor = FeatureDriftMonitor()
        monitor.fit(X_train)
        result = monitor.check(X_live)

        expected_names = {"feature_0", "feature_1", "feature_2"}
        assert set(result["psi_scores"].keys()) == expected_names


class TestColumnMismatch:
    """Different number of columns should log warning and compute on overlap."""

    def test_fewer_live_features(self):
        X_train = _make_data(200, 5, seed=60)
        X_live = _make_data(100, 3, seed=61)

        monitor = FeatureDriftMonitor()
        monitor.fit(X_train)
        result = monitor.check(X_live)

        assert result["n_features"] == 3  # Only overlap

    def test_more_live_features(self):
        X_train = _make_data(200, 3, seed=62)
        X_live = _make_data(100, 5, seed=63)

        monitor = FeatureDriftMonitor()
        monitor.fit(X_train)
        result = monitor.check(X_live)

        assert result["n_features"] == 3  # Only overlap


class TestComputePSI:
    """Unit tests for the static _compute_psi method."""

    def test_identical_distributions(self):
        """PSI of identical distributions should be near zero."""
        props = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.1])
        psi = FeatureDriftMonitor._compute_psi(props, props)
        assert abs(psi) < 1e-10

    def test_psi_non_negative(self):
        """PSI should always be non-negative."""
        rng = np.random.RandomState(70)
        for _ in range(50):
            a = rng.dirichlet(np.ones(10))
            b = rng.dirichlet(np.ones(10))
            psi = FeatureDriftMonitor._compute_psi(a, b)
            assert psi >= -1e-10, f"PSI was negative: {psi}"

    def test_psi_symmetric_property(self):
        """PSI is symmetric: PSI(a, b) == PSI(b, a)."""
        a = np.array([0.3, 0.3, 0.2, 0.1, 0.1])
        b = np.array([0.1, 0.1, 0.2, 0.3, 0.3])
        psi_ab = FeatureDriftMonitor._compute_psi(a, b)
        psi_ba = FeatureDriftMonitor._compute_psi(b, a)
        assert abs(psi_ab - psi_ba) < 1e-10

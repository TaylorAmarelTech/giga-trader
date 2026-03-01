"""
Tests for IsotonicCalibrator.
"""

import pytest
import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.phase_15_strategy.isotonic_calibrator import IsotonicCalibrator


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def calibration_data():
    """Generate raw probabilities and true labels for calibration."""
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X_train, y_train)

    # Raw probabilities on test set
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return y_pred_proba, y_test


@pytest.fixture
def fitted_calibrator(calibration_data):
    """An IsotonicCalibrator fitted on calibration data."""
    y_pred_proba, y_true = calibration_data
    # Use first half for fitting, second half for evaluation
    n = len(y_pred_proba) // 2
    cal = IsotonicCalibrator()
    cal.fit(y_pred_proba[:n], y_true[:n])
    return cal


# ─── Test 1: fit and calibrate work with valid data ─────────────────────────


class TestFitAndCalibrate:

    def test_fit_and_calibrate_valid(self, calibration_data):
        """fit() and calibrate() work end-to-end."""
        y_pred_proba, y_true = calibration_data
        n = len(y_pred_proba) // 2

        cal = IsotonicCalibrator()
        result = cal.fit(y_pred_proba[:n], y_true[:n])
        assert result is cal  # Returns self for chaining
        assert cal._fitted is True

        calibrated = cal.calibrate(y_pred_proba[n:])
        assert len(calibrated) == len(y_pred_proba[n:])

    def test_calibrate_before_fit_raises(self):
        """calibrate() raises RuntimeError if called before fit()."""
        cal = IsotonicCalibrator()
        with pytest.raises(RuntimeError, match="has not been fitted"):
            cal.calibrate(np.array([0.5, 0.6]))

    def test_length_mismatch_raises(self):
        """fit() raises ValueError on mismatched lengths."""
        cal = IsotonicCalibrator()
        with pytest.raises(ValueError, match="Length mismatch"):
            cal.fit(np.array([0.5, 0.6, 0.7]), np.array([0, 1]))


# ─── Test 2: Calibrated probabilities are in [0.01, 0.99] ──────────────────


class TestCalibratedRange:

    def test_calibrated_in_bounds(self, fitted_calibrator, calibration_data):
        """All calibrated probabilities are clipped to [0.01, 0.99]."""
        y_pred_proba, _ = calibration_data
        calibrated = fitted_calibrator.calibrate(y_pred_proba)
        assert np.all(calibrated >= 0.01)
        assert np.all(calibrated <= 0.99)

    def test_extreme_inputs_clipped(self, fitted_calibrator):
        """Extreme raw probabilities (0.0 and 1.0) are properly clipped."""
        extreme_proba = np.array([0.0, 0.001, 0.5, 0.999, 1.0])
        calibrated = fitted_calibrator.calibrate(extreme_proba)
        assert np.all(calibrated >= 0.01)
        assert np.all(calibrated <= 0.99)


# ─── Test 3: Calibrated probabilities preserve ordering (monotonic) ─────────


class TestMonotonicity:

    def test_monotonic_ordering(self, fitted_calibrator):
        """Isotonic regression preserves monotonic ordering."""
        # Generate sorted raw probabilities
        raw = np.linspace(0.05, 0.95, 50)
        calibrated = fitted_calibrator.calibrate(raw)

        # Isotonic regression guarantees non-decreasing output
        diffs = np.diff(calibrated)
        assert np.all(diffs >= -1e-10), (
            "Calibrated probabilities should be non-decreasing"
        )


# ─── Test 4: evaluate returns ECE values ────────────────────────────────────


class TestEvaluate:

    def test_evaluate_returns_ece(self, calibration_data):
        """evaluate() returns dict with ece_before, ece_after, improvement."""
        y_pred_proba, y_true = calibration_data
        n = len(y_pred_proba) // 2

        cal = IsotonicCalibrator()
        cal.fit(y_pred_proba[:n], y_true[:n])

        result = cal.evaluate(y_pred_proba[n:], y_true[n:])
        assert "ece_before" in result
        assert "ece_after" in result
        assert "improvement" in result

        # ECE values should be non-negative
        assert result["ece_before"] >= 0.0
        assert result["ece_after"] >= 0.0
        # Improvement = ece_before - ece_after
        assert result["improvement"] == pytest.approx(
            result["ece_before"] - result["ece_after"]
        )

    def test_evaluate_before_fit_raises(self):
        """evaluate() raises RuntimeError if called before fit()."""
        cal = IsotonicCalibrator()
        with pytest.raises(RuntimeError, match="has not been fitted"):
            cal.evaluate(np.array([0.5]), np.array([1]))


# ─── Test 5: Works with perfectly calibrated model (no change) ──────────────


class TestPerfectCalibration:

    def test_perfect_calibration_no_harm(self):
        """When probabilities are already calibrated, ECE should stay low."""
        np.random.seed(42)
        n = 1000
        # Generate "perfectly calibrated" probabilities:
        # For each probability p, the true label is Bernoulli(p)
        proba = np.random.uniform(0.1, 0.9, n)
        labels = (np.random.rand(n) < proba).astype(float)

        # Split into fit/eval
        n_fit = n // 2
        cal = IsotonicCalibrator()
        cal.fit(proba[:n_fit], labels[:n_fit])

        result = cal.evaluate(proba[n_fit:], labels[n_fit:])

        # Both ECEs should be relatively low for well-calibrated data
        assert result["ece_before"] < 0.10
        assert result["ece_after"] < 0.10
        # Calibration should not make things significantly worse
        assert result["improvement"] > -0.05


# ─── Test 6: Works with poorly calibrated model (sigmoid-squished) ──────────


class TestPoorCalibration:

    def test_poorly_calibrated_improves(self):
        """Sigmoid-squished probabilities should improve after calibration."""
        np.random.seed(42)
        n = 1000

        # True probabilities
        true_proba = np.random.uniform(0.1, 0.9, n)
        labels = (np.random.rand(n) < true_proba).astype(float)

        # Squish probabilities toward center (poorly calibrated)
        # This simulates an under-confident model
        squished = 0.3 + 0.4 * true_proba  # Maps [0,1] -> [0.3, 0.7]

        # Split
        n_fit = n // 2
        cal = IsotonicCalibrator()
        cal.fit(squished[:n_fit], labels[:n_fit])

        result = cal.evaluate(squished[n_fit:], labels[n_fit:])

        # The squished probabilities should have worse ECE than calibrated
        # Improvement should be positive (calibration helps)
        assert result["ece_before"] > result["ece_after"], (
            f"Expected improvement: before={result['ece_before']}, "
            f"after={result['ece_after']}"
        )
        assert result["improvement"] > 0


# ─── Additional edge case tests ─────────────────────────────────────────────


class TestEdgeCases:

    def test_compute_ece_empty(self):
        """ECE of empty arrays is 0."""
        ece = IsotonicCalibrator._compute_ece(np.array([]), np.array([]))
        assert ece == 0.0

    def test_repr(self):
        cal = IsotonicCalibrator()
        r = repr(cal)
        assert "not fitted" in r
        assert "clip" in r

    def test_invalid_out_of_bounds(self):
        with pytest.raises(ValueError, match="out_of_bounds must be"):
            IsotonicCalibrator(out_of_bounds="raise")

    def test_too_few_samples(self):
        """Fewer than 2 samples -> not fitted."""
        cal = IsotonicCalibrator()
        cal.fit(np.array([0.5]), np.array([1]))
        assert cal._fitted is False

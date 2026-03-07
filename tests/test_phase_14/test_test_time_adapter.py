"""Tests for TestTimeAdapter."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.phase_14_robustness.test_time_adapter import TestTimeAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n=200, n_features=8, seed=42):
    """Generate a simple binary-classification dataset."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * 0.3 > 0).astype(int)
    return X, y


def _make_fitted_model(X, y):
    """Return a fitted LogisticRegression on the given data."""
    model = LogisticRegression(C=1.0, max_iter=300, random_state=42)
    model.fit(X, y)
    return model


class _DummyModel:
    """Minimal mock model that returns controllable probabilities."""

    def __init__(self, proba_value: float = 0.7):
        self._p = proba_value

    def predict_proba(self, X):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        p1 = np.full(n, self._p)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_construction(self):
        model = _DummyModel()
        adapter = TestTimeAdapter(base_model=model)
        assert adapter.adaptation_rate == 0.01
        assert adapter.n_recent == 50
        assert adapter.entropy_weight == 0.1
        assert adapter.max_shift == 0.05

    def test_custom_params(self):
        adapter = TestTimeAdapter(
            base_model=_DummyModel(),
            adaptation_rate=0.05,
            n_recent=100,
            entropy_weight=0.2,
            max_shift=0.03,
        )
        assert adapter.adaptation_rate == 0.05
        assert adapter.n_recent == 100
        assert adapter.entropy_weight == 0.2
        assert adapter.max_shift == 0.03

    def test_invalid_adaptation_rate(self):
        with pytest.raises(ValueError, match="adaptation_rate"):
            TestTimeAdapter(base_model=_DummyModel(), adaptation_rate=-0.1)

    def test_invalid_n_recent(self):
        with pytest.raises(ValueError, match="n_recent"):
            TestTimeAdapter(base_model=_DummyModel(), n_recent=0)

    def test_invalid_entropy_weight(self):
        with pytest.raises(ValueError, match="entropy_weight"):
            TestTimeAdapter(base_model=_DummyModel(), entropy_weight=-1.0)

    def test_invalid_max_shift(self):
        with pytest.raises(ValueError, match="max_shift"):
            TestTimeAdapter(base_model=_DummyModel(), max_shift=0.6)


# ---------------------------------------------------------------------------
# predict_proba shape and range
# ---------------------------------------------------------------------------

class TestPredictProba:
    def test_shape_matches_base(self):
        X, y = _make_data()
        model = _make_fitted_model(X[:150], y[:150])
        adapter = TestTimeAdapter(base_model=model)
        adapter.fit_reference(X[:150])

        proba = adapter.predict_proba(X[150:])
        assert proba.shape == (50, 2)

    def test_probabilities_sum_to_one(self):
        X, y = _make_data()
        model = _make_fitted_model(X[:150], y[:150])
        adapter = TestTimeAdapter(base_model=model)
        adapter.fit_reference(X[:150])

        proba = adapter.predict_proba(X[150:])
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-7)

    def test_probabilities_in_valid_range(self):
        X, y = _make_data()
        model = _make_fitted_model(X[:150], y[:150])
        adapter = TestTimeAdapter(base_model=model)
        adapter.fit_reference(X[:150])

        proba = adapter.predict_proba(X[150:])
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0


# ---------------------------------------------------------------------------
# predict returns binary labels
# ---------------------------------------------------------------------------

class TestPredict:
    def test_returns_binary(self):
        X, y = _make_data()
        model = _make_fitted_model(X[:150], y[:150])
        adapter = TestTimeAdapter(base_model=model)
        adapter.fit_reference(X[:150])

        labels = adapter.predict(X[150:])
        assert labels.shape == (50,)
        assert set(np.unique(labels)).issubset({0, 1})


# ---------------------------------------------------------------------------
# max_shift respected -- adaptation never flips predictions
# ---------------------------------------------------------------------------

class TestMaxShiftRespected:
    def test_no_flip_high_confidence(self):
        """A model predicting 0.8 should never be pushed below 0.5."""
        model = _DummyModel(proba_value=0.8)
        adapter = TestTimeAdapter(
            base_model=model, max_shift=0.05, adaptation_rate=0.5,
        )
        rng = np.random.RandomState(99)
        X_ref = rng.randn(100, 5)
        adapter.fit_reference(X_ref)

        # Feed very different data to trigger drift
        X_shifted = rng.randn(50, 5) + 10.0
        adapter.update(X_shifted)

        proba = adapter.predict_proba(X_shifted)
        # All predictions should stay >= 0.5 (no flip)
        assert np.all(proba[:, 1] >= 0.5)

    def test_no_flip_low_confidence(self):
        """A model predicting 0.2 should never be pushed above 0.5."""
        model = _DummyModel(proba_value=0.2)
        adapter = TestTimeAdapter(
            base_model=model, max_shift=0.05, adaptation_rate=0.5,
        )
        rng = np.random.RandomState(77)
        X_ref = rng.randn(100, 5)
        adapter.fit_reference(X_ref)

        X_shifted = rng.randn(50, 5) + 10.0
        adapter.update(X_shifted)

        proba = adapter.predict_proba(X_shifted)
        assert np.all(proba[:, 1] < 0.5)

    def test_shift_magnitude_bounded(self):
        """Absolute shift must never exceed max_shift."""
        max_shift = 0.03
        model = _DummyModel(proba_value=0.65)
        adapter = TestTimeAdapter(
            base_model=model, max_shift=max_shift, adaptation_rate=1.0,
        )
        rng = np.random.RandomState(11)
        X_ref = rng.randn(100, 5)
        adapter.fit_reference(X_ref)

        X_shifted = rng.randn(30, 5) + 50.0  # extreme drift
        adapter.update(X_shifted)
        proba = adapter.predict_proba(X_shifted)

        base_p1 = 0.65
        shift = np.abs(proba[:, 1] - base_p1)
        assert np.all(shift <= max_shift + 1e-9)


# ---------------------------------------------------------------------------
# fit_reference
# ---------------------------------------------------------------------------

class TestFitReference:
    def test_stores_statistics(self):
        adapter = TestTimeAdapter(base_model=_DummyModel())
        assert adapter._ref_mean is None

        rng = np.random.RandomState(0)
        X = rng.randn(80, 6)
        adapter.fit_reference(X)

        assert adapter._ref_mean is not None
        assert adapter._ref_std is not None
        assert adapter._ref_mean.shape == (6,)
        assert adapter._ref_n == 80

    def test_returns_self(self):
        adapter = TestTimeAdapter(base_model=_DummyModel())
        rng = np.random.RandomState(0)
        result = adapter.fit_reference(rng.randn(20, 4))
        assert result is adapter


# ---------------------------------------------------------------------------
# update accumulates samples
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_accumulates_in_buffer(self):
        adapter = TestTimeAdapter(base_model=_DummyModel(), n_recent=30)
        rng = np.random.RandomState(0)

        adapter.update(rng.randn(10, 4))
        assert adapter._recent_buffer is not None
        assert adapter._recent_buffer.shape == (10, 4)

        adapter.update(rng.randn(15, 4))
        assert adapter._recent_buffer.shape == (25, 4)

    def test_buffer_respects_n_recent(self):
        adapter = TestTimeAdapter(base_model=_DummyModel(), n_recent=20)
        rng = np.random.RandomState(1)

        adapter.update(rng.randn(30, 4))
        assert adapter._recent_buffer.shape[0] == 20

    def test_welford_counts(self):
        adapter = TestTimeAdapter(base_model=_DummyModel())
        rng = np.random.RandomState(2)

        adapter.update(rng.randn(5, 3))
        assert adapter._run_n == 5

        adapter.update(rng.randn(7, 3))
        assert adapter._run_n == 12


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_returns_expected_keys(self):
        adapter = TestTimeAdapter(base_model=_DummyModel())
        diag = adapter.get_adaptation_diagnostics()
        expected_keys = {
            "n_predictions", "n_updates", "ema_shift",
            "mean_abs_shift", "drift_detected", "reference_fitted",
            "recent_buffer_size", "running_n",
        }
        assert expected_keys == set(diag.keys())

    def test_initial_values(self):
        adapter = TestTimeAdapter(base_model=_DummyModel())
        diag = adapter.get_adaptation_diagnostics()
        assert diag["n_predictions"] == 0
        assert diag["n_updates"] == 0
        assert diag["ema_shift"] == 0.0
        assert diag["reference_fitted"] is False
        assert diag["recent_buffer_size"] == 0

    def test_updates_after_usage(self):
        X, y = _make_data(n=100, n_features=5)
        model = _make_fitted_model(X[:70], y[:70])
        adapter = TestTimeAdapter(base_model=model)
        adapter.fit_reference(X[:70])
        adapter.update(X[70:80])
        adapter.predict_proba(X[80:])

        diag = adapter.get_adaptation_diagnostics()
        assert diag["n_predictions"] == 20
        assert diag["n_updates"] == 1
        assert diag["reference_fitted"] is True
        assert diag["recent_buffer_size"] == 10


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_clears_adaptation_state(self):
        adapter = TestTimeAdapter(base_model=_DummyModel())
        rng = np.random.RandomState(0)
        X = rng.randn(30, 4)

        adapter.fit_reference(X)
        adapter.update(X)
        adapter.predict_proba(X[:5])

        adapter.reset()
        diag = adapter.get_adaptation_diagnostics()

        assert diag["n_predictions"] == 0
        assert diag["n_updates"] == 0
        assert diag["ema_shift"] == 0.0
        assert diag["recent_buffer_size"] == 0
        assert diag["running_n"] == 0

    def test_preserves_reference(self):
        adapter = TestTimeAdapter(base_model=_DummyModel())
        rng = np.random.RandomState(0)
        adapter.fit_reference(rng.randn(50, 4))
        adapter.update(rng.randn(10, 4))

        adapter.reset()
        # Reference statistics should still be present
        assert adapter._ref_mean is not None
        assert adapter._ref_n == 50


# ---------------------------------------------------------------------------
# No adaptation when no drift
# ---------------------------------------------------------------------------

class TestNoDrift:
    def test_predictions_match_base_when_no_drift(self):
        """When test distribution == train distribution, adaptation is near zero."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 6)
        y = (X[:, 0] > 0).astype(int)
        model = _make_fitted_model(X[:150], y[:150])

        adapter = TestTimeAdapter(
            base_model=model, adaptation_rate=0.01, max_shift=0.05,
        )
        adapter.fit_reference(X[:150])

        # Use data from same distribution
        X_test = X[150:]
        base_proba = model.predict_proba(X_test)[:, 1]
        adapted_proba = adapter.predict_proba(X_test)[:, 1]

        # Shifts should be very small (< max_shift on all samples)
        diff = np.abs(adapted_proba - base_proba)
        assert np.all(diff <= 0.05 + 1e-9)
        # Average shift should be close to zero
        assert np.mean(diff) < 0.01


# ---------------------------------------------------------------------------
# Gentle adaptation under mild drift
# ---------------------------------------------------------------------------

class TestMildDrift:
    def test_gentle_adaptation(self):
        """Under mild drift, adaptation produces small but nonzero corrections."""
        rng = np.random.RandomState(42)
        X_train = rng.randn(200, 6)
        y_train = (X_train[:, 0] > 0).astype(int)
        model = _make_fitted_model(X_train, y_train)

        adapter = TestTimeAdapter(
            base_model=model, adaptation_rate=0.05, max_shift=0.05,
            entropy_weight=0.2,
        )
        adapter.fit_reference(X_train)

        # Mild drift: shift means by 1 std
        X_shifted = rng.randn(50, 6) + 1.0
        adapter.update(X_shifted)

        base_proba = model.predict_proba(X_shifted)[:, 1]
        adapted_proba = adapter.predict_proba(X_shifted)[:, 1]

        # Should not be identical (some adaptation happened)
        assert not np.allclose(adapted_proba, base_proba, atol=1e-10)
        # But shifts must be bounded
        assert np.all(np.abs(adapted_proba - base_proba) <= 0.05 + 1e-9)


# ---------------------------------------------------------------------------
# Edge case: single sample
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_sample_predict(self):
        model = _DummyModel(proba_value=0.7)
        adapter = TestTimeAdapter(base_model=model)
        rng = np.random.RandomState(0)
        adapter.fit_reference(rng.randn(50, 4))

        X_single = rng.randn(1, 4)
        proba = adapter.predict_proba(X_single)
        assert proba.shape == (1, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-7)

    def test_single_sample_update(self):
        adapter = TestTimeAdapter(base_model=_DummyModel())
        rng = np.random.RandomState(0)
        adapter.update(rng.randn(1, 4))
        assert adapter._run_n == 1
        assert adapter._recent_buffer.shape == (1, 4)

    def test_1d_input_handled(self):
        """A 1-D feature vector should be reshaped to (1, n_features)."""
        model = _DummyModel(proba_value=0.6)
        adapter = TestTimeAdapter(base_model=model)
        rng = np.random.RandomState(0)
        adapter.fit_reference(rng.randn(30, 5))

        proba = adapter.predict_proba(rng.randn(5))  # 1-D
        assert proba.shape == (1, 2)

    def test_no_reference_still_works(self):
        """Adapter should work even without fit_reference (no drift correction)."""
        model = _DummyModel(proba_value=0.7)
        adapter = TestTimeAdapter(base_model=model)
        rng = np.random.RandomState(0)

        proba = adapter.predict_proba(rng.randn(10, 4))
        assert proba.shape == (10, 2)
        # Without reference, drift correction is 0; only entropy applies
        assert np.all(proba[:, 1] >= 0.5)

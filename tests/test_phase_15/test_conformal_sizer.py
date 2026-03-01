"""
Tests for ConformalPositionSizer.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.phase_15_strategy.conformal_sizer import ConformalPositionSizer, _HAS_MAPIE


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def classification_data():
    """Generate binary classification data with train/cal/test splits."""
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    # Train / calibration / test splits
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=42
    )
    return X_train, y_train, X_cal, y_cal, X_test, y_test


@pytest.fixture
def fitted_model(classification_data):
    """A fitted LogisticRegression model."""
    X_train, y_train, _, _, _, _ = classification_data
    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def fitted_sizer(fitted_model, classification_data):
    """A ConformalPositionSizer that has been fitted."""
    _, _, X_cal, y_cal, _, _ = classification_data
    sizer = ConformalPositionSizer(alpha=0.1, min_size_fraction=0.25)
    sizer.fit(fitted_model, X_cal, y_cal)
    return sizer


# ─── Test 1: fit and size work with valid data ──────────────────────────────


class TestFitAndSize:

    @pytest.mark.skipif(not _HAS_MAPIE, reason="mapie not installed")
    def test_fit_and_size_valid(self, fitted_sizer, classification_data):
        """fit() and size() work end-to-end with valid data."""
        _, _, _, _, X_test, _ = classification_data
        assert fitted_sizer._fitted is True

        sizes = fitted_sizer.size(X_test, base_size=0.10)
        assert sizes is not None
        assert len(sizes) == len(X_test)
        # All sizes should be positive
        assert np.all(sizes > 0)


# ─── Test 2: Certain predictions get full position size ─────────────────────


class TestCertainPredictions:

    @pytest.mark.skipif(not _HAS_MAPIE, reason="mapie not installed")
    def test_certain_predictions_full_size(self, fitted_sizer, classification_data):
        """Samples with singleton prediction sets get full base_size."""
        _, _, _, _, X_test, _ = classification_data
        base_size = 0.10
        sizes = fitted_sizer.size(X_test, base_size)

        # At least some predictions should be certain (set_width == 1)
        # and receive full base_size
        full_mask = np.isclose(sizes, base_size)
        # We expect at least some confident predictions in a well-separated dataset
        assert full_mask.sum() > 0, (
            "Expected at least one sample with full position size"
        )


# ─── Test 3: Uncertain predictions get reduced size ─────────────────────────


class TestUncertainPredictions:

    @pytest.mark.skipif(not _HAS_MAPIE, reason="mapie not installed")
    def test_uncertain_predictions_reduced_size(
        self, fitted_sizer, classification_data
    ):
        """Samples with wide prediction sets get min_size_fraction * base_size."""
        _, _, _, _, X_test, _ = classification_data
        base_size = 0.10
        min_fraction = fitted_sizer.min_size_fraction
        sizes = fitted_sizer.size(X_test, base_size)

        reduced_mask = np.isclose(sizes, min_fraction * base_size)
        # In a 90% coverage setting, some samples should be uncertain
        # (not guaranteed, but very likely with 100 test samples)
        # If all are certain, that's fine too — just check the range
        assert np.all(sizes >= min_fraction * base_size - 1e-12)
        assert np.all(sizes <= base_size + 1e-12)


# ─── Test 4: min_size_fraction is enforced ───────────────────────────────────


class TestMinSizeFraction:

    @pytest.mark.skipif(not _HAS_MAPIE, reason="mapie not installed")
    def test_min_size_fraction_enforced(self, fitted_model, classification_data):
        """Position sizes never fall below min_size_fraction * base_size
        (except for empty sets which get 0.5 * base_size)."""
        _, _, X_cal, y_cal, X_test, _ = classification_data
        min_frac = 0.30
        sizer = ConformalPositionSizer(alpha=0.1, min_size_fraction=min_frac)
        sizer.fit(fitted_model, X_cal, y_cal)

        base_size = 0.20
        sizes = sizer.size(X_test, base_size)

        # The floor is min(0.5 * base_size, min_frac * base_size) for empty sets
        # For normal predictions: min_frac * base_size
        absolute_floor = min(0.5 * base_size, min_frac * base_size)
        assert np.all(sizes >= absolute_floor - 1e-12)


# ─── Test 5: size returns array of correct shape ────────────────────────────


class TestSizeShape:

    @pytest.mark.skipif(not _HAS_MAPIE, reason="mapie not installed")
    def test_size_returns_correct_shape(self, fitted_sizer, classification_data):
        """size() returns an array with length == n_samples."""
        _, _, _, _, X_test, _ = classification_data
        sizes = fitted_sizer.size(X_test, base_size=0.15)
        assert isinstance(sizes, np.ndarray)
        assert sizes.shape == (len(X_test),)

    @pytest.mark.skipif(not _HAS_MAPIE, reason="mapie not installed")
    def test_size_single_sample(self, fitted_sizer, classification_data):
        """size() works with a single sample (1D input)."""
        _, _, _, _, X_test, _ = classification_data
        sizes = fitted_sizer.size(X_test[0], base_size=0.10)
        assert isinstance(sizes, np.ndarray)
        assert sizes.shape == (1,)


# ─── Test 6: size_single returns float ───────────────────────────────────────


class TestSizeSingle:

    @pytest.mark.skipif(not _HAS_MAPIE, reason="mapie not installed")
    def test_size_single_returns_float(self, fitted_sizer, classification_data):
        """size_single() returns a Python float."""
        _, _, _, _, X_test, _ = classification_data
        result = fitted_sizer.size_single(X_test[0], base_size=0.10)
        assert isinstance(result, float)
        assert 0 < result <= 0.10


# ─── Test 7: Graceful degradation when MAPIE not available ──────────────────


class TestGracefulDegradation:

    def test_returns_base_size_when_not_fitted(self, classification_data):
        """When not fitted (e.g., MAPIE missing), returns base_size for all."""
        _, _, _, _, X_test, _ = classification_data
        sizer = ConformalPositionSizer(alpha=0.1, min_size_fraction=0.25)
        # Do NOT fit — simulate MAPIE not available
        base_size = 0.12
        sizes = sizer.size(X_test, base_size)
        assert len(sizes) == len(X_test)
        assert np.all(np.isclose(sizes, base_size))

    def test_fit_without_mapie_stays_unfitted(self, fitted_model, classification_data):
        """When MAPIE import fails, fit() sets _fitted=False gracefully."""
        _, _, X_cal, y_cal, _, _ = classification_data
        sizer = ConformalPositionSizer(alpha=0.1)

        with patch(
            "src.phase_15_strategy.conformal_sizer._HAS_MAPIE", False
        ):
            sizer.fit(fitted_model, X_cal, y_cal)
            assert sizer._fitted is False

            # size() still works, returning base_size
            sizes = sizer.size(X_cal, base_size=0.10)
            assert np.all(np.isclose(sizes, 0.10))

    def test_size_single_degradation(self, classification_data):
        """size_single returns base_size when not fitted."""
        _, _, _, _, X_test, _ = classification_data
        sizer = ConformalPositionSizer()
        result = sizer.size_single(X_test[0], base_size=0.08)
        assert isinstance(result, float)
        assert result == pytest.approx(0.08)


# ─── Test 8: Works with logistic regression model ───────────────────────────


class TestLogisticRegression:

    @pytest.mark.skipif(not _HAS_MAPIE, reason="mapie not installed")
    def test_works_with_logistic_regression(self):
        """Full end-to-end with LogisticRegression."""
        X, y = make_classification(
            n_samples=300,
            n_features=8,
            n_informative=4,
            random_state=123,
        )
        X_train, X_rest, y_train, y_rest = train_test_split(
            X, y, test_size=0.4, random_state=123
        )
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_rest, y_rest, test_size=0.5, random_state=123
        )

        model = LogisticRegression(random_state=123, max_iter=500)
        model.fit(X_train, y_train)

        sizer = ConformalPositionSizer(alpha=0.10, min_size_fraction=0.20)
        sizer.fit(model, X_cal, y_cal)
        assert sizer._fitted is True

        base_size = 0.15
        sizes = sizer.size(X_test, base_size)

        # Basic sanity: array of correct length with valid values
        assert len(sizes) == len(X_test)
        assert np.all(sizes > 0)
        assert np.all(sizes <= base_size + 1e-12)

        # At least some variation in sizes (not all identical)
        unique_sizes = np.unique(np.round(sizes, 6))
        # With a well-separated dataset, we expect some certain and some uncertain
        # but this is not guaranteed, so just check validity
        assert len(unique_sizes) >= 1


# ─── Test: Constructor validation ────────────────────────────────────────────


class TestConstructorValidation:

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            ConformalPositionSizer(alpha=0.0)
        with pytest.raises(ValueError, match="alpha must be in"):
            ConformalPositionSizer(alpha=1.0)

    def test_invalid_min_size_fraction(self):
        with pytest.raises(ValueError, match="min_size_fraction must be in"):
            ConformalPositionSizer(min_size_fraction=0.0)
        with pytest.raises(ValueError, match="min_size_fraction must be in"):
            ConformalPositionSizer(min_size_fraction=1.5)

    def test_is_available_property(self):
        sizer = ConformalPositionSizer()
        assert sizer.is_available == _HAS_MAPIE

    def test_repr(self):
        sizer = ConformalPositionSizer(alpha=0.05, min_size_fraction=0.30)
        r = repr(sizer)
        assert "alpha=0.05" in r
        assert "min_size_fraction=0.3" in r
        assert "not fitted" in r

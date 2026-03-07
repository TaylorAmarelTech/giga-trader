"""Tests for AdaptiveConformalSizer (ACI)."""

import numpy as np
import pytest

from src.phase_15_strategy.adaptive_conformal import AdaptiveConformalSizer


class TestACIInit:
    def test_default_construction(self):
        sizer = AdaptiveConformalSizer()
        assert sizer.target_alpha == 0.1
        assert sizer.gamma == 0.01

    def test_custom_params(self):
        sizer = AdaptiveConformalSizer(
            target_alpha=0.2, gamma=0.05, min_alpha=0.05, max_alpha=0.40,
        )
        assert sizer.target_alpha == 0.2
        assert sizer.gamma == 0.05

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            AdaptiveConformalSizer(target_alpha=0.0)
        with pytest.raises(ValueError):
            AdaptiveConformalSizer(target_alpha=1.0)

    def test_invalid_alpha_bounds(self):
        with pytest.raises(ValueError):
            AdaptiveConformalSizer(min_alpha=0.5, max_alpha=0.1)

    def test_invalid_min_size_fraction(self):
        with pytest.raises(ValueError):
            AdaptiveConformalSizer(min_size_fraction=0.0)
        with pytest.raises(ValueError):
            AdaptiveConformalSizer(min_size_fraction=1.5)


class TestACIGracefulDegradation:
    def test_unfitted_returns_base_size(self):
        sizer = AdaptiveConformalSizer()
        X = np.array([[1.0, 2.0, 3.0]])
        result = sizer.size_single(X, base_size=0.10)
        assert result == 0.10

    def test_is_available_property(self):
        sizer = AdaptiveConformalSizer()
        # Should not crash regardless of MAPIE availability
        assert isinstance(sizer.is_available, bool)


class TestACIAlphaAdaptation:
    def test_alpha_increases_on_errors(self):
        sizer = AdaptiveConformalSizer(target_alpha=0.1, gamma=0.1)
        initial_alpha = sizer.current_alpha
        # Simulate errors (predictions were wrong)
        for _ in range(5):
            sizer.update(y_pred=1, y_true=0)
        assert sizer.current_alpha > initial_alpha

    def test_alpha_decreases_on_successes(self):
        sizer = AdaptiveConformalSizer(
            target_alpha=0.1, gamma=0.1, max_alpha=0.50,
        )
        # First push alpha up with errors
        for _ in range(10):
            sizer.update(y_pred=1, y_true=0)
        high_alpha = sizer.current_alpha
        assert high_alpha > 0.1, "Alpha should have increased from errors"
        # Then many successes should bring it down
        for _ in range(50):
            sizer.update(y_pred=1, y_true=1)
        assert sizer.current_alpha < high_alpha

    def test_alpha_stays_in_bounds(self):
        sizer = AdaptiveConformalSizer(
            target_alpha=0.1, gamma=0.5,
            min_alpha=0.02, max_alpha=0.25,
        )
        # Lots of errors
        for _ in range(50):
            sizer.update(y_pred=1, y_true=0)
        assert sizer.current_alpha <= 0.25

        # Lots of successes
        for _ in range(50):
            sizer.update(y_pred=1, y_true=1)
        assert sizer.current_alpha >= 0.02

    def test_diagnostics_returns_dict(self):
        sizer = AdaptiveConformalSizer()
        sizer.update(y_pred=1, y_true=1)
        sizer.update(y_pred=0, y_true=1)
        diag = sizer.get_diagnostics()
        assert "current_alpha" in diag
        assert "empirical_coverage" in diag
        assert "n_observations" in diag
        assert diag["n_observations"] == 2


class TestACIUpdate:
    def test_coverage_tracking(self):
        sizer = AdaptiveConformalSizer()
        # 8 correct, 2 wrong -> 80% coverage
        for _ in range(8):
            sizer.update(y_pred=1, y_true=1)
        for _ in range(2):
            sizer.update(y_pred=1, y_true=0)
        diag = sizer.get_diagnostics()
        assert abs(diag["empirical_coverage"] - 0.8) < 0.01

    def test_lookback_window_respected(self):
        sizer = AdaptiveConformalSizer(lookback_window=10)
        # Add 20 observations -- only last 10 should count
        for _ in range(20):
            sizer.update(y_pred=1, y_true=1)
        diag = sizer.get_diagnostics()
        assert diag["n_observations"] <= 10


class TestACIRepr:
    def test_repr(self):
        sizer = AdaptiveConformalSizer()
        r = repr(sizer)
        assert "AdaptiveConformalSizer" in r

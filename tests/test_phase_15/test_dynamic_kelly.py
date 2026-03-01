"""
Tests for DynamicKellySizer.
"""

import pytest
import numpy as np

from src.phase_15_strategy.dynamic_kelly_sizer import DynamicKellySizer


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def default_sizer():
    """DynamicKellySizer with default parameters."""
    return DynamicKellySizer()


@pytest.fixture
def normal_returns():
    """Generate returns from a known distribution."""
    rng = np.random.RandomState(42)
    return rng.normal(loc=0.001, scale=0.015, size=500)


# ─── Test: Kelly fraction computation ────────────────────────────────────────


class TestKellyFraction:

    def test_no_edge_zero_fraction(self, default_sizer):
        """p=0.5, b=1.0 -> Kelly = 0.5 - 0.5/1.0 = 0.0."""
        f = default_sizer._kelly_fraction(0.5, 1.0)
        assert f == pytest.approx(0.0, abs=1e-10)

    def test_positive_edge(self, default_sizer):
        """p=0.6, b=1.5 -> Kelly = 0.6 - 0.4/1.5 = 0.333."""
        f = default_sizer._kelly_fraction(0.6, 1.5)
        expected = 0.6 - 0.4 / 1.5
        assert f == pytest.approx(expected, abs=1e-10)

    def test_large_edge(self, default_sizer):
        """p=0.8, b=2.0 -> Kelly = 0.8 - 0.2/2.0 = 0.7."""
        f = default_sizer._kelly_fraction(0.8, 2.0)
        expected = 0.8 - 0.2 / 2.0
        assert f == pytest.approx(expected, abs=1e-10)

    def test_negative_edge_floored_at_zero(self, default_sizer):
        """p=0.3, b=1.0 -> Kelly = 0.3 - 0.7 = -0.4 -> floored to 0.0."""
        f = default_sizer._kelly_fraction(0.3, 1.0)
        assert f == pytest.approx(0.0, abs=1e-10)

    def test_zero_win_loss_ratio(self, default_sizer):
        """win_loss_ratio=0 should return 0.0 (avoid div by zero)."""
        f = default_sizer._kelly_fraction(0.6, 0.0)
        assert f == 0.0

    def test_negative_win_loss_ratio(self, default_sizer):
        """Negative win_loss_ratio should return 0.0."""
        f = default_sizer._kelly_fraction(0.6, -1.0)
        assert f == 0.0

    def test_perfect_edge(self, default_sizer):
        """p=1.0 should give Kelly = 1.0 - 0/b = 1.0."""
        f = default_sizer._kelly_fraction(1.0, 2.0)
        assert f == pytest.approx(1.0, abs=1e-10)

    def test_high_win_loss_ratio(self, default_sizer):
        """Very high b with modest p should produce positive Kelly."""
        f = default_sizer._kelly_fraction(0.4, 5.0)
        expected = 0.4 - 0.6 / 5.0
        assert f == pytest.approx(expected, abs=1e-10)
        assert f > 0


# ─── Test: VIX scaling ──────────────────────────────────────────────────────


class TestVIXScaling:

    def test_low_vix_highest_scale(self, default_sizer):
        """VIX < 15 should use the highest scale (0.50)."""
        scale = default_sizer._vix_scale(12.0)
        assert scale == 0.50

    def test_normal_vix_scale(self, default_sizer):
        """VIX in [15, 25) should use 0.35."""
        scale = default_sizer._vix_scale(20.0)
        assert scale == 0.35

    def test_high_vix_scale(self, default_sizer):
        """VIX in [25, 35) should use 0.20."""
        scale = default_sizer._vix_scale(30.0)
        assert scale == 0.20

    def test_extreme_vix_scale(self, default_sizer):
        """VIX >= 35 should use the lowest scale (0.10)."""
        scale = default_sizer._vix_scale(50.0)
        assert scale == 0.10

    def test_exactly_at_boundary(self, default_sizer):
        """VIX=15.0 should be in the second bin (>=15, <25)."""
        scale = default_sizer._vix_scale(15.0)
        assert scale == 0.35

    def test_custom_bins_and_scales(self):
        """Custom VIX bins and scales should work correctly."""
        sizer = DynamicKellySizer(vix_bins=[20.0], vix_scales=[0.8, 0.2])
        assert sizer._vix_scale(15.0) == 0.8
        assert sizer._vix_scale(25.0) == 0.2

    def test_monotonically_decreasing(self, default_sizer):
        """Higher VIX should give lower scale."""
        vix_levels = [10, 20, 30, 50]
        scales = [default_sizer._vix_scale(v) for v in vix_levels]
        for i in range(len(scales) - 1):
            assert scales[i] >= scales[i + 1]


# ─── Test: size() end-to-end ────────────────────────────────────────────────


class TestSizeEndToEnd:

    def test_low_vix_higher_position(self, default_sizer):
        """Same edge, lower VIX -> larger position."""
        pos_low_vix = default_sizer.size(win_probability=0.65, vix_level=12.0)
        pos_high_vix = default_sizer.size(win_probability=0.65, vix_level=30.0)
        assert pos_low_vix > pos_high_vix

    def test_higher_edge_larger_position(self, default_sizer):
        """Same VIX, higher win probability -> larger position."""
        pos_low_edge = default_sizer.size(win_probability=0.55, vix_level=20.0)
        pos_high_edge = default_sizer.size(win_probability=0.75, vix_level=20.0)
        assert pos_high_edge > pos_low_edge

    def test_no_edge_returns_min(self, default_sizer):
        """p=0.5, b=1.0 -> Kelly=0 -> position=min_position."""
        pos = default_sizer.size(win_probability=0.5, vix_level=20.0, win_loss_ratio=1.0)
        assert pos == default_sizer.min_position

    def test_extreme_vix_small_position(self, default_sizer):
        """VIX=50 should produce a very small position even with good edge."""
        pos = default_sizer.size(win_probability=0.6, vix_level=50.0)
        # Kelly(0.6, 1.5) = 0.333, scale=0.10, position=0.033
        assert pos < 0.05


# ─── Test: position bounds clipping ─────────────────────────────────────────


class TestBoundsClipping:

    def test_never_below_min(self):
        """Position should never go below min_position."""
        sizer = DynamicKellySizer(min_position=0.05)
        pos = sizer.size(win_probability=0.5, vix_level=50.0, win_loss_ratio=1.0)
        assert pos >= 0.05

    def test_never_above_max(self):
        """Position should never exceed max_position."""
        sizer = DynamicKellySizer(max_position=0.15)
        pos = sizer.size(win_probability=0.95, vix_level=10.0, win_loss_ratio=5.0)
        assert pos <= 0.15

    def test_clipping_at_max(self):
        """Very high edge + low VIX should clip to max_position."""
        sizer = DynamicKellySizer(max_position=0.20)
        pos = sizer.size(win_probability=0.9, vix_level=10.0, win_loss_ratio=3.0)
        assert pos == pytest.approx(0.20, abs=1e-10)


# ─── Test: fit() ────────────────────────────────────────────────────────────


class TestFit:

    def test_fit_returns_self(self, default_sizer, normal_returns):
        """fit() should return self for method chaining."""
        result = default_sizer.fit(normal_returns)
        assert result is default_sizer

    def test_fit_sets_estimated_edge(self, default_sizer, normal_returns):
        """After fit, estimated_edge should be set."""
        default_sizer.fit(normal_returns)
        assert default_sizer.estimated_edge is not None
        assert 0.0 <= default_sizer.estimated_edge <= 1.0

    def test_fit_sets_estimated_odds(self, default_sizer, normal_returns):
        """After fit, estimated_odds should be set and positive."""
        default_sizer.fit(normal_returns)
        assert default_sizer.estimated_odds is not None
        assert default_sizer.estimated_odds > 0

    def test_fit_not_fitted_initially(self, default_sizer):
        """Before fit, estimated properties should be None."""
        assert default_sizer.estimated_edge is None
        assert default_sizer.estimated_odds is None

    def test_fit_with_few_returns(self):
        """Fewer than 20 returns should use neutral estimates."""
        sizer = DynamicKellySizer()
        sizer.fit(np.array([0.01, -0.005, 0.003]))
        assert sizer._fitted is True
        assert sizer._estimated_edge == 0.0
        assert sizer._estimated_odds == 1.0

    def test_fit_with_custom_win_rate(self, normal_returns):
        """Custom win_rate override should be used."""
        sizer = DynamicKellySizer()
        sizer.fit(normal_returns, win_rate=0.7)
        assert sizer._estimated_edge == 0.7

    def test_fit_handles_nans(self):
        """NaN values in returns should be stripped."""
        rng = np.random.RandomState(42)
        returns = rng.normal(0.001, 0.01, size=100)
        returns[10] = np.nan
        returns[50] = np.nan

        sizer = DynamicKellySizer()
        sizer.fit(returns)
        assert sizer._fitted is True
        assert sizer.estimated_edge is not None


# ─── Test: constructor validation ────────────────────────────────────────────


class TestConstructorValidation:

    def test_mismatched_bins_scales_raises(self):
        """vix_scales must have len(vix_bins)+1 entries."""
        with pytest.raises(ValueError, match="vix_scales"):
            DynamicKellySizer(vix_bins=[15.0, 25.0], vix_scales=[0.5, 0.3])

    def test_repr_not_fitted(self):
        sizer = DynamicKellySizer()
        r = repr(sizer)
        assert "not fitted" in r

    def test_repr_fitted(self, normal_returns):
        sizer = DynamicKellySizer()
        sizer.fit(normal_returns)
        r = repr(sizer)
        assert "fitted" in r
        assert "edge=" in r
        assert "odds=" in r

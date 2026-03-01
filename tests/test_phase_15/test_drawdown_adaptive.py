"""
Tests for DrawdownAdaptiveSizer.
"""

import pytest
import numpy as np

from src.phase_15_strategy.drawdown_adaptive_sizer import DrawdownAdaptiveSizer


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def default_sizer():
    """DrawdownAdaptiveSizer with default parameters."""
    return DrawdownAdaptiveSizer()


@pytest.fixture
def equity_with_drawdown():
    """Equity curve that peaks at 110 then drops to 100 (9.09% drawdown)."""
    return np.array([100, 102, 105, 108, 110, 107, 103, 100])


@pytest.fixture
def flat_equity():
    """Flat equity curve (no drawdown)."""
    return np.array([100, 100, 100, 100, 100])


@pytest.fixture
def rising_equity():
    """Monotonically rising equity curve (no drawdown)."""
    return np.array([100, 101, 102, 103, 104, 105])


# ─── Test: zero drawdown -> full position ────────────────────────────────────


class TestZeroDrawdown:

    def test_zero_dd_returns_base(self, default_sizer):
        """Zero drawdown should return base position."""
        pos = default_sizer.size(base_position=0.20, current_drawdown=0.0)
        assert pos == pytest.approx(0.20, abs=1e-10)

    def test_negative_dd_returns_base(self, default_sizer):
        """Negative drawdown (shouldn't happen, but defensive) returns base."""
        pos = default_sizer.size(base_position=0.15, current_drawdown=-0.01)
        assert pos == pytest.approx(0.15, abs=1e-10)

    def test_none_dd_returns_base(self, default_sizer):
        """None drawdown (not fitted, no explicit dd) returns base."""
        pos = default_sizer.size(base_position=0.20)
        assert pos == pytest.approx(0.20, abs=1e-10)


# ─── Test: max drawdown -> min position ──────────────────────────────────────


class TestMaxDrawdown:

    def test_at_max_dd_returns_min(self, default_sizer):
        """Drawdown exactly at max_drawdown_pct should return min_position."""
        pos = default_sizer.size(base_position=0.20, current_drawdown=0.10)
        assert pos == default_sizer.min_position

    def test_beyond_max_dd_returns_min(self, default_sizer):
        """Drawdown beyond max_drawdown_pct should still return min_position."""
        pos = default_sizer.size(base_position=0.20, current_drawdown=0.15)
        assert pos == default_sizer.min_position

    def test_custom_max_dd(self):
        """Custom max_drawdown should work."""
        sizer = DrawdownAdaptiveSizer(max_drawdown=0.05, min_position=0.01)
        pos = sizer.size(base_position=0.20, current_drawdown=0.05)
        assert pos == 0.01


# ─── Test: quadratic decay ──────────────────────────────────────────────────


class TestQuadraticDecay:

    def test_half_dd_gives_quarter_with_power_2(self):
        """50% of max_dd with power=2 -> (1-0.5)^2 = 0.25 of base."""
        sizer = DrawdownAdaptiveSizer(
            max_drawdown=0.10, power=2.0,
            min_position=0.001, max_position=0.50,
        )
        pos = sizer.size(base_position=0.20, current_drawdown=0.05)
        expected = 0.20 * 0.25  # 0.05
        assert pos == pytest.approx(expected, abs=1e-10)

    def test_quarter_dd_with_power_2(self):
        """25% of max_dd with power=2 -> (1-0.25)^2 = 0.5625 of base."""
        sizer = DrawdownAdaptiveSizer(
            max_drawdown=0.10, power=2.0,
            min_position=0.001, max_position=0.50,
        )
        pos = sizer.size(base_position=0.20, current_drawdown=0.025)
        expected = 0.20 * (0.75 ** 2)  # 0.20 * 0.5625 = 0.1125
        assert pos == pytest.approx(expected, abs=1e-10)

    def test_75_pct_dd_with_power_2(self):
        """75% of max_dd with power=2 -> (1-0.75)^2 = 0.0625 of base."""
        sizer = DrawdownAdaptiveSizer(
            max_drawdown=0.10, power=2.0,
            min_position=0.001, max_position=0.50,
        )
        pos = sizer.size(base_position=0.20, current_drawdown=0.075)
        expected = 0.20 * (0.25 ** 2)  # 0.20 * 0.0625 = 0.0125
        assert pos == pytest.approx(expected, abs=1e-10)

    def test_linear_decay_with_power_1(self):
        """power=1 -> linear scaling."""
        sizer = DrawdownAdaptiveSizer(
            max_drawdown=0.10, power=1.0,
            min_position=0.001, max_position=0.50,
        )
        pos = sizer.size(base_position=0.20, current_drawdown=0.05)
        expected = 0.20 * 0.5  # 0.10
        assert pos == pytest.approx(expected, abs=1e-10)

    def test_cubic_decay_with_power_3(self):
        """power=3 -> cubic (faster decay)."""
        sizer = DrawdownAdaptiveSizer(
            max_drawdown=0.10, power=3.0,
            min_position=0.001, max_position=0.50,
        )
        pos = sizer.size(base_position=0.20, current_drawdown=0.05)
        expected = 0.20 * (0.5 ** 3)  # 0.20 * 0.125 = 0.025
        assert pos == pytest.approx(expected, abs=1e-10)


# ─── Test: monotonic decay ──────────────────────────────────────────────────


class TestMonotonicDecay:

    def test_position_decreases_with_deeper_drawdown(self, default_sizer):
        """Larger drawdown should always give smaller (or equal) position."""
        dd_levels = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
        positions = [
            default_sizer.size(base_position=0.20, current_drawdown=dd)
            for dd in dd_levels
        ]
        for i in range(len(positions) - 1):
            assert positions[i] >= positions[i + 1], (
                f"Position at dd={dd_levels[i]} ({positions[i]}) should be >= "
                f"position at dd={dd_levels[i+1]} ({positions[i+1]})"
            )


# ─── Test: bounds clipping ──────────────────────────────────────────────────


class TestBoundsClipping:

    def test_never_below_min(self):
        """Position should never go below min_position."""
        sizer = DrawdownAdaptiveSizer(min_position=0.05)
        pos = sizer.size(base_position=0.20, current_drawdown=0.09)
        assert pos >= 0.05

    def test_never_above_max(self):
        """Position should never exceed max_position."""
        sizer = DrawdownAdaptiveSizer(max_position=0.15)
        pos = sizer.size(base_position=0.30, current_drawdown=0.0)
        assert pos <= 0.15

    def test_base_above_max_clips(self):
        """Base position larger than max should clip to max."""
        sizer = DrawdownAdaptiveSizer(max_position=0.20)
        pos = sizer.size(base_position=0.50, current_drawdown=0.0)
        assert pos == pytest.approx(0.20, abs=1e-10)

    def test_small_scaled_clips_to_min(self):
        """Very small scaled position should clip to min_position."""
        sizer = DrawdownAdaptiveSizer(
            max_drawdown=0.10, min_position=0.03, power=2.0
        )
        # dd=0.09 -> scale = (1-0.9)^2 = 0.01, pos=0.20*0.01=0.002 -> clips to 0.03
        pos = sizer.size(base_position=0.20, current_drawdown=0.09)
        assert pos == pytest.approx(0.03, abs=1e-10)


# ─── Test: fit() from equity curve ──────────────────────────────────────────


class TestFit:

    def test_fit_returns_self(self, default_sizer, rising_equity):
        """fit() should return self for method chaining."""
        result = default_sizer.fit(rising_equity)
        assert result is default_sizer

    def test_fit_rising_equity_zero_drawdown(self, default_sizer, rising_equity):
        """Rising equity curve should have zero drawdown."""
        default_sizer.fit(rising_equity)
        assert default_sizer.current_drawdown == pytest.approx(0.0, abs=1e-8)

    def test_fit_flat_equity_zero_drawdown(self, default_sizer, flat_equity):
        """Flat equity curve should have zero drawdown."""
        default_sizer.fit(flat_equity)
        assert default_sizer.current_drawdown == pytest.approx(0.0, abs=1e-8)

    def test_fit_with_drawdown(self, default_sizer, equity_with_drawdown):
        """Equity [100,102,...,110,107,103,100] has dd = (110-100)/110 ~ 0.0909."""
        default_sizer.fit(equity_with_drawdown)
        expected_dd = (110 - 100) / 110  # ~0.0909
        assert default_sizer.current_drawdown == pytest.approx(expected_dd, abs=1e-3)

    def test_fit_peak_equity(self, default_sizer, equity_with_drawdown):
        """Peak equity should be 110."""
        default_sizer.fit(equity_with_drawdown)
        assert default_sizer._peak == pytest.approx(110.0, abs=1e-3)

    def test_fit_then_size_uses_fitted_dd(self, equity_with_drawdown):
        """After fit(), size() without explicit dd should use fitted value."""
        sizer = DrawdownAdaptiveSizer(max_drawdown=0.10, min_position=0.01, max_position=0.50)
        sizer.fit(equity_with_drawdown)
        pos = sizer.size(base_position=0.20)
        # dd ~0.0909, scale = (1 - 0.909)^2 ~= 0.0083, pos = 0.20 * 0.0083 ~= 0.0017 -> clips to min
        assert pos == sizer.min_position

    def test_fit_explicit_dd_overrides(self, default_sizer, equity_with_drawdown):
        """Explicit current_drawdown should override the fitted value."""
        default_sizer.fit(equity_with_drawdown)
        pos_fitted = default_sizer.size(base_position=0.20)
        pos_zero = default_sizer.size(base_position=0.20, current_drawdown=0.0)
        assert pos_zero > pos_fitted

    def test_fit_single_point(self):
        """Single-point equity curve: fewer than 2 values -> not fitted."""
        sizer = DrawdownAdaptiveSizer()
        sizer.fit(np.array([100.0]))
        # Source requires >= 2 values; single-point won't fit
        assert sizer.current_drawdown is None

    def test_fit_empty_array(self):
        """Empty equity curve should be handled gracefully."""
        sizer = DrawdownAdaptiveSizer()
        sizer.fit(np.array([]))
        # With 0 elements, current_drawdown defaults to None or 0
        pos = sizer.size(base_position=0.20)
        # Should return base position (dd=0)
        assert pos == pytest.approx(0.20, abs=1e-10)


# ─── Test: constructor validation ────────────────────────────────────────────


class TestConstructorValidation:

    def test_invalid_max_drawdown(self):
        with pytest.raises(ValueError, match="max_drawdown"):
            DrawdownAdaptiveSizer(max_drawdown=0.0)
        with pytest.raises(ValueError, match="max_drawdown"):
            DrawdownAdaptiveSizer(max_drawdown=-0.05)

    def test_invalid_position_bounds(self):
        with pytest.raises(ValueError):
            DrawdownAdaptiveSizer(min_position=0.30, max_position=0.20)

    def test_invalid_power(self):
        with pytest.raises(ValueError, match="power"):
            DrawdownAdaptiveSizer(power=0.0)
        with pytest.raises(ValueError, match="power"):
            DrawdownAdaptiveSizer(power=-1.0)

    def test_repr(self):
        sizer = DrawdownAdaptiveSizer(max_drawdown=0.10, power=2.0)
        r = repr(sizer)
        assert "max_dd=0.1" in r
        assert "power=2.0" in r

    def test_repr_with_dd(self, equity_with_drawdown):
        sizer = DrawdownAdaptiveSizer()
        sizer.fit(equity_with_drawdown)
        r = repr(sizer)
        assert "dd=" in r


# ─── Test: properties ────────────────────────────────────────────────────────


class TestProperties:

    def test_current_drawdown_none_initially(self, default_sizer):
        assert default_sizer.current_drawdown is None

    def test_peak_none_initially(self, default_sizer):
        assert default_sizer._peak is None

    def test_properties_after_fit(self, default_sizer, equity_with_drawdown):
        default_sizer.fit(equity_with_drawdown)
        assert default_sizer.current_drawdown is not None
        assert default_sizer._peak is not None
        assert default_sizer.current_drawdown >= 0
        assert default_sizer._peak > 0

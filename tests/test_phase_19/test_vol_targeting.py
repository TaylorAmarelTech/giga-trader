"""Tests for volatility-targeting position sizing and VIX-conditional Kelly."""

import math
import numpy as np
import pytest

from src.phase_19_paper_trading.signal_generator import (
    vol_target_position_size,
    vix_adjusted_kelly,
)


# ─── Volatility Targeting Tests ──────────────────────────────────────────────

class TestVolTargetPositionSize:
    def test_normal_vol_returns_base_size(self):
        """When realized vol equals target vol, should return base size."""
        result = vol_target_position_size(0.15, target_vol=0.15, base_size=0.10)
        assert abs(result - 0.10) < 0.001

    def test_high_vol_reduces_size(self):
        """Higher vol should produce smaller position."""
        normal = vol_target_position_size(0.15, target_vol=0.15, base_size=0.10)
        high = vol_target_position_size(0.30, target_vol=0.15, base_size=0.10)
        assert high < normal

    def test_low_vol_increases_size(self):
        """Lower vol should produce larger position."""
        normal = vol_target_position_size(0.15, target_vol=0.15, base_size=0.10)
        low = vol_target_position_size(0.08, target_vol=0.15, base_size=0.10)
        assert low > normal

    def test_clamped_to_max(self):
        """Very low vol should be capped at max_size."""
        result = vol_target_position_size(0.01, target_vol=0.15, base_size=0.10, max_size=0.25)
        assert result == 0.25

    def test_clamped_to_min(self):
        """Very high vol should be floored at min_size."""
        result = vol_target_position_size(5.0, target_vol=0.15, base_size=0.10, min_size=0.02)
        assert result == 0.02

    def test_zero_vol_returns_base(self):
        """Zero vol should return base size (guard)."""
        result = vol_target_position_size(0.0, base_size=0.10)
        assert result == 0.10

    def test_negative_vol_returns_base(self):
        """Negative vol should return base size (guard)."""
        result = vol_target_position_size(-0.15, base_size=0.10)
        assert result == 0.10

    def test_nan_vol_returns_base(self):
        """NaN vol should return base size (guard)."""
        result = vol_target_position_size(float('nan'), base_size=0.10)
        assert result == 0.10

    def test_inf_vol_returns_base(self):
        """Inf vol should return base size (guard)."""
        result = vol_target_position_size(float('inf'), base_size=0.10)
        assert result == 0.10

    def test_inverse_relationship(self):
        """Doubling vol should roughly halve position size (within bounds)."""
        size_15 = vol_target_position_size(0.15, target_vol=0.15, base_size=0.10)
        size_30 = vol_target_position_size(0.30, target_vol=0.15, base_size=0.10)
        # 0.30 is 2x of 0.15, so position should be ~half
        ratio = size_30 / size_15
        assert abs(ratio - 0.5) < 0.05

    def test_custom_parameters(self):
        """Custom target_vol and base_size work correctly."""
        result = vol_target_position_size(
            0.20, target_vol=0.20, base_size=0.15
        )
        assert abs(result - 0.15) < 0.001

    def test_returns_float(self):
        result = vol_target_position_size(0.15)
        assert isinstance(result, float)


# ─── VIX-Conditional Kelly Tests ─────────────────────────────────────────────

class TestVixAdjustedKelly:
    def test_calm_market_highest_fraction(self):
        """VIX < 15 should give highest scaling (0.50)."""
        result = vix_adjusted_kelly(0.10, vix_level=12.0)
        assert abs(result - 0.05) < 0.001  # 0.10 * 0.50

    def test_normal_market(self):
        """VIX 15-25 should give 0.35 scaling."""
        result = vix_adjusted_kelly(0.10, vix_level=20.0)
        assert abs(result - 0.035) < 0.001  # 0.10 * 0.35

    def test_elevated_market(self):
        """VIX 25-35 should give 0.20 scaling."""
        result = vix_adjusted_kelly(0.10, vix_level=30.0)
        assert abs(result - 0.020) < 0.001  # 0.10 * 0.20

    def test_fear_market(self):
        """VIX > 35 should give 0.10 scaling."""
        result = vix_adjusted_kelly(0.10, vix_level=45.0)
        assert abs(result - 0.010) < 0.001  # 0.10 * 0.10

    def test_monotonically_decreasing(self):
        """Higher VIX should always produce smaller or equal position."""
        kelly = 0.20
        levels = [10, 15, 20, 25, 30, 35, 40, 50]
        sizes = [vix_adjusted_kelly(kelly, v) for v in levels]
        for i in range(len(sizes) - 1):
            assert sizes[i] >= sizes[i + 1], f"VIX {levels[i]} -> {levels[i+1]}: {sizes[i]} < {sizes[i+1]}"

    def test_clamped_to_max(self):
        """Large Kelly fraction * 0.50 should be capped at max_fraction."""
        result = vix_adjusted_kelly(1.0, vix_level=10.0, max_fraction=0.25)
        assert result == 0.25

    def test_clamped_to_min(self):
        """Tiny Kelly * fear scale should be floored at min_fraction."""
        result = vix_adjusted_kelly(0.01, vix_level=50.0, min_fraction=0.01)
        assert result == 0.01

    def test_zero_vix_default(self):
        """Zero VIX should use default scaling (normal regime)."""
        result = vix_adjusted_kelly(0.10, vix_level=0.0)
        assert abs(result - 0.035) < 0.001  # Default to 0.35 scale

    def test_nan_vix_default(self):
        """NaN VIX should use default scaling."""
        result = vix_adjusted_kelly(0.10, vix_level=float('nan'))
        assert abs(result - 0.035) < 0.001

    def test_negative_vix_default(self):
        """Negative VIX should use default scaling."""
        result = vix_adjusted_kelly(0.10, vix_level=-5.0)
        assert abs(result - 0.035) < 0.001

    def test_returns_float(self):
        result = vix_adjusted_kelly(0.10, vix_level=20.0)
        assert isinstance(result, float)

    def test_boundary_15(self):
        """VIX exactly 15 should use normal regime (0.35)."""
        result = vix_adjusted_kelly(0.10, vix_level=15.0)
        assert abs(result - 0.035) < 0.001

    def test_boundary_25(self):
        """VIX exactly 25 should use elevated regime (0.20)."""
        result = vix_adjusted_kelly(0.10, vix_level=25.0)
        assert abs(result - 0.020) < 0.001

    def test_boundary_35(self):
        """VIX exactly 35 should use fear regime (0.10)."""
        result = vix_adjusted_kelly(0.10, vix_level=35.0)
        assert abs(result - 0.010) < 0.001

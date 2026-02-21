"""
Tests for the synthetic weight penalty system.

Validates that:
  - Config fields exist and have correct defaults
  - Penalty is correctly applied to synthetic rows
  - Floor and ceiling bounds are enforced
  - Real rows are unaffected by the penalty
"""

import numpy as np
import pandas as pd
import pytest

from src.experiment_config import AntiOverfitConfig


# ─── Config Tests ────────────────────────────────────────────────────────────

class TestWeightPenaltyConfig:

    def test_default_penalty(self):
        config = AntiOverfitConfig()
        assert config.synthetic_weight_penalty == 0.5

    def test_default_floor(self):
        config = AntiOverfitConfig()
        assert config.synthetic_weight_floor == 0.10

    def test_default_ceiling(self):
        config = AntiOverfitConfig()
        assert config.synthetic_weight_ceiling == 0.60

    def test_custom_values(self):
        config = AntiOverfitConfig(
            synthetic_weight_penalty=0.3,
            synthetic_weight_floor=0.05,
            synthetic_weight_ceiling=0.40,
        )
        assert config.synthetic_weight_penalty == 0.3
        assert config.synthetic_weight_floor == 0.05
        assert config.synthetic_weight_ceiling == 0.40


# ─── Penalty Application Tests ──────────────────────────────────────────────

def _apply_penalty(
    weights: np.ndarray,
    sample_weight_augment: np.ndarray,
    real_weight: float = 0.6,
    penalty: float = 0.5,
    floor: float = 0.10,
    ceiling: float = 0.60,
) -> np.ndarray:
    """Replicate the penalty logic from experiment_runner.py."""
    weights = weights.copy()
    is_synthetic = sample_weight_augment < real_weight - 0.01
    n_synth = int(is_synthetic.sum())
    if n_synth > 0:
        weights[is_synthetic] = np.clip(
            weights[is_synthetic] * penalty, floor, ceiling
        )
    return weights


class TestPenaltyApplication:

    def test_synthetic_rows_penalized(self):
        """Synthetic rows should have lower weights after penalty."""
        weights = np.array([0.8, 0.8, 0.8, 0.8])  # All confidence=0.8
        augment = np.array([0.6, 0.6, 0.02, 0.02])  # First 2 real, last 2 synthetic
        result = _apply_penalty(weights, augment, real_weight=0.6)
        # Real rows unchanged
        assert result[0] == 0.8
        assert result[1] == 0.8
        # Synthetic: 0.8 * 0.5 = 0.4, clipped to [0.10, 0.60]
        assert result[2] == pytest.approx(0.4)
        assert result[3] == pytest.approx(0.4)

    def test_real_rows_unaffected(self):
        """Rows with sample_weight_augment >= real_weight should not be penalized."""
        weights = np.array([0.9, 0.7, 0.5, 0.3])
        augment = np.array([0.6, 0.6, 0.6, 0.6])  # All real
        result = _apply_penalty(weights, augment, real_weight=0.6)
        np.testing.assert_array_equal(result, weights)

    def test_floor_enforced(self):
        """Very low confidence synthetic samples should not go below floor."""
        weights = np.array([0.1])  # Very low confidence
        augment = np.array([0.02])  # Synthetic
        result = _apply_penalty(weights, augment, real_weight=0.6, penalty=0.5, floor=0.10)
        # 0.1 * 0.5 = 0.05, but floor is 0.10
        assert result[0] == pytest.approx(0.10)

    def test_ceiling_enforced(self):
        """High confidence synthetic samples should not exceed ceiling."""
        weights = np.array([1.0])  # Max confidence
        augment = np.array([0.02])  # Synthetic
        result = _apply_penalty(weights, augment, real_weight=0.6, penalty=0.9, ceiling=0.60)
        # 1.0 * 0.9 = 0.9, but ceiling is 0.60
        assert result[0] == pytest.approx(0.60)

    def test_no_synthetic_rows(self):
        """If no synthetic rows, all weights unchanged."""
        weights = np.array([0.8, 0.7, 0.6])
        augment = np.array([0.6, 0.6, 0.6])
        result = _apply_penalty(weights, augment, real_weight=0.6)
        np.testing.assert_array_equal(result, weights)

    def test_all_synthetic_rows(self):
        """If all rows are synthetic, all get penalized."""
        weights = np.array([0.8, 0.6, 0.4])
        augment = np.array([0.02, 0.02, 0.02])
        result = _apply_penalty(weights, augment, real_weight=0.6)
        expected = np.clip(weights * 0.5, 0.10, 0.60)
        np.testing.assert_array_almost_equal(result, expected)

    def test_mixed_weights(self):
        """Test with realistic mixed real/synthetic data."""
        n_real = 100
        n_synth = 80
        np.random.seed(42)

        weights = np.clip(np.abs(np.random.normal(0.65, 0.2, n_real + n_synth)), 0.3, 1.0)
        augment = np.concatenate([
            np.full(n_real, 0.6),  # Real
            np.full(n_synth, 0.02),  # Synthetic
        ])

        result = _apply_penalty(weights, augment, real_weight=0.6, penalty=0.5)

        # Real weights unchanged
        np.testing.assert_array_equal(result[:n_real], weights[:n_real])

        # Synthetic weights reduced
        assert np.mean(result[n_real:]) < np.mean(weights[n_real:])

        # All synthetic within bounds
        assert np.all(result[n_real:] >= 0.10)
        assert np.all(result[n_real:] <= 0.60)

    def test_penalty_of_one_no_change(self):
        """Penalty = 1.0 means no reduction (but ceiling still applies)."""
        weights = np.array([0.5])
        augment = np.array([0.02])
        result = _apply_penalty(
            weights, augment, real_weight=0.6,
            penalty=1.0, ceiling=0.60,
        )
        # 0.5 * 1.0 = 0.5, within [0.10, 0.60]
        assert result[0] == pytest.approx(0.5)

    def test_penalty_of_zero(self):
        """Penalty = 0.0 means all synthetic go to floor."""
        weights = np.array([0.8, 0.5])
        augment = np.array([0.02, 0.02])
        result = _apply_penalty(
            weights, augment, real_weight=0.6,
            penalty=0.0, floor=0.10,
        )
        assert result[0] == pytest.approx(0.10)
        assert result[1] == pytest.approx(0.10)


# ─── Integration Smoke Test ─────────────────────────────────────────────────

class TestIntegrationSmoke:

    def test_penalty_fields_in_dataclass(self):
        """All three penalty fields exist in AntiOverfitConfig."""
        config = AntiOverfitConfig()
        assert hasattr(config, "synthetic_weight_penalty")
        assert hasattr(config, "synthetic_weight_floor")
        assert hasattr(config, "synthetic_weight_ceiling")

    def test_penalty_bounds_sensible(self):
        """Floor < penalty * max_confidence_weight < ceiling by default."""
        config = AntiOverfitConfig()
        # Max confidence weight is 1.0
        # 1.0 * 0.5 = 0.5, which is within [0.10, 0.60]
        penalized_max = 1.0 * config.synthetic_weight_penalty
        assert config.synthetic_weight_floor < penalized_max
        assert penalized_max <= config.synthetic_weight_ceiling

    def test_floor_less_than_ceiling(self):
        config = AntiOverfitConfig()
        assert config.synthetic_weight_floor < config.synthetic_weight_ceiling

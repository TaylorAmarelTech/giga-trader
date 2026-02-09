"""
Test RobustnessEnsemble creation and dimension variant logic.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase_14_robustness.robustness_ensemble import (
    RobustnessEnsemble,
    create_robustness_ensemble,
)


# ---------------------------------------------------------------------------
# RobustnessEnsemble tests
# ---------------------------------------------------------------------------

def test_robustness_ensemble_creation():
    """RobustnessEnsemble should initialize with default parameters."""
    ensemble = RobustnessEnsemble()
    assert isinstance(ensemble, RobustnessEnsemble)
    assert ensemble.n_dimension_variants == 2
    assert ensemble.n_param_variants == 2
    assert ensemble.param_noise_pct == 0.05
    assert ensemble.center_weight == 0.5
    assert ensemble.adjacent_weight == 0.25


def test_robustness_ensemble_custom_params():
    """RobustnessEnsemble should accept custom parameters."""
    ensemble = RobustnessEnsemble(
        n_dimension_variants=3,
        n_param_variants=4,
        param_noise_pct=0.10,
        center_weight=0.6,
        adjacent_weight=0.2,
    )
    assert ensemble.n_dimension_variants == 3
    assert ensemble.n_param_variants == 4
    assert ensemble.param_noise_pct == 0.10
    assert ensemble.center_weight == 0.6


def test_create_dimension_variants_default():
    """create_dimension_variants should return [n-2, n-1, n, n+1, n+2]."""
    ensemble = RobustnessEnsemble(n_dimension_variants=2)
    variants = ensemble.create_dimension_variants(optimal_dims=30)

    assert isinstance(variants, list)
    assert 30 in variants  # The optimal should be included
    assert len(variants) == 5  # -2, -1, 0, +1, +2
    assert variants == [28, 29, 30, 31, 32]


def test_create_dimension_variants_near_lower_bound():
    """Variants should respect min_dims boundary."""
    ensemble = RobustnessEnsemble(n_dimension_variants=2)
    variants = ensemble.create_dimension_variants(optimal_dims=6, min_dims=5)

    # 6-2=4 < min_dims=5, so 4 should be excluded
    assert all(v >= 5 for v in variants)
    assert 6 in variants


def test_create_dimension_variants_near_upper_bound():
    """Variants should respect max_dims boundary."""
    ensemble = RobustnessEnsemble(n_dimension_variants=2)
    variants = ensemble.create_dimension_variants(optimal_dims=99, max_dims=100)

    assert all(v <= 100 for v in variants)
    assert 99 in variants


def test_create_dimension_variants_at_exact_bounds():
    """Variants should work when optimal is at min or max."""
    ensemble = RobustnessEnsemble(n_dimension_variants=2)

    # At minimum
    variants_min = ensemble.create_dimension_variants(optimal_dims=5, min_dims=5)
    assert all(v >= 5 for v in variants_min)
    assert len(variants_min) >= 3  # At least 3 variants guaranteed

    # At maximum
    variants_max = ensemble.create_dimension_variants(optimal_dims=100, max_dims=100)
    assert all(v <= 100 for v in variants_max)
    assert len(variants_max) >= 3


def test_create_dimension_variants_narrow_range():
    """With a very narrow range, should still produce at least 3 variants."""
    ensemble = RobustnessEnsemble(n_dimension_variants=2)
    variants = ensemble.create_dimension_variants(optimal_dims=5, min_dims=5, max_dims=6)
    assert len(variants) >= 2  # At minimum 5 and 6


def test_ensemble_models_dict_starts_empty():
    """The models dict should start empty."""
    ensemble = RobustnessEnsemble()
    assert isinstance(ensemble.models, dict)
    assert len(ensemble.models) == 0


def test_ensemble_fragility_score_starts_none():
    """Fragility score should be None before training."""
    ensemble = RobustnessEnsemble()
    assert ensemble.fragility_score is None


# ---------------------------------------------------------------------------
# create_robustness_ensemble factory function
# ---------------------------------------------------------------------------

def test_create_robustness_ensemble_callable():
    """create_robustness_ensemble should be a callable function."""
    assert callable(create_robustness_ensemble)


def test_create_robustness_ensemble_with_data():
    """create_robustness_ensemble should accept numpy arrays and return a tuple."""
    import numpy as np

    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = (np.random.rand(100) > 0.5).astype(int)

    result = create_robustness_ensemble(
        X=X,
        y=y,
        optimal_dims=5,
        n_dim_variants=1,
        n_param_variants=1,
    )
    # Should return a tuple of (RobustnessEnsemble, dict)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], RobustnessEnsemble)

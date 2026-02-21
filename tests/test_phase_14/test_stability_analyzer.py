"""
Tests for Wave 29: Multi-radius stability analysis.
"""

import pytest
import numpy as np
from src.phase_14_robustness.stability_analyzer import StabilityAnalyzer


class TestMultiRadiusStability:
    """Test multi-radius perturbation in StabilityAnalyzer."""

    def test_constant_score_fn_gives_high_stability(self):
        """A constant score_fn should get stability at the v3 cap (0.70).

        v3 caps stability at 0.70 when sensitivity is unmeasurably low,
        because zero sensitivity means we couldn't distinguish parameter
        quality, not that the model is perfectly robust.
        """
        analyzer = StabilityAnalyzer()

        def constant_fn(params):
            return 0.58

        result = analyzer.compute_stability_score(
            base_params={"l2_C": 0.01},
            base_score=0.58,
            param_ranges={"l2_C": (0.001, 1.0)},
            score_fn=constant_fn,
            n_samples=24,
        )
        assert result["stability_score"] >= 0.65, (
            f"Constant score_fn should give stability at v3 cap (~0.70), got {result['stability_score']:.3f}"
        )

    def test_highly_sensitive_fn_gives_low_stability(self):
        """A model whose AUC varies wildly with params should get low stability."""
        analyzer = StabilityAnalyzer()

        def sensitive_fn(params):
            # AUC varies linearly with C: 0.50 at C=0.001, 0.70 at C=1.0
            c = params["l2_C"]
            return 0.50 + 0.20 * (c - 0.001) / (1.0 - 0.001)

        result = analyzer.compute_stability_score(
            base_params={"l2_C": 0.01},
            base_score=0.50 + 0.20 * (0.01 - 0.001) / (1.0 - 0.001),
            param_ranges={"l2_C": (0.001, 1.0)},
            score_fn=sensitive_fn,
            n_samples=24,
        )
        assert result["stability_score"] < 0.60, (
            f"Highly sensitive fn should give stability < 0.60, got {result['stability_score']:.3f}"
        )

    def test_three_rings_present(self):
        """Result should contain per-ring details for local, moderate, and wide."""
        analyzer = StabilityAnalyzer()

        def dummy_fn(params):
            return 0.57 + np.random.normal(0, 0.001)

        result = analyzer.compute_stability_score(
            base_params={"l2_C": 0.01},
            base_score=0.57,
            param_ranges={"l2_C": (0.001, 1.0)},
            score_fn=dummy_fn,
            n_samples=24,
        )
        assert "per_ring" in result
        assert "local" in result["per_ring"]
        assert "moderate" in result["per_ring"]
        assert "wide" in result["per_ring"]

        # Each ring should have at least 4 scores
        for ring_name in ("local", "moderate", "wide"):
            assert result["per_ring"][ring_name]["n_scores"] >= 4

    def test_wide_ring_explores_full_range(self):
        """Wide ring should sample from across the full param range, not just near base."""
        analyzer = StabilityAnalyzer()
        sampled_values = []

        def tracking_fn(params):
            sampled_values.append(params["l2_C"])
            return 0.57

        analyzer.compute_stability_score(
            base_params={"l2_C": 0.01},
            base_score=0.57,
            param_ranges={"l2_C": (0.001, 1.0)},
            score_fn=tracking_fn,
            n_samples=24,
        )

        # Wide ring samples should include values far from base (0.01)
        # At least some should be > 0.1 (10x base) and some > 0.3
        assert max(sampled_values) > 0.1, (
            f"Wide ring should explore far from base=0.01, max was {max(sampled_values):.4f}"
        )

    def test_joint_perturbation_changes_all_params(self):
        """With multiple params, all should be perturbed simultaneously."""
        analyzer = StabilityAnalyzer()
        param_combos = []

        def tracking_fn(params):
            param_combos.append(params.copy())
            return 0.57

        base = {"gb_n_estimators": 100, "gb_max_depth": 3, "gb_learning_rate": 0.1}
        ranges = {
            "gb_n_estimators": (30, 150),
            "gb_max_depth": (2, 5),
            "gb_learning_rate": (0.01, 0.3),
        }

        analyzer.compute_stability_score(
            base_params=base,
            base_score=0.57,
            param_ranges=ranges,
            score_fn=tracking_fn,
            n_samples=24,
        )

        # At least some samples should have ALL params different from base
        joint_changes = 0
        for combo in param_combos:
            all_different = all(
                combo[k] != base[k] for k in base
            )
            if all_different:
                joint_changes += 1

        assert joint_changes >= 5, (
            f"Expected at least 5 joint perturbations, got {joint_changes} out of {len(param_combos)}"
        )

    def test_n_samples_controls_total_evaluations(self):
        """Total evaluations should be approximately n_samples."""
        analyzer = StabilityAnalyzer()
        call_count = 0

        def counting_fn(params):
            nonlocal call_count
            call_count += 1
            return 0.57

        analyzer.compute_stability_score(
            base_params={"l2_C": 0.01},
            base_score=0.57,
            param_ranges={"l2_C": (0.001, 1.0)},
            score_fn=counting_fn,
            n_samples=24,
        )

        # 3 rings × 8 samples = 24 calls
        assert 20 <= call_count <= 30, (
            f"Expected ~24 evaluations, got {call_count}"
        )

    def test_wide_ring_has_most_weight(self):
        """Wide ring should have 50% weight, making it hardest to fake stability."""
        analyzer = StabilityAnalyzer()

        # Score fn that is stable locally but unstable at wide range
        def locally_stable_fn(params):
            c = params["l2_C"]
            if abs(c - 0.01) < 0.01:  # Near base
                return 0.57
            else:  # Far from base
                return 0.50 + np.random.uniform(0, 0.04)

        result = analyzer.compute_stability_score(
            base_params={"l2_C": 0.01},
            base_score=0.57,
            param_ranges={"l2_C": (0.001, 1.0)},
            score_fn=locally_stable_fn,
            n_samples=24,
        )

        # Local ring should be stable, wide ring should not
        assert result["per_ring"]["local"]["stability"] > result["per_ring"]["wide"]["stability"], (
            "Local ring should be more stable than wide ring for locally-stable model"
        )
        # Overall stability should be dragged down by wide ring
        assert result["stability_score"] < 0.90, (
            f"Locally-stable but globally-unstable model should have stability < 0.90, "
            f"got {result['stability_score']:.3f}"
        )

    def test_backward_compat_perturbation_pct_accepted(self):
        """Constructor should still accept perturbation_pct without error."""
        analyzer = StabilityAnalyzer(perturbation_pct=0.10)
        assert analyzer.perturbation_pct == 0.10

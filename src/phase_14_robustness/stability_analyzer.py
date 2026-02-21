"""
GIGA TRADER - Hyperparameter Stability Analysis
=================================================
Analyze hyperparameter stability to detect fragile solutions.

A good solution should be on a "plateau" - small parameter changes
should not drastically affect performance.
"""

import os
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class StabilityAnalyzer:
    """
    Analyze hyperparameter stability to detect fragile solutions.

    A good solution should be on a "plateau" - small parameter changes
    should not drastically affect performance.

    Wave 29: Multi-radius perturbation with joint parameter changes.
    Tests three concentric rings (local ±15%, moderate ±50%, wide full-range)
    to ensure stability scores reflect genuine robustness across the entire
    parameter space, not just an artificially narrow neighborhood.
    """

    # Three perturbation rings: (name, perturbation_pct or None, weight)
    # Wide ring (None) samples from full param_ranges instead of base ± pct
    RINGS = [
        ("local",    0.15, 0.20),
        ("moderate", 0.50, 0.30),
        ("wide",     None, 0.50),
    ]

    def __init__(self, perturbation_pct: float = 0.05):
        # Kept for backward compat but no longer used internally
        self.perturbation_pct = perturbation_pct

    def compute_stability_score(
        self,
        base_params: Dict,
        base_score: float,
        param_ranges: Dict[str, Tuple[float, float]],
        score_fn,
        n_samples: int = 24,
    ) -> Dict:
        """
        Compute stability score using multi-radius joint perturbation.

        Tests three concentric rings around the base parameters:
        - Local (±15%): immediate neighborhood smoothness
        - Moderate (±50%): moderate-distance stability
        - Wide (10-90% of full range): broad robustness

        All parameters are perturbed jointly (simultaneously) to catch
        interaction effects. Each ring's sensitivity is weighted, with
        the wide ring carrying 50% of the total score.

        Returns:
            Dict with stability_score (0-1), per-ring details, and diagnostics
        """
        print("\n[STABILITY] Multi-radius hyperparameter sensitivity analysis...")

        samples_per_ring = max(4, n_samples // len(self.RINGS))
        ring_results = {}
        all_scores = [base_score]

        for ring_name, pct, weight in self.RINGS:
            ring_scores = []

            for _ in range(samples_per_ring):
                perturbed = base_params.copy()

                # Joint perturbation: change ALL params simultaneously
                for param, (low, high) in param_ranges.items():
                    base_val = base_params.get(param)
                    if base_val is None:
                        continue

                    if pct is not None:
                        # Local/moderate: multiplicative perturbation
                        delta = np.random.uniform(-pct, pct)
                        new_val = base_val * (1 + delta)
                    else:
                        # Wide: sample from 10-90% of full parameter range
                        range_low = low + 0.10 * (high - low)
                        range_high = low + 0.90 * (high - low)
                        new_val = np.random.uniform(range_low, range_high)

                    new_val = max(low, min(high, new_val))
                    if isinstance(base_val, int):
                        new_val = int(round(new_val))
                    perturbed[param] = new_val

                try:
                    score = score_fn(perturbed)
                    ring_scores.append(score)
                    all_scores.append(score)
                except (ValueError, TypeError, RuntimeError):
                    pass

            # Compute per-ring sensitivity
            avg_change = 0.0  # Wave 31: defensive init for dict construction below
            if ring_scores:
                avg_change = float(np.mean(np.abs(np.array(ring_scores) - base_score)))
                ring_sensitivity = avg_change / (base_score + 1e-6)
                ring_stability = max(0.0, 1.0 - ring_sensitivity * 40)
                ring_std = float(np.std(ring_scores))
                # v3: Cap stability at 0.70 when sensitivity is unmeasurably low.
                # Zero sensitivity means we couldn't distinguish parameter quality,
                # NOT that the model is perfectly robust.
                if ring_sensitivity < 0.001:
                    ring_stability = min(ring_stability, 0.70)
            else:
                ring_sensitivity = 0.0
                ring_stability = 0.5
                ring_std = 0.0

            ring_results[ring_name] = {
                "stability": ring_stability,
                "sensitivity": ring_sensitivity,
                "std": ring_std,
                "n_scores": len(ring_scores),
                "avg_change": avg_change if ring_scores else 0.0,
            }
            print(f"  {ring_name:>8} ring: stability={ring_stability:.3f} "
                  f"(sensitivity={ring_sensitivity:.4f}, n={len(ring_scores)})")

        # Weighted average across rings
        stability_score = sum(
            ring_results[name]["stability"] * w
            for name, _, w in self.RINGS
            if name in ring_results
        )
        stability_score = float(min(1.0, max(0.0, stability_score)))

        score_variance = float(np.var(all_scores))

        results = {
            "stability_score": stability_score,
            "score_variance": score_variance,
            "base_score": base_score,
            "n_samples": len(all_scores),
            "per_ring": ring_results,
        }

        print(f"  Weighted Stability Score: {stability_score:.3f} (1.0 = very stable)")
        print(f"  Score Variance: {score_variance:.6f}")

        if stability_score < 0.5:
            print("  [WARN] Solution may be fragile (low stability)")

        return results

    def detect_plateau(
        self,
        optuna_study,
        top_n: int = 10,
    ) -> bool:
        """
        Check if best solutions form a plateau (similar scores, different params).

        A plateau indicates a robust solution.
        """
        trials = sorted(
            [t for t in optuna_study.trials if t.value is not None],
            key=lambda t: t.value,
            reverse=True
        )[:top_n]

        if len(trials) < 3:
            return False

        scores = [t.value for t in trials]
        score_range = max(scores) - min(scores)

        # If top solutions have similar scores, it's a plateau
        is_plateau = score_range < 0.01  # Within 1% AUC

        if is_plateau:
            print(f"  [GOOD] Detected plateau: top {top_n} solutions within {score_range:.4f} AUC")
        else:
            print(f"  [WARN] No plateau: top {top_n} solutions span {score_range:.4f} AUC")

        return is_plateau

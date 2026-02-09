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
    """

    def __init__(self, perturbation_pct: float = 0.05):
        self.perturbation_pct = perturbation_pct

    def compute_stability_score(
        self,
        base_params: Dict,
        base_score: float,
        param_ranges: Dict[str, Tuple[float, float]],
        score_fn,
        n_samples: int = 20,
    ) -> Dict:
        """
        Compute stability score by perturbing parameters.

        Returns:
            Dict with stability_score (0-1) and per-parameter sensitivity
        """
        print("\n[STABILITY] Analyzing hyperparameter sensitivity...")

        sensitivities = {}
        all_scores = [base_score]

        for param, (low, high) in param_ranges.items():
            base_val = base_params.get(param)
            if base_val is None:
                continue

            param_scores = []

            # Sample perturbations
            for _ in range(n_samples // len(param_ranges)):
                # Random perturbation within +/- perturbation_pct
                perturbation = np.random.uniform(
                    -self.perturbation_pct,
                    self.perturbation_pct
                )
                new_val = base_val * (1 + perturbation)
                new_val = max(low, min(high, new_val))

                perturbed_params = base_params.copy()
                perturbed_params[param] = new_val

                try:
                    score = score_fn(perturbed_params)
                    param_scores.append(score)
                    all_scores.append(score)
                except (ValueError, TypeError, RuntimeError):
                    pass

            if len(param_scores) > 0:
                # Compute sensitivity for this parameter
                score_std = np.std(param_scores)
                score_change = np.mean(np.abs(np.array(param_scores) - base_score))
                sensitivities[param] = {
                    "std": score_std,
                    "avg_change": score_change,
                    "sensitivity": score_change / (base_score + 1e-6),
                }

        # Overall stability score (lower sensitivity = higher stability)
        if len(sensitivities) > 0:
            avg_sensitivity = np.mean([s["sensitivity"] for s in sensitivities.values()])
            stability_score = max(0, 1 - avg_sensitivity * 10)  # Scale factor
        else:
            stability_score = 0.5

        # Score variance across all perturbations
        score_variance = np.var(all_scores)

        results = {
            "stability_score": stability_score,
            "score_variance": score_variance,
            "base_score": base_score,
            "n_samples": len(all_scores),
            "per_param_sensitivity": sensitivities,
        }

        print(f"  Stability Score: {stability_score:.3f} (1.0 = very stable)")
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

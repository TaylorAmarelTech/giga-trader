"""
GIGA TRADER - Label Noise Robustness Test
==========================================
Tests model robustness by flipping a fraction of training labels,
retraining, and measuring AUC drop. Stable models tolerate noise well.

A robust model should degrade gracefully when labels are corrupted,
because it has learned the true underlying signal rather than
memorizing specific training examples.

Usage:
    from src.phase_14_robustness.label_noise_test import LabelNoiseTest

    test = LabelNoiseTest(noise_levels=[0.05, 0.10])
    result = test.run(
        model_factory=lambda: LogisticRegression(C=1.0),
        X=X_train,
        y=y_train,
        cv=3,
    )
    print(result["score"])  # 0-1, higher = more robust
"""

import logging
import numpy as np
from typing import Any, Callable, Dict, List, Optional

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


class LabelNoiseTest:
    """
    Label noise robustness test. Flips N% of labels, retrains the model,
    and measures how much AUC drops. Stable models tolerate noise well.

    Returns a score 0-1 (1 = maximally robust, tolerates noise perfectly).
    """

    def __init__(
        self,
        noise_levels: Optional[List[float]] = None,
        n_repeats: int = 3,
        random_state: int = 42,
    ):
        """
        Args:
            noise_levels: Fraction of labels to flip (default [0.05, 0.10])
            n_repeats: Number of repetitions per noise level for stability
            random_state: Random seed
        """
        self.noise_levels = noise_levels or [0.05, 0.10]
        self.n_repeats = n_repeats
        self.random_state = random_state

    def run(
        self,
        model_factory: Callable[[], Any],
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 3,
    ) -> Dict:
        """
        Run the label noise robustness test.

        Args:
            model_factory: Callable that returns a fresh (unfitted) model instance.
                          Called once per noise level per repeat.
            X: Feature matrix
            y: Binary labels (0/1)
            cv: Number of CV folds for evaluation

        Returns:
            Dict with keys:
              - "score": float 0-1 (1 = robust, 0 = fragile)
              - "base_auc": float (AUC with no noise)
              - "auc_drops": List[Dict] with keys "noise_level", "mean_auc", "auc_drop"
              - "max_drop": float (worst AUC drop across all noise levels)
              - "skipped": bool (True if test couldn't run)
              - "reason": str (if skipped)

        Scoring:
          score = 1.0 - min(max_drop / 0.20, 1.0)
          (AUC drop of 0.20+ saturates to score=0.0)
        """
        # Guard: insufficient samples
        if len(y) < 50:
            logger.info("Label noise test skipped: insufficient samples (%d < 50)", len(y))
            return {
                "score": -1.0,
                "base_auc": 0.0,
                "auc_drops": [],
                "max_drop": 0.0,
                "skipped": True,
                "reason": "insufficient samples",
            }

        # Guard: need both classes
        if len(np.unique(y)) < 2:
            logger.info("Label noise test skipped: single class in labels")
            return {
                "score": -1.0,
                "base_auc": 0.0,
                "auc_drops": [],
                "max_drop": 0.0,
                "skipped": True,
                "reason": "single class in labels",
            }

        # Step 1: Compute base AUC (no noise)
        try:
            base_model = model_factory()
            base_scores = cross_val_score(
                base_model, X, y, cv=cv, scoring="roc_auc"
            )
            base_auc = float(np.mean(base_scores))
        except Exception as e:
            logger.warning("Label noise test skipped: base model failed to fit: %s", e)
            return {
                "score": -1.0,
                "base_auc": 0.0,
                "auc_drops": [],
                "max_drop": 0.0,
                "skipped": True,
                "reason": f"base model fit failed: {e}",
            }

        # Guard: base model too weak
        if base_auc < 0.52:
            logger.info(
                "Label noise test skipped: base model too weak (AUC=%.4f < 0.52)",
                base_auc,
            )
            return {
                "score": -1.0,
                "base_auc": base_auc,
                "auc_drops": [],
                "max_drop": 0.0,
                "skipped": True,
                "reason": "base model too weak",
            }

        logger.info("Label noise test: base AUC=%.4f, testing %d noise levels", base_auc, len(self.noise_levels))

        # Step 2: Test each noise level
        rng = np.random.RandomState(self.random_state)
        auc_drops = []

        for noise_level in self.noise_levels:
            repeat_aucs = []

            for repeat_idx in range(self.n_repeats):
                # Derive a child RNG for reproducibility across repeats
                child_seed = self.random_state + int(noise_level * 10000) + repeat_idx
                child_rng = np.random.RandomState(child_seed)

                y_noisy = self._flip_labels(y, noise_level, child_rng)

                # Need both classes after flipping
                if len(np.unique(y_noisy)) < 2:
                    continue

                try:
                    model = model_factory()
                    scores = cross_val_score(
                        model, X, y_noisy, cv=cv, scoring="roc_auc"
                    )
                    repeat_aucs.append(float(np.mean(scores)))
                except Exception as e:
                    logger.debug(
                        "Label noise test: repeat %d at noise=%.2f failed: %s",
                        repeat_idx, noise_level, e,
                    )
                    continue

            if repeat_aucs:
                mean_auc = float(np.mean(repeat_aucs))
                auc_drop = base_auc - mean_auc
            else:
                mean_auc = 0.0
                auc_drop = base_auc  # Worst case: total failure

            auc_drops.append({
                "noise_level": noise_level,
                "mean_auc": mean_auc,
                "auc_drop": auc_drop,
                "n_successful_repeats": len(repeat_aucs),
            })

            logger.info(
                "  noise=%.2f: mean_auc=%.4f, auc_drop=%.4f (%d/%d repeats)",
                noise_level, mean_auc, auc_drop,
                len(repeat_aucs), self.n_repeats,
            )

        # Step 3: Compute overall score
        if auc_drops:
            max_drop = max(entry["auc_drop"] for entry in auc_drops)
        else:
            max_drop = 0.0

        # Score: 1.0 - min(max_drop / 0.20, 1.0)
        # AUC drop of 0.20+ saturates to score=0.0
        score = 1.0 - min(max_drop / 0.20, 1.0)
        score = float(max(0.0, min(1.0, score)))

        logger.info(
            "Label noise test: max_drop=%.4f, score=%.4f", max_drop, score
        )

        return {
            "score": score,
            "base_auc": base_auc,
            "auc_drops": auc_drops,
            "max_drop": max_drop,
            "skipped": False,
            "reason": "",
        }

    def _flip_labels(self, y: np.ndarray, noise_level: float, rng) -> np.ndarray:
        """
        Flip a random fraction of labels.

        Args:
            y: Original labels
            noise_level: Fraction to flip (e.g., 0.05 = 5%)
            rng: numpy random generator

        Returns:
            Copy of y with noise_level fraction of labels flipped (0->1, 1->0)
        """
        y_noisy = y.copy()
        n_flip = int(len(y) * noise_level)

        if n_flip == 0:
            return y_noisy

        flip_indices = rng.choice(len(y), size=n_flip, replace=False)
        y_noisy[flip_indices] = 1 - y_noisy[flip_indices]

        return y_noisy

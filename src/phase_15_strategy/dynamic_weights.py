"""
Dynamic Ensemble Weighting — weight models by recent OOS performance.

Instead of equal-weighting all ensemble members, this module computes
softmax weights from each model's rolling AUC (or other score), with a
configurable minimum weight floor to prevent any single model from being
completely silenced.

Algorithm:
  1. Track rolling performance scores per model (bounded by lookback).
  2. Compute mean score per model over the lookback window.
  3. Apply softmax with temperature: w_i = exp(mean_i / T) / sum(exp(mean_j / T)).
  4. Enforce floor: w_i = max(w_i, min_weight_fraction / N).
  5. Renormalize so weights sum to 1.0.

Models with no history receive equal weight (1/N).
"""

import logging
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DynamicEnsembleWeighter:
    """
    Dynamically weights ensemble models based on recent OOS performance.
    Uses softmax of rolling AUC scores with a minimum weight floor.
    """

    def __init__(
        self,
        lookback: int = 20,
        min_weight_fraction: float = 0.5,
        temperature: float = 1.0,
    ):
        """
        Args:
            lookback: Number of recent observations to consider for weighting.
            min_weight_fraction: Minimum weight as fraction of equal weight
                (0.5 = at least half of 1/N).
            temperature: Softmax temperature. Lower = more concentrated on
                the best model.
        """
        self.lookback = lookback
        self.min_weight_fraction = min_weight_fraction
        self.temperature = temperature
        self._history: Dict[str, List[float]] = defaultdict(list)

    def update(self, model_id: str, score: float):
        """
        Record a performance score for a model.

        Args:
            model_id: Unique model identifier.
            score: Performance metric (e.g., AUC, accuracy). Higher is better.
        """
        self._history[model_id].append(score)
        # Keep only last `lookback` scores
        if len(self._history[model_id]) > self.lookback:
            self._history[model_id] = self._history[model_id][-self.lookback :]

    def get_weights(self, model_ids: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute dynamic weights for models.

        Args:
            model_ids: List of model IDs to weight. If None, use all known
                models from history.

        Returns:
            Dict mapping model_id to weight (sums to 1.0).

        Weighting algorithm:
            1. Compute mean score per model over lookback window.
            2. Apply softmax with temperature:
               w_i = exp(mean_i / T) / sum(exp(mean_j / T))
            3. Apply floor: w_i = max(w_i, min_weight_fraction / N)
            4. Renormalize to sum to 1.0.

        Models with no history get equal weight (1/N).
        """
        if model_ids is None:
            model_ids = list(self._history.keys())

        n = len(model_ids)
        if n == 0:
            return {}
        if n == 1:
            return {model_ids[0]: 1.0}

        equal_weight = 1.0 / n
        floor = self.min_weight_fraction * equal_weight

        # Compute mean scores; models without history get NaN
        means = np.empty(n, dtype=np.float64)
        has_history = np.zeros(n, dtype=bool)
        for i, mid in enumerate(model_ids):
            scores = self._history.get(mid, [])
            if scores:
                means[i] = np.mean(scores)
                has_history[i] = True
            else:
                means[i] = np.nan

        # If no model has history, return equal weights
        if not has_history.any():
            return {mid: equal_weight for mid in model_ids}

        # Fill models without history with the overall mean of those that have
        overall_mean = np.nanmean(means)
        means = np.where(has_history, means, overall_mean)

        # Softmax with temperature (shift for numerical stability)
        scaled = means / self.temperature
        scaled -= np.max(scaled)
        exp_vals = np.exp(scaled)
        weights = exp_vals / np.sum(exp_vals)

        # Apply floor with iterative redistribution so that
        # renormalization never pushes any model below the floor.
        for _ in range(n):
            below = weights < floor
            if not below.any():
                break
            # Pin floored models at the floor value
            weights[below] = floor
            # Redistribute remaining weight among non-floored models
            above = ~below
            if not above.any():
                break
            locked_total = floor * below.sum()
            free_total = weights[above].sum()
            if free_total > 0:
                weights[above] *= (1.0 - locked_total) / free_total

        result = {mid: float(weights[i]) for i, mid in enumerate(model_ids)}

        logger.debug(
            "Dynamic weights (N=%d, T=%.2f): %s",
            n,
            self.temperature,
            {k: round(v, 4) for k, v in result.items()},
        )
        return result

    def get_weighted_prediction(
        self,
        predictions: Dict[str, float],
        model_ids: Optional[List[str]] = None,
    ) -> float:
        """
        Compute weighted average prediction.

        Args:
            predictions: Dict mapping model_id to prediction value.
            model_ids: Optional list of model IDs to include. If None,
                uses all keys from predictions.

        Returns:
            Weighted average prediction.
        """
        if model_ids is None:
            model_ids = list(predictions.keys())

        weights = self.get_weights(model_ids)

        weighted_sum = 0.0
        for mid in model_ids:
            weighted_sum += weights.get(mid, 0.0) * predictions[mid]

        return weighted_sum

    def reset(self):
        """Clear all history."""
        self._history.clear()

    @property
    def n_models(self) -> int:
        """Number of tracked models."""
        return len(self._history)

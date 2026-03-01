"""
Thompson Sampling Model Selector -- bandit-based online model weighting.

Uses Beta-Bernoulli conjugate priors to learn which model performs best
over time. Each model maintains a Beta(alpha, beta) posterior that
represents the belief about its success probability.

Replaces static 50/50 or learned BMA weights with an adaptive,
exploration-exploitation balanced approach.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ThompsonSamplingSelector:
    """
    Thompson Sampling for online model selection.

    Each model is represented by a Beta(alpha, beta) distribution where:
      - alpha tracks cumulative (decayed) successes
      - beta tracks cumulative (decayed) failures

    On each ``select()`` call, we draw one sample from each model's Beta
    posterior and pick the model with the highest draw (Thompson Sampling).

    ``get_weights()`` returns posterior means (alpha / (alpha + beta)),
    clamped by ``min_weight`` and renormalized, for use in ensemble
    combination.

    Parameters
    ----------
    model_ids : List[str]
        List of model identifiers.
    prior_alpha : float
        Prior successes (default 1.0 = uniform prior).
    prior_beta : float
        Prior failures (default 1.0 = uniform prior).
    decay : float
        Exponential decay for old observations (default 0.995).
        Applies multiplicative decay to alpha and beta each update
        to gradually forget old performance. Set to 1.0 for no decay.
    min_weight : float
        Minimum weight floor (default 0.1). Prevents any model from
        being completely zeroed out in ``get_weights()``.
    """

    def __init__(
        self,
        model_ids: List[str],
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        decay: float = 0.995,
        min_weight: float = 0.1,
    ):
        if not model_ids:
            raise ValueError("model_ids must not be empty")
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("prior_alpha and prior_beta must be positive")
        if not (0.0 < decay <= 1.0):
            raise ValueError("decay must be in (0, 1]")
        if min_weight < 0:
            raise ValueError("min_weight must be non-negative")

        self.model_ids = list(model_ids)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.decay = decay
        self.min_weight = min_weight

        # Posterior parameters per model
        self._alphas: Dict[str, float] = {
            mid: prior_alpha for mid in self.model_ids
        }
        self._betas: Dict[str, float] = {
            mid: prior_beta for mid in self.model_ids
        }
        self._n_updates = 0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update(self, model_id: str, reward: float) -> None:
        """
        Update posterior for a model after observing a reward.

        Parameters
        ----------
        model_id : str
            Which model generated the signal.
        reward : float
            Reward signal. Can be binary (0/1) or continuous in [0, 1].
            For continuous rewards, uses fractional update:
              alpha += reward
              beta  += (1 - reward)

        Raises
        ------
        ValueError
            If model_id is unknown or reward is outside [0, 1].
        """
        if model_id not in self._alphas:
            raise ValueError(
                f"Unknown model_id '{model_id}'. "
                f"Known models: {self.model_ids}"
            )
        if not (0.0 <= reward <= 1.0):
            raise ValueError(f"reward must be in [0, 1], got {reward}")

        # Apply decay to ALL models (aging of old observations)
        if self.decay < 1.0:
            for mid in self.model_ids:
                self._alphas[mid] *= self.decay
                self._betas[mid] *= self.decay
                # Keep a minimum prior floor so posteriors never collapse
                self._alphas[mid] = max(self._alphas[mid], 1e-6)
                self._betas[mid] = max(self._betas[mid], 1e-6)

        # Fractional Beta update for the observed model
        self._alphas[model_id] += reward
        self._betas[model_id] += 1.0 - reward

        self._n_updates += 1

        logger.debug(
            "Thompson update: model=%s reward=%.3f -> alpha=%.3f beta=%.3f",
            model_id,
            reward,
            self._alphas[model_id],
            self._betas[model_id],
        )

    def select(self, rng: Optional[np.random.Generator] = None) -> str:
        """
        Sample from posteriors and return the model with highest sample.

        This is the core Thompson Sampling step: draw one sample from each
        model's Beta(alpha, beta) posterior, then pick the model whose
        draw is largest.

        Parameters
        ----------
        rng : np.random.Generator, optional
            Random number generator for reproducibility. If None, uses
            the default numpy RNG.

        Returns
        -------
        str
            The selected model_id.
        """
        if rng is None:
            rng = np.random.default_rng()

        best_model = None
        best_sample = -1.0

        for mid in self.model_ids:
            sample = rng.beta(self._alphas[mid], self._betas[mid])
            if sample > best_sample:
                best_sample = sample
                best_model = mid

        logger.debug("Thompson select: chose %s (sample=%.4f)", best_model, best_sample)
        return best_model

    def get_weights(self) -> Dict[str, float]:
        """
        Get posterior mean weights for ensemble combination.

        Computes the posterior mean alpha / (alpha + beta) for each model,
        applies the ``min_weight`` floor, and renormalizes so weights
        sum to 1.0.

        Returns
        -------
        Dict[str, float]
            Model weights summing to 1.0.
        """
        n = len(self.model_ids)
        raw: Dict[str, float] = {}

        for mid in self.model_ids:
            a = self._alphas[mid]
            b = self._betas[mid]
            raw[mid] = a / (a + b)

        # Apply floor
        weights: Dict[str, float] = {}
        for mid in self.model_ids:
            weights[mid] = max(raw[mid], self.min_weight)

        # Renormalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            # Fallback to equal weights (should not happen)
            weights = {mid: 1.0 / n for mid in self.model_ids}

        return weights

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Return posterior statistics for each model.

        Returns
        -------
        Dict[str, Dict[str, float]]
            For each model: alpha, beta, mean, variance.
        """
        stats: Dict[str, Dict[str, float]] = {}
        for mid in self.model_ids:
            a = self._alphas[mid]
            b = self._betas[mid]
            mean = a / (a + b)
            variance = (a * b) / ((a + b) ** 2 * (a + b + 1))
            stats[mid] = {
                "alpha": a,
                "beta": b,
                "mean": mean,
                "variance": variance,
            }
        return stats

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: Path) -> None:
        """
        Save state to JSON file.

        Uses atomic write (temp + rename) to avoid corruption if
        another process reads mid-write.

        Parameters
        ----------
        path : Path
            Destination file path (JSON).
        """
        path = Path(path)
        state = {
            "model_ids": self.model_ids,
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
            "decay": self.decay,
            "min_weight": self.min_weight,
            "alphas": self._alphas,
            "betas": self._betas,
            "n_updates": self._n_updates,
        }

        # Atomic write: write to temp file, then rename
        tmp_path = path.with_suffix(".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        tmp_path.replace(path)

        logger.info("Thompson state saved to %s (%d updates)", path, self._n_updates)

    def load_state(self, path: Path) -> None:
        """
        Load state from JSON file.

        Parameters
        ----------
        path : Path
            Source file path (JSON). Must exist.

        Raises
        ------
        FileNotFoundError
            If path does not exist.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"State file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        self.model_ids = state["model_ids"]
        self.prior_alpha = state["prior_alpha"]
        self.prior_beta = state["prior_beta"]
        self.decay = state["decay"]
        self.min_weight = state["min_weight"]
        self._alphas = state["alphas"]
        self._betas = state["betas"]
        self._n_updates = state["n_updates"]

        logger.info(
            "Thompson state loaded from %s (%d updates, %d models)",
            path,
            self._n_updates,
            len(self.model_ids),
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        weights = self.get_weights()
        w_str = ", ".join(f"{k}={v:.3f}" for k, v in weights.items())
        return (
            f"ThompsonSamplingSelector(n_models={len(self.model_ids)}, "
            f"n_updates={self._n_updates}, weights=[{w_str}])"
        )

"""
GIGA TRADER - Weighted Model Evaluation Score (WMES)
=====================================================
Evaluates models on multiple dimensions beyond simple win rate.

Components:
  - Win Rate (traditional)
  - Robustness (stability across variations)
  - Profit Potential (risk-adjusted returns)
  - Noise Tolerance (performance on noisy data)
  - Plateau Stability (sensitivity to parameter changes)
  - Complexity Penalty (fewer features preferred)
"""

import os
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class WeightedModelEvaluator:
    """
    Evaluates models on multiple dimensions beyond simple win rate.

    Components:
      - Win Rate (traditional)
      - Robustness (stability across variations)
      - Profit Potential (risk-adjusted returns)
      - Noise Tolerance (performance on noisy data)
      - Plateau Stability (sensitivity to parameter changes)
      - Complexity Penalty (fewer features preferred)
    """

    def __init__(self, weights: Dict[str, float] = None):
        """Initialize with custom weights or defaults."""
        self.weights = weights or {
            "win_rate": 0.15,           # Traditional metric (reduced weight)
            "robustness": 0.25,         # Cross-validation stability
            "profit_potential": 0.20,   # Risk-adjusted returns
            "noise_tolerance": 0.15,    # Performance on noisy data
            "plateau_stability": 0.15,  # Parameter sensitivity
            "complexity_penalty": 0.10, # Fewer features preferred
        }

    def compute_wmes(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        returns: np.ndarray,
        cv_scores: List[float],
        n_features: int,
        hp_sensitivity: float,
        noise_scores: List[float] = None,
    ) -> Dict[str, float]:
        """
        Compute Weighted Model Evaluation Score.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            returns: Actual returns for each prediction
            cv_scores: Cross-validation AUC scores
            n_features: Number of features used
            hp_sensitivity: How much AUC changes with small HP changes (lower=better)
            noise_scores: Performance on noisy test data

        Returns:
            Dictionary with component scores and final WMES
        """
        scores = {}

        # 1. Win Rate (but capped to prevent over-weighting)
        buy_mask = y_pred == 1
        if buy_mask.sum() > 0:
            buy_returns = returns[buy_mask]
            raw_win_rate = (buy_returns > 0).mean()
            # Cap win rate contribution (suspicious if > 75%)
            scores["win_rate"] = min(raw_win_rate, 0.75) / 0.75
        else:
            scores["win_rate"] = 0.5

        # 2. Robustness (CV score stability)
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        # Penalize high variance in CV scores
        scores["robustness"] = cv_mean * (1 - min(cv_std / cv_mean, 0.5))

        # 3. Profit Potential (risk-adjusted)
        if buy_mask.sum() > 0:
            buy_returns = returns[buy_mask]
            avg_win = buy_returns[buy_returns > 0].mean() if (buy_returns > 0).any() else 0
            avg_loss = abs(buy_returns[buy_returns < 0].mean()) if (buy_returns < 0).any() else 0.001
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 1
            # Sharpe-like ratio
            sharpe = buy_returns.mean() / (buy_returns.std() + 1e-6)
            scores["profit_potential"] = min((profit_factor * 0.3 + sharpe * 0.7), 1.0)
        else:
            scores["profit_potential"] = 0

        # 4. Noise Tolerance
        if noise_scores is not None and len(noise_scores) > 0:
            # How well does the model perform on noisy data?
            noise_degradation = (cv_mean - np.mean(noise_scores)) / cv_mean
            scores["noise_tolerance"] = max(1 - noise_degradation, 0)
        else:
            scores["noise_tolerance"] = 0.5  # Neutral if not tested

        # 5. Plateau Stability (lower sensitivity = better)
        # hp_sensitivity should be in [0, 1] where 0 = very stable
        scores["plateau_stability"] = 1 - min(hp_sensitivity, 1.0)

        # 6. Complexity Penalty (fewer features = better, with diminishing returns)
        optimal_features = 30  # Sweet spot
        if n_features <= optimal_features:
            scores["complexity_penalty"] = 1.0
        else:
            # Penalize excess features
            excess = n_features - optimal_features
            scores["complexity_penalty"] = max(1 - (excess / 100), 0.5)

        # Compute weighted final score
        wmes = sum(scores[k] * self.weights[k] for k in self.weights)
        scores["wmes"] = wmes

        return scores

    def evaluate(
        self,
        y_test: np.ndarray = None,
        y_pred_proba: np.ndarray = None,
        y_train: np.ndarray = None,
        y_train_proba: np.ndarray = None,
        cv_scores: List[float] = None,
        n_features: int = 50,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Simplified evaluation interface for experiment engine.

        This is a wrapper around compute_wmes() with sensible defaults.

        Args:
            y_test: Test set labels
            y_pred_proba: Test set prediction probabilities
            y_train: Training set labels (optional)
            y_train_proba: Training set probabilities (optional)
            cv_scores: Cross-validation scores
            n_features: Number of features used

        Returns:
            Dictionary with wmes_score and component scores
        """
        from sklearn.metrics import roc_auc_score

        # Default CV scores if not provided
        if cv_scores is None:
            if y_test is not None and y_pred_proba is not None:
                try:
                    test_auc = roc_auc_score(y_test, y_pred_proba)
                    cv_scores = [test_auc] * 5  # Use test AUC as proxy
                except (ValueError, TypeError):
                    cv_scores = [0.5] * 5
            else:
                cv_scores = [0.5] * 5

        # Compute predictions from probabilities
        if y_pred_proba is not None:
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = np.zeros(len(y_test) if y_test is not None else 100)

        # Create synthetic returns based on predictions vs actuals
        if y_test is not None and y_pred_proba is not None:
            # Simulate returns: correct prediction = +0.01, wrong = -0.01
            correct = (y_pred == y_test).astype(float)
            returns = (correct * 2 - 1) * 0.01  # +/- 1%
        else:
            returns = np.random.normal(0, 0.01, len(y_pred))

        # Estimate hyperparameter sensitivity (default to moderate)
        hp_sensitivity = 0.3

        # Call compute_wmes with derived values
        if y_test is None:
            y_test = np.zeros(len(y_pred))
        if y_pred_proba is None:
            y_pred_proba = np.zeros(len(y_pred))

        result = self.compute_wmes(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_pred_proba,
            returns=returns,
            cv_scores=cv_scores,
            n_features=n_features,
            hp_sensitivity=hp_sensitivity,
            noise_scores=None,
        )

        # Add convenience alias
        result["wmes_score"] = result["wmes"]

        return result

    def analyze_hp_sensitivity(
        self,
        base_params: Dict,
        param_ranges: Dict[str, Tuple[float, float]],
        evaluate_fn,
        n_perturbations: int = 10,
    ) -> float:
        """
        Measure how sensitive the model is to small parameter changes.

        Returns:
            Sensitivity score (0 = very stable, 1 = very fragile)
        """
        base_score = evaluate_fn(base_params)
        score_changes = []

        for param, (low, high) in param_ranges.items():
            # Perturb each parameter by +/- 5%
            base_val = base_params.get(param)
            if base_val is None:
                continue

            for direction in [-0.05, 0.05]:
                perturbed = base_params.copy()
                new_val = base_val * (1 + direction)
                new_val = max(low, min(high, new_val))  # Clip to range
                perturbed[param] = new_val

                try:
                    perturbed_score = evaluate_fn(perturbed)
                    change = abs(perturbed_score - base_score) / (base_score + 1e-6)
                    score_changes.append(change)
                except (ValueError, TypeError, RuntimeError):
                    pass

        if len(score_changes) == 0:
            return 0.5  # Neutral

        # Average change indicates sensitivity
        avg_change = np.mean(score_changes)
        # Convert to 0-1 where high change = high sensitivity
        sensitivity = min(avg_change * 10, 1.0)  # Scale factor

        return sensitivity

    def evaluate_quick(self, cv_scores: List[float], n_features: int = 30) -> float:
        """
        Quick WMES estimate using only CV scores and feature count.

        Used by MultiFidelityEvaluator Tier 1 for fast screening without
        needing full predictions, returns, or noise tests.

        Returns:
            Estimated WMES in [0, 1]
        """
        # Robustness from CV scores (same formula as compute_wmes)
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        robustness = cv_mean * (1 - min(cv_std / (cv_mean + 1e-6), 0.5))

        # Complexity penalty (same formula as compute_wmes)
        optimal_features = 30
        if n_features <= optimal_features:
            complexity = 1.0
        else:
            excess = n_features - optimal_features
            complexity = max(1 - (excess / 100), 0.5)

        # Weighted estimate: robustness dominates, complexity minor, neutral for unmeasured
        return robustness * 0.6 + complexity * 0.2 + 0.5 * 0.2


def compute_weighted_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    returns: np.ndarray,
    cv_scores: List[float],
    n_features: int,
    hp_sensitivity: float = 0.0,
    noise_scores: List[float] = None,
) -> Dict[str, float]:
    """
    Convenience function to compute WMES.

    Returns comprehensive evaluation metrics.
    """
    evaluator = WeightedModelEvaluator()
    return evaluator.compute_wmes(
        y_true, y_pred, y_proba, returns,
        cv_scores, n_features, hp_sensitivity, noise_scores
    )

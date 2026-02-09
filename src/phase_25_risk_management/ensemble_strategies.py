"""
GIGA TRADER - Ensemble Strategies for Model Selection
=======================================================
Data structures and ensemble strategies for combining model predictions.

Usage:
    from src.phase_25_risk_management.ensemble_strategies import (
        ModelCandidate,
        EnsemblePrediction,
        WeightedAverageEnsemble,
        MedianEnsemble,
        VotingEnsemble,
        StackingEnsemble,
    )
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field

import numpy as np


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ModelCandidate:
    """A candidate model from the registry."""
    model_id: str
    model_path: str
    config: Dict

    # Performance metrics
    cv_auc: float = 0.0
    test_auc: float = 0.0
    backtest_sharpe: float = 0.0
    backtest_win_rate: float = 0.0
    wmes_score: float = 0.0

    # Robustness metrics
    stability_score: float = 0.0
    fragility_score: float = 1.0
    tier: int = 1  # 1=registry, 2=paper-eligible, 3=live-eligible

    # Entry/Exit configuration
    entry_window_start: int = 0
    entry_window_end: int = 120
    exit_window_start: int = 300
    exit_window_end: int = 385

    # Model object (loaded on demand)
    model: Any = None
    feature_cols: List[str] = field(default_factory=list)

    def score(self, weights: Dict[str, float] = None) -> float:
        """Calculate weighted score for ranking, with stability bonus."""
        weights = weights or {
            "cv_auc": 0.15,
            "test_auc": 0.20,
            "backtest_sharpe": 0.25,
            "wmes_score": 0.20,
            "stability_score": 0.20,
        }
        base = sum(
            weights.get(k, 0) * getattr(self, k, 0)
            for k in weights
        )
        # Penalize fragile models
        fragility_penalty = self.fragility_score * 0.1
        return base - fragility_penalty

    def matches_window(
        self,
        entry_window: Tuple[int, int] = None,
        exit_window: Tuple[int, int] = None,
        tolerance: int = 30,
    ) -> bool:
        """Check if model's windows match requested windows within tolerance."""
        if entry_window:
            if abs(self.entry_window_start - entry_window[0]) > tolerance:
                return False
            if abs(self.entry_window_end - entry_window[1]) > tolerance:
                return False
        if exit_window:
            if abs(self.exit_window_start - exit_window[0]) > tolerance:
                return False
            if abs(self.exit_window_end - exit_window[1]) > tolerance:
                return False
        return True


@dataclass
class EnsemblePrediction:
    """Prediction from an ensemble of models."""
    swing_probability: float
    timing_probability: float
    confidence: float
    direction: str  # "LONG", "SHORT", "HOLD"

    # Individual model predictions
    model_predictions: List[Dict] = field(default_factory=list)

    # Ensemble metadata
    n_models: int = 0
    agreement_ratio: float = 0.0  # Fraction of models agreeing on direction
    ensemble_method: str = "weighted_average"

    # Entry/Exit windows from best model
    entry_window: Tuple[int, int] = (30, 120)
    exit_window: Tuple[int, int] = (300, 385)

    # Position sizing suggestions
    suggested_position_pct: float = 0.10
    confidence_adjusted_position_pct: float = 0.10


# =============================================================================
# ENSEMBLE STRATEGIES
# =============================================================================

class EnsembleStrategy:
    """Base class for ensemble strategies."""

    def combine(
        self,
        predictions: List[Dict],
        weights: List[float],
    ) -> Tuple[float, float]:
        """Combine predictions into final probabilities."""
        raise NotImplementedError


class WeightedAverageEnsemble(EnsembleStrategy):
    """Weighted average of model predictions."""

    def combine(
        self,
        predictions: List[Dict],
        weights: List[float],
    ) -> Tuple[float, float]:
        if not predictions:
            return 0.5, 0.5

        weights = np.array(weights)
        weights = weights / weights.sum()

        swing_probas = np.array([p["swing_proba"] for p in predictions])
        timing_probas = np.array([p["timing_proba"] for p in predictions])

        swing_combined = np.sum(swing_probas * weights)
        timing_combined = np.sum(timing_probas * weights)

        return float(swing_combined), float(timing_combined)


class MedianEnsemble(EnsembleStrategy):
    """Median of model predictions (robust to outliers)."""

    def combine(
        self,
        predictions: List[Dict],
        weights: List[float],
    ) -> Tuple[float, float]:
        if not predictions:
            return 0.5, 0.5

        swing_probas = np.array([p["swing_proba"] for p in predictions])
        timing_probas = np.array([p["timing_proba"] for p in predictions])

        return float(np.median(swing_probas)), float(np.median(timing_probas))


class VotingEnsemble(EnsembleStrategy):
    """Voting ensemble with weighted votes."""

    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold

    def combine(
        self,
        predictions: List[Dict],
        weights: List[float],
    ) -> Tuple[float, float]:
        if not predictions:
            return 0.5, 0.5

        weights = np.array(weights)
        weights = weights / weights.sum()

        # Convert to votes
        swing_votes_up = np.array([
            1 if p["swing_proba"] > self.threshold else 0
            for p in predictions
        ])
        timing_votes_up = np.array([
            1 if p["timing_proba"] > self.threshold else 0
            for p in predictions
        ])

        # Weighted votes
        swing_vote_pct = np.sum(swing_votes_up * weights)
        timing_vote_pct = np.sum(timing_votes_up * weights)

        # Convert back to probability-like score
        swing_combined = 0.5 + (swing_vote_pct - 0.5) * 0.4  # Scale to 0.3-0.7 range
        timing_combined = 0.5 + (timing_vote_pct - 0.5) * 0.4

        return float(swing_combined), float(timing_combined)


class StackingEnsemble(EnsembleStrategy):
    """Stacking ensemble using meta-learner (if available)."""

    def __init__(self, meta_model=None):
        self.meta_model = meta_model

    def combine(
        self,
        predictions: List[Dict],
        weights: List[float],
    ) -> Tuple[float, float]:
        if not predictions:
            return 0.5, 0.5

        if self.meta_model is None:
            # Fall back to weighted average
            return WeightedAverageEnsemble().combine(predictions, weights)

        # Stack predictions as features for meta-model
        meta_features = np.array([
            [p["swing_proba"], p["timing_proba"]] for p in predictions
        ]).flatten()

        # Meta-model predicts final probabilities
        meta_pred = self.meta_model.predict_proba(meta_features.reshape(1, -1))[0]

        return float(meta_pred[1]), float(meta_pred[1])  # Assuming binary classification

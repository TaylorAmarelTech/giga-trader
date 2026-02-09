"""
GIGA TRADER - Signal Data Structures
=====================================
Core data structures for the enhanced signal generator.

Contains:
- SignalDirection enum
- StrategySignal dataclass
- EnsembleSignal dataclass
- StrategyPerformance class
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class SignalDirection(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class StrategySignal:
    """Signal from a single strategy."""
    direction: SignalDirection
    confidence: float  # 0.0 to 1.0
    raw_score: float   # -1.0 to 1.0
    strategy_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


@dataclass
class EnsembleSignal:
    """Combined signal from all strategies."""
    direction: SignalDirection
    confidence: float
    calibrated_confidence: float  # Adjusted for historical accuracy
    raw_score: float
    timestamp: datetime
    contributing_strategies: List[str]
    strategy_weights: Dict[str, float]
    drift_detected: bool = False
    drift_severity: float = 0.0
    regime: str = "NORMAL"
    metadata: Dict = field(default_factory=dict)


@dataclass
class StrategyPerformance:
    """Track strategy performance for robust weight calculation."""
    strategy_name: str
    total_signals: int = 0
    correct_signals: int = 0

    # Performance on real SPY
    real_spy_accuracy: float = 0.5
    real_spy_sharpe: float = 0.0
    real_spy_returns: List[float] = field(default_factory=list)

    # Performance on synthetic universes (critical for robustness)
    synthetic_accuracy: float = 0.5
    synthetic_sharpe: float = 0.0
    synthetic_variance: float = 0.0  # Lower = more robust
    universe_accuracies: Dict[str, float] = field(default_factory=dict)

    # Anti-overfitting metrics
    robustness_score: float = 0.5  # Combined real + synthetic
    overfitting_penalty: float = 0.0  # High if real >> synthetic

    # Weight in ensemble (determined by CV, not recent performance)
    cv_weight: float = 0.2  # Default equal weight for 5 strategies

    def update_robustness(self):
        """Calculate robustness score penalizing overfitting."""
        if self.synthetic_accuracy > 0:
            # Overfitting penalty: how much better on real vs synthetic
            overfit_gap = max(0, self.real_spy_accuracy - self.synthetic_accuracy)
            self.overfitting_penalty = overfit_gap * 2  # 2x penalty

            # Robustness = weighted average minus penalty minus variance
            self.robustness_score = (
                0.4 * self.real_spy_accuracy +
                0.6 * self.synthetic_accuracy -  # Weight synthetic more
                0.3 * self.overfitting_penalty -
                0.2 * min(1.0, self.synthetic_variance * 5)  # Variance penalty
            )
            self.robustness_score = max(0.0, min(1.0, self.robustness_score))

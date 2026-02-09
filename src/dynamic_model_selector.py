"""
GIGA TRADER - Dynamic Model Selector (SHIM)
=====================================
This module has been decomposed into:
  - src.phase_25_risk_management.ensemble_strategies
  - src.phase_25_risk_management.model_selector

This file re-exports all public names for backward compatibility.
"""

from src.phase_25_risk_management.ensemble_strategies import (
    ModelCandidate,
    EnsemblePrediction,
    EnsembleStrategy,
    WeightedAverageEnsemble,
    MedianEnsemble,
    VotingEnsemble,
    StackingEnsemble,
)
from src.phase_25_risk_management.model_selector import (
    DynamicModelSelector,
    EntryExitGridSearchRunner,
)

__all__ = [
    "ModelCandidate",
    "EnsemblePrediction",
    "EnsembleStrategy",
    "WeightedAverageEnsemble",
    "MedianEnsemble",
    "VotingEnsemble",
    "StackingEnsemble",
    "DynamicModelSelector",
    "EntryExitGridSearchRunner",
]

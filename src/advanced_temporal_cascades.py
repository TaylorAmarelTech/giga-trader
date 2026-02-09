"""
Backward-compatible shim. Code moved to src.phase_26_temporal.advanced_temporal_cascades.

All classes and functions are re-exported for backward compatibility.
"""
# Re-export everything from new location
from src.phase_26_temporal.advanced_temporal_cascades import *  # noqa: F401,F403

# Also re-export specific names for explicit imports
from src.phase_26_temporal.advanced_temporal_cascades import (  # noqa: F401
    CascadePrediction,
    MultiResolutionCascade,
    BackwardLookingCascade,
    IntermittentMaskedCascade,
    StochasticDepthCascade,
    CrossTemporalAttentionCascade,
    UnifiedTemporalEnsemble,
)

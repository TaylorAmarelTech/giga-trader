"""Phase 26: Temporal Cascades.

Temporal cascade model architectures for anti-overfitting through
temporal diversity. Multiple models trained on different temporal
slices provide robust, real-time prediction updates.
"""

from src.phase_26_temporal.temporal_cascade_models import (
    SwingDirection,
    DayMagnitude,
    DayRegime,
    IntradayPattern,
    OptimalWindow,
    TemporalPrediction,
    CascadeEnsemblePrediction,
    TemporalFeatureEngineer,
    TemporalSliceModel,
    TemporalTargetLabeler,
    TemporalCascadeEnsemble,
)

from src.phase_26_temporal.advanced_temporal_cascades import (
    CascadePrediction,
    MultiResolutionCascade,
    BackwardLookingCascade,
    IntermittentMaskedCascade,
    StochasticDepthCascade,
    CrossTemporalAttentionCascade,
    UnifiedTemporalEnsemble,
)

from src.phase_26_temporal.resolution_cascade import (
    ResolutionCascade,
    ResolutionCascadePrediction,
    ResolutionModelResult,
)

__all__ = [
    # temporal_cascade_models
    "SwingDirection",
    "DayMagnitude",
    "DayRegime",
    "IntradayPattern",
    "OptimalWindow",
    "TemporalPrediction",
    "CascadeEnsemblePrediction",
    "TemporalFeatureEngineer",
    "TemporalSliceModel",
    "TemporalTargetLabeler",
    "TemporalCascadeEnsemble",
    # advanced_temporal_cascades
    "CascadePrediction",
    "MultiResolutionCascade",
    "BackwardLookingCascade",
    "IntermittentMaskedCascade",
    "StochasticDepthCascade",
    "CrossTemporalAttentionCascade",
    "UnifiedTemporalEnsemble",
    # resolution_cascade (full-pipeline multi-resolution)
    "ResolutionCascade",
    "ResolutionCascadePrediction",
    "ResolutionModelResult",
]

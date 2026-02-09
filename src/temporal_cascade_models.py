"""
Backward-compatible shim. Code moved to src.phase_26_temporal.temporal_cascade_models.

All classes and functions are re-exported for backward compatibility.
"""
# Re-export everything from new location
from src.phase_26_temporal.temporal_cascade_models import *  # noqa: F401,F403

# Also re-export specific names for explicit imports
from src.phase_26_temporal.temporal_cascade_models import (  # noqa: F401
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

"""
Backward-compatible shim. Code moved to src.phase_12_model_training.temporal_regularization.

All classes and functions are re-exported for backward compatibility.
"""
# Re-export everything from new location
from src.phase_12_model_training.temporal_regularization import *  # noqa: F401,F403

# Also re-export specific names for explicit imports
from src.phase_12_model_training.temporal_regularization import (  # noqa: F401
    TemporalMaskingWrapper,
    TemporalFeatureAugmenter,
    TemporalDropoutCV,
    TemporalConsistencyRegularizer,
    apply_temporal_regularization,
    create_temporally_regularized_swing_model,
    create_temporally_regularized_timing_model,
    create_temporally_regularized_entry_exit_model,
)

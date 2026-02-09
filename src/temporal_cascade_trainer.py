"""
Backward-compatible shim. Code moved to src.phase_12_model_training.temporal_cascade_trainer.

All classes and functions are re-exported for backward compatibility.
"""
# Re-export everything from new location
from src.phase_12_model_training.temporal_cascade_trainer import *  # noqa: F401,F403

# Also re-export specific names for explicit imports
from src.phase_12_model_training.temporal_cascade_trainer import (  # noqa: F401
    TEMPORAL_CASCADE_CONFIG,
    prepare_intraday_data_dict,
    aggregate_to_daily,
    TemporalCascadeTrainResult,
    train_temporal_cascade,
    load_temporal_cascade,
    register_temporal_cascade_experiment,
    TemporalCascadeSignalGenerator,
)

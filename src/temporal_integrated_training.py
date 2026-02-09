"""
Backward-compatible shim. Code moved to src.phase_12_model_training.temporal_integrated_training.

All classes and functions are re-exported for backward compatibility.
"""
# Re-export everything from new location
from src.phase_12_model_training.temporal_integrated_training import *  # noqa: F401,F403

# Also re-export specific names for explicit imports
from src.phase_12_model_training.temporal_integrated_training import (  # noqa: F401
    ModelType,
    ModelStep,
    TEMPORAL_SLICES,
    TemporalModelRecord,
    TemporalModelRegistry,
    TemporalIntegratedTrainer,
    reset_model_registry,
    train_all_temporal_models,
)

"""
Backward-compatible shim. Code moved to src.phase_12_model_training.training_pipeline_v2.

All classes and functions are re-exported for backward compatibility.
"""
# Re-export everything from new location
from src.phase_12_model_training.training_pipeline_v2 import *  # noqa: F401,F403

# Also re-export specific names for explicit imports
from src.phase_12_model_training.training_pipeline_v2 import (  # noqa: F401
    DataLoader,
    FeatureEngineer,
    TargetCreator,
    Preprocessor,
    ScalerFactory,
    FeatureSelector,
    DimReducerFactory,
    ModelFactory,
    SampleWeighter,
    CrossValidator,
    TrainingPipelineV2,
    run_minimal_grid,
    run_focused_grid,
    run_quick_experiment,
    train_from_registry,
)

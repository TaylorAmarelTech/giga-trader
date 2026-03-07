"""Phase 12: Model Training.

Core model training modules including timing models, temporal cascade
training, temporal regularization, integrated training, and the
training pipeline v2.
"""

from src.phase_12_model_training.timing_models import (
    EntryTimeModel,
    ExitTimeModel,
    PositionSizeModel,
    StopTakeProfitModel,
    BatchScheduleModel,
    GuardrailModel,
    EntryExitTimingModel,
    create_entry_exit_model,
)

from src.phase_12_model_training.temporal_cascade_trainer import (
    TEMPORAL_CASCADE_CONFIG,
    prepare_intraday_data_dict,
    aggregate_to_daily,
    TemporalCascadeTrainResult,
    train_temporal_cascade,
    load_temporal_cascade,
    register_temporal_cascade_experiment,
    TemporalCascadeSignalGenerator,
)

from src.phase_12_model_training.temporal_regularization import (
    TemporalMaskingWrapper,
    TemporalFeatureAugmenter,
    TemporalDropoutCV,
    TemporalConsistencyRegularizer,
    apply_temporal_regularization,
    create_temporally_regularized_swing_model,
    create_temporally_regularized_timing_model,
    create_temporally_regularized_entry_exit_model,
)

from src.phase_12_model_training.temporal_integrated_training import (
    ModelType,
    ModelStep,
    TEMPORAL_SLICES,
    TemporalModelRecord,
    TemporalModelRegistry,
    TemporalIntegratedTrainer,
    reset_model_registry,
    train_all_temporal_models,
)

from src.phase_12_model_training.quantile_forest_wrapper import (
    QuantileForestClassifier,
)

from src.phase_12_model_training.stacking_ensemble import (
    StackingEnsembleClassifier,
)

from src.phase_12_model_training.kan_model import KANClassifier

from src.phase_12_model_training.mamba_model import MambaClassifier

from src.phase_12_model_training.training_pipeline_v2 import (
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

__all__ = [
    # timing_models
    "EntryTimeModel",
    "ExitTimeModel",
    "PositionSizeModel",
    "StopTakeProfitModel",
    "BatchScheduleModel",
    "GuardrailModel",
    "EntryExitTimingModel",
    "create_entry_exit_model",
    # temporal_cascade_trainer
    "TEMPORAL_CASCADE_CONFIG",
    "prepare_intraday_data_dict",
    "aggregate_to_daily",
    "TemporalCascadeTrainResult",
    "train_temporal_cascade",
    "load_temporal_cascade",
    "register_temporal_cascade_experiment",
    "TemporalCascadeSignalGenerator",
    # temporal_regularization
    "TemporalMaskingWrapper",
    "TemporalFeatureAugmenter",
    "TemporalDropoutCV",
    "TemporalConsistencyRegularizer",
    "apply_temporal_regularization",
    "create_temporally_regularized_swing_model",
    "create_temporally_regularized_timing_model",
    "create_temporally_regularized_entry_exit_model",
    # temporal_integrated_training
    "ModelType",
    "ModelStep",
    "TEMPORAL_SLICES",
    "TemporalModelRecord",
    "TemporalModelRegistry",
    "TemporalIntegratedTrainer",
    "reset_model_registry",
    "train_all_temporal_models",
    # quantile_forest_wrapper
    "QuantileForestClassifier",
    # stacking_ensemble
    "StackingEnsembleClassifier",
    # training_pipeline_v2
    "DataLoader",
    "FeatureEngineer",
    "TargetCreator",
    "Preprocessor",
    "ScalerFactory",
    "FeatureSelector",
    "DimReducerFactory",
    "ModelFactory",
    "SampleWeighter",
    "CrossValidator",
    "TrainingPipelineV2",
    "run_minimal_grid",
    "run_focused_grid",
    "run_quick_experiment",
    "train_from_registry",
    # kan_model
    "KANClassifier",
    # mamba_model
    "MambaClassifier",
]

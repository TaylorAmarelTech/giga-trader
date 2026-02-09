"""
GIGA TRADER - Entry/Exit Timing Model (Shim)
==============================================
Re-exports from decomposed phase modules:
  - phase_05_targets.timing_targets: TargetLabeler
  - phase_06_features_intraday.timing_features: TimingFeatureEngineer
  - phase_12_model_training.timing_models: All model classes + helpers
"""

# Phase 05: Target labeling
from src.phase_05_targets.timing_targets import TargetLabeler

# Phase 06: Timing feature engineering
from src.phase_06_features_intraday.timing_features import TimingFeatureEngineer

# Phase 12: All timing models
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

__all__ = [
    "TargetLabeler",
    "TimingFeatureEngineer",
    "EntryTimeModel",
    "ExitTimeModel",
    "PositionSizeModel",
    "StopTakeProfitModel",
    "BatchScheduleModel",
    "GuardrailModel",
    "EntryExitTimingModel",
    "create_entry_exit_model",
]

"""
GIGA TRADER - Unified Experiment Engine (Shim)
===============================================
Re-exports from decomposed phase modules:
  - phase_11_cv_splitting.walk_forward_cv: WalkForwardCV
  - phase_21_continuous.experiment_tracking: Data structures + utilities
  - phase_21_continuous.experiment_runner: UnifiedExperimentRunner, ExperimentEngine, main
"""

# Phase 11: Walk-forward CV
from src.phase_11_cv_splitting.walk_forward_cv import WalkForwardCV

# Phase 21: Experiment tracking data structures
from src.phase_21_continuous.experiment_tracking import (
    ENGINE_CONFIG,
    ExperimentStatus,
    ExperimentResult,
    ModelRecord,
    ExperimentGenerator,
    ExperimentHistory,
    ModelRegistry,
    compute_realistic_backtest_metrics,
    calibrate_probabilities,
)

# Phase 21: Experiment runner
from src.phase_21_continuous.experiment_runner import (
    UnifiedExperimentRunner,
    ExperimentEngine,
    main,
)

__all__ = [
    "ENGINE_CONFIG",
    "WalkForwardCV",
    "ExperimentStatus",
    "ExperimentResult",
    "ModelRecord",
    "ExperimentGenerator",
    "ExperimentHistory",
    "ModelRegistry",
    "compute_realistic_backtest_metrics",
    "calibrate_probabilities",
    "UnifiedExperimentRunner",
    "ExperimentEngine",
    "main",
]

if __name__ == "__main__":
    main()

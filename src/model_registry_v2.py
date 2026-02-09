"""
GIGA TRADER - Model Registry v2 (Shim)
========================================
This module has been decomposed into:
  - src/phase_18_persistence/registry_enums.py
  - src/phase_18_persistence/registry_configs.py
  - src/phase_18_persistence/model_registry.py
  - src/phase_18_persistence/grid_search_generator.py

All public names are re-exported here for backward compatibility.
"""

# Re-export everything from the decomposed modules
from src.phase_18_persistence.registry_enums import (  # noqa: F401
    DataSource,
    TimeResolution,
    DataPeriod,
    MarketHours,
    SyntheticDataMethod,
    AugmentationMethod,
    OutlierMethod,
    MissingValueMethod,
    ScalingMethod,
    TransformMethod,
    FeatureSelectionMethod,
    DimReductionMethod,
    ModelType,
    EnsembleMethod,
    CascadeType,
    TemporalEncoding,
    TargetType,
    TargetDefinition,
    LabelSmoothingMethod,
    CVMethod,
    ScoringMetric,
    SampleWeightMethod,
    ModelStatus,
)

from src.phase_18_persistence.registry_configs import (  # noqa: F401
    DataConfig,
    SyntheticDataConfig,
    AugmentationConfig,
    PreprocessConfig,
    FeatureConfig,
    TargetConfig,
    FeatureSelectionConfig,
    DimReductionConfig,
    ModelConfig,
    CascadeConfig,
    SampleWeightConfig,
    TrainingConfig,
    EvaluationConfig,
    ModelMetrics,
    ModelArtifacts,
    ModelEntry,
)

from src.phase_18_persistence.model_registry import (  # noqa: F401
    ModelRegistryV2,
    get_registry,
)

from src.phase_18_persistence.grid_search_generator import (  # noqa: F401
    GridSearchConfigGenerator,
    create_model_entry,
    list_all_options,
    print_all_options,
    estimate_grid_size,
    create_quick_experiment,
    print_pipeline_steps,
    PIPELINE_STEPS,
)

# Preserve project_root and logger for any code that imported them
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
logger = logging.getLogger("MODEL_REGISTRY_V2")

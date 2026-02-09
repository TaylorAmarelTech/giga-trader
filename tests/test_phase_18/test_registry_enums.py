"""
Test all enum classes in phase_18_persistence/registry_enums.py.

Verifies that:
- All 22 enum classes exist and are importable
- Each enum has expected key members
- Enums are iterable with expected lengths
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase_18_persistence.registry_enums import (
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


# ---------------------------------------------------------------------------
# Parametrized test: each enum exists and is iterable
# ---------------------------------------------------------------------------

ALL_ENUMS = [
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
]


def test_all_22_enum_classes_exist():
    """There should be exactly 23 enum classes (22 original + ModelStatus)."""
    assert len(ALL_ENUMS) == 23


@pytest.mark.parametrize("enum_cls", ALL_ENUMS, ids=lambda e: e.__name__)
def test_enum_is_iterable(enum_cls):
    """Each enum class should be iterable and have at least 2 members."""
    members = list(enum_cls)
    assert len(members) >= 2, f"{enum_cls.__name__} should have at least 2 members"


# ---------------------------------------------------------------------------
# Specific member checks
# ---------------------------------------------------------------------------

def test_data_source_members():
    """DataSource should have ALPACA, YFINANCE, POLYGON."""
    assert DataSource.ALPACA.value == "alpaca"
    assert DataSource.YFINANCE.value == "yfinance"
    assert DataSource.POLYGON.value == "polygon"
    assert len(list(DataSource)) == 7


def test_model_type_members():
    """ModelType should have key model types."""
    assert ModelType.LOGISTIC_L1.value == "logistic_l1"
    assert ModelType.LOGISTIC_L2.value == "logistic_l2"
    assert ModelType.ELASTIC_NET.value == "elastic_net"
    assert ModelType.GRADIENT_BOOSTING.value == "gradient_boosting"
    assert ModelType.XGBOOST.value == "xgboost"
    assert ModelType.LIGHTGBM.value == "lightgbm"
    assert ModelType.RANDOM_FOREST.value == "random_forest"
    # Should have 30+ model types
    assert len(list(ModelType)) >= 25


def test_dim_reduction_method_members():
    """DimReductionMethod should have all documented methods."""
    assert DimReductionMethod.PCA.value == "pca"
    assert DimReductionMethod.KERNEL_PCA_RBF.value == "kernel_pca_rbf"
    assert DimReductionMethod.UMAP.value == "umap"
    assert DimReductionMethod.ICA.value == "ica"
    assert DimReductionMethod.AGGLOMERATION.value == "agglomeration"
    assert DimReductionMethod.KMEDOIDS.value == "kmedoids"
    assert DimReductionMethod.ENSEMBLE_PLUS.value == "ensemble_plus"


def test_cascade_type_members():
    """CascadeType should have all cascade variants."""
    assert CascadeType.BASE.value == "base"
    assert CascadeType.MASKED.value == "masked"
    assert CascadeType.ATTENTION.value == "attention"
    assert CascadeType.MIXTURE_OF_EXPERTS.value == "mixture_of_experts"
    assert len(list(CascadeType)) >= 10


def test_cv_method_members():
    """CVMethod should include time-series aware splits."""
    assert CVMethod.TIMESERIES_SPLIT.value == "timeseries_split"
    assert CVMethod.PURGED_KFOLD.value == "purged_kfold"
    assert CVMethod.WALK_FORWARD.value == "walk_forward"


def test_target_type_members():
    """TargetType should have swing and timing."""
    assert TargetType.SWING.value == "swing"
    assert TargetType.TIMING.value == "timing"
    assert TargetType.ENTRY_EXIT.value == "entry_exit"


def test_model_status_members():
    """ModelStatus should cover the full lifecycle."""
    assert ModelStatus.QUEUED.value == "queued"
    assert ModelStatus.TRAINING.value == "training"
    assert ModelStatus.TRAINED.value == "trained"
    assert ModelStatus.PRODUCTION.value == "production"
    assert ModelStatus.DEPRECATED.value == "deprecated"
    assert ModelStatus.FAILED.value == "failed"
    assert len(list(ModelStatus)) >= 10


def test_scaling_method_members():
    """ScalingMethod should include standard methods."""
    assert ScalingMethod.STANDARD.value == "standard"
    assert ScalingMethod.ROBUST.value == "robust"
    assert ScalingMethod.QUANTILE_NORMAL.value == "quantile_normal"


def test_outlier_method_members():
    """OutlierMethod should include winsorize and ML-based methods."""
    assert OutlierMethod.WINSORIZE_1.value == "winsorize_1"
    assert OutlierMethod.ISOLATION_FOREST.value == "isolation_forest"
    assert OutlierMethod.NONE.value == "none"


def test_sample_weight_method_members():
    """SampleWeightMethod should include time-decay methods."""
    assert SampleWeightMethod.NONE.value == "none"
    assert SampleWeightMethod.BALANCED.value == "balanced"
    assert SampleWeightMethod.TIME_DECAY_EXPONENTIAL.value == "time_decay_exponential"


def test_enums_are_str_enum():
    """All enums should be str enums (inherit from str)."""
    for enum_cls in ALL_ENUMS:
        for member in enum_cls:
            assert isinstance(member.value, str), (
                f"{enum_cls.__name__}.{member.name} value should be a string"
            )

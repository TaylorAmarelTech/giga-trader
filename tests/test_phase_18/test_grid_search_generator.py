"""
Test GridSearchConfigGenerator, list_all_options, PIPELINE_STEPS,
and create_model_entry utility.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase_18_persistence.grid_search_generator import (
    GridSearchConfigGenerator,
    list_all_options,
    create_model_entry,
    PIPELINE_STEPS,
)
from src.phase_18_persistence.registry_enums import (
    TargetType,
    ModelType,
    CascadeType,
    DimReductionMethod,
    ModelStatus,
)
from src.phase_18_persistence.registry_configs import ModelEntry


# ---------------------------------------------------------------------------
# GridSearchConfigGenerator tests
# ---------------------------------------------------------------------------

def test_generator_creation():
    """GridSearchConfigGenerator should initialize with empty dimensions."""
    gen = GridSearchConfigGenerator()
    assert isinstance(gen, GridSearchConfigGenerator)
    assert len(gen.dimensions) == 0
    assert len(gen.constraints) == 0


def test_add_dimension():
    """Adding a dimension should store the options."""
    gen = GridSearchConfigGenerator()
    gen.add_dimension("model_type", ["logistic_l2", "gradient_boosting"])
    assert "model_type" in gen.dimensions
    assert len(gen.dimensions["model_type"]) == 2


def test_count_combinations():
    """count_combinations should compute the Cartesian product size."""
    gen = GridSearchConfigGenerator()
    gen.add_dimension("a", [1, 2, 3])
    gen.add_dimension("b", ["x", "y"])
    assert gen.count_combinations() == 6  # 3 x 2


def test_create_minimal_grid():
    """create_minimal_grid should produce a generator with known dimensions."""
    gen = GridSearchConfigGenerator.create_minimal_grid()
    total = gen.count_combinations()
    # 2 resolutions x 5 dim reduction x 2 group modes x 2 models x 4 cascades = 160
    assert total == 160


def test_create_standard_grid():
    """create_standard_grid should produce a large grid."""
    gen = GridSearchConfigGenerator.create_standard_grid()
    total = gen.count_combinations()
    assert total > 500


def test_generate_configs_produces_model_entries():
    """generate_configs should produce a list of ModelEntry objects."""
    gen = GridSearchConfigGenerator()
    gen.add_dimension("model_config.model_type", [
        ModelType.LOGISTIC_L2.value,
        ModelType.GRADIENT_BOOSTING.value,
    ])
    gen.add_dimension("cascade_config.cascade_type", [
        CascadeType.BASE.value,
    ])

    configs = gen.generate_configs(
        target_type=TargetType.SWING.value,
        max_configs=5,
    )
    assert len(configs) == 2  # 2 models x 1 cascade
    for cfg in configs:
        assert isinstance(cfg, ModelEntry)
        assert cfg.target_type == TargetType.SWING.value


def test_generate_configs_with_max_configs():
    """max_configs should limit the number of generated configurations."""
    gen = GridSearchConfigGenerator()
    gen.add_dimension("a", list(range(10)))
    gen.add_dimension("b", list(range(10)))
    # 10x10 = 100, but limited to 5
    configs = gen.generate_configs(
        target_type=TargetType.SWING.value,
        max_configs=5,
    )
    assert len(configs) == 5


def test_add_constraint():
    """Adding a constraint should filter invalid combinations."""
    gen = GridSearchConfigGenerator()
    gen.add_dimension("dim_a", ["x", "y"])
    gen.add_dimension("dim_b", ["p", "q", "r"])
    # When dim_a=x, dim_b must be p
    gen.add_constraint("dim_a", "x", "dim_b", ["p"])

    configs = gen.generate_configs(target_type=TargetType.SWING.value)
    # Without constraint: 6 combos
    # With constraint: (x,p) + (y,p) + (y,q) + (y,r) = 4
    assert len(configs) == 4


def test_get_summary():
    """get_summary should return a string describing the grid."""
    gen = GridSearchConfigGenerator.create_minimal_grid()
    summary = gen.get_summary()
    assert isinstance(summary, str)
    assert "GRID SEARCH CONFIGURATION SUMMARY" in summary
    assert "Total dimensions:" in summary


# ---------------------------------------------------------------------------
# list_all_options tests
# ---------------------------------------------------------------------------

def test_list_all_options_returns_dict():
    """list_all_options should return a dict."""
    options = list_all_options()
    assert isinstance(options, dict)


def test_list_all_options_has_23_keys():
    """list_all_options should return dict with 23 keys (one per enum category)."""
    options = list_all_options()
    assert len(options) == 23


def test_list_all_options_values_are_lists():
    """Each value in list_all_options should be a non-empty list of strings."""
    options = list_all_options()
    for key, values in options.items():
        assert isinstance(values, list), f"{key} should map to a list"
        assert len(values) >= 2, f"{key} should have at least 2 options"
        for v in values:
            assert isinstance(v, str), f"Options for {key} should be strings"


def test_list_all_options_contains_key_categories():
    """list_all_options should have all major categories."""
    options = list_all_options()
    expected_keys = [
        "data_source",
        "model_type",
        "dim_reduction_method",
        "cascade_type",
        "cv_method",
        "target_type",
        "scaling_method",
        "outlier_method",
        "scoring_metric",
        "model_status",
    ]
    for key in expected_keys:
        assert key in options, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# PIPELINE_STEPS tests
# ---------------------------------------------------------------------------

def test_pipeline_steps_is_dict():
    """PIPELINE_STEPS should be a dict."""
    assert isinstance(PIPELINE_STEPS, dict)


def test_pipeline_steps_has_entries():
    """PIPELINE_STEPS should have 13 entries."""
    assert len(PIPELINE_STEPS) == 13


def test_pipeline_steps_entries_have_required_fields():
    """Each pipeline step should have name, config_class, description, key_parameters."""
    for step_num, info in PIPELINE_STEPS.items():
        assert isinstance(step_num, int), "Step number should be int"
        assert "name" in info, f"Step {step_num} missing 'name'"
        assert "config_class" in info, f"Step {step_num} missing 'config_class'"
        assert "description" in info, f"Step {step_num} missing 'description'"
        assert "key_parameters" in info, f"Step {step_num} missing 'key_parameters'"
        assert isinstance(info["key_parameters"], list)


def test_pipeline_steps_covers_key_steps():
    """PIPELINE_STEPS should cover data loading, model architecture, etc."""
    names = {info["name"] for info in PIPELINE_STEPS.values()}
    assert "Data Loading" in names
    assert "Model Architecture" in names
    assert "Training Procedure" in names


# ---------------------------------------------------------------------------
# create_model_entry tests
# ---------------------------------------------------------------------------

def test_create_model_entry_returns_model_entry():
    """create_model_entry should return a ModelEntry."""
    entry = create_model_entry(target_type=TargetType.SWING.value)
    assert isinstance(entry, ModelEntry)
    assert entry.target_type == TargetType.SWING.value


def test_create_model_entry_with_model_type():
    """create_model_entry should set model_type correctly."""
    entry = create_model_entry(
        target_type=TargetType.TIMING.value,
        model_type=ModelType.XGBOOST.value,
    )
    assert entry.model_config.model_type == ModelType.XGBOOST.value


def test_create_model_entry_with_cascade_type():
    """create_model_entry should set cascade_type correctly."""
    entry = create_model_entry(
        target_type=TargetType.SWING.value,
        cascade_type=CascadeType.ATTENTION.value,
    )
    assert entry.cascade_config.cascade_type == CascadeType.ATTENTION.value

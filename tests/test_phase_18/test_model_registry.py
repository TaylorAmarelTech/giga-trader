"""
Test ModelRegistryV2 creation, register/get operations, and summary.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase_18_persistence.model_registry import ModelRegistryV2
from src.phase_18_persistence.registry_configs import ModelEntry
from src.phase_18_persistence.registry_enums import (
    ModelStatus,
    TargetType,
    ModelType,
    CascadeType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_registry(tmp_path):
    """Create a ModelRegistryV2 with a temporary directory."""
    registry_dir = tmp_path / "registry"
    models_dir = tmp_path / "artifacts"
    return ModelRegistryV2(registry_dir=registry_dir, models_dir=models_dir)


@pytest.fixture
def sample_entry():
    """Create a sample ModelEntry for testing."""
    entry = ModelEntry(target_type=TargetType.SWING.value)
    entry.model_config.model_type = ModelType.GRADIENT_BOOSTING.value
    entry.cascade_config.cascade_type = CascadeType.BASE.value
    entry.metrics.cv_auc = 0.75
    entry.metrics.test_auc = 0.72
    entry.status = ModelStatus.TRAINED.value
    entry.tags = ["test", "v1"]
    return entry


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_registry_creation(tmp_registry):
    """ModelRegistryV2 should initialize with empty models dict."""
    assert isinstance(tmp_registry, ModelRegistryV2)
    assert len(tmp_registry.models) == 0
    assert tmp_registry.registry_dir.exists()
    assert tmp_registry.models_dir.exists()


def test_register_model(tmp_registry, sample_entry):
    """Registering a model should add it to the registry."""
    model_id = tmp_registry.register(sample_entry)
    assert model_id == sample_entry.model_id
    assert model_id in tmp_registry.models
    assert len(tmp_registry.models) == 1


def test_get_model(tmp_registry, sample_entry):
    """Getting a registered model should return the correct entry."""
    tmp_registry.register(sample_entry)
    retrieved = tmp_registry.get(sample_entry.model_id)
    assert retrieved is not None
    assert retrieved.model_id == sample_entry.model_id
    assert retrieved.target_type == TargetType.SWING.value
    assert retrieved.metrics.cv_auc == 0.75


def test_get_nonexistent_model(tmp_registry):
    """Getting a non-existent model should return None."""
    result = tmp_registry.get("nonexistent_model_id")
    assert result is None


def test_register_multiple_models(tmp_registry):
    """Multiple models can be registered."""
    for i in range(5):
        entry = ModelEntry(target_type=TargetType.SWING.value)
        entry.model_id = f"test_model_{i}"  # Explicit unique ID
        entry.metrics.cv_auc = 0.65 + i * 0.02
        entry.status = ModelStatus.TRAINED.value
        tmp_registry.register(entry)

    assert len(tmp_registry.models) == 5


def test_summary_returns_string(tmp_registry, sample_entry):
    """summary() should return a non-empty string."""
    tmp_registry.register(sample_entry)
    result = tmp_registry.summary()
    assert isinstance(result, str)
    assert len(result) > 0
    assert "MODEL REGISTRY V2 SUMMARY" in result
    assert "Total models: 1" in result


def test_summary_empty_registry(tmp_registry):
    """summary() should work even with no models."""
    result = tmp_registry.summary()
    assert isinstance(result, str)
    assert "Total models: 0" in result


def test_update_model(tmp_registry, sample_entry):
    """Updating a registered model should modify its attributes."""
    tmp_registry.register(sample_entry)
    tmp_registry.update(sample_entry.model_id, status=ModelStatus.PRODUCTION.value)
    retrieved = tmp_registry.get(sample_entry.model_id)
    assert retrieved.status == ModelStatus.PRODUCTION.value


def test_update_nonexistent_model_raises(tmp_registry):
    """Updating a non-existent model should raise KeyError."""
    with pytest.raises(KeyError):
        tmp_registry.update("nonexistent", status="failed")


def test_delete_model(tmp_registry, sample_entry):
    """Deleting a model should remove it from the registry."""
    tmp_registry.register(sample_entry)
    assert len(tmp_registry.models) == 1
    tmp_registry.delete(sample_entry.model_id)
    assert len(tmp_registry.models) == 0


def test_query_by_target_type(tmp_registry):
    """Querying by target_type should filter correctly."""
    configs = [
        ("swing_model_a", TargetType.SWING.value),
        ("timing_model_b", TargetType.TIMING.value),
        ("swing_model_c", TargetType.SWING.value),
    ]
    for model_id, target in configs:
        entry = ModelEntry(target_type=target)
        entry.model_id = model_id  # Explicit unique ID
        entry.status = ModelStatus.TRAINED.value
        tmp_registry.register(entry)

    swing_models = tmp_registry.query(target_type=TargetType.SWING.value)
    assert len(swing_models) == 2

    timing_models = tmp_registry.query(target_type=TargetType.TIMING.value)
    assert len(timing_models) == 1


def test_persistence_across_instances(tmp_path):
    """Registry should persist to disk and reload."""
    registry_dir = tmp_path / "registry"
    models_dir = tmp_path / "artifacts"

    # Create and populate
    reg1 = ModelRegistryV2(registry_dir=registry_dir, models_dir=models_dir)
    entry = ModelEntry(target_type=TargetType.SWING.value)
    entry.metrics.cv_auc = 0.80
    reg1.register(entry)
    saved_id = entry.model_id

    # Reload from same directory
    reg2 = ModelRegistryV2(registry_dir=registry_dir, models_dir=models_dir)
    assert len(reg2.models) == 1
    assert saved_id in reg2.models

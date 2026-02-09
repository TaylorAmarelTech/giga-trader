"""
Test StateManager creation, save/load, and atomic write behavior.
"""

import sys
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.state_manager import StateManager, SystemState


# ---------------------------------------------------------------------------
# SystemState tests
# ---------------------------------------------------------------------------

def test_system_state_creation():
    """SystemState should initialize with defaults."""
    state = SystemState()
    assert state.version == 1
    assert state.orchestrator_running is False
    assert state.mode == "IDLE"
    assert state.experiments_completed == 0
    assert state.best_auc == 0.0
    assert isinstance(state.errors, list)
    assert isinstance(state.custom_data, dict)


def test_system_state_to_dict():
    """to_dict should produce a serializable dictionary."""
    state = SystemState(experiments_completed=10, best_auc=0.78)
    d = state.to_dict()
    assert isinstance(d, dict)
    assert d["experiments_completed"] == 10
    assert d["best_auc"] == 0.78
    assert "version" in d
    assert "updated_at" in d


def test_system_state_from_dict():
    """from_dict should recreate a SystemState from a dictionary."""
    data = {
        "version": 1,
        "mode": "TRAINING",
        "experiments_completed": 5,
        "best_auc": 0.72,
        "errors": [],
        "custom_data": {"key": "value"},
        "updated_at": "2026-01-29T12:00:00",
    }
    state = SystemState.from_dict(data)
    assert state.mode == "TRAINING"
    assert state.experiments_completed == 5
    assert state.best_auc == 0.72
    assert state.custom_data["key"] == "value"


def test_system_state_from_dict_ignores_unknown_fields():
    """from_dict should ignore unknown fields gracefully."""
    data = {
        "version": 1,
        "mode": "IDLE",
        "unknown_field": "should_be_ignored",
    }
    state = SystemState.from_dict(data)
    assert state.mode == "IDLE"
    assert not hasattr(state, "unknown_field")


# ---------------------------------------------------------------------------
# StateManager tests
# ---------------------------------------------------------------------------

def test_state_manager_creation(tmp_path):
    """StateManager should initialize with a state directory."""
    sm = StateManager(state_dir=str(tmp_path))
    assert isinstance(sm, StateManager)
    assert isinstance(sm.state, SystemState)


def test_state_manager_update(tmp_path):
    """update should modify state fields."""
    sm = StateManager(state_dir=str(tmp_path))
    sm.update(experiments_completed=10, best_auc=0.75)

    assert sm.state.experiments_completed == 10
    assert sm.state.best_auc == 0.75


def test_state_manager_update_custom_data(tmp_path):
    """update with custom_data should merge rather than replace."""
    sm = StateManager(state_dir=str(tmp_path))
    sm.update(custom_data={"key1": "val1"})
    sm.update(custom_data={"key2": "val2"})

    assert sm.state.custom_data["key1"] == "val1"
    assert sm.state.custom_data["key2"] == "val2"


def test_state_manager_update_unknown_keys(tmp_path):
    """Unknown keys should be stored in custom_data."""
    sm = StateManager(state_dir=str(tmp_path))
    sm.update(unknown_key="some_value")

    assert sm.state.custom_data["unknown_key"] == "some_value"


def test_state_manager_save(tmp_path):
    """save should write state to disk."""
    sm = StateManager(state_dir=str(tmp_path))
    sm.update(mode="TRAINING", best_auc=0.80)
    result = sm.save()

    assert result is True
    assert sm.state_file.exists()


def test_state_manager_atomic_write(tmp_path):
    """After save, the state file should exist (atomic write completed)."""
    sm = StateManager(state_dir=str(tmp_path))
    sm.update(experiments_completed=42)
    sm.save()

    # Verify the file exists and is valid JSON
    assert sm.state_file.exists()
    with open(sm.state_file, "r") as f:
        data = json.load(f)
    assert data["experiments_completed"] == 42
    assert "_checksum" in data


def test_state_manager_load(tmp_path):
    """StateManager should load persisted state on initialization."""
    # Save state
    sm1 = StateManager(state_dir=str(tmp_path))
    sm1.update(mode="BACKTESTING", experiments_completed=25, best_auc=0.76)
    sm1.save()

    # Create new manager from same directory
    sm2 = StateManager(state_dir=str(tmp_path))
    assert sm2.state.mode == "BACKTESTING"
    assert sm2.state.experiments_completed == 25
    assert sm2.state.best_auc == 0.76


def test_state_manager_save_and_load_round_trip(tmp_path):
    """State should survive a full save/load cycle."""
    sm1 = StateManager(state_dir=str(tmp_path))
    sm1.update(
        mode="LIVE_TRADING",
        experiments_completed=100,
        models_trained=50,
        best_auc=0.82,
        custom_data={"strategy": "momentum", "version": 2},
    )
    sm1.save()

    sm2 = StateManager(state_dir=str(tmp_path))
    assert sm2.state.mode == "LIVE_TRADING"
    assert sm2.state.experiments_completed == 100
    assert sm2.state.models_trained == 50
    assert sm2.state.best_auc == 0.82
    assert sm2.state.custom_data["strategy"] == "momentum"
    assert sm2.state.custom_data["version"] == 2


def test_state_manager_backup(tmp_path):
    """create_backup should create a backup file."""
    sm = StateManager(state_dir=str(tmp_path))
    sm.update(mode="TESTING")
    sm.save()

    backup_path = sm.create_backup()
    assert backup_path is not None
    assert Path(backup_path).exists()


def test_state_manager_no_existing_state(tmp_path):
    """StateManager should start fresh when no state file exists."""
    sm = StateManager(state_dir=str(tmp_path / "new_dir"))
    assert sm.state.mode == "IDLE"
    assert sm.state.experiments_completed == 0


def test_state_manager_checksum_in_file(tmp_path):
    """Saved state file should contain a _checksum field."""
    sm = StateManager(state_dir=str(tmp_path))
    sm.update(best_auc=0.65)
    sm.save()

    with open(sm.state_file, "r") as f:
        data = json.load(f)
    assert "_checksum" in data
    assert isinstance(data["_checksum"], str)
    assert len(data["_checksum"]) == 32  # MD5 hex digest length

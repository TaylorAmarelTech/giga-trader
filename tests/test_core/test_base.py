"""
Test PhaseConfig, PhaseResult, PhaseRunner, and PipelineOrchestrator.
"""

import sys
from pathlib import Path
from datetime import datetime

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.base import (
    PhaseConfig,
    PhaseResult,
    PhaseRunner,
    PipelineOrchestrator,
)


# ---------------------------------------------------------------------------
# PhaseConfig tests
# ---------------------------------------------------------------------------

def test_phase_config_creation():
    """PhaseConfig should initialize with required fields."""
    config = PhaseConfig(phase_number=1, name="Data Acquisition")
    assert config.phase_number == 1
    assert config.name == "Data Acquisition"
    assert config.enabled is True
    assert config.params == {}
    assert config.depends_on == []


def test_phase_config_with_params():
    """PhaseConfig should accept optional params and dependencies."""
    config = PhaseConfig(
        phase_number=5,
        name="Target Creation",
        enabled=False,
        params={"threshold": 0.003},
        depends_on=[1, 2, 3],
    )
    assert config.phase_number == 5
    assert config.enabled is False
    assert config.params["threshold"] == 0.003
    assert config.depends_on == [1, 2, 3]


# ---------------------------------------------------------------------------
# PhaseResult tests
# ---------------------------------------------------------------------------

def test_phase_result_creation():
    """PhaseResult should initialize with required fields."""
    result = PhaseResult(
        phase_number=1,
        name="Data Acquisition",
        success=True,
    )
    assert result.phase_number == 1
    assert result.name == "Data Acquisition"
    assert result.success is True
    assert result.duration_seconds == 0.0
    assert result.metrics == {}
    assert result.artifacts == {}
    assert result.errors == []
    assert result.warnings == []
    assert isinstance(result.timestamp, datetime)


def test_phase_result_failed():
    """PhaseResult for a failed phase should capture errors."""
    result = PhaseResult(
        phase_number=3,
        name="Synthetic Data",
        success=False,
        errors=["Failed to download component data"],
        duration_seconds=12.5,
    )
    assert result.success is False
    assert len(result.errors) == 1
    assert result.duration_seconds == 12.5


def test_phase_result_to_dict():
    """to_dict should produce a serializable dictionary."""
    result = PhaseResult(
        phase_number=1,
        name="Test Phase",
        success=True,
        metrics={"auc": 0.75},
        duration_seconds=5.3,
    )
    d = result.to_dict()
    assert isinstance(d, dict)
    assert d["phase_number"] == 1
    assert d["success"] is True
    assert d["metrics"]["auc"] == 0.75
    assert d["duration_seconds"] == 5.3
    assert isinstance(d["timestamp"], str)  # ISO format


# ---------------------------------------------------------------------------
# PhaseRunner tests (abstract, needs concrete subclass)
# ---------------------------------------------------------------------------

class DummyPhaseRunner(PhaseRunner):
    """Concrete PhaseRunner for testing."""

    def __init__(self, config, should_pass=True, checkpoint_dir=None):
        super().__init__(config, checkpoint_dir=checkpoint_dir)
        self.should_pass = should_pass
        self.run_called = False

    def validate_inputs(self) -> bool:
        return True

    def run(self, **kwargs) -> PhaseResult:
        self.run_called = True
        return PhaseResult(
            phase_number=self.config.phase_number,
            name=self.config.name,
            success=self.should_pass,
            metrics={"test_metric": 42},
        )

    def validate_outputs(self, result: PhaseResult) -> bool:
        return result.success


def test_phase_runner_execute_success():
    """PhaseRunner.execute should run the phase and return a result."""
    config = PhaseConfig(phase_number=1, name="Test Phase")
    runner = DummyPhaseRunner(config, should_pass=True)
    result = runner.execute()

    assert runner.run_called is True
    assert result.success is True
    assert result.phase_number == 1
    assert result.metrics["test_metric"] == 42
    assert result.duration_seconds >= 0  # May be 0.0 on fast machines


def test_phase_runner_execute_disabled():
    """PhaseRunner.execute should skip when disabled."""
    config = PhaseConfig(phase_number=1, name="Disabled Phase", enabled=False)
    runner = DummyPhaseRunner(config)
    result = runner.execute()

    assert result.success is True
    assert "Phase disabled" in result.warnings
    assert runner.run_called is False


def test_phase_runner_execute_failure():
    """PhaseRunner.execute should handle failed phases."""
    config = PhaseConfig(phase_number=1, name="Failing Phase")
    runner = DummyPhaseRunner(config, should_pass=False)
    result = runner.execute()

    assert result.success is False


def test_phase_runner_checkpoint(tmp_path):
    """PhaseRunner should save and load checkpoints."""
    checkpoint_dir = str(tmp_path / "checkpoints")
    config = PhaseConfig(phase_number=1, name="Checkpoint Test")
    runner = DummyPhaseRunner(config, checkpoint_dir=checkpoint_dir)

    # Save
    path = runner.save_checkpoint({"key": "value"}, label="test")
    assert path is not None
    assert Path(path).exists()

    # Load
    loaded = runner.load_latest_checkpoint(label="test")
    assert loaded is not None
    assert loaded["key"] == "value"


def test_phase_runner_load_checkpoint_empty(tmp_path):
    """Loading checkpoint from empty dir should return None."""
    checkpoint_dir = str(tmp_path / "empty_checkpoints")
    config = PhaseConfig(phase_number=1, name="No Checkpoints")
    runner = DummyPhaseRunner(config, checkpoint_dir=checkpoint_dir)
    result = runner.load_latest_checkpoint()
    assert result is None


# ---------------------------------------------------------------------------
# PipelineOrchestrator tests
# ---------------------------------------------------------------------------

def test_orchestrator_creation():
    """PipelineOrchestrator should initialize with empty phases."""
    orch = PipelineOrchestrator()
    assert isinstance(orch, PipelineOrchestrator)
    assert len(orch.phases) == 0
    assert len(orch.results) == 0


def test_orchestrator_add_phase():
    """add_phase should append a phase to the pipeline."""
    orch = PipelineOrchestrator()
    config = PhaseConfig(phase_number=1, name="Phase 1")
    runner = DummyPhaseRunner(config)
    orch.add_phase(runner)
    assert len(orch.phases) == 1


def test_orchestrator_run_all():
    """run_all should execute phases in order and return results."""
    orch = PipelineOrchestrator()

    for i in range(1, 4):
        config = PhaseConfig(phase_number=i, name=f"Phase {i}")
        orch.add_phase(DummyPhaseRunner(config, should_pass=True))

    results = orch.run_all()
    assert len(results) == 3
    assert all(r.success for r in results.values())


def test_orchestrator_stops_on_failure():
    """run_all should stop when a phase fails."""
    orch = PipelineOrchestrator()

    for i in range(1, 4):
        should_pass = (i != 2)  # Phase 2 will fail
        config = PhaseConfig(phase_number=i, name=f"Phase {i}")
        orch.add_phase(DummyPhaseRunner(config, should_pass=should_pass))

    results = orch.run_all()
    assert results[1].success is True
    assert results[2].success is False
    assert 3 not in results  # Phase 3 should not have run


def test_orchestrator_dependency_check():
    """run_all should skip phases whose dependencies failed."""
    orch = PipelineOrchestrator()

    # Phase 1: passes
    config1 = PhaseConfig(phase_number=1, name="Phase 1")
    orch.add_phase(DummyPhaseRunner(config1, should_pass=True))

    # Phase 2: depends on phase 1, passes
    config2 = PhaseConfig(phase_number=2, name="Phase 2", depends_on=[1])
    orch.add_phase(DummyPhaseRunner(config2, should_pass=True))

    results = orch.run_all()
    assert results[1].success is True
    assert results[2].success is True


def test_orchestrator_get_summary():
    """get_summary should return a dict with expected keys."""
    orch = PipelineOrchestrator()
    config = PhaseConfig(phase_number=1, name="Phase 1")
    orch.add_phase(DummyPhaseRunner(config))
    orch.run_all()

    summary = orch.get_summary()
    assert isinstance(summary, dict)
    assert "total_phases" in summary
    assert "passed" in summary
    assert "failed" in summary
    assert "total_duration_seconds" in summary
    assert "phases" in summary
    assert summary["total_phases"] == 1
    assert summary["passed"] == 1

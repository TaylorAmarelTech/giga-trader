"""
Phase Runner Base Classes
=========================
Adapted from archive/orphaned_2026-02-03/core/base.py.

Provides a consistent framework for implementing pipeline phases with:
- Input/output validation
- Checkpoint/recovery support
- Timing and metrics collection
- Error handling
"""

import time
import pickle
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger("PHASE_RUNNER")

project_root = Path(__file__).parent.parent.parent


@dataclass
class PhaseConfig:
    """Configuration for a pipeline phase."""
    phase_number: int
    name: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[int] = field(default_factory=list)


@dataclass
class PhaseResult:
    """Result of executing a pipeline phase."""
    phase_number: int
    name: str
    success: bool
    duration_seconds: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "phase_number": self.phase_number,
            "name": self.name,
            "success": self.success,
            "duration_seconds": self.duration_seconds,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "errors": self.errors,
            "warnings": self.warnings,
            "timestamp": self.timestamp.isoformat(),
        }


class PhaseRunner(ABC):
    """
    Abstract base class for pipeline phase execution.

    Subclasses implement:
    - validate_inputs(): Check preconditions
    - run(): Execute the phase logic
    - validate_outputs(): Verify quality gates

    The execute() method orchestrates the full flow with error handling,
    timing, and optional checkpointing.
    """

    def __init__(self, config: PhaseConfig, checkpoint_dir: Optional[str] = None):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else (
            project_root / "models" / "checkpoints" / f"phase_{config.phase_number:02d}"
        )
        self.logger = logging.getLogger(f"Phase_{config.phase_number:02d}_{config.name}")

    @abstractmethod
    def validate_inputs(self) -> bool:
        """Validate that all required inputs are available."""
        pass

    @abstractmethod
    def run(self, **kwargs) -> PhaseResult:
        """Execute the phase logic."""
        pass

    @abstractmethod
    def validate_outputs(self, result: PhaseResult) -> bool:
        """Validate that outputs meet quality gates."""
        pass

    def execute(self, **kwargs) -> PhaseResult:
        """
        Full phase execution with validation, timing, and error handling.

        Flow:
        1. Validate inputs
        2. Run phase logic
        3. Validate outputs
        4. Return result
        """
        start_time = time.time()
        phase_name = f"Phase {self.config.phase_number}: {self.config.name}"

        self.logger.info(f"{'='*60}")
        self.logger.info(f"Starting {phase_name}")
        self.logger.info(f"{'='*60}")

        # Check if disabled
        if not self.config.enabled:
            self.logger.info(f"{phase_name} is disabled, skipping")
            return PhaseResult(
                phase_number=self.config.phase_number,
                name=self.config.name,
                success=True,
                warnings=["Phase disabled"],
            )

        # Validate inputs
        try:
            if not self.validate_inputs():
                return PhaseResult(
                    phase_number=self.config.phase_number,
                    name=self.config.name,
                    success=False,
                    errors=["Input validation failed"],
                    duration_seconds=time.time() - start_time,
                )
        except Exception as e:
            self.logger.error(f"Input validation error: {e}")
            return PhaseResult(
                phase_number=self.config.phase_number,
                name=self.config.name,
                success=False,
                errors=[f"Input validation error: {str(e)}"],
                duration_seconds=time.time() - start_time,
            )

        # Run phase
        try:
            result = self.run(**kwargs)
            result.duration_seconds = time.time() - start_time
        except Exception as e:
            self.logger.error(f"Phase execution error: {e}", exc_info=True)
            return PhaseResult(
                phase_number=self.config.phase_number,
                name=self.config.name,
                success=False,
                errors=[f"Execution error: {str(e)}"],
                duration_seconds=time.time() - start_time,
            )

        # Validate outputs
        try:
            if result.success and not self.validate_outputs(result):
                result.success = False
                result.errors.append("Output validation failed")
        except Exception as e:
            self.logger.warning(f"Output validation error: {e}")
            result.warnings.append(f"Output validation error: {str(e)}")

        # Log result
        status = "SUCCESS" if result.success else "FAILED"
        self.logger.info(
            f"{phase_name}: {status} in {result.duration_seconds:.1f}s"
        )
        if result.errors:
            for err in result.errors:
                self.logger.error(f"  Error: {err}")
        if result.warnings:
            for warn in result.warnings:
                self.logger.warning(f"  Warning: {warn}")

        return result

    def save_checkpoint(self, state: Any, label: str = "checkpoint"):
        """Save a checkpoint for crash recovery."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.checkpoint_dir / f"{label}_{timestamp}.pkl"

        try:
            with open(path, "wb") as f:
                pickle.dump(state, f)
            self.logger.debug(f"Checkpoint saved: {path}")
            return str(path)
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")
            return None

    def load_latest_checkpoint(self, label: str = "checkpoint") -> Optional[Any]:
        """Load the most recent checkpoint."""
        if not self.checkpoint_dir.exists():
            return None

        checkpoints = sorted(
            self.checkpoint_dir.glob(f"{label}_*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not checkpoints:
            return None

        try:
            with open(checkpoints[0], "rb") as f:
                state = pickle.load(f)
            self.logger.info(f"Loaded checkpoint: {checkpoints[0]}")
            return state
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            return None


class PipelineOrchestrator:
    """
    Orchestrates sequential execution of multiple PhaseRunners.

    Handles:
    - Dependency ordering
    - Phase skip/enable control
    - Result collection
    - Overall pipeline status
    """

    def __init__(self, phases: Optional[List[PhaseRunner]] = None):
        self.phases: List[PhaseRunner] = phases or []
        self.results: Dict[int, PhaseResult] = {}
        self.logger = logging.getLogger("PIPELINE")

    def add_phase(self, phase: PhaseRunner):
        """Add a phase to the pipeline."""
        self.phases.append(phase)

    def run_all(self, **kwargs) -> Dict[int, PhaseResult]:
        """Run all phases in order."""
        self.logger.info(f"Starting pipeline with {len(self.phases)} phases")

        for phase in sorted(self.phases, key=lambda p: p.config.phase_number):
            # Check dependencies
            deps_met = all(
                self.results.get(dep, PhaseResult(0, "", False)).success
                for dep in phase.config.depends_on
            )

            if not deps_met:
                self.logger.warning(
                    f"Skipping Phase {phase.config.phase_number} "
                    f"(dependencies not met: {phase.config.depends_on})"
                )
                self.results[phase.config.phase_number] = PhaseResult(
                    phase_number=phase.config.phase_number,
                    name=phase.config.name,
                    success=False,
                    errors=["Dependencies not met"],
                )
                continue

            result = phase.execute(**kwargs)
            self.results[phase.config.phase_number] = result

            if not result.success:
                self.logger.error(
                    f"Phase {phase.config.phase_number} failed, "
                    f"stopping pipeline"
                )
                break

        return self.results

    def get_summary(self) -> Dict:
        """Get pipeline execution summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r.success)
        failed = sum(1 for r in self.results.values() if not r.success)
        total_time = sum(r.duration_seconds for r in self.results.values())

        return {
            "total_phases": total,
            "passed": passed,
            "failed": failed,
            "total_duration_seconds": total_time,
            "phases": {
                num: r.to_dict() for num, r in self.results.items()
            },
        }

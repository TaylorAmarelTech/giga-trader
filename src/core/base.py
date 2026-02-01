"""
Base classes for pipeline phases.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import pickle


@dataclass
class PhaseConfig:
    """Configuration for a pipeline phase."""
    phase_num: int
    phase_name: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhaseResult:
    """Result of running a pipeline phase."""
    phase_num: int
    phase_name: str
    success: bool
    start_time: datetime
    end_time: datetime
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Path] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()


class PhaseRunner(ABC):
    """
    Abstract base class for pipeline phase runners.

    Each phase should implement:
    - validate_inputs(): Check preconditions
    - run(): Execute phase steps
    - validate_outputs(): Verify results
    """

    def __init__(self, config: PhaseConfig):
        self.config = config
        self._checkpoint_dir = Path("models/checkpoints")
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def validate_inputs(self) -> bool:
        """
        Validate that all required inputs are present and valid.

        Returns:
            True if inputs are valid, False otherwise
        """
        pass

    @abstractmethod
    def run(self) -> PhaseResult:
        """
        Execute all steps in this phase.

        Returns:
            PhaseResult with success status and metrics
        """
        pass

    @abstractmethod
    def validate_outputs(self) -> bool:
        """
        Validate that outputs meet quality gates.

        Returns:
            True if outputs are valid, False otherwise
        """
        pass

    def save_checkpoint(self, state: Dict[str, Any], step: int) -> Path:
        """
        Save a checkpoint of the current state.

        Args:
            state: Dictionary containing state to save
            step: Current step number

        Returns:
            Path to saved checkpoint file
        """
        filename = (
            f"checkpoint_p{self.config.phase_num:02d}_"
            f"s{step:03d}_{datetime.now():%Y%m%d_%H%M%S}.pkl"
        )
        filepath = self._checkpoint_dir / filename
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        return filepath

    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint for this phase.

        Returns:
            Checkpoint state dict or None if no checkpoint exists
        """
        pattern = f"checkpoint_p{self.config.phase_num:02d}_*.pkl"
        checkpoints = sorted(self._checkpoint_dir.glob(pattern))
        if checkpoints:
            with open(checkpoints[-1], "rb") as f:
                return pickle.load(f)
        return None

    def execute(self) -> PhaseResult:
        """
        Full execution flow: validate inputs, run, validate outputs.

        Returns:
            PhaseResult with complete execution details
        """
        from .logging import get_logger

        logger = get_logger(f"phase_{self.config.phase_num:02d}")
        start_time = datetime.now()

        try:
            # Validate inputs
            logger.info(f"Phase {self.config.phase_num}: Validating inputs...")
            if not self.validate_inputs():
                return PhaseResult(
                    phase_num=self.config.phase_num,
                    phase_name=self.config.phase_name,
                    success=False,
                    start_time=start_time,
                    end_time=datetime.now(),
                    errors=["Input validation failed"],
                )

            # Run phase
            logger.info(f"Phase {self.config.phase_num}: Running...")
            result = self.run()

            # Validate outputs
            if result.success:
                logger.info(f"Phase {self.config.phase_num}: Validating outputs...")
                if not self.validate_outputs():
                    result.success = False
                    result.errors.append("Output validation failed")

            result.end_time = datetime.now()
            logger.info(
                f"Phase {self.config.phase_num}: "
                f"{'Completed' if result.success else 'Failed'} "
                f"in {result.duration_seconds:.1f}s"
            )
            return result

        except Exception as e:
            logger.error(f"Phase {self.config.phase_num} failed: {e}")
            return PhaseResult(
                phase_num=self.config.phase_num,
                phase_name=self.config.phase_name,
                success=False,
                start_time=start_time,
                end_time=datetime.now(),
                errors=[str(e)],
            )

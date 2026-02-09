"""
GIGA TRADER - Experiment Progress Tracker
==========================================
Tracks experiment progress in real-time for dashboard visibility.

This module provides a shared progress tracker that:
  - Records current experiment step and substep
  - Tracks metrics as they're computed
  - Monitors last activity timestamp for stuck detection
  - Persists to JSON for dashboard consumption
"""

import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum


project_root = Path(__file__).parent.parent


class ExperimentStep(Enum):
    """Experiment pipeline steps."""
    IDLE = "idle"
    LOADING_DATA = "loading_data"
    FEATURE_ENGINEERING = "feature_engineering"
    ANTI_OVERFIT = "anti_overfit"
    DIM_REDUCTION = "dim_reduction"
    TRAIN_TEST_SPLIT = "train_test_split"
    MODEL_TRAINING = "model_training"
    CROSS_VALIDATION = "cross_validation"
    EVALUATION = "evaluation"
    BACKTEST = "backtest"
    COMPLETE = "complete"
    FAILED = "failed"


STEP_ORDER = [
    ExperimentStep.LOADING_DATA,
    ExperimentStep.FEATURE_ENGINEERING,
    ExperimentStep.ANTI_OVERFIT,
    ExperimentStep.DIM_REDUCTION,
    ExperimentStep.TRAIN_TEST_SPLIT,
    ExperimentStep.MODEL_TRAINING,
    ExperimentStep.CROSS_VALIDATION,
    ExperimentStep.EVALUATION,
    ExperimentStep.BACKTEST,
    ExperimentStep.COMPLETE,
]

STEP_DESCRIPTIONS = {
    ExperimentStep.IDLE: "Waiting for next experiment",
    ExperimentStep.LOADING_DATA: "Loading market data from Alpaca",
    ExperimentStep.FEATURE_ENGINEERING: "Engineering features (premarket, afterhours, patterns)",
    ExperimentStep.ANTI_OVERFIT: "Running anti-overfit augmentation (synthetic universes, cross-assets)",
    ExperimentStep.DIM_REDUCTION: "Reducing dimensions (UMAP, Kernel PCA, ICA)",
    ExperimentStep.TRAIN_TEST_SPLIT: "Splitting data for training/testing",
    ExperimentStep.MODEL_TRAINING: "Training ML model",
    ExperimentStep.CROSS_VALIDATION: "Running cross-validation",
    ExperimentStep.EVALUATION: "Computing WMES and stability metrics",
    ExperimentStep.BACKTEST: "Running backtest simulation",
    ExperimentStep.COMPLETE: "Experiment completed successfully",
    ExperimentStep.FAILED: "Experiment failed",
}

# Expected duration ranges per step (min_seconds, typical_seconds, max_seconds)
# Used by dashboard to show "this step normally takes X" instead of "STUCK"
STEP_EXPECTED_DURATIONS = {
    ExperimentStep.IDLE: (0, 0, 0),
    ExperimentStep.LOADING_DATA: (10, 30, 120),
    ExperimentStep.FEATURE_ENGINEERING: (15, 60, 300),
    ExperimentStep.ANTI_OVERFIT: (60, 600, 3600),        # 1-60 min (downloads + synthetic universes)
    ExperimentStep.DIM_REDUCTION: (10, 60, 300),
    ExperimentStep.TRAIN_TEST_SPLIT: (1, 5, 30),
    ExperimentStep.MODEL_TRAINING: (30, 300, 3600),      # 0.5-60 min (depends on model type)
    ExperimentStep.CROSS_VALIDATION: (60, 600, 7200),    # 1-120 min (leak-proof CV is expensive)
    ExperimentStep.EVALUATION: (30, 300, 3600),           # 0.5-60 min (includes stability + fragility)
    ExperimentStep.BACKTEST: (5, 30, 180),
    ExperimentStep.COMPLETE: (0, 0, 0),
    ExperimentStep.FAILED: (0, 0, 0),
}


@dataclass
class ExperimentProgress:
    """Current experiment progress state."""

    # Status
    is_running: bool = False
    step: str = "idle"
    step_number: int = 0
    total_steps: int = 10
    step_description: str = "Waiting for next experiment"

    # Current experiment info
    experiment_id: str = ""
    experiment_type: str = ""
    experiment_name: str = ""
    started_at: str = ""

    # Timing
    last_activity: str = ""
    seconds_since_activity: float = 0.0
    elapsed_seconds: float = 0.0
    estimated_remaining: float = 0.0

    # Substep progress
    substep: str = ""
    substep_progress: float = 0.0  # 0-100

    # Live metrics (updated as computed)
    live_metrics: Dict[str, Any] = field(default_factory=dict)

    # Recent history
    recent_experiments: List[Dict] = field(default_factory=list)

    # Statistics
    experiments_today: int = 0
    experiments_total: int = 0
    avg_experiment_duration: float = 0.0
    success_rate: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


class ExperimentProgressTracker:
    """
    Singleton tracker for experiment progress.

    Usage:
        tracker = ExperimentProgressTracker.instance()
        tracker.start_experiment(experiment_id, experiment_type)
        tracker.set_step(ExperimentStep.LOADING_DATA)
        tracker.update_substep("Loading 5 years of data", 25)
        tracker.record_metric("n_samples", 1250)
        tracker.complete_experiment(result)
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.progress = ExperimentProgress()
        self.progress_file = project_root / "logs" / "experiment_progress.json"
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        self._start_time: Optional[datetime] = None
        self._step_times: Dict[str, float] = {}
        self._load()

    @classmethod
    def instance(cls) -> "ExperimentProgressTracker":
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load(self):
        """Load existing progress from disk."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file) as f:
                    data = json.load(f)
                    # Restore recent experiments and stats
                    self.progress.recent_experiments = data.get("recent_experiments", [])[-20:]
                    self.progress.experiments_today = data.get("experiments_today", 0)
                    self.progress.experiments_total = data.get("experiments_total", 0)
                    self.progress.avg_experiment_duration = data.get("avg_experiment_duration", 0)
                    self.progress.success_rate = data.get("success_rate", 0)
            except Exception:
                pass

    def _save(self):
        """Save progress to disk."""
        try:
            now = datetime.now()
            self.progress.last_activity = now.isoformat()

            # Compute seconds since activity
            if self._start_time:
                self.progress.elapsed_seconds = (now - self._start_time).total_seconds()

            with open(self.progress_file, "w") as f:
                json.dump(self.progress.to_dict(), f, indent=2, default=str)
        except Exception:
            pass

    def touch(self):
        """Update last activity timestamp (proves we're not stuck)."""
        self.progress.last_activity = datetime.now().isoformat()
        self._save()

    def start_experiment(self, experiment_id: str, experiment_type: str, experiment_name: str = ""):
        """Mark experiment as started."""
        with self._lock:
            self._start_time = datetime.now()
            self.progress.is_running = True
            self.progress.experiment_id = experiment_id
            self.progress.experiment_type = experiment_type
            self.progress.experiment_name = experiment_name or experiment_type
            self.progress.started_at = self._start_time.isoformat()
            self.progress.step = ExperimentStep.LOADING_DATA.value
            self.progress.step_number = 1
            self.progress.step_description = STEP_DESCRIPTIONS[ExperimentStep.LOADING_DATA]
            self.progress.live_metrics = {}
            self.progress.substep = ""
            self.progress.substep_progress = 0.0
            self._step_times = {}
            self._save()

    def set_step(self, step: ExperimentStep, substep: str = ""):
        """Update current step."""
        with self._lock:
            self.progress.step = step.value
            self.progress.step_description = STEP_DESCRIPTIONS.get(step, step.value)
            self.progress.substep = substep
            self.progress.substep_progress = 0.0

            # Calculate step number
            try:
                idx = STEP_ORDER.index(step)
                self.progress.step_number = idx + 1
            except ValueError:
                pass

            # Track step timing
            now = time.time()
            self._step_times[step.value] = now

            self._save()

    def update_substep(self, substep: str, progress_pct: float = 0.0):
        """Update substep progress."""
        with self._lock:
            self.progress.substep = substep
            self.progress.substep_progress = min(100, max(0, progress_pct))
            self._save()

    def record_metric(self, name: str, value: Any):
        """Record a live metric."""
        with self._lock:
            self.progress.live_metrics[name] = value
            self._save()

    def record_metrics(self, metrics: Dict[str, Any]):
        """Record multiple metrics at once."""
        with self._lock:
            self.progress.live_metrics.update(metrics)
            self._save()

    def complete_experiment(self, success: bool, result_summary: Dict = None):
        """Mark experiment as complete."""
        with self._lock:
            now = datetime.now()
            duration = (now - self._start_time).total_seconds() if self._start_time else 0

            # Record to history
            history_entry = {
                "experiment_id": self.progress.experiment_id,
                "experiment_type": self.progress.experiment_type,
                "experiment_name": self.progress.experiment_name,
                "started_at": self.progress.started_at,
                "completed_at": now.isoformat(),
                "duration_seconds": round(duration, 1),
                "success": success,
                "metrics": dict(self.progress.live_metrics),
            }
            if result_summary:
                history_entry.update(result_summary)

            self.progress.recent_experiments.append(history_entry)
            self.progress.recent_experiments = self.progress.recent_experiments[-20:]

            # Update statistics
            self.progress.experiments_today += 1
            self.progress.experiments_total += 1

            # Update average duration
            if self.progress.experiments_total > 0:
                prev_total = self.progress.avg_experiment_duration * (self.progress.experiments_total - 1)
                self.progress.avg_experiment_duration = (prev_total + duration) / self.progress.experiments_total

            # Update success rate
            successful = sum(1 for e in self.progress.recent_experiments if e.get("success"))
            self.progress.success_rate = successful / len(self.progress.recent_experiments) if self.progress.recent_experiments else 0

            # Reset running state
            self.progress.is_running = False
            self.progress.step = ExperimentStep.COMPLETE.value if success else ExperimentStep.FAILED.value
            self.progress.step_description = STEP_DESCRIPTIONS[ExperimentStep.COMPLETE if success else ExperimentStep.FAILED]
            self.progress.step_number = self.progress.total_steps if success else 0

            self._save()

    def fail_experiment(self, error_message: str):
        """Mark experiment as failed."""
        self.record_metric("error", error_message)
        self.complete_experiment(success=False, result_summary={"error": error_message})

    def get_progress(self) -> Dict:
        """Get current progress for API."""
        with self._lock:
            data = self.progress.to_dict()

            # Look up expected duration for current step
            step_enum = None
            for s in ExperimentStep:
                if s.value == self.progress.step:
                    step_enum = s
                    break
            expected = STEP_EXPECTED_DURATIONS.get(step_enum, (60, 300, 1800))
            data["step_expected_min"] = expected[0]
            data["step_expected_typical"] = expected[1]
            data["step_expected_max"] = expected[2]

            # Add computed fields with step-aware stuck thresholds
            if self.progress.last_activity:
                try:
                    last = datetime.fromisoformat(self.progress.last_activity)
                    secs = (datetime.now() - last).total_seconds()
                    data["seconds_since_activity"] = secs
                    # Stuck thresholds scale with expected step duration
                    # Warning: 2x the typical duration for this step (min 10 min)
                    # Critical: 3x the max duration for this step (min 30 min)
                    warn_threshold = max(600, expected[1] * 2)
                    crit_threshold = max(1800, expected[2] * 3)
                    data["stuck_warning"] = secs > warn_threshold
                    data["stuck_critical"] = secs > crit_threshold
                    data["stuck_warn_threshold"] = warn_threshold
                    data["stuck_crit_threshold"] = crit_threshold
                except (ValueError, TypeError):
                    data["seconds_since_activity"] = 0
                    data["stuck_warning"] = False
                    data["stuck_critical"] = False

            # Estimate remaining time
            if self.progress.is_running and self.progress.avg_experiment_duration > 0:
                data["estimated_remaining"] = max(0, self.progress.avg_experiment_duration - self.progress.elapsed_seconds)

            return data

    def reset_daily_count(self):
        """Reset daily experiment count (call at midnight)."""
        with self._lock:
            self.progress.experiments_today = 0
            self._save()


# Convenience function - reads directly from file for cross-process communication
def get_experiment_progress() -> Dict:
    """
    Get current experiment progress for dashboard.

    Reads directly from the JSON file to support cross-process communication.
    The orchestrator writes progress, and the dashboard reads it.
    """
    progress_file = project_root / "logs" / "experiment_progress.json"

    if progress_file.exists():
        try:
            with open(progress_file) as f:
                data = json.load(f)

            # Look up expected duration for current step
            step_str = data.get("step", "idle")
            step_enum = None
            for s in ExperimentStep:
                if s.value == step_str:
                    step_enum = s
                    break
            expected = STEP_EXPECTED_DURATIONS.get(step_enum, (60, 300, 1800))
            data["step_expected_min"] = expected[0]
            data["step_expected_typical"] = expected[1]
            data["step_expected_max"] = expected[2]

            # Add computed fields with step-aware stuck thresholds
            if data.get("last_activity"):
                try:
                    last = datetime.fromisoformat(data["last_activity"])
                    secs = (datetime.now() - last).total_seconds()
                    data["seconds_since_activity"] = secs
                    # Stuck thresholds scale with expected step duration
                    # Warning: 2x the typical duration (min 10 min)
                    # Critical: 3x the max duration (min 30 min)
                    warn_threshold = max(600, expected[1] * 2)
                    crit_threshold = max(1800, expected[2] * 3)
                    data["stuck_warning"] = secs > warn_threshold
                    data["stuck_critical"] = secs > crit_threshold
                    data["stuck_warn_threshold"] = warn_threshold
                    data["stuck_crit_threshold"] = crit_threshold
                except Exception:
                    data["seconds_since_activity"] = 0
                    data["stuck_warning"] = False
                    data["stuck_critical"] = False

            # Estimate remaining time
            if data.get("is_running") and data.get("avg_experiment_duration", 0) > 0:
                elapsed = data.get("elapsed_seconds", 0)
                data["estimated_remaining"] = max(0, data["avg_experiment_duration"] - elapsed)

            return data
        except Exception:
            pass

    # Return default progress if file doesn't exist or can't be read
    return ExperimentProgress().to_dict()

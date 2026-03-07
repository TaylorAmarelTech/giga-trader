"""
GIGA TRADER - Automated Scheduler
===================================
Provides scheduled retraining, trading loop, and maintenance tasks.

Usage:
    from src.phase_22_automation.scheduler import (
        TaskScheduler,
        ScheduledTask,
        TaskPriority,
    )
"""

import logging
import threading
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("GigaTrader.Scheduler")


# =============================================================================
# ENUMS & DATA STRUCTURES
# =============================================================================

class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ScheduledTask:
    """A scheduled task with timing, priority, and execution metadata."""
    name: str
    callback: Callable
    interval_seconds: int = 3600
    priority: TaskPriority = TaskPriority.NORMAL
    enabled: bool = True

    # Time window restrictions (None = any time)
    earliest: Optional[dt_time] = None  # Earliest time of day to run
    latest: Optional[dt_time] = None  # Latest time of day to run
    weekdays_only: bool = False  # Only run Mon-Fri

    # Execution tracking
    last_run: Optional[datetime] = None
    last_status: TaskStatus = TaskStatus.PENDING
    last_error: Optional[str] = None
    run_count: int = 0
    fail_count: int = 0
    max_retries: int = 3
    retry_delay_seconds: int = 60

    # Dependencies
    depends_on: List[str] = field(default_factory=list)

    def is_due(self, now: Optional[datetime] = None) -> bool:
        """Check if task is due for execution."""
        now = now or datetime.now()
        if not self.enabled:
            return False
        if self.last_run is None:
            return self._in_time_window(now)
        elapsed = (now - self.last_run).total_seconds()
        return elapsed >= self.interval_seconds and self._in_time_window(now)

    def _in_time_window(self, now: datetime) -> bool:
        """Check if current time is within allowed window."""
        if self.weekdays_only and now.weekday() >= 5:
            return False
        current_time = now.time()
        if self.earliest and current_time < self.earliest:
            return False
        if self.latest and current_time > self.latest:
            return False
        return True


@dataclass
class TaskResult:
    """Result of a task execution."""
    task_name: str
    status: TaskStatus
    started_at: datetime
    finished_at: datetime
    duration_seconds: float
    error: Optional[str] = None
    result: Optional[dict] = None


# =============================================================================
# TASK SCHEDULER
# =============================================================================

class TaskScheduler:
    """
    Priority-based task scheduler for automated trading operations.

    Manages scheduled tasks like:
      - Model retraining (daily/weekly)
      - Signal generation (every N minutes during market hours)
      - Health checks (every 60 seconds)
      - Data downloads (daily after market close)
      - Portfolio reconciliation (daily)
    """

    def __init__(self, check_interval: int = 10):
        """
        Args:
            check_interval: Seconds between scheduler ticks.
        """
        self.check_interval = check_interval
        self.tasks: Dict[str, ScheduledTask] = {}
        self.history: List[TaskResult] = []
        self.max_history: int = 1000
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ── Task management ──────────────────────────────────────────────────

    def add_task(self, task: ScheduledTask) -> None:
        """Register a scheduled task."""
        with self._lock:
            self.tasks[task.name] = task
        logger.info(f"Registered task: {task.name} (interval={task.interval_seconds}s, priority={task.priority.name})")

    def remove_task(self, name: str) -> bool:
        """Remove a task by name. Returns True if found."""
        with self._lock:
            if name in self.tasks:
                del self.tasks[name]
                return True
            return False

    def enable_task(self, name: str) -> None:
        """Enable a task."""
        with self._lock:
            if name in self.tasks:
                self.tasks[name].enabled = True

    def disable_task(self, name: str) -> None:
        """Disable a task without removing it."""
        with self._lock:
            if name in self.tasks:
                self.tasks[name].enabled = False

    def get_task(self, name: str) -> Optional[ScheduledTask]:
        """Get task by name."""
        return self.tasks.get(name)

    # ── Execution ────────────────────────────────────────────────────────

    def _get_due_tasks(self, now: Optional[datetime] = None) -> List[ScheduledTask]:
        """Get tasks that are due, sorted by priority."""
        now = now or datetime.now()
        due = []
        with self._lock:
            for task in self.tasks.values():
                if not task.is_due(now):
                    continue
                # Check dependencies
                deps_met = all(
                    self.tasks.get(dep, ScheduledTask(name="", callback=lambda: None)).last_status == TaskStatus.COMPLETED
                    for dep in task.depends_on
                    if dep in self.tasks
                )
                if deps_met:
                    due.append(task)
        due.sort(key=lambda t: t.priority.value)
        return due

    def _execute_task(self, task: ScheduledTask) -> TaskResult:
        """Execute a single task with error handling and retries."""
        started = datetime.now()
        task.last_status = TaskStatus.RUNNING

        last_exception = None
        for attempt in range(task.max_retries):
            try:
                result_data = task.callback()
                finished = datetime.now()
                duration = (finished - started).total_seconds()

                task.last_run = finished
                task.last_status = TaskStatus.COMPLETED
                task.last_error = None
                task.run_count += 1

                result = TaskResult(
                    task_name=task.name,
                    status=TaskStatus.COMPLETED,
                    started_at=started,
                    finished_at=finished,
                    duration_seconds=duration,
                    result=result_data if isinstance(result_data, dict) else None,
                )
                logger.info(f"Task {task.name} completed in {duration:.1f}s")
                return result

            except Exception as exc:
                last_exception = exc
                logger.warning(f"Task {task.name} attempt {attempt+1}/{task.max_retries} failed: {exc}")
                if attempt < task.max_retries - 1:
                    _time.sleep(task.retry_delay_seconds)

        # All retries exhausted
        finished = datetime.now()
        duration = (finished - started).total_seconds()
        task.last_run = finished
        task.last_status = TaskStatus.FAILED
        task.last_error = str(last_exception)
        task.fail_count += 1

        result = TaskResult(
            task_name=task.name,
            status=TaskStatus.FAILED,
            started_at=started,
            finished_at=finished,
            duration_seconds=duration,
            error=str(last_exception),
        )
        logger.error(f"Task {task.name} failed after {task.max_retries} attempts: {last_exception}")
        return result

    def tick(self, now: Optional[datetime] = None) -> List[TaskResult]:
        """Run one scheduler tick: find and execute due tasks."""
        due_tasks = self._get_due_tasks(now)
        results = []
        for task in due_tasks:
            result = self._execute_task(task)
            results.append(result)
            with self._lock:
                self.history.append(result)
                if len(self.history) > self.max_history:
                    self.history = self.history[-self.max_history:]
        return results

    # ── Background loop ──────────────────────────────────────────────────

    def start(self) -> None:
        """Start the scheduler in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Scheduler started")

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.check_interval + 5)
        logger.info("Scheduler stopped")

    @property
    def is_running(self) -> bool:
        """Whether the scheduler loop is active."""
        return self._running

    def _run_loop(self) -> None:
        """Internal scheduler loop."""
        while self._running:
            try:
                self.tick()
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
            _time.sleep(self.check_interval)

    # ── Reporting ────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        """Get scheduler status summary."""
        with self._lock:
            tasks_info = {}
            for name, task in self.tasks.items():
                tasks_info[name] = {
                    "enabled": task.enabled,
                    "priority": task.priority.name,
                    "interval_seconds": task.interval_seconds,
                    "last_run": task.last_run.isoformat() if task.last_run else None,
                    "last_status": task.last_status.value,
                    "run_count": task.run_count,
                    "fail_count": task.fail_count,
                }
            recent = self.history[-10:] if self.history else []
            return {
                "running": self._running,
                "n_tasks": len(self.tasks),
                "tasks": tasks_info,
                "recent_history": [
                    {
                        "task": r.task_name,
                        "status": r.status.value,
                        "duration": r.duration_seconds,
                        "time": r.finished_at.isoformat(),
                    }
                    for r in recent
                ],
            }

    def get_history(self, task_name: Optional[str] = None, limit: int = 50) -> List[TaskResult]:
        """Get task execution history, optionally filtered by task name."""
        with self._lock:
            hist = self.history
            if task_name:
                hist = [r for r in hist if r.task_name == task_name]
            return hist[-limit:]

    def __repr__(self) -> str:
        n_enabled = sum(1 for t in self.tasks.values() if t.enabled)
        return f"TaskScheduler(tasks={len(self.tasks)}, enabled={n_enabled}, running={self._running})"


# =============================================================================
# PRESET TASK FACTORIES
# =============================================================================

def create_health_check_task(callback: Callable) -> ScheduledTask:
    """Create a health check task (every 60s, critical priority)."""
    return ScheduledTask(
        name="health_check",
        callback=callback,
        interval_seconds=60,
        priority=TaskPriority.CRITICAL,
    )


def create_signal_generation_task(callback: Callable) -> ScheduledTask:
    """Create a signal generation task (every 15min during market hours)."""
    return ScheduledTask(
        name="signal_generation",
        callback=callback,
        interval_seconds=900,
        priority=TaskPriority.HIGH,
        earliest=dt_time(9, 30),
        latest=dt_time(15, 45),
        weekdays_only=True,
    )


def create_retraining_task(callback: Callable) -> ScheduledTask:
    """Create a model retraining task (daily after market close)."""
    return ScheduledTask(
        name="model_retraining",
        callback=callback,
        interval_seconds=86400,
        priority=TaskPriority.NORMAL,
        earliest=dt_time(17, 0),
        latest=dt_time(23, 0),
        weekdays_only=True,
    )


def create_data_download_task(callback: Callable) -> ScheduledTask:
    """Create a data download task (daily after close)."""
    return ScheduledTask(
        name="data_download",
        callback=callback,
        interval_seconds=86400,
        priority=TaskPriority.HIGH,
        earliest=dt_time(16, 15),
        latest=dt_time(20, 0),
        weekdays_only=True,
    )


def create_reconciliation_task(callback: Callable) -> ScheduledTask:
    """Create an EOD reconciliation task."""
    return ScheduledTask(
        name="eod_reconciliation",
        callback=callback,
        interval_seconds=86400,
        priority=TaskPriority.HIGH,
        earliest=dt_time(16, 5),
        latest=dt_time(17, 0),
        weekdays_only=True,
        depends_on=["data_download"],
    )

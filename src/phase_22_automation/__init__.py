"""Phase: Automation & Orchestration.

The orchestrator classes live in src/giga_orchestrator.py (main entry point).
This phase re-exports key components for programmatic access.
"""

from src.giga_orchestrator import (
    GigaOrchestrator,
    SystemStatus,
    MarketHours,
    TrainingEngine,
    ExperimentRunner,
    ExperimentGateChecker,
    TradingEngine,
    HealthMonitor,
    StatusDisplay,
    ORCHESTRATOR_CONFIG,
)
from src.phase_22_automation.scheduler import (
    TaskScheduler,
    ScheduledTask,
    TaskPriority,
    TaskStatus,
    TaskResult,
    create_health_check_task,
    create_signal_generation_task,
    create_retraining_task,
    create_data_download_task,
    create_reconciliation_task,
)

__all__ = [
    "GigaOrchestrator",
    "SystemStatus",
    "MarketHours",
    "TrainingEngine",
    "ExperimentRunner",
    "ExperimentGateChecker",
    "TradingEngine",
    "HealthMonitor",
    "StatusDisplay",
    "ORCHESTRATOR_CONFIG",
    "TaskScheduler",
    "ScheduledTask",
    "TaskPriority",
    "TaskStatus",
    "TaskResult",
    "create_health_check_task",
    "create_signal_generation_task",
    "create_retraining_task",
    "create_data_download_task",
    "create_reconciliation_task",
]

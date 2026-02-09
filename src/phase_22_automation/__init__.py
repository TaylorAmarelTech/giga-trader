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
]

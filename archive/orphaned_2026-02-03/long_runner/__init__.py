"""
GIGA TRADER - Long-Running Experiment System
=============================================
A comprehensive wrapper system for running continuous experiments
for 30+ days with Claude Code CLI process management.

Components:
- ProcessManager: Manages Claude Code CLI subprocess lifecycle
- GridSearchController: Manages experiment parameter combinations
- StateManager: Persistence and recovery
- MonitoringDaemon: Health monitoring and alerting
- Orchestrator: Main coordination loop
"""

from .process_manager import ProcessManager, ClaudeCodeProcess
from .grid_search import GridSearchController, ParameterGrid
from .state_manager import StateManager
from .monitoring import MonitoringDaemon, AlertManager
from .orchestrator import LongRunningOrchestrator

__all__ = [
    "ProcessManager",
    "ClaudeCodeProcess",
    "GridSearchController",
    "ParameterGrid",
    "StateManager",
    "MonitoringDaemon",
    "AlertManager",
    "LongRunningOrchestrator",
]

__version__ = "1.0.0"

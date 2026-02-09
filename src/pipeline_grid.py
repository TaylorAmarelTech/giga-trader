"""
GIGA TRADER - Pipeline Grid Configuration (SHIM)
=========================================================
This module has been decomposed into:
  - src.phase_23_analytics.grid_config
  - src.phase_23_analytics.grid_search
  - src.phase_23_analytics.multi_objective

This file re-exports all public names for backward compatibility.
"""

from src.phase_23_analytics.grid_config import (
    GridDimensions,
    GridConfig,
)
from src.phase_23_analytics.grid_search import (
    PipelineGridSearch,
    QuickPresets,
)
from src.phase_23_analytics.multi_objective import (
    EntryExitPredictor,
    MultiObjectiveOptimizer,
    IntegratedGridSearch,
)

__all__ = [
    "GridDimensions",
    "GridConfig",
    "PipelineGridSearch",
    "QuickPresets",
    "EntryExitPredictor",
    "MultiObjectiveOptimizer",
    "IntegratedGridSearch",
]

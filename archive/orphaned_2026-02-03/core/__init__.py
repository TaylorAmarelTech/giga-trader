"""
Giga Trader - Core Module
Base classes and interfaces for the ML trading pipeline.
"""

from .base import PhaseRunner, PhaseResult, PhaseConfig
from .exceptions import (
    GigaTraderError,
    DataAcquisitionError,
    PreprocessingError,
    FeatureEngineeringError,
    ModelTrainingError,
    ValidationError,
    TradingError,
)
from .logging import setup_logging, get_logger

__all__ = [
    "PhaseRunner",
    "PhaseResult",
    "PhaseConfig",
    "GigaTraderError",
    "DataAcquisitionError",
    "PreprocessingError",
    "FeatureEngineeringError",
    "ModelTrainingError",
    "ValidationError",
    "TradingError",
    "setup_logging",
    "get_logger",
]

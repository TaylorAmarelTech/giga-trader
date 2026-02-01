"""
Custom exceptions for the Giga Trader pipeline.
"""


class GigaTraderError(Exception):
    """Base exception for all Giga Trader errors."""
    pass


# Phase 1-2: Data
class DataAcquisitionError(GigaTraderError):
    """Raised when data download or acquisition fails."""
    pass


class PreprocessingError(GigaTraderError):
    """Raised when data preprocessing fails."""
    pass


class DataValidationError(GigaTraderError):
    """Raised when data fails validation checks."""
    pass


# Phase 3-5: Synthetic & Targets
class SyntheticDataError(GigaTraderError):
    """Raised when synthetic data generation fails."""
    pass


class TargetCreationError(GigaTraderError):
    """Raised when target variable creation fails."""
    pass


# Phase 6-10: Features
class FeatureEngineeringError(GigaTraderError):
    """Raised when feature engineering fails."""
    pass


class FeatureSelectionError(GigaTraderError):
    """Raised when feature selection fails."""
    pass


# Phase 11-14: Model
class ModelTrainingError(GigaTraderError):
    """Raised when model training fails."""
    pass


class ValidationError(GigaTraderError):
    """Raised when model validation fails."""
    pass


class RobustnessError(GigaTraderError):
    """Raised when robustness testing fails."""
    pass


# Phase 15-17: Strategy
class StrategyError(GigaTraderError):
    """Raised when strategy development fails."""
    pass


class BacktestError(GigaTraderError):
    """Raised when backtesting fails."""
    pass


# Phase 18-27: Production
class TradingError(GigaTraderError):
    """Raised when trading execution fails."""
    pass


class RiskManagementError(GigaTraderError):
    """Raised when risk limits are violated."""
    pass


class MonitoringError(GigaTraderError):
    """Raised when monitoring systems fail."""
    pass

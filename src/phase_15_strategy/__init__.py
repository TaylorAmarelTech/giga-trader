"""Phase: Strategy & Signal Generation."""

from src.phase_15_strategy.signal_data import (
    SignalDirection,
    StrategySignal,
    EnsembleSignal,
    StrategyPerformance,
)
from src.phase_15_strategy.signal_detectors import (
    ADWINDriftDetector,
    ConfidenceCalibrator,
)
from src.phase_15_strategy.trading_strategies import (
    BaseStrategy,
    MomentumStrategy,
    ContrarianStrategy,
    RegimeFollowerStrategy,
    MeanReversionStrategy,
    LeadLagStrategy,
)
from src.phase_15_strategy.ensemble_signal_generator import (
    RobustEnsembleSignalGenerator,
    get_enhanced_signal_generator,
)

__all__ = [
    # Data structures
    "SignalDirection",
    "StrategySignal",
    "EnsembleSignal",
    "StrategyPerformance",
    # Detectors
    "ADWINDriftDetector",
    "ConfidenceCalibrator",
    # Strategies
    "BaseStrategy",
    "MomentumStrategy",
    "ContrarianStrategy",
    "RegimeFollowerStrategy",
    "MeanReversionStrategy",
    "LeadLagStrategy",
    # Ensemble
    "RobustEnsembleSignalGenerator",
    "get_enhanced_signal_generator",
]

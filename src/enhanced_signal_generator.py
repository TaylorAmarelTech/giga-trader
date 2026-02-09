"""
GIGA TRADER - Enhanced Signal Generator with Robust Anti-Overfitting
=====================================================================
SHIM: This module re-exports from src.phase_15_strategy submodules.

Multi-strategy ensemble with validation against synthetic SPY universes.

STRATEGIES:
- Momentum: Follow strong trends with sentiment confirmation
- Contrarian: Fade extreme sentiment (mean reversion)
- Regime Follower: Adapt to detected market regimes
- Mean Reversion: Trade when sentiment deviates from moving average
- Lead/Lag: Use sector leaders to predict SPY direction
"""

# Re-export everything from the decomposed phase_15_strategy modules
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
    "SignalDirection",
    "StrategySignal",
    "EnsembleSignal",
    "StrategyPerformance",
    "ADWINDriftDetector",
    "ConfidenceCalibrator",
    "BaseStrategy",
    "MomentumStrategy",
    "ContrarianStrategy",
    "RegimeFollowerStrategy",
    "MeanReversionStrategy",
    "LeadLagStrategy",
    "RobustEnsembleSignalGenerator",
    "get_enhanced_signal_generator",
]

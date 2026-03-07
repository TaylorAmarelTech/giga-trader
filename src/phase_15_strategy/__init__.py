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
from src.phase_15_strategy.meta_labeler import (
    MetaLabeler,
    half_kelly_fraction,
)
from src.phase_15_strategy.dynamic_weights import (
    DynamicEnsembleWeighter,
)
from src.phase_15_strategy.regime_router import (
    RegimeRouter,
)
from src.phase_15_strategy.conformal_sizer import (
    ConformalPositionSizer,
)
from src.phase_15_strategy.adaptive_conformal import (
    AdaptiveConformalSizer,
)
from src.phase_15_strategy.isotonic_calibrator import (
    IsotonicCalibrator,
)
from src.phase_15_strategy.bayesian_averaging import (
    BayesianModelAverager,
)
from src.phase_15_strategy.cvar_position_sizer import (
    CVaRPositionSizer,
)
from src.phase_15_strategy.thompson_selector import (
    ThompsonSamplingSelector,
)
from src.phase_15_strategy.dynamic_kelly_sizer import (
    DynamicKellySizer,
)
from src.phase_15_strategy.drawdown_adaptive_sizer import (
    DrawdownAdaptiveSizer,
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
    # Meta-labeling
    "MetaLabeler",
    "half_kelly_fraction",
    # Dynamic weighting
    "DynamicEnsembleWeighter",
    # Regime routing
    "RegimeRouter",
    # Conformal position sizing
    "ConformalPositionSizer",
    # Adaptive conformal inference
    "AdaptiveConformalSizer",
    # Isotonic calibration
    "IsotonicCalibrator",
    # Bayesian model averaging
    "BayesianModelAverager",
    # CVaR position sizing
    "CVaRPositionSizer",
    # Thompson Sampling model selection
    "ThompsonSamplingSelector",
    # Dynamic Kelly sizing
    "DynamicKellySizer",
    # Drawdown-adaptive sizing
    "DrawdownAdaptiveSizer",
]

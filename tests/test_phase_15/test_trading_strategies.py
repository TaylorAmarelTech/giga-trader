"""
Test trading strategy classes and signal generation.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase_15_strategy.signal_data import (
    SignalDirection,
    StrategySignal,
    StrategyPerformance,
)
from src.phase_15_strategy.trading_strategies import (
    MomentumStrategy,
    ContrarianStrategy,
    RegimeFollowerStrategy,
    MeanReversionStrategy,
    LeadLagStrategy,
)


# ---------------------------------------------------------------------------
# SignalDirection enum tests
# ---------------------------------------------------------------------------

def test_signal_direction_values():
    """SignalDirection should have the 5 expected values."""
    assert SignalDirection.STRONG_BUY.value == "STRONG_BUY"
    assert SignalDirection.BUY.value == "BUY"
    assert SignalDirection.NEUTRAL.value == "NEUTRAL"
    assert SignalDirection.SELL.value == "SELL"
    assert SignalDirection.STRONG_SELL.value == "STRONG_SELL"
    assert len(list(SignalDirection)) == 5


# ---------------------------------------------------------------------------
# StrategySignal tests
# ---------------------------------------------------------------------------

def test_strategy_signal_creation():
    """StrategySignal dataclass should initialize correctly."""
    signal = StrategySignal(
        direction=SignalDirection.BUY,
        confidence=0.75,
        raw_score=0.45,
        strategy_name="momentum",
    )
    assert signal.direction == SignalDirection.BUY
    assert signal.confidence == 0.75
    assert signal.raw_score == 0.45
    assert signal.strategy_name == "momentum"
    assert signal.timestamp is not None


# ---------------------------------------------------------------------------
# Helper to generate signals with enough history
# ---------------------------------------------------------------------------

def _warm_up_strategy(strategy, n_signals=10):
    """Feed a strategy enough data to get past the lookback period."""
    signals = []
    for i in range(n_signals):
        sentiment_data = {
            "net_sentiment": np.random.uniform(-0.5, 0.5),
            "expectation": np.random.uniform(-0.3, 0.3),
        }
        market_data = {
            "momentum_5d": np.random.uniform(-0.02, 0.02),
            "rsi_14": np.random.uniform(30, 70),
            "sma_ratio": np.random.uniform(0.98, 1.02),
            "spy_return_5d": np.random.uniform(-0.02, 0.02),
            "vol_ratio": np.random.uniform(0.8, 1.2),
        }
        signal = strategy.generate_signal(
            sentiment_data=sentiment_data,
            market_data=market_data,
            regime="NORMAL",
        )
        signals.append(signal)
    return signals


# ---------------------------------------------------------------------------
# MomentumStrategy tests
# ---------------------------------------------------------------------------

def test_momentum_strategy_creation():
    """MomentumStrategy should initialize correctly."""
    strategy = MomentumStrategy(lookback=5, threshold=0.3)
    assert strategy.name == "momentum"
    assert strategy.lookback == 5
    assert strategy.threshold == 0.3


def test_momentum_strategy_generate_signal():
    """MomentumStrategy should generate StrategySignal objects."""
    strategy = MomentumStrategy()
    signals = _warm_up_strategy(strategy, 15)

    for signal in signals:
        assert isinstance(signal, StrategySignal)
        assert isinstance(signal.direction, SignalDirection)
        assert 0 <= signal.confidence <= 1.0
        assert signal.strategy_name == "momentum"


def test_momentum_returns_neutral_initially():
    """With insufficient history, MomentumStrategy should return NEUTRAL."""
    strategy = MomentumStrategy(lookback=5)
    signal = strategy.generate_signal(
        sentiment_data={"net_sentiment": 0.5},
        market_data={"momentum_5d": 0.01},
        regime="NORMAL",
    )
    assert signal.direction == SignalDirection.NEUTRAL


# ---------------------------------------------------------------------------
# ContrarianStrategy tests
# ---------------------------------------------------------------------------

def test_contrarian_strategy_creation():
    """ContrarianStrategy should initialize correctly."""
    strategy = ContrarianStrategy()
    assert strategy.name == "contrarian"


def test_contrarian_strategy_generate_signal():
    """ContrarianStrategy should generate valid signals."""
    strategy = ContrarianStrategy()
    signals = _warm_up_strategy(strategy, 15)

    for signal in signals:
        assert isinstance(signal, StrategySignal)
        assert signal.strategy_name == "contrarian"


# ---------------------------------------------------------------------------
# RegimeFollowerStrategy tests
# ---------------------------------------------------------------------------

def test_regime_follower_creation():
    """RegimeFollowerStrategy should initialize correctly."""
    strategy = RegimeFollowerStrategy()
    assert strategy.name == "regime_follower"


def test_regime_follower_generate_signal():
    """RegimeFollowerStrategy should generate valid signals."""
    strategy = RegimeFollowerStrategy()
    signals = _warm_up_strategy(strategy, 10)

    for signal in signals:
        assert isinstance(signal, StrategySignal)
        assert signal.strategy_name == "regime_follower"


# ---------------------------------------------------------------------------
# MeanReversionStrategy tests
# ---------------------------------------------------------------------------

def test_mean_reversion_creation():
    """MeanReversionStrategy should initialize correctly."""
    strategy = MeanReversionStrategy()
    assert strategy.name == "mean_reversion"


def test_mean_reversion_generate_signal():
    """MeanReversionStrategy should generate valid signals."""
    strategy = MeanReversionStrategy()
    signals = _warm_up_strategy(strategy, 15)

    for signal in signals:
        assert isinstance(signal, StrategySignal)
        assert signal.strategy_name == "mean_reversion"


# ---------------------------------------------------------------------------
# LeadLagStrategy tests
# ---------------------------------------------------------------------------

def test_lead_lag_creation():
    """LeadLagStrategy should initialize correctly."""
    strategy = LeadLagStrategy()
    assert strategy.name == "lead_lag"


def test_lead_lag_generate_signal():
    """LeadLagStrategy should generate valid signals."""
    strategy = LeadLagStrategy()
    signals = _warm_up_strategy(strategy, 10)

    for signal in signals:
        assert isinstance(signal, StrategySignal)
        assert signal.strategy_name == "lead_lag"


# ---------------------------------------------------------------------------
# StrategyPerformance tests
# ---------------------------------------------------------------------------

def test_strategy_performance_creation():
    """StrategyPerformance should initialize with defaults."""
    perf = StrategyPerformance(strategy_name="test_strategy")
    assert perf.strategy_name == "test_strategy"
    assert perf.total_signals == 0
    assert perf.real_spy_accuracy == 0.5
    assert perf.robustness_score == 0.5


def test_strategy_performance_update_robustness():
    """update_robustness should compute a valid robustness score."""
    perf = StrategyPerformance(strategy_name="test")
    perf.real_spy_accuracy = 0.65
    perf.synthetic_accuracy = 0.60
    perf.synthetic_variance = 0.05
    perf.update_robustness()

    assert 0 <= perf.robustness_score <= 1.0
    assert perf.overfitting_penalty >= 0


# ---------------------------------------------------------------------------
# Cross-strategy consistency
# ---------------------------------------------------------------------------

ALL_STRATEGIES = [
    MomentumStrategy,
    ContrarianStrategy,
    RegimeFollowerStrategy,
    MeanReversionStrategy,
    LeadLagStrategy,
]


@pytest.mark.parametrize("strategy_cls", ALL_STRATEGIES, ids=lambda s: s.__name__)
def test_all_strategies_have_generate_signal(strategy_cls):
    """All strategies should have a generate_signal method."""
    strategy = strategy_cls()
    assert hasattr(strategy, "generate_signal")
    assert callable(strategy.generate_signal)


@pytest.mark.parametrize("strategy_cls", ALL_STRATEGIES, ids=lambda s: s.__name__)
def test_all_strategies_have_name(strategy_cls):
    """All strategies should have a non-empty name."""
    strategy = strategy_cls()
    assert isinstance(strategy.name, str)
    assert len(strategy.name) > 0


@pytest.mark.parametrize("strategy_cls", ALL_STRATEGIES, ids=lambda s: s.__name__)
def test_all_strategies_have_performance(strategy_cls):
    """All strategies should have a StrategyPerformance attribute."""
    strategy = strategy_cls()
    assert isinstance(strategy.performance, StrategyPerformance)

"""Phase: Backtesting."""

from src.phase_16_backtesting.portfolio import (
    Trade,
    Portfolio,
)
from src.phase_16_backtesting.backtest_core import (
    BacktestEngine,
    run_full_backtest,
)
from src.phase_16_backtesting.backtest_variants import (
    WalkForwardBacktest,
    MonteCarloSimulator,
)
from src.phase_16_backtesting.backtesting_harness import (
    BacktestingHarness,
    BacktestResult,
    MarketRegime,
    HISTORICAL_REGIMES,
)

__all__ = [
    # Portfolio
    "Trade",
    "Portfolio",
    # Backtest engine
    "BacktestEngine",
    "run_full_backtest",
    # Variants
    "WalkForwardBacktest",
    "MonteCarloSimulator",
    # Harness
    "BacktestingHarness",
    "BacktestResult",
    "MarketRegime",
    "HISTORICAL_REGIMES",
]

"""
GIGA TRADER - Backtest Engine
==============================
SHIM: This module re-exports from src.phase_16_backtesting submodules.

Tests trading configurations on historical data with realistic simulation.

Features:
  - Walk-forward backtesting
  - Realistic slippage and commission modeling
  - Position sizing with batch entries
  - Stop loss and take profit execution
  - Performance metrics (Sharpe, Sortino, max drawdown, etc.)
  - Regime analysis (bull/bear/sideways markets)
  - Monte Carlo stress testing

Usage:
    from src.backtest_engine import BacktestEngine

    engine = BacktestEngine(initial_capital=100000)
    results = engine.run_backtest(
        daily_data=df_daily,
        intraday_data=df_1min,
        swing_model=swing_model,
        timing_model=timing_model,
        entry_exit_model=entry_exit_model,
        config=grid_config,
    )
"""

# Re-export everything from the decomposed phase_16_backtesting modules
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

__all__ = [
    "Trade",
    "Portfolio",
    "BacktestEngine",
    "run_full_backtest",
    "WalkForwardBacktest",
    "MonteCarloSimulator",
]

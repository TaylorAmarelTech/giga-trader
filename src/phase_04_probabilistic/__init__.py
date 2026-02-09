"""Phase: Probabilistic / Monte Carlo.

Re-exports Monte Carlo simulation from phase_16 (backtesting)
and synthetic data generation from phase_03.
"""

from src.phase_16_backtesting.backtest_variants import MonteCarloSimulator
from src.phase_03_synthetic_data.synthetic_universe import SyntheticSPYGenerator

__all__ = [
    "MonteCarloSimulator",
    "SyntheticSPYGenerator",
]

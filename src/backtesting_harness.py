"""
GIGA TRADER - Comprehensive Backtesting Harness (SHIM)
=================================================
This module has been moved to:
  - src.phase_16_backtesting.backtesting_harness

This file re-exports all public names for backward compatibility.
"""

from src.phase_16_backtesting.backtesting_harness import (
    BacktestingHarness,
    BacktestResult,
    MarketRegime,
    HISTORICAL_REGIMES,
    main,
)

__all__ = [
    "BacktestingHarness",
    "BacktestResult",
    "MarketRegime",
    "HISTORICAL_REGIMES",
    "main",
]

if __name__ == "__main__":
    import sys
    sys.exit(main())

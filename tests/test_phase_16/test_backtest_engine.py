"""
Test BacktestEngine creation and basic operation.
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase_16_backtesting.backtest_core import BacktestEngine
from src.phase_16_backtesting.portfolio import Portfolio, Trade


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """Create a BacktestEngine with default settings."""
    return BacktestEngine(
        initial_capital=100000,
        commission_per_share=0.005,
        slippage_pct=0.0001,
        max_position_pct=0.25,
    )


@pytest.fixture
def mock_daily_data():
    """Create mock daily data for backtesting."""
    np.random.seed(42)
    n = 100
    dates = pd.bdate_range(start="2025-01-02", periods=n, freq="B")
    close = 450.0 + np.cumsum(np.random.normal(0.05, 1.0, n))
    df = pd.DataFrame({
        "open": close + np.random.normal(0, 0.3, n),
        "high": close + abs(np.random.normal(0, 0.8, n)),
        "low": close - abs(np.random.normal(0, 0.8, n)),
        "close": close,
        "volume": np.random.randint(50_000_000, 150_000_000, n).astype(float),
    }, index=dates)
    return df


@pytest.fixture
def mock_predictions(mock_daily_data):
    """Create mock swing and timing predictions aligned with the data."""
    np.random.seed(99)
    n = len(mock_daily_data)
    swing = pd.Series(np.random.uniform(0.3, 0.8, n), index=mock_daily_data.index)
    timing = pd.Series(np.random.uniform(0.3, 0.8, n), index=mock_daily_data.index)
    return swing, timing


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_engine_creation(engine):
    """BacktestEngine should initialize with given parameters."""
    assert isinstance(engine, BacktestEngine)
    assert engine.initial_capital == 100000
    assert engine.commission_per_share == 0.005
    assert engine.slippage_pct == 0.0001
    assert engine.max_position_pct == 0.25


def test_engine_creation_defaults():
    """BacktestEngine should have sensible defaults."""
    engine = BacktestEngine()
    assert engine.initial_capital == 100000
    assert engine.portfolio is None
    assert engine.results == {}


def test_engine_has_run_backtest_method(engine):
    """BacktestEngine should have a run_backtest method."""
    assert hasattr(engine, "run_backtest")
    assert callable(engine.run_backtest)


def test_engine_run_backtest_basic(engine, mock_daily_data, mock_predictions):
    """run_backtest should execute without error and return a dict."""
    swing_pred, timing_pred = mock_predictions
    result = engine.run_backtest(
        daily_data=mock_daily_data,
        swing_predictions=swing_pred,
        timing_predictions=timing_pred,
    )
    assert isinstance(result, dict)


def test_engine_portfolio_initialized_after_run(engine, mock_daily_data, mock_predictions):
    """After running backtest, engine.portfolio should be a Portfolio."""
    swing_pred, timing_pred = mock_predictions
    engine.run_backtest(
        daily_data=mock_daily_data,
        swing_predictions=swing_pred,
        timing_predictions=timing_pred,
    )
    assert isinstance(engine.portfolio, Portfolio)

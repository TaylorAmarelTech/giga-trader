"""
Test Trade and Portfolio classes from phase_16_backtesting.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase_16_backtesting.portfolio import Trade, Portfolio


# ---------------------------------------------------------------------------
# Trade tests
# ---------------------------------------------------------------------------

def test_trade_creation():
    """Trade should initialize with required fields."""
    trade = Trade(
        trade_id=1,
        entry_date=datetime(2025, 1, 6, 10, 30),
        direction="LONG",
        entry_price=450.0,
        position_size=100,
        entry_cost=45000.0,
    )
    assert trade.trade_id == 1
    assert trade.direction == "LONG"
    assert trade.entry_price == 450.0
    assert trade.position_size == 100
    assert trade.entry_cost == 45000.0
    assert trade.is_open is True
    assert trade.exit_date is None


def test_trade_close_long():
    """Closing a LONG trade should calculate correct P&L."""
    trade = Trade(
        trade_id=1,
        entry_date=datetime(2025, 1, 6, 10, 0),
        direction="LONG",
        entry_price=450.0,
        position_size=100,
        entry_cost=45000.0,
    )
    trade.close(
        exit_date=datetime(2025, 1, 6, 15, 0),
        exit_price=452.0,
        reason="take_profit",
        commission=0.50,
        slippage=0.10,
    )
    assert trade.is_open is False
    assert trade.exit_price == 452.0
    assert trade.exit_reason == "take_profit"
    # P&L: (452 - 450) * 100 - 0.50 - 0.10 = 199.40
    expected_pnl = (452.0 - 450.0) * 100 - 0.50 - 0.10
    assert abs(trade.realized_pnl - expected_pnl) < 0.01


def test_trade_close_short():
    """Closing a SHORT trade should calculate correct P&L."""
    trade = Trade(
        trade_id=2,
        entry_date=datetime(2025, 1, 6, 10, 0),
        direction="SHORT",
        entry_price=450.0,
        position_size=100,
        entry_cost=45000.0,
    )
    trade.close(
        exit_date=datetime(2025, 1, 6, 15, 0),
        exit_price=448.0,
        reason="signal",
        commission=0.0,
        slippage=0.0,
    )
    # SHORT P&L: (450 - 448) * 100 = 200
    assert trade.realized_pnl == 200.0
    assert trade.is_open is False


def test_trade_return_pct():
    """return_pct should be P&L / entry_cost."""
    trade = Trade(
        trade_id=1,
        entry_date=datetime(2025, 1, 6, 10, 0),
        direction="LONG",
        entry_price=450.0,
        position_size=100,
        entry_cost=45000.0,
    )
    trade.close(
        exit_date=datetime(2025, 1, 6, 15, 0),
        exit_price=454.5,
        reason="signal",
    )
    # P&L = (454.5 - 450) * 100 = 450
    # return_pct = 450 / 45000 = 0.01
    assert abs(trade.return_pct - 0.01) < 0.001


def test_trade_holding_period():
    """holding_period should return timedelta."""
    entry = datetime(2025, 1, 6, 10, 0)
    exit_dt = datetime(2025, 1, 6, 15, 30)
    trade = Trade(
        trade_id=1,
        entry_date=entry,
        direction="LONG",
        entry_price=450.0,
        position_size=100,
        entry_cost=45000.0,
    )
    assert trade.holding_period is None  # Open trade

    trade.close(exit_date=exit_dt, exit_price=451.0, reason="eod")
    assert trade.holding_period == exit_dt - entry


def test_trade_to_dict():
    """to_dict should return a dict with expected keys."""
    trade = Trade(
        trade_id=1,
        entry_date=datetime(2025, 1, 6, 10, 0),
        direction="LONG",
        entry_price=450.0,
        position_size=100,
        entry_cost=45000.0,
    )
    d = trade.to_dict()
    assert isinstance(d, dict)
    assert d["trade_id"] == 1
    assert d["direction"] == "LONG"
    assert d["entry_price"] == 450.0


# ---------------------------------------------------------------------------
# Portfolio tests
# ---------------------------------------------------------------------------

def test_portfolio_initialization():
    """Portfolio should initialize with defaults."""
    portfolio = Portfolio()
    assert portfolio.initial_capital == 100000
    assert portfolio.cash == 100000
    assert portfolio.position_count == 0
    assert len(portfolio.open_trades) == 0
    assert len(portfolio.closed_trades) == 0


def test_portfolio_custom_initialization():
    """Portfolio should accept custom parameters."""
    portfolio = Portfolio(
        initial_capital=50000,
        commission_per_share=0.01,
        slippage_pct=0.0002,
    )
    assert portfolio.initial_capital == 50000
    assert portfolio.cash == 50000
    assert portfolio.commission_per_share == 0.01
    assert portfolio.slippage_pct == 0.0002


def test_portfolio_open_trade():
    """open_trade should create a Trade and reduce cash."""
    portfolio = Portfolio(initial_capital=100000)
    trade = portfolio.open_trade(
        date=datetime(2025, 1, 6, 10, 0),
        direction="LONG",
        price=450.0,
        position_value=10000.0,
    )
    assert trade is not None
    assert portfolio.position_count == 1
    assert portfolio.cash < 100000  # Cash should be reduced


def test_portfolio_close_trade():
    """Closing a trade should move it to closed_trades and update cash."""
    portfolio = Portfolio(initial_capital=100000, slippage_pct=0.0)
    trade = portfolio.open_trade(
        date=datetime(2025, 1, 6, 10, 0),
        direction="LONG",
        price=450.0,
        position_value=10000.0,
    )
    cash_after_open = portfolio.cash

    portfolio.close_trade(
        trade=trade,
        date=datetime(2025, 1, 6, 15, 0),
        price=455.0,
        reason="take_profit",
    )
    assert portfolio.position_count == 0
    assert len(portfolio.closed_trades) == 1
    assert portfolio.cash > cash_after_open  # Profitable trade


def test_portfolio_cannot_open_when_insufficient_cash():
    """open_trade should return None if not enough cash."""
    portfolio = Portfolio(initial_capital=1000)
    trade = portfolio.open_trade(
        date=datetime(2025, 1, 6, 10, 0),
        direction="LONG",
        price=450.0,
        position_value=50000.0,  # Way more than available cash
    )
    assert trade is None
    assert portfolio.position_count == 0


def test_portfolio_equity():
    """equity should include cash + open positions."""
    portfolio = Portfolio(initial_capital=100000, slippage_pct=0.0)
    assert portfolio.equity == 100000

    portfolio.open_trade(
        date=datetime(2025, 1, 6, 10, 0),
        direction="LONG",
        price=450.0,
        position_value=10000.0,
    )
    # Equity should still be approximately 100000 (minus commissions)
    assert portfolio.equity > 90000


def test_portfolio_record_daily():
    """record_daily should add to equity curve and compute daily returns."""
    portfolio = Portfolio(initial_capital=100000)
    portfolio.record_daily(datetime(2025, 1, 6), 450.0)
    assert len(portfolio.equity_curve) == 1
    assert len(portfolio.daily_returns) == 1
    assert portfolio.daily_returns[0] == 0  # First day has 0 return


def test_portfolio_drawdown_tracking():
    """Portfolio should track max drawdown."""
    portfolio = Portfolio(initial_capital=100000, slippage_pct=0.0)
    portfolio.record_daily(datetime(2025, 1, 6), 450.0)

    # Simulate a loss
    portfolio.cash = 95000  # Direct mutation for testing
    portfolio.record_daily(datetime(2025, 1, 7), 440.0)

    assert portfolio.max_drawdown > 0


def test_portfolio_multiple_trades():
    """Portfolio should handle multiple sequential trades."""
    portfolio = Portfolio(initial_capital=100000, slippage_pct=0.0)

    for i in range(3):
        trade = portfolio.open_trade(
            date=datetime(2025, 1, 6 + i, 10, 0),
            direction="LONG",
            price=450.0 + i,
            position_value=5000.0,
        )
        if trade:
            portfolio.close_trade(
                trade=trade,
                date=datetime(2025, 1, 6 + i, 15, 0),
                price=452.0 + i,
                reason="eod",
            )

    assert len(portfolio.closed_trades) == 3
    assert portfolio.position_count == 0

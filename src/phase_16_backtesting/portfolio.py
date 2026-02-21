"""
GIGA TRADER - Portfolio and Trade Classes
==========================================
Core portfolio state management and trade record keeping for backtesting.

Contains:
- Trade class (single trade record with all details)
- Portfolio class (portfolio state tracking during backtest)
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# =============================================================================
# 1. TRADE RECORD
# =============================================================================

class Trade:
    """Represents a single trade with all details."""

    def __init__(
        self,
        trade_id: int,
        entry_date: datetime,
        direction: str,  # "LONG" or "SHORT"
        entry_price: float,
        position_size: float,  # Number of shares/contracts
        entry_cost: float,  # Total $ invested
    ):
        self.trade_id = trade_id
        self.entry_date = entry_date
        self.direction = direction
        self.entry_price = entry_price
        self.position_size = position_size
        self.entry_cost = entry_cost

        # Exit info (filled on close)
        self.exit_date: Optional[datetime] = None
        self.exit_price: Optional[float] = None
        self.exit_reason: Optional[str] = None  # "signal", "stop_loss", "take_profit", "eod"

        # Risk parameters
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.trailing_stop: Optional[float] = None

        # Batch info
        self.batches: List[Dict] = []  # [{time, price, size}, ...]
        self.avg_entry_price: float = entry_price

        # P&L
        self.realized_pnl: float = 0.0
        self.commission: float = 0.0
        self.slippage: float = 0.0

    def close(
        self,
        exit_date: datetime,
        exit_price: float,
        reason: str,
        commission: float = 0.0,
        slippage: float = 0.0,
    ):
        """Close the trade."""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = reason
        self.commission = commission
        self.slippage = slippage

        # Calculate P&L
        if self.direction == "LONG":
            gross_pnl = (self.exit_price - self.avg_entry_price) * self.position_size
        else:  # SHORT
            gross_pnl = (self.avg_entry_price - self.exit_price) * self.position_size

        self.realized_pnl = gross_pnl - self.commission - self.slippage

    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    @property
    def return_pct(self) -> float:
        if self.entry_cost == 0:
            return 0.0
        return self.realized_pnl / self.entry_cost

    @property
    def holding_period(self) -> Optional[timedelta]:
        if self.exit_date is None:
            return None
        return self.exit_date - self.entry_date

    def to_dict(self) -> Dict:
        return {
            "trade_id": self.trade_id,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "position_size": self.position_size,
            "entry_cost": self.entry_cost,
            "realized_pnl": self.realized_pnl,
            "return_pct": self.return_pct,
            "exit_reason": self.exit_reason,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
        }


# =============================================================================
# 2. PORTFOLIO STATE
# =============================================================================

class Portfolio:
    """Tracks portfolio state during backtest."""

    def __init__(
        self,
        initial_capital: float = 100000,
        commission_per_share: float = 0.005,
        slippage_pct: float = 0.0001,
        dynamic_slippage: bool = True,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_per_share = commission_per_share
        self.slippage_pct = slippage_pct
        self.dynamic_slippage = dynamic_slippage

        # Positions
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.trade_count = 0

        # Daily tracking
        self.equity_curve: List[Dict] = []
        self.daily_returns: List[float] = []

        # Statistics
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0
        self.total_commission = 0.0
        self.total_slippage = 0.0

        # Last known close for mark-to-market
        self._last_close: float = 0.0

    @property
    def equity(self) -> float:
        """Current portfolio value (mark-to-market)."""
        if self._last_close > 0 and self.open_trades:
            open_value = 0.0
            for t in self.open_trades:
                if t.direction == "LONG":
                    open_value += t.position_size * self._last_close
                else:
                    open_value += t.entry_cost + (t.entry_price - self._last_close) * t.position_size
            return self.cash + open_value
        # Fallback when no market price available yet
        open_value = sum(t.entry_cost for t in self.open_trades)
        return self.cash + open_value

    @property
    def open_position_value(self) -> float:
        return sum(t.entry_cost for t in self.open_trades)

    @property
    def position_count(self) -> int:
        return len(self.open_trades)

    def calculate_slippage(self, order_value: float, volatility: float = 0.0) -> float:
        """Calculate slippage percentage based on order size and volatility.

        Args:
            order_value: Dollar value of the order
            volatility: Recent annualized volatility (0-1 scale)

        Returns:
            Slippage as a fraction (e.g. 0.0002 = 2 basis points)
        """
        if not self.dynamic_slippage:
            return self.slippage_pct

        base = self.slippage_pct  # 1 bps default

        # Size impact: orders > $50K get up to 2x base slippage
        size_factor = 1.0 + min(1.0, order_value / 100_000)

        # Volatility impact: high vol (>20% annualized) increases slippage
        vol_factor = 1.0 + max(0.0, (volatility - 0.15)) * 5.0

        return base * size_factor * vol_factor

    def can_open_position(self, cost: float) -> bool:
        """Check if we have enough cash."""
        return self.cash >= cost

    def open_trade(
        self,
        date: datetime,
        direction: str,
        price: float,
        position_value: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        volatility: float = 0.0,
    ) -> Optional[Trade]:
        """Open a new trade."""
        # Apply slippage (dynamic or fixed)
        slip = self.calculate_slippage(position_value, volatility)
        if direction == "LONG":
            entry_price = price * (1 + slip)
        else:
            entry_price = price * (1 - slip)

        # Calculate position size
        position_size = position_value / entry_price
        commission = position_size * self.commission_per_share

        # Check if we have enough cash
        total_cost = position_value + commission
        if not self.can_open_position(total_cost):
            return None

        # Create trade
        self.trade_count += 1
        trade = Trade(
            trade_id=self.trade_count,
            entry_date=date,
            direction=direction,
            entry_price=entry_price,
            position_size=position_size,
            entry_cost=position_value,
        )
        trade.stop_loss = stop_loss
        trade.take_profit = take_profit
        trade.commission = commission

        # Update portfolio
        self.cash -= total_cost
        self.total_commission += commission
        self.total_slippage += abs(entry_price - price) * position_size
        self.open_trades.append(trade)

        return trade

    def close_trade(
        self,
        trade: Trade,
        date: datetime,
        price: float,
        reason: str,
        volatility: float = 0.0,
    ):
        """Close an existing trade."""
        if trade not in self.open_trades:
            return

        # Apply slippage (dynamic or fixed)
        slip = self.calculate_slippage(trade.entry_cost, volatility)
        if trade.direction == "LONG":
            exit_price = price * (1 - slip)
        else:
            exit_price = price * (1 + slip)

        # Commission
        commission = trade.position_size * self.commission_per_share

        # Close trade
        trade.close(
            exit_date=date,
            exit_price=exit_price,
            reason=reason,
            commission=commission,
            slippage=abs(exit_price - price) * trade.position_size,
        )

        # Update portfolio
        if trade.direction == "LONG":
            proceeds = exit_price * trade.position_size
        else:
            # For short: initial_cost + (entry - exit) * size
            proceeds = trade.entry_cost + (trade.entry_price - exit_price) * trade.position_size

        self.cash += proceeds - commission
        self.total_commission += commission
        self.total_slippage += abs(exit_price - price) * trade.position_size

        # Move to closed
        self.open_trades.remove(trade)
        self.closed_trades.append(trade)

    def record_daily(self, date: datetime, close_price: float):
        """Record daily equity and update drawdown."""
        self._last_close = close_price
        # Calculate current equity with mark-to-market
        open_value = 0
        for trade in self.open_trades:
            if trade.direction == "LONG":
                open_value += trade.position_size * close_price
            else:
                # Short position value
                open_value += trade.entry_cost + (trade.entry_price - close_price) * trade.position_size

        current_equity = self.cash + open_value

        # Record
        self.equity_curve.append({
            "date": date,
            "equity": current_equity,
            "cash": self.cash,
            "positions_value": open_value,
            "n_positions": len(self.open_trades),
        })

        # Daily return
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]["equity"]
            daily_return = (current_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            self.daily_returns.append(daily_return)
        else:
            self.daily_returns.append(0)

        # Drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, drawdown)

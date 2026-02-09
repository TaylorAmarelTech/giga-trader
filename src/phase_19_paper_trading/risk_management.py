"""
GIGA TRADER - Alpaca Paper Trading: Risk Management and Order Management
=========================================================================
Risk management enforcement and order execution.

Components:
  - RiskManager class
  - OrderManager class
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Local imports
from src.phase_19_paper_trading.alpaca_client import (
    TRADING_CONFIG,
    dynamic_thresholds,
    SignalType,
    TradingSignal,
    Position,
    TradeRecord,
    AlpacaPaperClient,
)

# Alpaca imports
try:
    from alpaca.trading.enums import OrderSide
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# Position history tracker
try:
    from src.position_tracker import get_tracker, PositionHistoryTracker
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False

logger = logging.getLogger("GigaTrader")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. RISK MANAGER
# ═══════════════════════════════════════════════════════════════════════════════
class RiskManager:
    """
    Risk management for trading.

    Enforces:
      - Position size limits
      - Daily loss limits
      - Drawdown limits
      - Trade count limits
      - Time-based restrictions
    """

    def __init__(self, config: Dict = None):
        self.config = config or TRADING_CONFIG
        self.daily_pnl = 0.0
        self.peak_equity = 0.0
        self.daily_trades = 0
        self.is_halted = False
        self.halt_reason = ""
        self.trade_log: List[TradeRecord] = []

    def reset_daily(self):
        """Reset daily counters."""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.is_halted = False
        self.halt_reason = ""
        logger.info("Daily risk counters reset")

    def update_pnl(self, pnl: float, equity: float):
        """Update P&L and check limits."""
        self.daily_pnl += pnl

        if equity > self.peak_equity:
            self.peak_equity = equity

        # Check daily loss limit
        daily_loss_pct = -self.daily_pnl / self.peak_equity if self.peak_equity > 0 else 0
        if daily_loss_pct >= self.config["max_daily_loss_pct"]:
            self.is_halted = True
            self.halt_reason = f"Daily loss limit reached: {daily_loss_pct:.2%}"
            logger.warning(self.halt_reason)

        # Check drawdown limit
        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        if drawdown >= self.config["max_drawdown_pct"]:
            self.is_halted = True
            self.halt_reason = f"Max drawdown reached: {drawdown:.2%}"
            logger.warning(self.halt_reason)

    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed."""
        if self.is_halted:
            return False, self.halt_reason

        # Check trade count
        if self.daily_trades >= self.config["max_daily_trades"]:
            return False, f"Daily trade limit reached: {self.daily_trades}"

        # Check time
        now = datetime.now().time()
        if now < self.config["market_open"]:
            return False, "Market not open yet"
        if now > self.config["no_new_trades_after"]:
            return False, "No new trades allowed after 3:30 PM"

        return True, "OK"

    def should_force_close(self) -> bool:
        """Check if positions should be force closed."""
        now = datetime.now().time()
        return now >= self.config["force_close_time"]

    def validate_order(
        self,
        signal: TradingSignal,
        account: Dict,
        current_position: Optional[Position],
    ) -> Tuple[bool, str, int]:
        """
        Validate order before execution.

        Returns:
            Tuple of (is_valid, reason, adjusted_quantity)
        """
        can_trade, reason = self.can_trade()
        if not can_trade:
            return False, reason, 0

        equity = account["equity"]
        buying_power = account["buying_power"]

        # Calculate position value
        position_value = equity * signal.position_size_pct

        # Check buying power
        if position_value > buying_power:
            position_value = buying_power * 0.95  # Leave 5% buffer

        # Calculate quantity
        if signal.entry_price and signal.entry_price > 0:
            quantity = int(position_value / signal.entry_price)
        else:
            return False, "Invalid entry price", 0

        if quantity <= 0:
            return False, "Insufficient buying power", 0

        # Check if we already have a position
        if current_position:
            if signal.signal_type == SignalType.BUY and current_position.side == "long":
                return False, "Already have long position", 0
            if signal.signal_type == SignalType.SELL and current_position.side == "short":
                return False, "Already have short position", 0

        self.daily_trades += 1
        return True, "OK", quantity

    def update_trailing_stop(
        self,
        position: Position,
        current_price: float
    ) -> Optional[float]:
        """Update trailing stop for position."""
        if position.trailing_stop is None:
            # Initialize trailing stop
            if position.side == "long":
                return current_price * (1 - self.config["trailing_stop_pct"])
            else:
                return current_price * (1 + self.config["trailing_stop_pct"])

        # Update trailing stop
        if position.side == "long":
            new_stop = current_price * (1 - self.config["trailing_stop_pct"])
            if new_stop > position.trailing_stop:
                return new_stop
        else:
            new_stop = current_price * (1 + self.config["trailing_stop_pct"])
            if new_stop < position.trailing_stop:
                return new_stop

        return position.trailing_stop

    def check_stop_conditions(
        self,
        position: Position,
        current_price: float
    ) -> Tuple[bool, str]:
        """Check if stop loss or take profit triggered."""
        if position.stop_loss:
            if position.side == "long" and current_price <= position.stop_loss:
                return True, "stop_loss"
            if position.side == "short" and current_price >= position.stop_loss:
                return True, "stop_loss"

        if position.take_profit:
            if position.side == "long" and current_price >= position.take_profit:
                return True, "take_profit"
            if position.side == "short" and current_price <= position.take_profit:
                return True, "take_profit"

        if position.trailing_stop:
            if position.side == "long" and current_price <= position.trailing_stop:
                return True, "trailing_stop"
            if position.side == "short" and current_price >= position.trailing_stop:
                return True, "trailing_stop"

        return False, ""


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ORDER MANAGER
# ═══════════════════════════════════════════════════════════════════════════════
class OrderManager:
    """
    Order execution and management.

    Handles:
      - Order submission (market/limit)
      - Order tracking
      - Fill confirmation
      - Order cancellation
    """

    def __init__(self, client: AlpacaPaperClient):
        self.client = client
        self.pending_orders: Dict[str, Dict] = {}
        self.filled_orders: List[Dict] = []

    def execute_signal(
        self,
        signal: TradingSignal,
        quantity: int,
    ) -> Optional[str]:
        """Execute trading signal."""
        if signal.signal_type == SignalType.HOLD:
            return None

        side = OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL

        if TRADING_CONFIG["use_limit_orders"] and signal.entry_price:
            # Use limit order
            if side == OrderSide.BUY:
                limit_price = signal.entry_price * (1 + TRADING_CONFIG["limit_offset_pct"])
            else:
                limit_price = signal.entry_price * (1 - TRADING_CONFIG["limit_offset_pct"])

            order_id = self.client.submit_limit_order(
                symbol=signal.symbol,
                qty=quantity,
                side=side,
                limit_price=limit_price,
            )
        else:
            # Use market order
            order_id = self.client.submit_market_order(
                symbol=signal.symbol,
                qty=quantity,
                side=side,
            )

        if order_id:
            self.pending_orders[order_id] = {
                "signal": signal,
                "quantity": quantity,
                "submitted_at": datetime.now(),
            }
            logger.info(f"Order submitted: {side.value} {quantity} {signal.symbol}")

        return order_id

    def close_position(self, position: Position, reason: str = "", tracker=None) -> Optional[str]:
        """Close an existing position."""
        side = OrderSide.SELL if position.side == "long" else OrderSide.BUY

        # Calculate P&L before closing
        pnl = 0.0
        profit_pct = 0.0
        if position.entry_price and position.current_price:
            if position.side == "long":
                profit_pct = (position.current_price - position.entry_price) / position.entry_price
                pnl = (position.current_price - position.entry_price) * position.quantity
            else:
                profit_pct = (position.entry_price - position.current_price) / position.entry_price
                pnl = (position.entry_price - position.current_price) * position.quantity

            # Update dynamic thresholds with trade result
            dynamic_thresholds.update_trade_history(profit_pct)
            logger.info(f"Trade P&L: {profit_pct:.2%}")

        order_id = self.client.submit_market_order(
            symbol=position.symbol,
            qty=position.quantity,
            side=side,
        )

        if order_id:
            logger.info(f"Closing position: {position.symbol} ({reason})")

            # Record trade in history tracker for dashboard
            if tracker and TRACKER_AVAILABLE:
                tracker.record_trade(
                    symbol=position.symbol,
                    side=side.value.lower(),
                    quantity=position.quantity,
                    price=position.current_price,
                    pnl=pnl,
                    signal_type=reason,
                    confidence=0.0,
                )

        return order_id

    def check_pending_orders(self) -> List[str]:
        """Check status of pending orders and handle timeouts."""
        filled = []
        cancelled = []

        for order_id, order_info in list(self.pending_orders.items()):
            status = self.client.get_order_status(order_id)

            if status == "filled":
                filled.append(order_id)
                self.filled_orders.append(order_info)
                del self.pending_orders[order_id]
                logger.info(f"Order filled: {order_id}")

            elif status in ["cancelled", "expired", "rejected"]:
                cancelled.append(order_id)
                del self.pending_orders[order_id]
                logger.info(f"Order {status}: {order_id}")

            else:
                # Check timeout
                age = (datetime.now() - order_info["submitted_at"]).total_seconds()
                if age > TRADING_CONFIG["order_timeout_seconds"]:
                    self.client.cancel_order(order_id)
                    del self.pending_orders[order_id]
                    logger.info(f"Order cancelled (timeout): {order_id}")

        return filled

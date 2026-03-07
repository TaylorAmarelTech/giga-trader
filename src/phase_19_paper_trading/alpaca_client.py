"""
GIGA TRADER - Alpaca Paper Trading: Client, Data Classes, and Configuration
============================================================================
Alpaca API connection, order execution, data classes, and trading configuration.

Components:
  - SignalType enum
  - TradingSignal dataclass
  - Position dataclass (with pnl_pct property)
  - TradeRecord dataclass
  - DynamicThresholds class
  - AlpacaPaperClient class
  - TRADING_CONFIG constants
"""

import os
import sys
import logging
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# Setup path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Alpaca imports
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        StopOrderRequest,
        StopLimitOrderRequest,
        GetOrdersRequest,
    )
    from alpaca.trading.enums import (
        OrderSide,
        OrderType,
        TimeInForce,
        OrderStatus,
        QueryOrderStatus,
    )
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.live import StockDataStream
    from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("[WARN] Alpaca SDK not installed. Run: pip install alpaca-py")

logger = logging.getLogger("GigaTrader")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
TRADING_CONFIG = {
    # Trading parameters
    "symbol": "SPY",
    "max_position_pct": 0.25,  # Max 25% of portfolio in single position
    "min_position_pct": 0.05,  # Min 5% position size
    "max_daily_trades": 10,  # Max trades per day
    "max_daily_loss_pct": 0.02,  # Stop trading after 2% daily loss

    # Signal thresholds
    "entry_threshold": 0.65,  # Min probability to enter
    "exit_threshold": 0.45,  # Exit when probability drops below
    "strong_signal_threshold": 0.75,  # Full position size

    # Risk management
    "stop_loss_pct": 0.01,  # 1% stop loss
    "take_profit_pct": 0.02,  # 2% take profit
    "trailing_stop_pct": 0.005,  # 0.5% trailing stop
    "max_drawdown_pct": 0.05,  # 5% max drawdown before halting

    # Timing
    "market_open": dt_time(9, 30),
    "market_close": dt_time(16, 0),
    "no_new_trades_after": dt_time(15, 30),  # No new trades in last 30 min
    "force_close_time": dt_time(15, 55),  # Force close all positions

    # Execution
    "use_limit_orders": True,
    "limit_offset_pct": 0.001,  # 0.1% limit offset
    "order_timeout_seconds": 60,  # Cancel unfilled orders after 60s

    # Model paths
    "model_dir": project_root / "models" / "production",
}


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════
class DynamicThresholds:
    """
    Dynamically adjust trading thresholds based on market conditions.

    Factors considered:
      1. Market volatility (ATR) - higher vol = tighter thresholds
      2. Recent performance - losing streak = more conservative
      3. Time of day - more conservative near close
      4. VIX proxy - market fear level
    """

    def __init__(self):
        self.recent_trades = []  # List of (profit_pct, timestamp)
        self.volatility_history = []  # Recent volatility readings
        self.base_config = TRADING_CONFIG.copy()

    def update_trade_history(self, profit_pct: float):
        """Record a completed trade."""
        self.recent_trades.append({
            "profit_pct": profit_pct,
            "timestamp": datetime.now()
        })
        # Keep only last 20 trades
        self.recent_trades = self.recent_trades[-20:]

    def update_volatility(self, atr_pct: float):
        """Update volatility reading (ATR as % of price)."""
        self.volatility_history.append({
            "atr_pct": atr_pct,
            "timestamp": datetime.now()
        })
        # Keep last hour of readings
        cutoff = datetime.now() - timedelta(hours=1)
        self.volatility_history = [v for v in self.volatility_history if v["timestamp"] > cutoff]

    def get_adjusted_thresholds(self, current_price: float = None) -> dict:
        """
        Get dynamically adjusted thresholds.

        Returns dict with adjusted values for:
          - entry_threshold
          - stop_loss_pct
          - take_profit_pct
          - max_position_pct
        """
        adjustments = {}

        # 1. Time-based adjustments
        now = datetime.now().time()
        minutes_to_close = (dt_time(16, 0).hour * 60 + dt_time(16, 0).minute) - (now.hour * 60 + now.minute)

        if 0 < minutes_to_close < 30:
            # Last 30 min - more conservative
            adjustments["entry_threshold_mult"] = 1.15  # Higher threshold
            adjustments["position_mult"] = 0.5  # Smaller positions
        elif 0 < minutes_to_close < 60:
            # Last hour - slightly more conservative
            adjustments["entry_threshold_mult"] = 1.05
            adjustments["position_mult"] = 0.75
        else:
            adjustments["entry_threshold_mult"] = 1.0
            adjustments["position_mult"] = 1.0

        # 2. Performance-based adjustments (recent win rate)
        if len(self.recent_trades) >= 5:
            recent_wins = sum(1 for t in self.recent_trades[-5:] if t["profit_pct"] > 0)
            win_rate = recent_wins / 5

            if win_rate < 0.3:
                # Losing streak - be more conservative
                adjustments["entry_threshold_mult"] *= 1.2
                adjustments["position_mult"] *= 0.6
                adjustments["stop_loss_mult"] = 1.2  # Tighter stops
            elif win_rate > 0.7:
                # Winning streak - can be slightly more aggressive
                adjustments["entry_threshold_mult"] *= 0.95
                adjustments["position_mult"] *= 1.1  # Max 110%

        # 3. Volatility-based adjustments
        if len(self.volatility_history) >= 5:
            avg_vol = np.mean([v["atr_pct"] for v in self.volatility_history[-5:]])
            normal_vol = 0.005  # 0.5% as baseline

            vol_ratio = avg_vol / normal_vol
            if vol_ratio > 2.0:
                # High volatility - widen stops, smaller positions
                adjustments["stop_loss_mult"] = adjustments.get("stop_loss_mult", 1.0) * 1.5
                adjustments["take_profit_mult"] = 1.5
                adjustments["position_mult"] *= 0.7
            elif vol_ratio < 0.5:
                # Low volatility - tighter stops
                adjustments["stop_loss_mult"] = adjustments.get("stop_loss_mult", 1.0) * 0.7
                adjustments["take_profit_mult"] = 0.8

        # Calculate final adjusted values
        result = {
            "entry_threshold": min(0.85, self.base_config["entry_threshold"] * adjustments.get("entry_threshold_mult", 1.0)),
            "stop_loss_pct": self.base_config["stop_loss_pct"] * adjustments.get("stop_loss_mult", 1.0),
            "take_profit_pct": self.base_config["take_profit_pct"] * adjustments.get("take_profit_mult", 1.0),
            "max_position_pct": self.base_config["max_position_pct"] * adjustments.get("position_mult", 1.0),
            "adjustments_applied": adjustments,
        }

        return result


# Global instance
dynamic_thresholds = DynamicThresholds()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════
class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


@dataclass
class TradingSignal:
    """Trading signal from ML model."""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    probability: float
    confidence: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size_pct: float = 0.1
    metadata: Dict = field(default_factory=dict)


@dataclass
class Position:
    """Current position information."""
    symbol: str
    side: str  # "long" or "short"
    quantity: int
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0
        if self.side == "long":
            return (self.current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.current_price) / self.entry_price


@dataclass
class TradeRecord:
    """Record of executed trade."""
    trade_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    timestamp: datetime
    order_type: str
    signal_probability: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    hold_time_minutes: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ALPACA PAPER CLIENT
# ═══════════════════════════════════════════════════════════════════════════════
class AlpacaPaperClient:
    """
    Alpaca paper trading client wrapper.

    Handles:
      - API connection (paper trading)
      - Account info and buying power
      - Order execution
      - Position management
    """

    def __init__(self, api_key: str = None, secret_key: str = None):
        if not ALPACA_AVAILABLE:
            raise ImportError("Alpaca SDK not installed")

        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")

        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API keys not found. Set ALPACA_API_KEY and ALPACA_SECRET_KEY")

        # Paper trading client (paper=True)
        self.trading_client = TradingClient(
            self.api_key,
            self.secret_key,
            paper=True
        )

        # Data client
        self.data_client = StockHistoricalDataClient(
            self.api_key,
            self.secret_key
        )

        logger.info("Alpaca Paper Trading client initialized")

    def get_account(self) -> Dict:
        """Get account information."""
        account = self.trading_client.get_account()
        return {
            "equity": float(account.equity),
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "day_trade_count": account.daytrade_count,
            "pattern_day_trader": account.pattern_day_trader,
            "trading_blocked": account.trading_blocked,
            "account_blocked": account.account_blocked,
        }

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol."""
        try:
            pos = self.trading_client.get_open_position(symbol)
            return Position(
                symbol=symbol,
                side="long" if float(pos.qty) > 0 else "short",
                quantity=abs(int(pos.qty)),
                entry_price=float(pos.avg_entry_price),
                entry_time=datetime.now(),  # Alpaca doesn't provide this directly
                current_price=float(pos.current_price),
                unrealized_pnl=float(pos.unrealized_pl),
            )
        except Exception:
            return None

    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        positions = self.trading_client.get_all_positions()
        return [
            Position(
                symbol=pos.symbol,
                side="long" if float(pos.qty) > 0 else "short",
                quantity=abs(int(pos.qty)),
                entry_price=float(pos.avg_entry_price),
                entry_time=datetime.now(),
                current_price=float(pos.current_price),
                unrealized_pnl=float(pos.unrealized_pl),
            )
            for pos in positions
        ]

    def get_latest_price(self, symbol: str) -> float:
        """Get latest price for symbol."""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = self.data_client.get_stock_latest_quote(request)
            if symbol in quote:
                return float(quote[symbol].ask_price)
        except Exception as e:
            logger.warning(f"Failed to get quote for {symbol}: {e}")
        return 0.0

    def submit_market_order(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
    ) -> Optional[str]:
        """Submit market order."""
        try:
            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
            order = self.trading_client.submit_order(request)
            logger.info(f"Market order submitted: {side.value} {qty} {symbol}, ID: {order.id}")
            return str(order.id)
        except Exception as e:
            logger.error(f"Failed to submit market order: {e}")
            return None

    def submit_limit_order(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
        limit_price: float,
    ) -> Optional[str]:
        """Submit limit order."""
        try:
            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                limit_price=round(limit_price, 2),
                time_in_force=TimeInForce.DAY,
            )
            order = self.trading_client.submit_order(request)
            logger.info(f"Limit order submitted: {side.value} {qty} {symbol} @ ${limit_price:.2f}, ID: {order.id}")
            return str(order.id)
        except Exception as e:
            logger.error(f"Failed to submit limit order: {e}")
            return None

    def submit_stop_order(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
        stop_price: float,
    ) -> Optional[str]:
        """Submit stop order."""
        try:
            request = StopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                stop_price=round(stop_price, 2),
                time_in_force=TimeInForce.DAY,
            )
            order = self.trading_client.submit_order(request)
            logger.info(f"Stop order submitted: {side.value} {qty} {symbol} @ ${stop_price:.2f}")
            return str(order.id)
        except Exception as e:
            logger.error(f"Failed to submit stop order: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_order_status(self, order_id: str) -> Optional[str]:
        """Get order status."""
        try:
            order = self.trading_client.get_order_by_id(order_id)
            return order.status.value
        except Exception:
            return None

    def close_position(self, symbol: str) -> bool:
        """Close all positions for symbol."""
        try:
            self.trading_client.close_position(symbol)
            logger.info(f"Position closed: {symbol}")
            return True
        except Exception as e:
            logger.warning(f"Failed to close position {symbol}: {e}")
            return False

    def close_all_positions(self) -> bool:
        """Close all open positions."""
        try:
            self.trading_client.close_all_positions(cancel_orders=True)
            logger.info("All positions closed")
            return True
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return False

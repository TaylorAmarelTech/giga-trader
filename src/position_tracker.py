"""
Position & Trade History Tracker
================================
Tracks positions, trades, and P&L history for dashboard visualization.

Stores data in JSON files for easy loading by the dashboard.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import threading
import time

project_root = Path(__file__).parent.parent
logger = logging.getLogger("POSITION_TRACKER")


@dataclass
class PositionSnapshot:
    """Snapshot of a position at a point in time."""
    timestamp: str
    symbol: str
    side: str  # "long" or "short"
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    market_value: float


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    timestamp: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float
    pnl: float = 0.0
    signal_type: str = ""
    confidence: float = 0.0


@dataclass
class EquitySnapshot:
    """Snapshot of account equity."""
    timestamp: str
    equity: float
    cash: float
    buying_power: float
    daily_pnl: float
    total_positions: int


class PositionHistoryTracker:
    """
    Tracks and persists position, trade, and equity history.

    Data is stored in:
      - logs/position_history.json
      - logs/trade_history.json
      - logs/equity_history.json
    """

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or (project_root / "logs")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.position_file = self.data_dir / "position_history.json"
        self.trade_file = self.data_dir / "trade_history.json"
        self.equity_file = self.data_dir / "equity_history.json"

        # In-memory caches
        self._positions: List[PositionSnapshot] = []
        self._trades: List[TradeRecord] = []
        self._equity: List[EquitySnapshot] = []

        # Load existing data
        self._load_all()

        # Limit history size
        self._max_position_records = 10000  # ~7 days at 1/min
        self._max_trade_records = 1000
        self._max_equity_records = 10000

    def _load_all(self):
        """Load all history from files."""
        self._positions = self._load_json(self.position_file, [])
        self._trades = self._load_json(self.trade_file, [])
        self._equity = self._load_json(self.equity_file, [])

    def _load_json(self, path: Path, default: Any) -> Any:
        """Load JSON file or return default."""
        try:
            if path.exists():
                with open(path, "r") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
        return default

    def _save_json(self, path: Path, data: Any):
        """Save data to JSON file."""
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving {path}: {e}")

    def record_position(self, position: Any):
        """Record a position snapshot."""
        snapshot = PositionSnapshot(
            timestamp=datetime.now().isoformat(),
            symbol=getattr(position, "symbol", "SPY"),
            side=getattr(position, "side", "long"),
            quantity=float(getattr(position, "qty", getattr(position, "quantity", 0))),
            entry_price=float(getattr(position, "avg_entry_price", getattr(position, "entry_price", 0))),
            current_price=float(getattr(position, "current_price", 0)),
            unrealized_pnl=float(getattr(position, "unrealized_pl", getattr(position, "unrealized_pnl", 0))),
            market_value=float(getattr(position, "market_value", 0)),
        )

        self._positions.append(asdict(snapshot))

        # Trim if too large
        if len(self._positions) > self._max_position_records:
            self._positions = self._positions[-self._max_position_records:]

        self._save_json(self.position_file, self._positions)

    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        pnl: float = 0.0,
        signal_type: str = "",
        confidence: float = 0.0,
    ):
        """Record a completed trade."""
        trade = TradeRecord(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            pnl=pnl,
            signal_type=signal_type,
            confidence=confidence,
        )

        self._trades.append(asdict(trade))

        # Trim if too large
        if len(self._trades) > self._max_trade_records:
            self._trades = self._trades[-self._max_trade_records:]

        self._save_json(self.trade_file, self._trades)

    def record_equity(
        self,
        equity: float,
        cash: float,
        buying_power: float,
        daily_pnl: float,
        total_positions: int,
    ):
        """Record equity snapshot."""
        snapshot = EquitySnapshot(
            timestamp=datetime.now().isoformat(),
            equity=equity,
            cash=cash,
            buying_power=buying_power,
            daily_pnl=daily_pnl,
            total_positions=total_positions,
        )

        self._equity.append(asdict(snapshot))

        # Trim if too large
        if len(self._equity) > self._max_equity_records:
            self._equity = self._equity[-self._max_equity_records:]

        self._save_json(self.equity_file, self._equity)

    def get_position_history(self, symbol: str = None, hours: int = 24) -> List[Dict]:
        """Get position history for the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        cutoff_str = cutoff.isoformat()

        positions = [
            p for p in self._positions
            if p["timestamp"] >= cutoff_str
            and (symbol is None or p["symbol"] == symbol)
        ]

        return positions

    def get_trade_history(self, symbol: str = None, days: int = 7) -> List[Dict]:
        """Get trade history for the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.isoformat()

        trades = [
            t for t in self._trades
            if t["timestamp"] >= cutoff_str
            and (symbol is None or t["symbol"] == symbol)
        ]

        return trades

    def get_equity_history(self, hours: int = 24) -> List[Dict]:
        """Get equity history for the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        cutoff_str = cutoff.isoformat()

        return [e for e in self._equity if e["timestamp"] >= cutoff_str]

    def get_pnl_series(self, hours: int = 24) -> List[Dict]:
        """Get P&L time series for charting."""
        equity = self.get_equity_history(hours)
        return [
            {"timestamp": e["timestamp"], "daily_pnl": e["daily_pnl"]}
            for e in equity
        ]

    def get_stats(self) -> Dict:
        """Get summary statistics."""
        trades = self.get_trade_history(days=30)

        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_trade": 0,
            }

        pnls = [t.get("pnl", 0) for t in trades]
        wins = sum(1 for p in pnls if p > 0)

        return {
            "total_trades": len(trades),
            "win_rate": wins / len(trades) if trades else 0,
            "total_pnl": sum(pnls),
            "avg_trade": sum(pnls) / len(pnls) if pnls else 0,
            "best_trade": max(pnls) if pnls else 0,
            "worst_trade": min(pnls) if pnls else 0,
        }


class AlpacaPositionFetcher:
    """Fetches position data from Alpaca API."""

    def __init__(self, client=None):
        self.client = client

    def fetch_current_positions(self) -> List[Dict]:
        """Fetch current positions from Alpaca."""
        if not self.client:
            return []

        try:
            positions = self.client.get_all_positions()
            return [
                {
                    "symbol": p.symbol,
                    "side": p.side,
                    "quantity": float(p.qty),
                    "entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "unrealized_pnl": float(p.unrealized_pl),
                    "market_value": float(p.market_value),
                }
                for p in positions
            ]
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []

    def fetch_account_info(self) -> Dict:
        """Fetch account information from Alpaca."""
        if not self.client:
            return {}

        try:
            account = self.client.get_account()
            return {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
            }
        except Exception as e:
            logger.error(f"Error fetching account: {e}")
            return {}

    def fetch_order_history(self, days: int = 7) -> List[Dict]:
        """Fetch recent order history from Alpaca."""
        if not self.client:
            return []

        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            start = datetime.now() - timedelta(days=days)
            request = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                after=start,
            )

            orders = self.client.trading_client.get_orders(filter=request)
            return [
                {
                    "timestamp": o.filled_at.isoformat() if o.filled_at else o.created_at.isoformat(),
                    "symbol": o.symbol,
                    "side": o.side.value,
                    "quantity": float(o.filled_qty),
                    "price": float(o.filled_avg_price) if o.filled_avg_price else 0,
                    "status": o.status.value,
                }
                for o in orders
            ]
        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            return []


# Global tracker instance
_tracker = None


def get_tracker() -> PositionHistoryTracker:
    """Get or create the global tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = PositionHistoryTracker()
    return _tracker

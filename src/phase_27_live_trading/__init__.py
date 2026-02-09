"""Phase: Live Trading.

Re-exports paper trading components for live trading promotion.
In production, the paper trading bot runs with real money by
switching to the live Alpaca API credentials.
"""

from src.phase_19_paper_trading.alpaca_client import (
    AlpacaPaperClient,
    SignalType,
    TradingSignal,
)
from src.phase_19_paper_trading.signal_generator import SignalGenerator
from src.phase_19_paper_trading.risk_management import RiskManager, OrderManager
from src.phase_19_paper_trading.trading_bot import TradingBot

__all__ = [
    "AlpacaPaperClient",
    "SignalType",
    "TradingSignal",
    "SignalGenerator",
    "RiskManager",
    "OrderManager",
    "TradingBot",
]

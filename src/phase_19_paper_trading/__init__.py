"""
Phase 19: Paper Trading
========================
Modular paper trading system split into 4 submodules:
  - alpaca_client: API client, data classes, configuration, dynamic thresholds
  - signal_generator: ML-based signal generation
  - risk_management: Risk manager and order manager
  - trading_bot: Main trading loop, status dashboard, entry point
"""

from src.phase_19_paper_trading.alpaca_client import (
    TRADING_CONFIG,
    DynamicThresholds,
    dynamic_thresholds,
    SignalType,
    TradingSignal,
    Position,
    TradeRecord,
    AlpacaPaperClient,
    ALPACA_AVAILABLE,
)

from src.phase_19_paper_trading.signal_generator import (
    SignalGenerator,
)

from src.phase_19_paper_trading.risk_management import (
    RiskManager,
    OrderManager,
)

from src.phase_19_paper_trading.trading_bot import (
    setup_logging,
    TradingBot,
    print_status,
    main,
)

__all__ = [
    # Configuration
    "TRADING_CONFIG",
    "DynamicThresholds",
    "dynamic_thresholds",
    # Data classes
    "SignalType",
    "TradingSignal",
    "Position",
    "TradeRecord",
    # Client
    "AlpacaPaperClient",
    "ALPACA_AVAILABLE",
    # Signal generation
    "SignalGenerator",
    # Risk management
    "RiskManager",
    "OrderManager",
    # Trading bot
    "setup_logging",
    "TradingBot",
    "print_status",
    "main",
]

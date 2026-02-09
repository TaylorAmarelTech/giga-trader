"""
GIGA TRADER - Alpaca Paper Trading Module (Shim)
=================================================
This file re-exports everything from the modular phase_19_paper_trading package.
All code has been decomposed into:
  - src/phase_19_paper_trading/alpaca_client.py
  - src/phase_19_paper_trading/signal_generator.py
  - src/phase_19_paper_trading/risk_management.py
  - src/phase_19_paper_trading/trading_bot.py

Any existing code doing `from src.paper_trading import X` will continue to work.

Usage:
    .venv/Scripts/python.exe src/paper_trading.py
"""

import sys
from pathlib import Path

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Re-export everything from the modular package
from src.phase_19_paper_trading.alpaca_client import (  # noqa: F401
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

from src.phase_19_paper_trading.signal_generator import (  # noqa: F401
    SignalGenerator,
)

from src.phase_19_paper_trading.risk_management import (  # noqa: F401
    RiskManager,
    OrderManager,
)

from src.phase_19_paper_trading.trading_bot import (  # noqa: F401
    setup_logging,
    PaperPerformanceTracker,
    PredictionRecord,
    TradingBot,
    print_status,
    main,
)


if __name__ == "__main__":
    sys.exit(main())

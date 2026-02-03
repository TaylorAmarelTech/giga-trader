"""
GIGA TRADER - Alpaca Paper Trading Module
==========================================
Real-time paper trading integration with Alpaca.

Components:
  - AlpacaPaperClient: API connection and order execution
  - SignalGenerator: Real-time ML signal generation
  - OrderManager: Order execution with slippage/commission
  - PositionTracker: Track open positions and P&L
  - RiskManager: Position limits, stop losses, drawdown limits
  - TradingBot: Main trading loop

Usage:
    .venv/Scripts/python.exe src/paper_trading.py
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from queue import Queue

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import joblib

load_dotenv(project_root / ".env")

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

# Supervision imports
try:
    from src.supervision import (
        TradingSupervisionService,
        SupervisionConfig,
        SupervisionLevel,
        ForceCloseManager,
        ForceCloseConfig,
    )
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    print("[INFO] Supervision module not yet available - will use basic force close")

# Position history tracker
try:
    from src.position_tracker import get_tracker, PositionHistoryTracker
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False
    print("[INFO] Position tracker not available")

# Dynamic model selector
try:
    from src.dynamic_model_selector import DynamicModelSelector, EnsemblePrediction
    DYNAMIC_SELECTOR_AVAILABLE = True
except ImportError:
    DYNAMIC_SELECTOR_AVAILABLE = False
    print("[INFO] Dynamic model selector not available")


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

        if minutes_to_close < 30:
            # Last 30 min - more conservative
            adjustments["entry_threshold_mult"] = 1.15  # Higher threshold
            adjustments["position_mult"] = 0.5  # Smaller positions
        elif minutes_to_close < 60:
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
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════
def setup_logging():
    """Setup logging configuration."""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"trading_{datetime.now().strftime('%Y%m%d')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger("GigaTrader")

logger = setup_logging()


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


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SIGNAL GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════
class SignalGenerator:
    """
    Generate trading signals from trained ML models.

    Supports two modes:
    1. Static mode: Load a single hardcoded model (legacy)
    2. Dynamic mode: Use DynamicModelSelector to select/ensemble from registry

    Dynamic mode provides:
    - Automatic model selection based on performance
    - Intelligent ensembling of multiple models
    - Adaptive entry/exit window matching
    """

    def __init__(self, model_dir: Path = None, use_dynamic_selector: bool = True):
        self.model_dir = model_dir or TRADING_CONFIG["model_dir"]
        self.models = {}
        self.scaler = None
        self.dim_state = None
        self.feature_cols = None
        self.use_leak_proof = False  # Initialize here - will be updated in _load_models() if leak-proof model loaded
        self.model_config = None

        # Dynamic model selector - MANDATORY for proper signal generation
        # No fallback to static models to ensure all signals use properly trained/validated models
        self.use_dynamic_selector = use_dynamic_selector and DYNAMIC_SELECTOR_AVAILABLE
        self.dynamic_selector = None

        if self.use_dynamic_selector:
            self._init_dynamic_selector()

        # If dynamic selector is not available or empty, try to load static models
        # but warn that this is a degraded mode that should be fixed
        if not self.dynamic_selector or not self.dynamic_selector.candidates:
            logger.warning("=" * 60)
            logger.warning("DEGRADED MODE: Dynamic model selector not available")
            logger.warning("Signals may not use properly validated models")
            logger.warning("Run 'python scripts/run_grid_search.py' to populate registry")
            logger.warning("=" * 60)
            self._load_models()

            # Verify we have BOTH swing AND timing models
            has_swing = "swing_pipeline" in self.models or "swing_l2" in self.models or "swing" in self.models
            has_timing = "timing_pipeline" in self.models or "timing_l2" in self.models or "timing" in self.models

            if not has_swing or not has_timing:
                raise ValueError(
                    "CRITICAL: No valid models available. Both swing AND timing models are required. "
                    "Run grid search to train and validate models: python scripts/run_grid_search.py"
                )

    def _init_dynamic_selector(self):
        """Initialize the dynamic model selector from registry.

        REQUIREMENTS (per CLAUDE.md):
        - min_test_auc > 0.58 (target, allow 0.55 minimum)
        - min_wmes > 0.55 (target, allow 0.50 minimum)
        - Models must have both swing AND timing predictions
        """
        try:
            self.dynamic_selector = DynamicModelSelector(
                min_test_auc=0.55,  # Minimum required AUC (0.58 is target)
                min_wmes=0.50,      # Minimum WMES (0.55 is target)
                max_models_to_load=10,
                ensemble_method="weighted_average",
            )
            n_loaded = self.dynamic_selector.load_from_registry()

            if n_loaded > 0:
                logger.info(f"Dynamic selector loaded {n_loaded} qualified model candidates")

                # Verify models meet quality standards
                status = self.dynamic_selector.get_status()
                top_models = status.get("top_models", [])

                if top_models:
                    best_auc = max(m.get("test_auc", 0) for m in top_models)
                    best_wmes = max(m.get("wmes", 0) for m in top_models)

                    if best_auc < 0.58:
                        logger.warning(f"Best model AUC ({best_auc:.3f}) below target (0.58)")
                    if best_wmes < 0.55:
                        logger.warning(f"Best model WMES ({best_wmes:.3f}) below target (0.55)")

                # Get available entry/exit windows
                windows = self.dynamic_selector.get_available_windows()
                if windows:
                    logger.info(f"Available entry/exit windows: {len(windows)}")
                    for w in windows[:3]:
                        logger.info(f"  Entry: {w['entry_window']}, Exit: {w['exit_window']}, Score: {w['best_score']:.3f}")
            else:
                logger.warning("No models in registry meet minimum quality requirements")
                logger.warning("Run grid search to train validated models: python scripts/run_grid_search.py")
                self.dynamic_selector = None

        except Exception as e:
            logger.error(f"Failed to initialize dynamic selector: {e}")
            self.dynamic_selector = None

    def _load_models(self):
        """Load trained models from disk."""
        self.use_leak_proof = False  # Track which model type is loaded

        try:
            # PREFER leak-proof model (better model with correct feature handling)
            leak_proof_path = self.model_dir / "spy_leak_proof_models.joblib"
            if leak_proof_path.exists():
                data = joblib.load(leak_proof_path)

                # Leak-proof model uses sklearn Pipelines that handle transformation internally
                if "swing_pipeline" in data:
                    self.models["swing_pipeline"] = data["swing_pipeline"]
                    logger.info("Loaded swing pipeline (leak-proof)")

                if "timing_pipeline" in data:
                    self.models["timing_pipeline"] = data["timing_pipeline"]
                    logger.info("Loaded timing pipeline (leak-proof)")

                # Feature columns are the RAW feature names (before transformation)
                if "feature_columns" in data:
                    self.feature_cols = data["feature_columns"]
                    logger.info(f"Loaded {len(self.feature_cols)} raw feature columns (leak-proof)")

                if "config" in data:
                    self.model_config = data["config"]

                self.use_leak_proof = True
                logger.info(f"Models loaded from {leak_proof_path} (LEAK-PROOF)")
                return

            # Fallback: Try loading from combined model file (legacy format)
            combined_path = self.model_dir / "spy_robust_models.joblib"
            if combined_path.exists():
                data = joblib.load(combined_path)

                # Extract models
                if "models" in data:
                    models_data = data["models"]
                    # Swing models (ensemble of L2 + GB)
                    if "swing" in models_data:
                        self.models["swing_l2"] = models_data["swing"]["l2"]
                        self.models["swing_gb"] = models_data["swing"]["gb"]
                        logger.info("Loaded swing models (L2 + GB ensemble)")

                    # Timing models
                    if "timing" in models_data:
                        self.models["timing_l2"] = models_data["timing"]["l2"]
                        self.models["timing_gb"] = models_data["timing"]["gb"]
                        logger.info("Loaded timing models (L2 + GB ensemble)")

                    # Entry/exit model
                    if "entry_exit_timing" in models_data:
                        self.models["entry_exit"] = models_data["entry_exit_timing"]
                        logger.info("Loaded entry/exit timing model")

                # Extract scaler
                if "scaler" in data:
                    self.scaler = data["scaler"]
                    logger.info("Loaded scaler")

                # Extract dim reduction state
                if "dim_reduction_state" in data:
                    self.dim_state = data["dim_reduction_state"]
                    logger.info("Loaded dimensionality reduction state")

                # Extract feature columns (NOTE: These are TRANSFORMED names in legacy model)
                if "feature_cols" in data:
                    self.feature_cols = data["feature_cols"]
                    logger.info(f"Loaded {len(self.feature_cols)} feature columns (legacy - transformed)")

                # Try to get original raw feature names from dim_state
                if self.dim_state and "var_selector" in self.dim_state:
                    # The var_selector knows the expected input dimension
                    expected_n = self.dim_state["var_selector"].n_features_in_
                    logger.info(f"Legacy model expects {expected_n} raw input features")

                # Store config for reference
                if "config" in data:
                    self.model_config = data["config"]

                logger.info(f"Models loaded from {combined_path} (LEGACY)")
                return

            # Fallback: try loading individual files (legacy format)
            swing_path = self.model_dir / "swing_model.joblib"
            timing_path = self.model_dir / "timing_model.joblib"
            scaler_path = self.model_dir / "scaler.joblib"
            dim_path = self.model_dir / "dim_reduction_state.joblib"
            features_path = self.model_dir / "feature_cols.joblib"

            if swing_path.exists():
                self.models["swing"] = joblib.load(swing_path)
                logger.info("Loaded swing model (legacy)")

            if timing_path.exists():
                self.models["timing"] = joblib.load(timing_path)
                logger.info("Loaded timing model (legacy)")

            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded scaler (legacy)")

            if dim_path.exists():
                self.dim_state = joblib.load(dim_path)
                logger.info("Loaded dim reduction state (legacy)")

            if features_path.exists():
                self.feature_cols = joblib.load(features_path)
                logger.info(f"Loaded {len(self.feature_cols)} feature columns (legacy)")

            # Load entry/exit model if available
            entry_exit_path = self.model_dir / "entry_exit_model.joblib"
            if entry_exit_path.exists():
                self.models["entry_exit"] = joblib.load(entry_exit_path)
                logger.info("Loaded entry/exit timing model (legacy)")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")

    def prepare_features(self, df_1min: pd.DataFrame) -> np.ndarray:
        """
        Prepare features from 1-minute data.

        This mirrors the feature engineering in train_robust_model.py
        """
        # Import feature engineering functions
        try:
            from src.train_robust_model import engineer_all_features, add_rolling_features

            # Prepare the data by adding required columns
            df = df_1min.copy()

            # Ensure timestamp is a column, not index
            if "timestamp" not in df.columns:
                # Reset index to make it a column
                df = df.reset_index()
                # If the index was named something else (like the Alpaca timestamp column)
                # find the datetime column and rename it
                if "timestamp" not in df.columns:
                    for col in df.columns:
                        if pd.api.types.is_datetime64_any_dtype(df[col]):
                            df = df.rename(columns={col: "timestamp"})
                            break

            # Add "date" column (required by engineer_all_features)
            if "timestamp" not in df.columns:
                logger.error("No timestamp column found in data")
                return np.array([])
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Convert to EST for session detection (Alpaca returns UTC)
            if df["timestamp"].dt.tz is not None:
                df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
            else:
                # Assume UTC if no timezone, convert to EST
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")

            # Create the columns expected by engineer_all_features
            df["date"] = df["timestamp"].dt.date
            df["time"] = df["timestamp"].dt.time
            df["hour"] = df["timestamp"].dt.hour
            df["minute"] = df["timestamp"].dt.minute

            # Add "session" column based on time of day (EST)
            # Premarket: 4:00 AM - 9:30 AM
            # Regular: 9:30 AM - 4:00 PM
            # Afterhours: 4:00 PM - 8:00 PM
            def get_session(row):
                h, m = row["hour"], row["minute"]
                if h < 4:
                    return "closed"
                elif h < 9 or (h == 9 and m < 30):
                    return "premarket"
                elif h < 16:
                    return "regular"
                elif h < 20:
                    return "afterhours"
                else:
                    return "closed"

            df["session"] = df.apply(get_session, axis=1)

            # Engineer daily features
            df_daily = engineer_all_features(df, swing_threshold=0.0025)
            df_daily = add_rolling_features(df_daily)

            if df_daily.empty:
                logger.error("Feature engineering returned empty dataframe")
                return np.array([])

            # Add anti-overfit features (MAG10, cross-assets, component streaks)
            # These are required because the model was trained with them
            try:
                from src.anti_overfit import integrate_anti_overfit
                df_daily, _ = integrate_anti_overfit(
                    df_daily,
                    spy_1min=df,  # Pass the 1min data for context
                    use_synthetic=False,  # Skip synthetic during inference
                    use_cross_assets=True,
                    use_breadth_streaks=True,
                    use_mag_breadth=True,
                )
            except Exception as e:
                logger.warning(f"Anti-overfit integration failed: {e}")

            # Get all numeric feature columns (the raw features)
            numeric_cols = df_daily.select_dtypes(include=[np.number]).columns
            exclude = ["target_up", "target_timing", "day_return", "sample_weight",
                       "is_up_day", "is_down_day", "low_before_high", "high_minutes", "low_minutes"]
            all_feature_cols = [c for c in numeric_cols if c not in exclude]

            if len(all_feature_cols) == 0:
                logger.error("No numeric features found in df_daily")
                return np.array([])

            # ─────────────────────────────────────────────────────────────────────
            # LEAK-PROOF MODEL: Use saved feature columns and return raw features
            # The pipeline handles transformation internally
            # ─────────────────────────────────────────────────────────────────────
            if self.use_leak_proof and self.feature_cols:
                # Select only the features that were used during training, in the same order
                available_features = set(all_feature_cols)
                missing_features = [f for f in self.feature_cols if f not in available_features]
                if missing_features:
                    logger.warning(f"Missing {len(missing_features)} features: {missing_features[:5]}...")

                # Build feature array in the correct order, filling missing with 0
                X_list = []
                for feat in self.feature_cols:
                    if feat in df_daily.columns:
                        X_list.append(df_daily[feat].iloc[-1])
                    else:
                        X_list.append(0.0)  # Fill missing features with 0

                X = np.array(X_list).reshape(1, -1)

                # Handle NaN values
                if np.isnan(X).any():
                    nan_count = np.isnan(X).sum()
                    nan_indices = np.where(np.isnan(X[0]))[0]
                    nan_features = [self.feature_cols[i] for i in nan_indices[:5]]
                    logger.warning(f"Found {nan_count} NaN values in features: {nan_features}")
                    X = np.nan_to_num(X, nan=0.0)

                logger.debug(f"Prepared {X.shape[1]} features for leak-proof model")
                return X

            # ─────────────────────────────────────────────────────────────────────
            # LEGACY MODEL: Apply dimensionality reduction manually
            # ─────────────────────────────────────────────────────────────────────
            feature_cols = all_feature_cols
            X = df_daily[feature_cols].iloc[-1:].values

            # Handle NaN values - fill with 0
            if np.isnan(X).any():
                nan_count = np.isnan(X).sum()
                logger.warning(f"Found {nan_count} NaN values in features, filling with 0")
                X = np.nan_to_num(X, nan=0.0)

            # Apply dimensionality reduction (handles scaling internally via pre_transform_scaler)
            if self.dim_state:
                from src.train_robust_model import reduce_dimensions
                try:
                    # Check if we need to adjust feature count
                    var_selector = self.dim_state.get("var_selector")
                    expected_features = var_selector.n_features_in_ if var_selector else None

                    if expected_features and X.shape[1] != expected_features:
                        if X.shape[1] > expected_features:
                            # Too many features - truncate (use first N)
                            logger.warning(f"Feature count mismatch: {X.shape[1]} vs {expected_features}, truncating")
                            X = X[:, :expected_features]
                            feature_cols = feature_cols[:expected_features]
                        else:
                            # Too few features - pad with zeros
                            logger.warning(f"Feature count mismatch: {X.shape[1]} vs {expected_features}, padding")
                            padding = np.zeros((X.shape[0], expected_features - X.shape[1]))
                            X = np.hstack([X, padding])

                    X, _, _ = reduce_dimensions(X, feature_cols, fit=False, state=self.dim_state)
                except Exception as e:
                    logger.error(f"Dimension reduction failed: {e}")
                    return np.array([])
            elif self.scaler:
                # Fallback: just scale if no dim_state
                X = self.scaler.transform(X)

            return X

        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return np.array([])

    def generate_signal(
        self,
        df_1min: pd.DataFrame,
        current_price: float,
        entry_window: Tuple[int, int] = None,
        exit_window: Tuple[int, int] = None,
    ) -> TradingSignal:
        """
        Generate trading signal from current data.

        Args:
            df_1min: Recent 1-minute OHLCV data
            current_price: Current market price
            entry_window: Optional entry window override (start_min, end_min)
            exit_window: Optional exit window override (start_min, end_min)

        Returns:
            TradingSignal with recommendation
        """
        timestamp = datetime.now()
        symbol = TRADING_CONFIG["symbol"]

        # Default hold signal
        default_signal = TradingSignal(
            timestamp=timestamp,
            symbol=symbol,
            signal_type=SignalType.HOLD,
            probability=0.5,
            confidence=0.0,
        )

        # ═══════════════════════════════════════════════════════════════════
        # DYNAMIC MODEL SELECTOR PATH (REQUIRED)
        # Uses registry to select/ensemble best validated models
        # This is the ONLY proper path for production signal generation
        # ═══════════════════════════════════════════════════════════════════
        if self.dynamic_selector and self.dynamic_selector.candidates:
            return self._generate_signal_dynamic(
                df_1min, current_price, entry_window, exit_window, default_signal
            )

        # ═══════════════════════════════════════════════════════════════════
        # STATIC MODEL PATH (DEGRADED MODE - NOT RECOMMENDED)
        # Uses single model without proper registry validation
        # WARNING: This path should only be used during initial setup
        # Run grid search to populate registry: python scripts/run_grid_search.py
        # ═══════════════════════════════════════════════════════════════════
        logger.warning("DEGRADED MODE: Using static model without registry validation")

        # Check if we have BOTH swing AND timing models (required)
        has_swing = "swing_pipeline" in self.models or "swing_l2" in self.models or "swing" in self.models
        has_timing = "timing_pipeline" in self.models or "timing_l2" in self.models or "timing" in self.models

        if not has_swing:
            logger.error("No swing model available - cannot generate signal")
            return default_signal

        if not has_timing:
            logger.error("No timing model available - cannot generate valid signal")
            return default_signal

        try:
            # Calculate and update volatility for dynamic thresholds
            if len(df_1min) >= 14:
                df_vol = df_1min.copy()
                if "high" in df_vol.columns and "low" in df_vol.columns and "close" in df_vol.columns:
                    tr = np.maximum(
                        df_vol["high"] - df_vol["low"],
                        np.maximum(
                            abs(df_vol["high"] - df_vol["close"].shift(1)),
                            abs(df_vol["low"] - df_vol["close"].shift(1))
                        )
                    )
                    atr = tr.rolling(14).mean().iloc[-1]
                    atr_pct = atr / current_price if current_price > 0 else 0
                    dynamic_thresholds.update_volatility(atr_pct)

            # Prepare features
            X = self.prepare_features(df_1min)

            if len(X) == 0:
                return default_signal

            # ─────────────────────────────────────────────────────────────────────
            # LEAK-PROOF MODEL: Use sklearn Pipeline (handles transformation internally)
            # ─────────────────────────────────────────────────────────────────────
            if self.use_leak_proof and "swing_pipeline" in self.models:
                swing_proba = self.models["swing_pipeline"].predict_proba(X)[0, 1]
                confidence_penalty = 1.0

                # Timing model - REQUIRED for proper signal generation
                if "timing_pipeline" not in self.models:
                    logger.warning("Timing model not available - cannot generate valid signal")
                    return default_signal  # Reject signal without timing validation

                timing_proba = self.models["timing_pipeline"].predict_proba(X)[0, 1]
                timing_disagreement = 0.0

                logger.debug(f"Leak-proof: swing={swing_proba:.3f}, timing={timing_proba:.3f}")

            # ─────────────────────────────────────────────────────────────────────
            # LEGACY MODEL: Manual ensemble of L2 + GB
            # ─────────────────────────────────────────────────────────────────────
            elif "swing_l2" in self.models and "swing_gb" in self.models:
                proba_l2 = self.models["swing_l2"].predict_proba(X)[0, 1]
                proba_gb = self.models["swing_gb"].predict_proba(X)[0, 1]

                # Calculate model disagreement
                disagreement = abs(proba_l2 - proba_gb)

                # Weighted ensemble with disagreement penalty
                # When models agree strongly, use simple average
                # When models disagree, weight towards the more confident model
                if disagreement > 0.2:
                    # Significant disagreement - use the more extreme (confident) prediction
                    # but reduce overall confidence
                    dist_l2 = abs(proba_l2 - 0.5)
                    dist_gb = abs(proba_gb - 0.5)
                    if dist_l2 > dist_gb:
                        swing_proba = 0.6 * proba_l2 + 0.4 * proba_gb
                    else:
                        swing_proba = 0.4 * proba_l2 + 0.6 * proba_gb
                    # Apply disagreement penalty to confidence later
                    confidence_penalty = 1 - (disagreement * 0.5)
                else:
                    swing_proba = (proba_l2 + proba_gb) / 2
                    confidence_penalty = 1.0

                # Log ensemble details for debugging
                logger.debug(f"Ensemble: L2={proba_l2:.3f}, GB={proba_gb:.3f}, disagree={disagreement:.3f}, final={swing_proba:.3f}")

                # Legacy timing models - REQUIRED
                timing_proba = None  # Will be set below or signal rejected
                timing_disagreement = 0.0
            else:
                swing_proba = self.models["swing"].predict_proba(X)[0, 1]
                confidence_penalty = 1.0
                timing_proba = None  # Will be set below or signal rejected
                timing_disagreement = 0.0

            # Legacy timing ensemble - REQUIRED for valid signal
            if not self.use_leak_proof and "timing_l2" in self.models and "timing_gb" in self.models:
                proba_l2 = self.models["timing_l2"].predict_proba(X)[0, 1]
                proba_gb = self.models["timing_gb"].predict_proba(X)[0, 1]
                timing_disagreement = abs(proba_l2 - proba_gb)

                if timing_disagreement > 0.2:
                    dist_l2 = abs(proba_l2 - 0.5)
                    dist_gb = abs(proba_gb - 0.5)
                    if dist_l2 > dist_gb:
                        timing_proba = 0.6 * proba_l2 + 0.4 * proba_gb
                    else:
                        timing_proba = 0.4 * proba_l2 + 0.6 * proba_gb
                else:
                    timing_proba = (proba_l2 + proba_gb) / 2
            elif "timing" in self.models:
                timing_proba = self.models["timing"].predict_proba(X)[0, 1]

            # REQUIRE timing model - reject signal if no timing validation
            if timing_proba is None:
                logger.warning("No timing model available - cannot generate valid signal")
                return default_signal

            # Get dynamic thresholds based on current market conditions
            dyn_thresholds = dynamic_thresholds.get_adjusted_thresholds(current_price)
            entry_threshold = dyn_thresholds["entry_threshold"]
            stop_loss_pct = dyn_thresholds["stop_loss_pct"]
            take_profit_pct = dyn_thresholds["take_profit_pct"]
            max_position_pct = dyn_thresholds["max_position_pct"]

            # Determine signal type using dynamic thresholds
            if swing_proba >= entry_threshold:
                # Bullish signal
                if timing_proba >= 0.5:
                    # Good timing (low before high expected)
                    signal_type = SignalType.BUY
                else:
                    # Wait for better entry
                    signal_type = SignalType.HOLD
            elif swing_proba <= (1 - entry_threshold):
                # Bearish signal (could short or avoid)
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD

            # Calculate position size based on confidence with dynamic limits
            # Apply disagreement penalty from ensemble
            base_confidence = abs(swing_proba - 0.5) * 2  # 0-1 scale
            confidence = base_confidence * confidence_penalty  # Reduce if models disagree
            strong_threshold = TRADING_CONFIG["strong_signal_threshold"]
            min_position_pct = TRADING_CONFIG["min_position_pct"]

            if swing_proba >= strong_threshold and confidence_penalty > 0.8:
                # Only use max position if models agree
                position_size = max_position_pct
            else:
                # Scale position size with confidence, capped by dynamic max
                position_size = min_position_pct + \
                    (max_position_pct - min_position_pct) * confidence

            # Calculate stop loss and take profit using dynamic thresholds
            if signal_type == SignalType.BUY:
                stop_loss = current_price * (1 - stop_loss_pct)
                take_profit = current_price * (1 + take_profit_pct)
            elif signal_type == SignalType.SELL:
                stop_loss = current_price * (1 + stop_loss_pct)
                take_profit = current_price * (1 - take_profit_pct)
            else:
                stop_loss = None
                take_profit = None

            # Log dynamic adjustments if any were applied
            if dyn_thresholds.get("adjustments_applied"):
                logger.debug(f"Dynamic adjustments: {dyn_thresholds['adjustments_applied']}")

            # Use entry/exit model for refined timing if available
            entry_exit_decision = {}
            if "entry_exit" in self.models and signal_type in [SignalType.BUY, SignalType.SELL]:
                try:
                    entry_exit_decision = self.models["entry_exit"].predict(
                        X,
                        swing_proba=swing_proba,
                        timing_proba=timing_proba,
                        current_price=current_price
                    )
                    if entry_exit_decision.get("stop_loss"):
                        stop_loss = entry_exit_decision["stop_loss"]
                    if entry_exit_decision.get("take_profit"):
                        take_profit = entry_exit_decision["take_profit"]
                    if entry_exit_decision.get("position_size_pct"):
                        position_size = entry_exit_decision["position_size_pct"]
                except Exception as e:
                    logger.warning(f"Entry/exit model prediction failed: {e}")

            return TradingSignal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=signal_type,
                probability=swing_proba,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size_pct=position_size,
                metadata={
                    "timing_proba": timing_proba,
                    "entry_exit_decision": entry_exit_decision,
                }
            )

        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return default_signal

    def _generate_signal_dynamic(
        self,
        df_1min: pd.DataFrame,
        current_price: float,
        entry_window: Tuple[int, int],
        exit_window: Tuple[int, int],
        default_signal: TradingSignal,
    ) -> TradingSignal:
        """
        Generate signal using dynamic model selector.

        Uses the registry to select/ensemble the best models for current conditions.
        """
        timestamp = datetime.now()
        symbol = TRADING_CONFIG["symbol"]

        try:
            # Calculate volatility for dynamic thresholds
            if len(df_1min) >= 14:
                df_vol = df_1min.copy()
                if "high" in df_vol.columns and "low" in df_vol.columns and "close" in df_vol.columns:
                    tr = np.maximum(
                        df_vol["high"] - df_vol["low"],
                        np.maximum(
                            abs(df_vol["high"] - df_vol["close"].shift(1)),
                            abs(df_vol["low"] - df_vol["close"].shift(1))
                        )
                    )
                    atr = tr.rolling(14).mean().iloc[-1]
                    atr_pct = atr / current_price if current_price > 0 else 0
                    dynamic_thresholds.update_volatility(atr_pct)

            # Prepare features
            X = self.prepare_features(df_1min)

            if len(X) == 0:
                return default_signal

            # Get feature names for the dynamic selector
            feature_names = self.feature_cols if self.feature_cols else []

            # Get prediction from dynamic selector (ensembles best models)
            prediction = self.dynamic_selector.predict(
                features=X,
                feature_names=feature_names,
                entry_window=entry_window,
                exit_window=exit_window,
                n_models=5,
            )

            swing_proba = prediction.swing_probability
            timing_proba = prediction.timing_probability
            confidence = prediction.confidence
            direction = prediction.direction

            # Log ensemble details
            logger.info(
                f"Dynamic ensemble: direction={direction}, swing={swing_proba:.3f}, "
                f"timing={timing_proba:.3f}, confidence={confidence:.3f}, "
                f"n_models={prediction.n_models}, agreement={prediction.agreement_ratio:.2f}"
            )

            # Get dynamic thresholds
            dyn_thresholds = dynamic_thresholds.get_adjusted_thresholds(current_price)
            entry_threshold = dyn_thresholds["entry_threshold"]
            stop_loss_pct = dyn_thresholds["stop_loss_pct"]
            take_profit_pct = dyn_thresholds["take_profit_pct"]
            max_position_pct = dyn_thresholds["max_position_pct"]

            # Convert direction to signal type
            if direction == "LONG" and swing_proba >= entry_threshold:
                signal_type = SignalType.BUY
            elif direction == "SHORT" and swing_proba <= (1 - entry_threshold):
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD

            # Use ensemble's position sizing suggestion, capped by risk limits
            position_size = min(
                prediction.confidence_adjusted_position_pct,
                max_position_pct
            )
            position_size = max(position_size, TRADING_CONFIG["min_position_pct"])

            # Calculate stop loss and take profit
            if signal_type == SignalType.BUY:
                stop_loss = current_price * (1 - stop_loss_pct)
                take_profit = current_price * (1 + take_profit_pct)
            elif signal_type == SignalType.SELL:
                stop_loss = current_price * (1 + stop_loss_pct)
                take_profit = current_price * (1 - take_profit_pct)
            else:
                stop_loss = None
                take_profit = None

            return TradingSignal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=signal_type,
                probability=swing_proba,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size_pct=position_size,
                metadata={
                    "timing_proba": timing_proba,
                    "ensemble_method": prediction.ensemble_method,
                    "n_models": prediction.n_models,
                    "agreement_ratio": prediction.agreement_ratio,
                    "entry_window": prediction.entry_window,
                    "exit_window": prediction.exit_window,
                    "model_predictions": prediction.model_predictions[:3],  # Top 3
                    "dynamic_selector": True,
                }
            )

        except Exception as e:
            logger.error(f"Dynamic signal generation failed: {e}")
            return default_signal

    def get_selector_status(self) -> Dict:
        """Get status of the dynamic model selector."""
        if self.dynamic_selector:
            return self.dynamic_selector.get_status()
        return {"dynamic_selector": False, "mode": "static"}


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


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TRADING BOT
# ═══════════════════════════════════════════════════════════════════════════════
class TradingBot:
    """
    Main trading bot that ties everything together.

    Runs the main trading loop:
      1. Fetch latest data
      2. Generate signals
      3. Validate with risk manager
      4. Execute orders
      5. Monitor positions
      6. Update stops
    """

    def __init__(self):
        self.client = AlpacaPaperClient()
        self.signal_generator = SignalGenerator()
        self.risk_manager = RiskManager()
        self.order_manager = OrderManager(self.client)

        self.is_running = False
        self.last_signal_time = None
        self.signal_interval = 60  # Generate signal every 60 seconds

        # Initialize position tracker for dashboard history
        self.position_tracker = None
        if TRACKER_AVAILABLE:
            self.position_tracker = get_tracker()
            logger.info("Position history tracker initialized")

        # Initialize supervision service
        self.supervisor = None
        if SUPERVISION_AVAILABLE:
            try:
                self.supervisor = TradingSupervisionService(
                    alpaca_client=self.client,
                    config=SupervisionConfig(
                        level=SupervisionLevel.STANDARD,
                        force_close_enabled=True,
                        circuit_breakers_enabled=True,
                        reconciliation_enabled=True,
                        feature_validation_enabled=True,
                        model_health_enabled=True,
                    ),
                )
                logger.info("Supervision service initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize supervision: {e}")
                self.supervisor = None

        logger.info("Trading Bot initialized")

    def fetch_latest_data(self, lookback_days: int = 30) -> pd.DataFrame:
        """Fetch latest 1-minute data."""
        try:
            end = datetime.now()
            start = end - timedelta(days=lookback_days)

            request = StockBarsRequest(
                symbol_or_symbols=TRADING_CONFIG["symbol"],
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
            )

            bars = self.client.data_client.get_stock_bars(request)
            df = bars.df.reset_index()

            # Rename columns to match expected format
            df = df.rename(columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            })

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            return pd.DataFrame()

    def run_once(self):
        """Run one iteration of the trading loop."""
        try:
            # Get account info
            account = self.client.get_account()

            # Get current position
            position = self.client.get_position(TRADING_CONFIG["symbol"])

            # Get current price
            current_price = self.client.get_latest_price(TRADING_CONFIG["symbol"])

            if current_price <= 0:
                logger.warning("Could not get current price")
                return

            # Record equity snapshot for dashboard
            if self.position_tracker:
                self.position_tracker.record_equity(
                    equity=account["equity"],
                    cash=account["cash"],
                    buying_power=account["buying_power"],
                    daily_pnl=self.risk_manager.daily_pnl,
                    total_positions=1 if position else 0,
                )

            # Record position snapshot for dashboard
            if self.position_tracker and position:
                self.position_tracker.record_position(position)

            # Update risk manager
            if position:
                self.risk_manager.update_pnl(position.unrealized_pnl, account["equity"])

            # ═══════════════════════════════════════════════════════════════
            # SUPERVISION CHECK (runs BEFORE any trading logic)
            # ═══════════════════════════════════════════════════════════════
            if self.supervisor:
                # Get positions for supervision check
                positions = [position] if position else []

                # Pre-trade check handles force close, circuit breakers, reconciliation
                supervision_status = self.supervisor.pre_trade_check(positions)

                if not supervision_status.trading_allowed:
                    logger.warning(f"Trading blocked by supervision: {supervision_status.blocking_reasons}")
                    return

                if supervision_status.warnings:
                    for warning in supervision_status.warnings:
                        logger.warning(f"Supervision warning: {warning}")

                # Log time to force close
                if supervision_status.time_to_force_close is not None and supervision_status.time_to_force_close < 15:
                    logger.info(f"Force close in {supervision_status.time_to_force_close:.1f} minutes (urgency: {supervision_status.force_close_urgency})")

            # Fallback force close check (if supervision not available)
            elif self.risk_manager.should_force_close():
                if position:
                    self.order_manager.close_position(position, "force_close_eod", tracker=self.position_tracker)
                    logger.info("Force closing position at end of day")
                return

            # Check and update stops for existing position
            if position:
                position.current_price = current_price

                # Update trailing stop
                new_trailing = self.risk_manager.update_trailing_stop(position, current_price)
                if new_trailing and new_trailing != position.trailing_stop:
                    position.trailing_stop = new_trailing
                    logger.info(f"Trailing stop updated: ${new_trailing:.2f}")

                # Check stop conditions
                should_close, reason = self.risk_manager.check_stop_conditions(position, current_price)
                if should_close:
                    self.order_manager.close_position(position, reason, tracker=self.position_tracker)
                    logger.info(f"Position closed: {reason}")
                    return

            # Generate new signal
            should_generate = (
                self.last_signal_time is None or
                (datetime.now() - self.last_signal_time).total_seconds() >= self.signal_interval
            )

            if should_generate:
                df_1min = self.fetch_latest_data()

                if len(df_1min) > 0:
                    signal = self.signal_generator.generate_signal(df_1min, current_price)
                    self.last_signal_time = datetime.now()

                    logger.info(
                        f"Signal: {signal.signal_type.value}, "
                        f"Prob: {signal.probability:.3f}, "
                        f"Conf: {signal.confidence:.3f}"
                    )

                    # Validate and execute
                    if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                        is_valid, reason, quantity = self.risk_manager.validate_order(
                            signal, account, position
                        )

                        if is_valid:
                            self.order_manager.execute_signal(signal, quantity)
                        else:
                            logger.info(f"Order rejected: {reason}")

                    # Check if signal says to close existing position
                    elif signal.signal_type == SignalType.CLOSE and position:
                        self.order_manager.close_position(position, "signal_close", tracker=self.position_tracker)

            # Check pending orders
            self.order_manager.check_pending_orders()

        except Exception as e:
            logger.error(f"Error in trading loop: {e}")

    def run(self, interval_seconds: int = 10):
        """Run the main trading loop."""
        logger.info("=" * 60)
        logger.info("GIGA TRADER - Paper Trading Bot Started")
        logger.info("=" * 60)

        # Log account info
        account = self.client.get_account()
        logger.info(f"Account Equity: ${account['equity']:,.2f}")
        logger.info(f"Buying Power: ${account['buying_power']:,.2f}")
        logger.info(f"Trading Symbol: {TRADING_CONFIG['symbol']}")

        self.is_running = True
        self.risk_manager.peak_equity = account["equity"]

        try:
            while self.is_running:
                self.run_once()
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("Shutting down trading bot...")
            self.is_running = False

        finally:
            # Clean shutdown
            positions = self.client.get_all_positions()
            if positions:
                logger.info(f"Open positions at shutdown: {len(positions)}")
                for pos in positions:
                    logger.info(f"  {pos.symbol}: {pos.quantity} @ ${pos.entry_price:.2f}")

    def stop(self):
        """Stop the trading bot."""
        self.is_running = False
        logger.info("Trading bot stop requested")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. STATUS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
def print_status(bot: TradingBot):
    """Print current trading status."""
    print("\n" + "=" * 60)
    print("GIGA TRADER - STATUS")
    print("=" * 60)

    # Account
    account = bot.client.get_account()
    print(f"\n[ACCOUNT]")
    print(f"  Equity: ${account['equity']:,.2f}")
    print(f"  Buying Power: ${account['buying_power']:,.2f}")
    print(f"  Cash: ${account['cash']:,.2f}")

    # Position
    position = bot.client.get_position(TRADING_CONFIG["symbol"])
    print(f"\n[POSITION]")
    if position:
        print(f"  Symbol: {position.symbol}")
        print(f"  Side: {position.side}")
        print(f"  Quantity: {position.quantity}")
        print(f"  Entry Price: ${position.entry_price:.2f}")
        print(f"  Current Price: ${position.current_price:.2f}")
        print(f"  Unrealized P&L: ${position.unrealized_pnl:.2f} ({position.pnl_pct:.2%})")
    else:
        print("  No open position")

    # Risk
    print(f"\n[RISK MANAGER]")
    print(f"  Daily P&L: ${bot.risk_manager.daily_pnl:.2f}")
    print(f"  Daily Trades: {bot.risk_manager.daily_trades}")
    print(f"  Is Halted: {bot.risk_manager.is_halted}")
    if bot.risk_manager.halt_reason:
        print(f"  Halt Reason: {bot.risk_manager.halt_reason}")

    # Models
    print(f"\n[MODELS]")
    print(f"  Loaded: {list(bot.signal_generator.models.keys())}")

    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    """Main entry point for paper trading."""
    print("\n" + "=" * 60)
    print("GIGA TRADER - Paper Trading")
    print("=" * 60)

    if not ALPACA_AVAILABLE:
        print("\n[ERROR] Alpaca SDK not installed.")
        print("Run: pip install alpaca-py")
        return 1

    try:
        # Initialize bot
        bot = TradingBot()

        # Print initial status
        print_status(bot)

        # Run trading loop
        print("\nStarting trading loop (Ctrl+C to stop)...")
        bot.run(interval_seconds=10)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
GIGA TRADER - Alpaca Paper Trading: Trading Bot
=================================================
Main trading bot loop, status dashboard, and entry point.

Components:
  - setup_logging() function
  - TradingBot class
  - print_status() function
  - main() function
"""

import sys
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# Setup path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# Local imports
from src.phase_19_paper_trading.alpaca_client import (
    TRADING_CONFIG,
    SignalType,
    AlpacaPaperClient,
    ALPACA_AVAILABLE,
)
from src.phase_19_paper_trading.signal_generator import SignalGenerator
from src.phase_19_paper_trading.risk_management import RiskManager, OrderManager

# Alpaca imports
try:
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
except ImportError:
    pass

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


logger = logging.getLogger("GigaTrader")


# ═══════════════════════════════════════════════════════════════════════════════
# 4b. PAPER PERFORMANCE TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PredictionRecord:
    """A single model prediction with its outcome."""
    timestamp: str
    signal_type: str  # BUY, SELL, HOLD
    swing_probability: float
    timing_probability: float
    confidence: float
    entry_price: float
    # Filled in when position closes
    exit_price: float = 0.0
    actual_return: float = 0.0
    predicted_correct: bool = False
    closed_at: str = ""
    close_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class PaperPerformanceTracker:
    """
    Tracks real-world performance of model predictions during paper trading.

    Records every signal, compares predictions to actual outcomes, and
    persists cumulative statistics to logs/paper_performance.json for
    dashboard consumption.
    """

    def __init__(self, log_dir: Path = None):
        self.log_dir = log_dir or (project_root / "logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.perf_file = self.log_dir / "paper_performance.json"

        self.predictions: List[PredictionRecord] = []
        self.open_prediction: Optional[PredictionRecord] = None
        self._load()

    def _load(self):
        """Load existing performance data."""
        if self.perf_file.exists():
            try:
                with open(self.perf_file) as f:
                    data = json.load(f)
                    self.predictions = [
                        PredictionRecord(**p) for p in data.get("predictions", [])
                    ]
                    # Restore open prediction if any
                    open_pred = data.get("open_prediction")
                    if open_pred:
                        self.open_prediction = PredictionRecord(**open_pred)
            except Exception as e:
                logger.warning(f"Could not load paper performance: {e}")

    def _save(self):
        """Persist performance data to disk."""
        try:
            data = {
                "predictions": [p.to_dict() for p in self.predictions],
                "open_prediction": self.open_prediction.to_dict() if self.open_prediction else None,
                "summary": self.get_summary(),
                "updated_at": datetime.now().isoformat(),
            }
            self.perf_file.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.warning(f"Could not save paper performance: {e}")

    def record_signal(self, signal, entry_price: float):
        """Record a new trading signal/prediction."""
        pred = PredictionRecord(
            timestamp=datetime.now().isoformat(),
            signal_type=signal.signal_type.value,
            swing_probability=getattr(signal, "probability", 0.0),
            timing_probability=getattr(signal, "timing_probability", 0.0),
            confidence=getattr(signal, "confidence", 0.0),
            entry_price=entry_price,
        )
        self.open_prediction = pred
        self._save()

    def record_close(self, exit_price: float, reason: str = ""):
        """Record position close and evaluate prediction accuracy."""
        if not self.open_prediction:
            return

        pred = self.open_prediction
        pred.exit_price = exit_price
        pred.closed_at = datetime.now().isoformat()
        pred.close_reason = reason

        # Calculate actual return
        if pred.entry_price > 0:
            if pred.signal_type == "BUY":
                pred.actual_return = (exit_price - pred.entry_price) / pred.entry_price
                pred.predicted_correct = pred.actual_return > 0
            elif pred.signal_type == "SELL":
                pred.actual_return = (pred.entry_price - exit_price) / pred.entry_price
                pred.predicted_correct = pred.actual_return > 0

        self.predictions.append(pred)
        self.open_prediction = None
        self._save()

        logger.info(
            f"[PERF] {pred.signal_type} {'CORRECT' if pred.predicted_correct else 'WRONG'} "
            f"return={pred.actual_return:.4f} conf={pred.confidence:.3f}"
        )

    def get_summary(self) -> dict:
        """Get cumulative performance summary."""
        if not self.predictions:
            return {
                "total_trades": 0,
                "correct_predictions": 0,
                "accuracy": 0.0,
                "avg_return": 0.0,
                "total_return": 0.0,
                "avg_confidence": 0.0,
                "sharpe": 0.0,
                "win_rate_by_confidence": {},
            }

        closed = [p for p in self.predictions if p.exit_price > 0]
        if not closed:
            return {"total_trades": 0, "correct_predictions": 0, "accuracy": 0.0}

        correct = sum(1 for p in closed if p.predicted_correct)
        returns = [p.actual_return for p in closed]
        confidences = [p.confidence for p in closed]

        # Win rate by confidence bucket
        buckets = {}
        for p in closed:
            bucket = f"{int(p.confidence * 10) / 10:.1f}"
            if bucket not in buckets:
                buckets[bucket] = {"total": 0, "correct": 0}
            buckets[bucket]["total"] += 1
            if p.predicted_correct:
                buckets[bucket]["correct"] += 1

        win_rate_by_conf = {
            k: round(v["correct"] / v["total"], 3) if v["total"] > 0 else 0
            for k, v in sorted(buckets.items())
        }

        # Sharpe ratio
        ret_array = np.array(returns)
        sharpe = float(np.mean(ret_array) / np.std(ret_array) * np.sqrt(252)) if np.std(ret_array) > 0 else 0.0

        return {
            "total_trades": len(closed),
            "correct_predictions": correct,
            "accuracy": round(correct / len(closed), 4),
            "avg_return": round(float(np.mean(returns)), 6),
            "total_return": round(float(np.sum(returns)), 6),
            "avg_confidence": round(float(np.mean(confidences)), 4),
            "sharpe": round(sharpe, 3),
            "best_trade": round(float(max(returns)), 6),
            "worst_trade": round(float(min(returns)), 6),
            "win_rate_by_confidence": win_rate_by_conf,
        }


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

        # Initialize performance tracker (real-world model accuracy)
        self.performance_tracker = PaperPerformanceTracker()
        logger.info("Paper performance tracker initialized")

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
                    self.performance_tracker.record_close(current_price, "force_close_eod")
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
                    self.performance_tracker.record_close(current_price, reason)
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
                            # Record prediction for performance tracking
                            self.performance_tracker.record_signal(signal, current_price)
                        else:
                            logger.info(f"Order rejected: {reason}")

                    # Check if signal says to close existing position
                    elif signal.signal_type == SignalType.CLOSE and position:
                        self.order_manager.close_position(position, "signal_close", tracker=self.position_tracker)
                        # Record close for performance tracking
                        self.performance_tracker.record_close(current_price, "signal_close")

            # Check pending orders
            self.order_manager.check_pending_orders()

            # Write dashboard data every cycle
            self._write_dashboard_data(account, position, current_price)

        except Exception as e:
            logger.error(f"Error in trading loop: {e}")

    def _write_dashboard_data(self, account: dict, position, current_price: float):
        """Write current state to JSON in orchestrator-compatible format for dashboard."""
        try:
            logs_dir = project_root / "logs"
            logs_dir.mkdir(exist_ok=True)

            # Determine position string
            pos_str = "FLAT"
            pos_pnl = 0.0
            if position:
                side = getattr(position, "side", "long")
                pos_str = side.upper() if isinstance(side, str) else "LONG"
                pos_pnl = float(getattr(position, "unrealized_pnl", 0))

            # Calculate uptime
            if not hasattr(self, "_start_time"):
                self._start_time = datetime.now()
            uptime = (datetime.now() - self._start_time).total_seconds()

            # Build status in orchestrator format (what the dashboard JS expects)
            status = {
                "mode": "TRADING" if self.is_running else "IDLE",
                "last_update": datetime.now().isoformat(),
                "uptime_seconds": int(uptime),
                "trading": {
                    "active": self.is_running,
                    "position": pos_str,
                    "position_pnl": round(pos_pnl, 2),
                    "daily_pnl": round(self.risk_manager.daily_pnl, 2),
                    "trades_today": self.risk_manager.daily_trades,
                },
                "model": {
                    "loaded": self.signal_generator.models_loaded if hasattr(self.signal_generator, "models_loaded") else True,
                    "last_train": "",
                    "accuracy": getattr(self.signal_generator, "model_auc", 0.0),
                },
                "health": {
                    "status": "HEALTHY",
                    "consecutive_errors": 0,
                    "last_error": "",
                },
                "experiment_gates": {
                    "gates_passed": True,
                    "experiments_completed": 0,
                    "experiments_required": 0,
                    "models_above_threshold": 0,
                    "models_required": 0,
                    "best_model_auc": 0.0,
                },
                "components": {
                    "trading_bot": "RUNNING" if self.is_running else "STOPPED",
                    "signal_generator": "RUNNING" if self.is_running else "STOPPED",
                    "risk_manager": "RUNNING" if self.is_running else "STOPPED",
                    "training_engine": "STOPPED",
                    "experiment_engine": "STOPPED",
                    "monitor": "RUNNING",
                },
                # Extra fields for richer display
                "account": {
                    "equity": account.get("equity", 0),
                    "cash": account.get("cash", 0),
                    "buying_power": account.get("buying_power", 0),
                },
                "performance": self.performance_tracker.get_summary(),
                "current_price": current_price,
            }

            # Try to populate gate info
            try:
                from src.giga_orchestrator import ExperimentGateChecker
                checker = ExperimentGateChecker()
                trading_mode = "paper"
                passed, gate_status = checker.check_gates(trading_mode=trading_mode)
                status["experiment_gates"] = {
                    "gates_passed": passed,
                    "experiments_completed": gate_status.get("completed_experiments", 0),
                    "experiments_required": gate_status.get("min_experiments_required", 0),
                    "models_above_threshold": gate_status.get("models_above_threshold", 0),
                    "models_required": gate_status.get("min_models_required", 0),
                    "best_model_auc": gate_status.get("best_model_auc", 0.0),
                }
            except Exception:
                pass

            (logs_dir / "status.json").write_text(json.dumps(status, indent=2, default=str))
        except Exception as e:
            logger.debug(f"Could not write dashboard data: {e}")

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

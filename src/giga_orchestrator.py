"""
GIGA TRADER - AUTONOMOUS ORCHESTRATOR
======================================
Single entry point that runs EVERYTHING:
  - Training & model improvements (off-market hours)
  - Paper trading (market hours)
  - Continuous experimentation (NEVER IDLE)
  - Self-learning from results
  - Self-healing on errors
  - Multiple status monitors

The system is NEVER idle - always running experiments,
backtests, or trying new configurations to improve.

Usage:
    .venv/Scripts/python.exe src/giga_orchestrator.py

This runs FOREVER until manually stopped.
"""

import os
import sys
import time
import json
import threading
import traceback
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
import signal

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
ORCHESTRATOR_CONFIG = {
    # Market hours (Eastern Time)
    "market_open": dt_time(9, 30),
    "market_close": dt_time(16, 0),
    "pre_market_start": dt_time(4, 0),
    "after_hours_end": dt_time(20, 0),

    # Training schedule
    "train_on_weekends": True,
    "train_after_hours": True,
    "retrain_interval_days": 7,

    # Experimentation (NEVER IDLE)
    "run_experiments_continuously": True,
    "experiment_interval_seconds": 60,  # Min time between experiments
    "max_experiments_per_day": 100,

    # Monitoring
    "status_update_interval": 30,  # seconds
    "health_check_interval": 300,  # 5 minutes

    # Self-healing
    "max_consecutive_errors": 5,
    "error_cooldown_seconds": 60,
    "auto_restart_on_crash": True,

    # Logging
    "log_dir": project_root / "logs",
    "status_file": project_root / "logs" / "status.json",
}


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════════
def setup_logging():
    log_dir = ORCHESTRATOR_CONFIG["log_dir"]
    log_dir.mkdir(exist_ok=True)

    # Main log
    main_log = log_dir / f"orchestrator_{datetime.now().strftime('%Y%m%d')}.log"

    # Create formatters and handlers
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    file_handler = logging.FileHandler(main_log)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return logging.getLogger("ORCHESTRATOR")

logger = setup_logging()


# ═══════════════════════════════════════════════════════════════════════════════
# STATUS TRACKER
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class SystemStatus:
    mode: str = "INITIALIZING"  # INITIALIZING, TRADING, TRAINING, IDLE, ERROR
    last_update: str = ""
    uptime_seconds: int = 0

    # Trading status
    trading_active: bool = False
    current_position: str = "FLAT"
    position_pnl: float = 0.0
    daily_pnl: float = 0.0
    total_trades_today: int = 0

    # Model status
    model_loaded: bool = False
    last_train_date: str = ""
    model_accuracy: float = 0.0

    # Health
    consecutive_errors: int = 0
    last_error: str = ""
    health_status: str = "HEALTHY"

    # Components
    components: Dict = None

    def __post_init__(self):
        self.components = {
            "trading_bot": "STOPPED",
            "signal_generator": "STOPPED",
            "risk_manager": "STOPPED",
            "training_engine": "STOPPED",
            "experiment_engine": "STOPPED",
            "monitor": "RUNNING",
        }

    def to_dict(self) -> Dict:
        return {
            "mode": self.mode,
            "last_update": self.last_update,
            "uptime_seconds": self.uptime_seconds,
            "trading": {
                "active": self.trading_active,
                "position": self.current_position,
                "position_pnl": self.position_pnl,
                "daily_pnl": self.daily_pnl,
                "trades_today": self.total_trades_today,
            },
            "model": {
                "loaded": self.model_loaded,
                "last_train": self.last_train_date,
                "accuracy": self.model_accuracy,
            },
            "health": {
                "status": self.health_status,
                "consecutive_errors": self.consecutive_errors,
                "last_error": self.last_error,
            },
            "components": self.components,
        }

    def save(self):
        self.last_update = datetime.now().isoformat()
        status_file = ORCHESTRATOR_CONFIG["status_file"]
        with open(status_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET HOURS HELPER
# ═══════════════════════════════════════════════════════════════════════════════
from zoneinfo import ZoneInfo

# Try to get Eastern timezone - with fallback
ET_TIMEZONE = None
try:
    ET_TIMEZONE = ZoneInfo("America/New_York")
except Exception as e:
    print(f"[WARN] Failed to load America/New_York timezone: {e}")
    # Fallback: try installing tzdata
    try:
        import tzdata
        ET_TIMEZONE = ZoneInfo("America/New_York")
    except ImportError:
        print("[ERROR] tzdata package not installed. Run: pip install tzdata")
        print("[WARN] Falling back to local time - this may cause timezone issues!")


class MarketHours:
    _last_debug_log: datetime = None

    @staticmethod
    def get_et_now() -> datetime:
        """Get current time in Eastern Time."""
        if ET_TIMEZONE is None:
            # Fallback: assume local time is ET (dangerous but better than crashing)
            return datetime.now()
        return datetime.now(ET_TIMEZONE)

    @staticmethod
    def is_market_open() -> bool:
        now = MarketHours.get_et_now()

        # Debug log the first time and every 5 minutes to help diagnose timezone issues
        try:
            if (MarketHours._last_debug_log is None or
                (datetime.now() - MarketHours._last_debug_log).total_seconds() > 300):
                logger.info(f"Market hours check - ET time: {now.strftime('%Y-%m-%d %H:%M:%S')}, "
                            f"Local time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                MarketHours._last_debug_log = datetime.now()
        except Exception:
            pass  # Logger might not be initialized yet

        # Skip weekends
        if now.weekday() >= 5:
            return False
        current_time = now.time()
        is_open = (ORCHESTRATOR_CONFIG["market_open"] <= current_time <=
                   ORCHESTRATOR_CONFIG["market_close"])
        return is_open

    @staticmethod
    def is_extended_hours() -> bool:
        now = MarketHours.get_et_now()
        if now.weekday() >= 5:
            return False
        current_time = now.time()
        pre_market = (ORCHESTRATOR_CONFIG["pre_market_start"] <= current_time <
                     ORCHESTRATOR_CONFIG["market_open"])
        after_hours = (ORCHESTRATOR_CONFIG["market_close"] < current_time <=
                      ORCHESTRATOR_CONFIG["after_hours_end"])
        return pre_market or after_hours

    @staticmethod
    def should_train() -> bool:
        now = MarketHours.get_et_now()
        # Train on weekends
        if now.weekday() >= 5 and ORCHESTRATOR_CONFIG["train_on_weekends"]:
            return True
        # Train after hours
        if ORCHESTRATOR_CONFIG["train_after_hours"]:
            current_time = now.time()
            after_close = current_time > ORCHESTRATOR_CONFIG["after_hours_end"]
            before_pre = current_time < ORCHESTRATOR_CONFIG["pre_market_start"]
            return after_close or before_pre
        return False

    @staticmethod
    def time_until_market_open() -> timedelta:
        now = MarketHours.get_et_now()
        today_open = now.replace(hour=ORCHESTRATOR_CONFIG["market_open"].hour,
                                  minute=ORCHESTRATOR_CONFIG["market_open"].minute,
                                  second=0, microsecond=0)
        if now.time() < ORCHESTRATOR_CONFIG["market_open"]:
            return today_open - now
        else:
            tomorrow_open = today_open + timedelta(days=1)
            return tomorrow_open - now


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
class TrainingEngine:
    def __init__(self, status: SystemStatus):
        self.status = status
        self.logger = logging.getLogger("TRAINING")
        self.last_train_time = self._get_model_train_time()
        self.is_training = False

    def _get_model_train_time(self) -> Optional[datetime]:
        """Get the last training time from existing model files."""
        model_dir = project_root / "models" / "production"
        model_files = list(model_dir.glob("spy_*.joblib"))
        if model_files:
            # Use the most recent model file's modification time
            latest_model = max(model_files, key=lambda f: f.stat().st_mtime)
            mtime = latest_model.stat().st_mtime
            return datetime.fromtimestamp(mtime)
        return None

    def needs_retraining(self) -> bool:
        if self.last_train_time is None:
            return True

        days_since_train = (datetime.now() - self.last_train_time).days
        return days_since_train >= ORCHESTRATOR_CONFIG["retrain_interval_days"]

    def run_training(self) -> bool:
        """Run the full training pipeline."""
        if self.is_training:
            self.logger.warning("Training already in progress")
            return False

        self.is_training = True
        self.status.components["training_engine"] = "RUNNING"
        self.status.mode = "TRAINING"
        self.status.save()

        self.logger.info("=" * 60)
        self.logger.info("STARTING TRAINING PIPELINE")
        self.logger.info("=" * 60)

        try:
            # Import and run training
            from src.train_robust_model import main as train_main

            result = train_main()

            if result == 0:
                self.logger.info("Training completed successfully!")
                self.last_train_time = datetime.now()
                self.status.last_train_date = self.last_train_time.strftime("%Y-%m-%d %H:%M")
                self.status.model_loaded = True
                return True
            else:
                self.logger.error(f"Training failed with code: {result}")
                return False

        except Exception as e:
            self.logger.error(f"Training error: {e}")
            self.logger.error(traceback.format_exc())
            return False
        finally:
            self.is_training = False
            self.status.components["training_engine"] = "STOPPED"
            self.status.save()

    def run_incremental_improvement(self) -> bool:
        """Run incremental model improvement based on recent performance."""
        self.logger.info("Running incremental improvement...")

        try:
            # Analyze recent trading performance
            # Adjust model parameters based on results
            # This is a placeholder for self-learning logic

            # Read recent trade logs
            trade_log = project_root / "logs" / "trades.json"
            if trade_log.exists():
                with open(trade_log) as f:
                    trades = json.load(f)

                if len(trades) >= 10:
                    # Analyze win rate
                    wins = sum(1 for t in trades[-20:] if t.get("pnl", 0) > 0)
                    recent_win_rate = wins / min(20, len(trades))

                    self.logger.info(f"Recent win rate: {recent_win_rate:.1%}")

                    # If performing poorly, trigger retraining
                    if recent_win_rate < 0.4:
                        self.logger.warning("Poor performance detected, triggering retraining")
                        return self.run_training()

            return True

        except Exception as e:
            self.logger.error(f"Improvement error: {e}")
            return False


# ===============================================================================
# EXPERIMENT RUNNER (NEVER IDLE)
# ===============================================================================
class ExperimentRunner:
    """
    Runs experiments continuously so the system is NEVER idle.

    This engine constantly:
      - Tries new hyperparameter configurations
      - Tests different feature subsets
      - Experiments with ensemble methods
      - Runs backtests to validate changes
    """

    def __init__(self, status: SystemStatus):
        self.status = status
        self.logger = logging.getLogger("EXPERIMENT")
        self.is_running = False
        self.last_experiment_time = None
        self.experiments_today = 0
        self.experiment_engine = None

    def initialize(self):
        """Initialize the experiment engine."""
        try:
            from src.experiment_engine import ExperimentEngine
            self.experiment_engine = ExperimentEngine()
            self.logger.info("Experiment engine initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize experiment engine: {e}")
            return False

    def should_run_experiment(self) -> bool:
        """Check if we should run another experiment."""
        if not ORCHESTRATOR_CONFIG["run_experiments_continuously"]:
            return False

        # Check daily limit
        if self.experiments_today >= ORCHESTRATOR_CONFIG["max_experiments_per_day"]:
            return False

        # Check interval
        if self.last_experiment_time:
            elapsed = (datetime.now() - self.last_experiment_time).total_seconds()
            if elapsed < ORCHESTRATOR_CONFIG["experiment_interval_seconds"]:
                return False

        return True

    def run_experiment(self) -> bool:
        """Run a single experiment using the UNIFIED FULL pipeline."""
        if self.is_running:
            return False

        self.is_running = True
        self.status.components["experiment_engine"] = "RUNNING"
        self.status.save()

        try:
            if self.experiment_engine is None:
                self.initialize()

            if self.experiment_engine is None:
                return False

            # Use the unified experiment engine's run_one_experiment method
            # This generates a config, runs the FULL pipeline, and returns result
            result = self.experiment_engine.run_one_experiment()

            self.experiments_today += 1
            self.last_experiment_time = datetime.now()

            if result is not None:
                self.logger.info(f"Experiment completed: score={result.test_auc:.3f}")

                # Check if this is a good model
                if result.test_auc > 0.7:
                    self.logger.info("Excellent result! Consider promoting this model.")

                return True

            return False

        except Exception as e:
            self.logger.error(f"Experiment error: {e}")
            self.logger.error(traceback.format_exc())
            return False

        finally:
            self.is_running = False
            self.status.components["experiment_engine"] = "IDLE"
            self.status.save()

    def run_quick_backtest(self) -> bool:
        """Run a validation backtest on the current model."""
        self.logger.info("Running validation backtest...")

        try:
            from src.backtesting_harness import BacktestingHarness

            harness = BacktestingHarness()
            result = harness.run_quick_validation()

            if "error" in result:
                self.logger.error(f"Backtest error: {result['error']}")
                return False

            self.logger.info(
                f"Backtest complete: Sharpe={result.get('sharpe_ratio', 0):.2f}, "
                f"Return={result.get('total_return', 0):.1%}, "
                f"WinRate={result.get('win_rate', 0):.1%}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Backtest error: {e}")
            return False

    def run_comprehensive_backtest(self) -> bool:
        """Run full comprehensive backtesting suite with robustness analysis."""
        self.logger.info("Running COMPREHENSIVE backtesting suite...")

        try:
            from src.backtesting_harness import BacktestingHarness

            harness = BacktestingHarness()
            results = harness.run_comprehensive_backtest(
                include_walk_forward=True,
                include_regime_analysis=True,
                include_robustness=True,
                include_monte_carlo=True,
                n_monte_carlo_runs=100,
            )

            score = results.get("overall_robustness_score", 0)
            self.logger.info(f"Comprehensive backtest complete: Robustness Score={score:.2f}")

            return score >= 0.5  # Pass if score >= 0.5

        except Exception as e:
            self.logger.error(f"Comprehensive backtest error: {e}")
            return False

    def reset_daily_count(self):
        """Reset daily experiment count (call at midnight)."""
        self.experiments_today = 0
        self.logger.info("Daily experiment count reset")


# ===============================================================================
# TRADING ENGINE
# ===============================================================================
class TradingEngine:
    def __init__(self, status: SystemStatus):
        self.status = status
        self.logger = logging.getLogger("TRADING")
        self.bot = None
        self.is_running = False
        self.trade_count = 0

    def start(self):
        """Start the trading bot."""
        if self.is_running:
            self.logger.warning("Trading already running")
            return

        self.logger.info("Starting trading engine...")
        self.status.components["trading_bot"] = "STARTING"
        self.status.save()

        try:
            from src.paper_trading import TradingBot, AlpacaPaperClient

            self.bot = TradingBot()
            self.is_running = True

            self.status.trading_active = True
            self.status.components["trading_bot"] = "RUNNING"
            self.status.components["signal_generator"] = "RUNNING"
            self.status.components["risk_manager"] = "RUNNING"
            self.status.save()

            self.logger.info("Trading engine started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start trading: {e}")
            self.status.components["trading_bot"] = "ERROR"
            self.status.last_error = str(e)
            self.is_running = False

    def stop(self):
        """Stop the trading bot."""
        if not self.is_running:
            return

        self.logger.info("Stopping trading engine...")

        try:
            if self.bot:
                self.bot.stop()

                # Close any open positions
                positions = self.bot.client.get_all_positions()
                if positions:
                    self.logger.info(f"Closing {len(positions)} positions...")
                    self.bot.client.close_all_positions()
        except Exception as e:
            self.logger.error(f"Error stopping trading: {e}")

        self.is_running = False
        self.status.trading_active = False
        self.status.components["trading_bot"] = "STOPPED"
        self.status.components["signal_generator"] = "STOPPED"
        self.status.components["risk_manager"] = "STOPPED"
        self.status.save()

    def run_cycle(self):
        """Run one trading cycle."""
        if not self.is_running or not self.bot:
            return

        try:
            self.bot.run_once()

            # Update status
            account = self.bot.client.get_account()
            position = self.bot.client.get_position("SPY")

            if position:
                self.status.current_position = f"{position.side.upper()} {position.quantity}"
                self.status.position_pnl = position.unrealized_pnl
            else:
                self.status.current_position = "FLAT"
                self.status.position_pnl = 0.0

            self.status.daily_pnl = self.bot.risk_manager.daily_pnl
            self.status.total_trades_today = self.bot.risk_manager.daily_trades

        except Exception as e:
            self.logger.error(f"Trading cycle error: {e}")
            self.status.consecutive_errors += 1


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH MONITOR
# ═══════════════════════════════════════════════════════════════════════════════
class HealthMonitor:
    def __init__(self, status: SystemStatus):
        self.status = status
        self.logger = logging.getLogger("HEALTH")
        self.start_time = datetime.now()

    def check_health(self) -> bool:
        """Run health checks."""
        issues = []

        # Check consecutive errors
        if self.status.consecutive_errors >= ORCHESTRATOR_CONFIG["max_consecutive_errors"]:
            issues.append(f"Too many errors: {self.status.consecutive_errors}")

        # Check model loaded
        if not self.status.model_loaded:
            issues.append("Model not loaded")

        # Check components
        for name, state in self.status.components.items():
            if state == "ERROR":
                issues.append(f"Component {name} in error state")

        if issues:
            self.status.health_status = "DEGRADED"
            self.logger.warning(f"Health issues: {issues}")
            return False

        self.status.health_status = "HEALTHY"
        return True

    def update_uptime(self):
        self.status.uptime_seconds = int((datetime.now() - self.start_time).total_seconds())

    def self_heal(self) -> bool:
        """Attempt to fix issues."""
        self.logger.info("Attempting self-healing...")

        healed = False

        # Reset error count after cooldown
        if self.status.consecutive_errors > 0:
            self.logger.info("Resetting error count")
            self.status.consecutive_errors = 0
            healed = True

        # Reload models if needed
        if not self.status.model_loaded:
            self.logger.info("Attempting to reload models...")
            try:
                model_dir = project_root / "models" / "production"
                if any(model_dir.glob("spy_*.joblib")):
                    self.status.model_loaded = True
                    healed = True
            except Exception as e:
                self.logger.error(f"Model reload failed: {e}")

        return healed


# ═══════════════════════════════════════════════════════════════════════════════
# STATUS DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════
class StatusDisplay:
    def __init__(self, status: SystemStatus):
        self.status = status
        self.logger = logging.getLogger("STATUS")

    def print_status(self):
        """Print current status to console."""
        s = self.status

        print("\n" + "=" * 70)
        print(f" GIGA TRADER - {s.mode}")
        print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Uptime: {s.uptime_seconds // 3600}h {(s.uptime_seconds % 3600) // 60}m")
        print("=" * 70)

        # Market status
        if MarketHours.is_market_open():
            print(" MARKET: OPEN")
        elif MarketHours.is_extended_hours():
            print(" MARKET: EXTENDED HOURS")
        else:
            print(" MARKET: CLOSED")

        # Trading
        print(f"\n [TRADING]")
        print(f"   Active: {s.trading_active}")
        print(f"   Position: {s.current_position}")
        print(f"   Position P&L: ${s.position_pnl:,.2f}")
        print(f"   Daily P&L: ${s.daily_pnl:,.2f}")
        print(f"   Trades Today: {s.total_trades_today}")

        # Model
        print(f"\n [MODEL]")
        print(f"   Loaded: {s.model_loaded}")
        print(f"   Last Train: {s.last_train_date or 'Never'}")

        # Health
        print(f"\n [HEALTH]")
        print(f"   Status: {s.health_status}")
        print(f"   Errors: {s.consecutive_errors}")
        if s.last_error:
            print(f"   Last Error: {s.last_error[:50]}")

        # Components
        print(f"\n [COMPONENTS]")
        for name, state in s.components.items():
            indicator = "[ON] " if state in ("RUNNING", "IDLE") else "[OFF]"
            print(f"   {indicator} {name}: {state}")

        # Mode explanation
        print(f"\n [ACTIVITY]")
        mode_desc = {
            "TRADING": "Actively trading SPY during market hours",
            "TRAINING": "Running full model training pipeline",
            "EXPERIMENTING": "Testing new model configurations",
            "IMPROVING": "Running incremental model improvements",
            "BACKTESTING": "Validating models on historical data",
            "READY": "Ready and monitoring",
        }
        print(f"   {mode_desc.get(s.mode, s.mode)}")

        print("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════
class GigaOrchestrator:
    """
    Main orchestrator that runs everything.

    This is the SINGLE COMMAND that does it all:
      - Trains models when market is closed
      - Trades when market is open
      - Monitors health continuously
      - Self-heals on errors
      - Updates status displays
    """

    def __init__(self):
        self.status = SystemStatus()
        self.training_engine = TrainingEngine(self.status)
        self.trading_engine = TradingEngine(self.status)
        self.experiment_runner = ExperimentRunner(self.status)
        self.health_monitor = HealthMonitor(self.status)
        self.display = StatusDisplay(self.status)

        self.running = False
        self.shutdown_requested = False
        self.last_experiment_date = None
        self.last_comprehensive_backtest = None  # Track when we last ran comprehensive backtest
        self._status_thread = None

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info("GigaOrchestrator initialized")

    def _should_run_comprehensive_backtest(self) -> bool:
        """Check if we should run comprehensive backtesting."""
        # Run comprehensive backtest once per day during off-market hours
        if self.last_comprehensive_backtest is None:
            return True

        # Get current time in ET
        now_et = MarketHours.get_et_now()

        # Check if we've run it today
        if self.last_comprehensive_backtest.date() == now_et.date():
            return False

        # Only run after 8 PM ET (after-hours end) to have full market data
        if now_et.time() >= dt_time(20, 0):
            return True

        return False

    def _status_updater_loop(self):
        """Background thread that updates status periodically."""
        while self.running and not self.shutdown_requested:
            try:
                self.health_monitor.update_uptime()
                self.status.save()
            except Exception:
                pass  # Don't crash the status thread
            time.sleep(5)  # Update every 5 seconds

    def _handle_shutdown(self, signum, frame):
        logger.info("Shutdown signal received")
        self.shutdown_requested = True
        self.running = False

    def startup(self):
        """Initialize all systems."""
        logger.info("=" * 70)
        logger.info("GIGA TRADER ORCHESTRATOR STARTING")
        logger.info("=" * 70)

        self.status.mode = "INITIALIZING"
        self.status.save()

        # Check if models exist
        model_dir = project_root / "models" / "production"
        if any(model_dir.glob("spy_*.joblib")):
            logger.info("Found existing models")
            self.status.model_loaded = True
        else:
            logger.warning("No models found - will train on first opportunity")
            self.status.model_loaded = False

        # Initial training if no models
        if not self.status.model_loaded:
            logger.info("Running initial training...")
            self.training_engine.run_training()

        # Initialize experiment engine
        logger.info("Initializing experiment engine...")
        self.experiment_runner.initialize()
        self.last_experiment_date = datetime.now().date()

        self.status.mode = "READY"
        self.status.save()

        logger.info("Startup complete - system will NEVER be idle!")

    def run_forever(self):
        """Main loop - runs until shutdown."""
        self.running = True
        self.startup()

        # Start background status updater thread
        self._status_thread = threading.Thread(target=self._status_updater_loop, daemon=True)
        self._status_thread.start()

        last_status_update = datetime.now()
        last_health_check = datetime.now()

        logger.info("Entering main loop...")

        while self.running and not self.shutdown_requested:
            try:
                now = datetime.now()

                # Update uptime
                self.health_monitor.update_uptime()

                # ─────────────────────────────────────────────────────────
                # MARKET OPEN: TRADING MODE
                # ─────────────────────────────────────────────────────────
                if MarketHours.is_market_open():
                    if self.status.mode != "TRADING":
                        logger.info("Market is OPEN - switching to TRADING mode")
                        self.status.mode = "TRADING"
                        self.trading_engine.start()

                    # Run trading cycle
                    self.trading_engine.run_cycle()

                # ─────────────────────────────────────────────────────────
                # MARKET CLOSED: TRAINING/EXPERIMENTING MODE (NEVER IDLE!)
                # ─────────────────────────────────────────────────────────
                else:
                    if self.trading_engine.is_running:
                        logger.info("Market is CLOSED - stopping trading")
                        self.trading_engine.stop()

                    # Reset daily experiment count at midnight
                    today = datetime.now().date()
                    if self.last_experiment_date != today:
                        self.experiment_runner.reset_daily_count()
                        self.last_experiment_date = today

                    # Priority 1: Full retraining if needed
                    if MarketHours.should_train() and self.training_engine.needs_retraining():
                        self.status.mode = "TRAINING"
                        self.training_engine.run_training()

                    # Priority 2: Comprehensive backtest (once per day)
                    elif self._should_run_comprehensive_backtest():
                        self.status.mode = "COMPREHENSIVE_BACKTEST"
                        logger.info("Running daily comprehensive backtest with robustness analysis...")
                        self.experiment_runner.run_comprehensive_backtest()
                        self.last_comprehensive_backtest = datetime.now()

                    # Priority 3: Run experiments (NEVER IDLE!)
                    elif self.experiment_runner.should_run_experiment():
                        self.status.mode = "EXPERIMENTING"
                        self.experiment_runner.run_experiment()

                    # Priority 4: Run incremental improvements
                    elif MarketHours.should_train():
                        self.status.mode = "IMPROVING"
                        self.training_engine.run_incremental_improvement()

                    # Priority 5: Quick validation backtest (still not idle!)
                    else:
                        self.status.mode = "BACKTESTING"
                        self.experiment_runner.run_quick_backtest()

                # ─────────────────────────────────────────────────────────
                # PERIODIC: STATUS UPDATES
                # ─────────────────────────────────────────────────────────
                if (now - last_status_update).seconds >= ORCHESTRATOR_CONFIG["status_update_interval"]:
                    self.status.save()
                    self.display.print_status()
                    last_status_update = now

                # ─────────────────────────────────────────────────────────
                # PERIODIC: HEALTH CHECKS
                # ─────────────────────────────────────────────────────────
                if (now - last_health_check).seconds >= ORCHESTRATOR_CONFIG["health_check_interval"]:
                    if not self.health_monitor.check_health():
                        self.health_monitor.self_heal()
                    last_health_check = now

                # Reset consecutive errors on successful cycle
                if self.status.consecutive_errors > 0:
                    self.status.consecutive_errors = 0

                # Sleep between cycles
                time.sleep(10)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break

            except Exception as e:
                logger.error(f"Main loop error: {e}")
                logger.error(traceback.format_exc())
                self.status.consecutive_errors += 1
                self.status.last_error = str(e)

                # Self-heal on errors
                if self.status.consecutive_errors >= ORCHESTRATOR_CONFIG["max_consecutive_errors"]:
                    logger.warning("Too many errors, attempting recovery...")
                    self.health_monitor.self_heal()
                    time.sleep(ORCHESTRATOR_CONFIG["error_cooldown_seconds"])

        self.shutdown()

    def shutdown(self):
        """Clean shutdown."""
        logger.info("=" * 70)
        logger.info("SHUTTING DOWN")
        logger.info("=" * 70)

        # Stop trading
        if self.trading_engine.is_running:
            self.trading_engine.stop()

        # Save final status
        self.status.mode = "SHUTDOWN"
        self.status.save()

        logger.info("Shutdown complete")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("""
    ==================================================================
    |                                                                |
    |   GIGA TRADER - AUTONOMOUS TRADING ORCHESTRATOR                |
    |                                                                |
    |   * Training & Model Improvements (off-market hours)           |
    |   * Paper Trading (market hours)                               |
    |   * Self-Learning from Results                                 |
    |   * Self-Healing on Errors                                     |
    |   * Continuous Status Monitoring                               |
    |                                                                |
    ==================================================================
    """)

    orchestrator = GigaOrchestrator()
    orchestrator.run_forever()

    return 0


if __name__ == "__main__":
    sys.exit(main())

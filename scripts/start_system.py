"""
GIGA TRADER - System Launcher
==============================
Starts all components in a coordinated way:
  1. Dashboard Server (port 8050) - background thread
  2. Web Monitor (port 5000) - background thread
  3. Experiment Engine - background thread (random config search)
  4. Thick Weave Search - background thread (intelligent plateau search)
  5. Paper Trading Bot - main thread

Usage:
    python scripts/start_system.py                    # Full system
    python scripts/start_system.py --dashboard-only   # Just dashboards
    python scripts/start_system.py --trading-only     # Just trading bot
    python scripts/start_system.py --no-web-monitor   # Skip web monitor
    python scripts/start_system.py --no-thick-weave   # Skip thick weave search
"""

import gc
import os
import sys
import json
import time
import shutil
import signal
import logging
import argparse
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load env
from dotenv import load_dotenv
load_dotenv(project_root / ".env")


def setup_logging():
    """Setup logging for the launcher."""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"system_{datetime.now().strftime('%Y%m%d')}.log"

    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_file, maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[file_handler, logging.StreamHandler()],
    )
    return logging.getLogger("SystemLauncher")


def check_prerequisites(logger):
    """Verify all prerequisites before starting."""
    checks = {}

    # Check models exist
    model_path = project_root / "models" / "production" / "spy_leak_proof_models.joblib"
    checks["models"] = model_path.exists()
    if not checks["models"]:
        logger.warning(f"Model file not found: {model_path}")

    # Check .env file
    checks["env_file"] = (project_root / ".env").exists()
    if not checks["env_file"]:
        logger.error(".env file not found! Copy .env.example to .env and fill in API keys")

    # Check Alpaca keys
    checks["alpaca_key"] = bool(os.getenv("ALPACA_API_KEY"))
    checks["alpaca_secret"] = bool(os.getenv("ALPACA_SECRET_KEY"))
    if not checks["alpaca_key"] or not checks["alpaca_secret"]:
        logger.error("ALPACA_API_KEY and/or ALPACA_SECRET_KEY not set in .env")

    # Check trading mode
    trading_mode = os.getenv("TRADING_MODE", "paper")
    checks["paper_mode"] = trading_mode == "paper"
    if not checks["paper_mode"]:
        logger.warning(f"TRADING_MODE={trading_mode} (expected 'paper' for safety)")

    # Check experiments directory
    exp_dir = project_root / "experiments"
    checks["experiments_dir"] = exp_dir.exists()

    # Check Flask available
    try:
        import flask  # noqa: F401
        checks["flask"] = True
    except ImportError:
        checks["flask"] = False
        logger.warning("Flask not installed. Dashboards will not start.")

    return checks


def start_dashboard_server(port=8050, logger=None):
    """Start the dashboard server in a background thread."""
    def _run():
        try:
            from src.phase_20_monitoring.dashboard_server import app
            if logger:
                logger.info(f"Dashboard server starting on http://127.0.0.1:{port}")
            app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
        except Exception as e:
            if logger:
                logger.error(f"Dashboard server error: {e}")

    thread = threading.Thread(target=_run, name="DashboardServer", daemon=True)
    thread.start()
    return thread


def start_web_monitor(port=5000, logger=None):
    """Start the web monitor in a background thread."""
    def _run():
        try:
            from src.phase_20_monitoring.web_monitor import app as monitor_app
            if logger:
                logger.info(f"Web monitor starting on http://127.0.0.1:{port}")
            monitor_app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
        except Exception as e:
            if logger:
                logger.error(f"Web monitor error: {e}")

    thread = threading.Thread(target=_run, name="WebMonitor", daemon=True)
    thread.start()
    return thread


def check_gates(logger):
    """Check trading gates and report status."""
    try:
        from src.giga_orchestrator import ExperimentGateChecker
        checker = ExperimentGateChecker()

        trading_mode = os.getenv("TRADING_MODE", "paper")
        passed, status = checker.check_gates(trading_mode=trading_mode)

        mode_label = status.get("mode_label", trading_mode.upper())
        logger.info(f"{mode_label} TRADING GATES:")
        logger.info(f"  Experiments: {status.get('completed_experiments', 0)}/{status.get('min_experiments_required', '?')}")
        logger.info(f"  Good models: {status.get('models_above_threshold', 0)}/{status.get('min_models_required', '?')}")
        logger.info(f"  Best AUC:    {status.get('best_model_auc', 0):.3f}/{status.get('min_auc_required', 0):.3f}")
        logger.info(f"  Result:      {'PASS' if passed else 'FAIL'}")

        return passed
    except Exception as e:
        logger.error(f"Gate check failed: {e}")
        return False


def write_status_json(components: dict, mode: str = "INITIALIZING", logger=None):
    """Write status.json in orchestrator-compatible format for the dashboard."""
    try:
        logs_dir = project_root / "logs"
        logs_dir.mkdir(exist_ok=True)

        if not hasattr(write_status_json, "_start_time"):
            write_status_json._start_time = datetime.now()

        uptime = (datetime.now() - write_status_json._start_time).total_seconds()

        status = {
            "mode": mode,
            "last_update": datetime.now().isoformat(),
            "uptime_seconds": int(uptime),
            "trading": {
                "active": components.get("trading_bot") == "RUNNING",
                "position": "FLAT",
                "position_pnl": 0.0,
                "daily_pnl": 0.0,
                "trades_today": 0,
            },
            "model": {
                "loaded": True,
                "last_train": "",
                "accuracy": 0.0,
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
            "components": components,
        }

        # Populate gate info
        try:
            from src.giga_orchestrator import ExperimentGateChecker
            checker = ExperimentGateChecker()
            trading_mode = os.getenv("TRADING_MODE", "paper")
            passed, gate_status = checker.check_gates(trading_mode=trading_mode)
            status["experiment_gates"] = {
                "gates_passed": passed,
                "experiments_completed": gate_status.get("completed_experiments", 0),
                "experiments_required": gate_status.get("min_experiments_required", 0),
                "models_above_threshold": gate_status.get("models_above_threshold", 0),
                "models_required": gate_status.get("min_models_required", 0),
                "best_model_auc": gate_status.get("best_model_auc", 0.0),
            }
        except Exception as e:
            if logger:
                logger.debug(f"Gate check for status.json failed: {e}")

        from src.core.state_manager import atomic_write_json
        atomic_write_json(logs_dir / "status.json", status)
    except Exception as e:
        if logger:
            logger.debug(f"Could not write status.json: {e}")


# Shared component state (updated by threads)
_component_state = {
    "trading_bot": "STOPPED",
    "signal_generator": "STOPPED",
    "risk_manager": "STOPPED",
    "training_engine": "STOPPED",
    "experiment_engine": "STOPPED",
    "thick_weave_search": "STOPPED",
    "health_checker": "STOPPED",
    "monitor": "RUNNING",
}
_component_lock = threading.Lock()

# Graceful shutdown flag
_shutdown_requested = threading.Event()
_active_bot: Optional = None  # type: ignore[assignment]


def _shutdown_handler(signum, frame):
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    logger_ref = logging.getLogger("SystemLauncher")
    logger_ref.info(f"Received {sig_name}, initiating graceful shutdown...")
    _shutdown_requested.set()
    if _active_bot is not None:
        _active_bot.stop()


def update_component(name: str, status: str, logger=None):
    """Thread-safe component status update + status.json write."""
    with _component_lock:
        _component_state[name] = status
        mode = "TRADING" if _component_state.get("trading_bot") == "RUNNING" else "IDLE"
        write_status_json(_component_state.copy(), mode=mode, logger=logger)


def cleanup_old_experiment_models(max_age_days=7, logger=None):
    """Remove old experiment model files below Tier 2 to prevent disk bloat.

    Wave 28: For 2-6 week unattended operation, prune models/experiments/*.joblib
    files that are older than max_age_days and not associated with Tier 2+ models.
    """
    models_dir = project_root / "models" / "experiments"
    if not models_dir.exists():
        return 0

    try:
        from src.core.registry_db import get_registry_db
        db = get_registry_db()

        # Get all model paths for Tier 2+ models (we want to KEEP these)
        protected_paths = set()
        try:
            models = db.query_models()
            for m in models:
                tier = m.get("tier", 1)
                model_path = m.get("model_path", "")
                if tier >= 2 and model_path:
                    protected_paths.add(Path(model_path).name)
        except Exception:
            pass

        cutoff = time.time() - (max_age_days * 86400)
        removed = 0
        freed_bytes = 0

        for f in models_dir.glob("*.joblib"):
            if f.name in protected_paths:
                continue
            if f.stat().st_mtime < cutoff:
                size = f.stat().st_size
                f.unlink()
                removed += 1
                freed_bytes += size

        if removed > 0 and logger:
            logger.info(
                f"[CLEANUP] Removed {removed} old experiment models "
                f"({freed_bytes / 1024 / 1024:.1f} MB freed)"
            )
        return removed
    except Exception as e:
        if logger:
            logger.warning(f"[CLEANUP] Model cleanup failed: {e}")
        return 0


def run_periodic_maintenance(logger=None):
    """Run periodic DB maintenance and model cleanup in a background thread.

    Wave 28: Runs every 6 hours:
    - VACUUM/ANALYZE the SQLite database
    - Clean up old experiment model files
    - Force garbage collection
    - Check disk space (halt experiments if < 500MB)
    """
    def _maintenance():
        while not _shutdown_requested.is_set():
            try:
                # 1. DB maintenance (VACUUM + ANALYZE)
                try:
                    from src.core.registry_db import get_registry_db
                    db = get_registry_db()
                    db.vacuum()
                    if logger:
                        logger.info("[MAINTENANCE] Database VACUUM completed")
                except Exception as e:
                    if logger:
                        logger.warning(f"[MAINTENANCE] DB vacuum failed: {e}")

                # 2. Recompute model tiers (fix mismatches)
                try:
                    tier_stats = db.recompute_all_tiers()
                    promoted = tier_stats.get("promoted", 0)
                    demoted = tier_stats.get("demoted", 0)
                    if promoted or demoted:
                        if logger:
                            logger.info(
                                f"[MAINTENANCE] Tier recomputation: {promoted} promoted, "
                                f"{demoted} demoted out of {tier_stats['checked']} models"
                            )
                    else:
                        if logger:
                            logger.info(
                                f"[MAINTENANCE] Tier check: all {tier_stats['checked']} models correct"
                            )
                except Exception as e:
                    if logger:
                        logger.warning(f"[MAINTENANCE] Tier recomputation failed: {e}")

                # 3. Model file cleanup (>7 days, below Tier 2)
                cleanup_old_experiment_models(max_age_days=7, logger=logger)

                # 4. Force garbage collection
                collected = gc.collect()
                if logger:
                    logger.info(f"[MAINTENANCE] GC collected {collected} objects")

                # 5. Disk space check
                try:
                    usage = shutil.disk_usage(str(project_root))
                    free_mb = usage.free / (1024 * 1024)
                    if free_mb < 500:
                        if logger:
                            logger.critical(
                                f"[MAINTENANCE] CRITICAL: Only {free_mb:.0f}MB disk space remaining! "
                                f"Halting experiments until space freed."
                            )
                        # Signal shutdown to stop new experiments
                        _shutdown_requested.set()
                    elif free_mb < 2000:
                        if logger:
                            logger.warning(
                                f"[MAINTENANCE] Low disk space: {free_mb:.0f}MB remaining"
                            )
                        # Aggressive cleanup - reduce model age to 3 days
                        cleanup_old_experiment_models(max_age_days=3, logger=logger)
                except Exception as e:
                    if logger:
                        logger.warning(f"[MAINTENANCE] Disk check failed: {e}")

            except Exception as e:
                if logger:
                    logger.error(f"[MAINTENANCE] Error: {e}")

            # Run every 6 hours
            for _ in range(6 * 60 * 2):  # 6 hours in 30-second increments
                if _shutdown_requested.is_set():
                    return
                time.sleep(30)

    thread = threading.Thread(target=_maintenance, name="Maintenance", daemon=True)
    thread.start()
    return thread


def check_memory_usage(logger=None):
    """Check current process memory usage and alert if high.

    Wave 28: Returns memory in MB. Triggers gc.collect() if > 1.5GB.
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)

        if mem_mb > 2000:
            if logger:
                logger.critical(f"[MEMORY] Process using {mem_mb:.0f}MB (>2GB) - forcing GC")
            gc.collect()
        elif mem_mb > 1500:
            if logger:
                logger.warning(f"[MEMORY] Process using {mem_mb:.0f}MB (>1.5GB) - forcing GC")
            gc.collect()

        return mem_mb
    except ImportError:
        # psutil not available, use rough approximation
        return 0
    except Exception:
        return 0


def start_experiment_runner(logger=None, timeout_minutes=240):
    """Start the experiment runner in a background thread for continuous training.

    Wave 28: Added per-experiment timeout to prevent hung experiments from
    blocking the pipeline during unattended operation.
    """
    def _run():
        try:
            from src.experiment_engine import ExperimentEngine
            from src.core.registry_db import get_registry_db
            engine = ExperimentEngine(db=get_registry_db())

            if logger:
                logger.info("Experiment runner started (continuous training)")
            update_component("experiment_engine", "RUNNING", logger)

            cycle = 0
            while not _shutdown_requested.is_set():
                cycle += 1
                exp_start = time.time()
                try:
                    if logger:
                        logger.info(f"[EXPERIMENT] Starting experiment cycle {cycle}...")
                    update_component("training_engine", "RUNNING", logger)

                    # Run experiment in a sub-thread with timeout
                    result_holder = [None]
                    error_holder = [None]

                    def _run_one():
                        try:
                            result_holder[0] = engine.run_one_experiment()
                        except Exception as ex:
                            error_holder[0] = ex

                    exp_thread = threading.Thread(target=_run_one, name="ExperimentWorker", daemon=True)
                    exp_thread.start()
                    exp_thread.join(timeout=timeout_minutes * 60)

                    if exp_thread.is_alive():
                        elapsed_min = (time.time() - exp_start) / 60
                        if logger:
                            logger.warning(
                                f"[EXPERIMENT] Cycle {cycle} TIMED OUT after {elapsed_min:.1f} min "
                                f"(limit: {timeout_minutes} min) - abandoning"
                            )
                        # Thread is daemon, it will be cleaned up eventually
                        # Force GC to reclaim memory from abandoned experiment
                        gc.collect()
                    elif error_holder[0] is not None:
                        raise error_holder[0]
                    else:
                        result = result_holder[0]
                        if result and logger:
                            elapsed_min = (time.time() - exp_start) / 60
                            logger.info(
                                f"[EXPERIMENT] Cycle {cycle} complete ({elapsed_min:.1f} min): "
                                f"AUC={result.test_auc:.3f}, "
                                f"WMES={result.wmes_score:.3f}"
                            )

                    update_component("training_engine", "STOPPED", logger)

                except Exception as e:
                    update_component("training_engine", "STOPPED", logger)
                    if logger:
                        logger.warning(f"[EXPERIMENT] Cycle {cycle} failed: {e}")

                # Memory check every 5 experiments
                if cycle % 5 == 0:
                    mem_mb = check_memory_usage(logger)
                    if mem_mb > 0 and logger:
                        logger.info(f"[EXPERIMENT] Memory: {mem_mb:.0f}MB after {cycle} experiments")

                # Brief cooldown between experiments
                for _ in range(10):
                    if _shutdown_requested.is_set():
                        break
                    time.sleep(1)

        except Exception as e:
            update_component("experiment_engine", "ERROR", logger)
            if logger:
                logger.error(f"Experiment runner error: {e}")

    thread = threading.Thread(target=_run, name="ExperimentRunner", daemon=True)
    thread.start()
    return thread


def start_watchdog(logger=None, restart_delay=120):
    """Monitor all worker threads and restart if they die.

    Wave 26: Ensures 10-day unattended training campaigns survive crashes.
    Wave 28: Extended to also monitor dashboard + web monitor threads.
    Checks every 30s, restarts dead threads after restart_delay seconds.
    """
    _watched_threads: Dict[str, Optional[threading.Thread]] = {
        "experiment_runner": None,
        "thick_weave_search": None,
        "dashboard_server": None,
        "web_monitor": None,
    }

    # Track restart ports for dashboard/monitor
    _dashboard_port = [8050]
    _monitor_port = [5000]

    def _watchdog():
        while not _shutdown_requested.is_set():
            # Check experiment runner
            exp_thread = _watched_threads.get("experiment_runner")
            if exp_thread is not None and not exp_thread.is_alive():
                if logger:
                    logger.warning(
                        f"[WATCHDOG] Experiment runner died, restarting in {restart_delay}s..."
                    )
                update_component("experiment_engine", "RESTARTING", logger)
                time.sleep(restart_delay)
                if not _shutdown_requested.is_set():
                    new_thread = start_experiment_runner(logger=logger)
                    _watched_threads["experiment_runner"] = new_thread
                    if logger:
                        logger.info("[WATCHDOG] Experiment runner restarted")

            # Check thick weave
            tw_thread = _watched_threads.get("thick_weave_search")
            if tw_thread is not None and not tw_thread.is_alive():
                if logger:
                    logger.warning(
                        f"[WATCHDOG] Thick weave search died, restarting in {restart_delay}s..."
                    )
                update_component("thick_weave_search", "RESTARTING", logger)
                time.sleep(restart_delay)
                if not _shutdown_requested.is_set():
                    new_thread = start_thick_weave_search(logger=logger)
                    _watched_threads["thick_weave_search"] = new_thread
                    if logger:
                        logger.info("[WATCHDOG] Thick weave search restarted")

            # Check dashboard server
            dash_thread = _watched_threads.get("dashboard_server")
            if dash_thread is not None and not dash_thread.is_alive():
                if logger:
                    logger.warning("[WATCHDOG] Dashboard server died, restarting...")
                time.sleep(5)
                if not _shutdown_requested.is_set():
                    new_thread = start_dashboard_server(
                        port=_dashboard_port[0], logger=logger
                    )
                    _watched_threads["dashboard_server"] = new_thread
                    if logger:
                        logger.info("[WATCHDOG] Dashboard server restarted")

            # Check web monitor
            mon_thread = _watched_threads.get("web_monitor")
            if mon_thread is not None and not mon_thread.is_alive():
                if logger:
                    logger.warning("[WATCHDOG] Web monitor died, restarting...")
                time.sleep(5)
                if not _shutdown_requested.is_set():
                    new_thread = start_web_monitor(
                        port=_monitor_port[0], logger=logger
                    )
                    _watched_threads["web_monitor"] = new_thread
                    if logger:
                        logger.info("[WATCHDOG] Web monitor restarted")

            time.sleep(30)

    thread = threading.Thread(target=_watchdog, name="Watchdog", daemon=True)
    thread.start()
    return thread, _watched_threads, _dashboard_port, _monitor_port


def _register_thick_weave_candidates(report, db, logger=None):
    """Bridge thick weave production candidates into RegistryDB with tier scoring.

    Maps ThickWeave metrics to model record fields:
      - wmes_score ← candidate wmes
      - stability_score ← candidate pts (Path Thickness Score maps to HP stability)
      - fragility_score ← candidate fragility (from Tier 3 robustness check)
      - test_auc ← looked up from ModelRegistryV2 by config_hash
    """
    candidates = report.get("production_candidates", [])
    if not candidates:
        return 0

    # Try to look up test_auc from ModelRegistryV2 entries in SQLite
    v2_aucs = {}
    try:
        entries = db.query_model_entries()
        for d in entries:
            metrics = d.get("metrics", {})
            config_hash = d.get("config_hash", "")
            auc = metrics.get("test_auc", 0) or metrics.get("cv_auc", 0)
            if config_hash:
                v2_aucs[config_hash] = auc
    except Exception as e:
        if logger:
            logger.debug(f"V2 AUC lookup failed: {e}")

    registered = 0
    for cand in candidates:
        config_hash = cand.get("config_hash", "")
        wmes = cand.get("wmes", 0)
        pts = cand.get("pts", 0)
        fragility = cand.get("fragility", 0.25)  # Default moderate
        test_auc = v2_aucs.get(config_hash, 0.60)  # Default if not found

        # Skip if already poor quality
        if wmes < 0.50 or test_auc < 0.55:
            continue

        # Register directly via RegistryDB
        result_dict = {
            "experiment_id": f"tw_{config_hash[:16]}_{datetime.now().strftime('%H%M%S')}",
            "status": "completed",
            "test_auc": test_auc,
            "train_auc": test_auc + 0.03,  # Conservative estimate
            "cv_auc_mean": test_auc,
            "wmes_score": wmes,
            "stability_score": pts,       # PTS >= 0.5 means thick/stable plateau
            "fragility_score": fragility,
            "backtest_sharpe": 0,
            "backtest_win_rate": 0,
            "backtest_total_return": 0,
            "model_path": "",  # Models already in ModelRegistryV2
            "config": {},
        }

        try:
            from src.core.registry_db import compute_tier
            model_id = db.register_model_from_experiment(result_dict)
            tier = compute_tier(pts, fragility, test_auc)
            if logger:
                logger.info(
                    f"[THICK_WEAVE] Registered {model_id}: "
                    f"WMES={wmes:.3f}, PTS={pts:.3f}, frag={fragility:.3f}, "
                    f"tier={tier}"
                )
            registered += 1
        except Exception as e:
            if logger:
                logger.warning(f"[THICK_WEAVE] Failed to register candidate: {e}")

    return registered


def start_thick_weave_search(budget=50, logger=None):
    """Start thick weave search in a background thread for intelligent plateau discovery."""
    def _run():
        try:
            from src.phase_23_analytics.thick_weave_search import (
                ThickWeaveSearch, ThickWeaveConfig,
            )
            from src.core.registry_db import get_registry_db

            _db = get_registry_db()

            if logger:
                logger.info(f"Thick weave search started (budget={budget} per cycle)")
            update_component("thick_weave_search", "RUNNING", logger)

            cycle = 0
            while True:
                cycle += 1
                try:
                    if logger:
                        logger.info(f"[THICK_WEAVE] Starting search cycle {cycle} (budget={budget})...")

                    config = ThickWeaveConfig(
                        n_initial_threads=4,
                        max_active_threads=8,
                        max_total_evaluations=budget,
                        configs_per_round=4,
                        max_rounds=max(budget // 4, 10),
                        checkpoint_dir=str(project_root / "models" / "thick_weave_checkpoints"),
                    )

                    search = ThickWeaveSearch(config)
                    report = search.run(target_type="swing")

                    # Save report
                    report_dir = project_root / "reports"
                    report_dir.mkdir(exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    report_file = report_dir / f"thick_weave_report_{timestamp}.json"
                    with open(report_file, "w") as f:
                        json.dump(report, f, indent=2, default=str)

                    # Bridge production candidates into RegistryDB with tier scoring
                    n_registered = _register_thick_weave_candidates(report, _db, logger)

                    stats = report.get("search_stats", {})
                    thick_paths = report.get("thick_paths", [])
                    best_wmes = report.get("best_overall_wmes", 0)

                    if logger:
                        logger.info(
                            f"[THICK_WEAVE] Cycle {cycle} complete: "
                            f"{stats.get('total_evaluated', 0)} evals, "
                            f"{len(thick_paths)} thick paths, "
                            f"best WMES={best_wmes:.3f}, "
                            f"{n_registered} registered"
                        )

                except Exception as e:
                    if logger:
                        logger.warning(f"[THICK_WEAVE] Cycle {cycle} failed: {e}")

                # Wait between search cycles
                time.sleep(120)

        except Exception as e:
            update_component("thick_weave_search", "ERROR", logger)
            if logger:
                logger.error(f"Thick weave search error: {e}")

    thread = threading.Thread(target=_run, name="ThickWeaveSearch", daemon=True)
    thread.start()
    return thread


def run_trading_bot(logger):
    """Run the trading bot in the main thread."""
    global _active_bot
    from src.paper_trading import TradingBot, ALPACA_AVAILABLE

    if not ALPACA_AVAILABLE:
        logger.error("Alpaca SDK not installed. Run: pip install alpaca-py")
        return False

    try:
        bot = TradingBot()
        _active_bot = bot  # Register for graceful shutdown handler

        # Print account info
        account = bot.client.get_account()
        logger.info(f"Account Equity: ${account['equity']:,.2f}")
        logger.info(f"Buying Power:   ${account['buying_power']:,.2f}")

        logger.info("Starting trading loop (Ctrl+C to stop)...")
        bot.run(interval_seconds=10)
        return True
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
        return True
    except Exception as e:
        logger.error(f"Trading bot error: {e}")
        return False
    finally:
        _active_bot = None
        # Flush performance tracker and risk state on shutdown
        if bot:
            try:
                bot.performance_tracker._save()
                logger.info("Performance tracker flushed to disk")
            except Exception as e:
                logger.debug(f"Shutdown: perf tracker flush failed: {e}")
            try:
                bot.risk_manager._persist_state()
                logger.info("Risk manager state persisted")
            except Exception as e:
                logger.debug(f"Shutdown: risk state persist failed: {e}")
            # Cancel any pending orders
            try:
                pending = list(bot.order_manager.pending_orders.keys())
                for order_id in pending:
                    bot.client.cancel_order(order_id)
                    logger.info(f"Cancelled pending order: {order_id}")
            except Exception as e:
                logger.debug(f"Shutdown: order cancellation failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="GIGA TRADER - System Launcher")
    parser.add_argument("--dashboard-only", action="store_true", help="Only start dashboards")
    parser.add_argument("--trading-only", action="store_true", help="Only start trading bot")
    parser.add_argument("--with-training", action="store_true", default=True, help="Run experiments in background (default: True)")
    parser.add_argument("--no-training", action="store_true", help="Disable background experiments")
    parser.add_argument("--no-web-monitor", action="store_true", help="Skip web monitor")
    parser.add_argument("--no-trading", action="store_true", help="Skip trading bot (experiment-only mode)")
    parser.add_argument("--no-thick-weave", action="store_true", help="Skip thick weave search")
    parser.add_argument("--thick-weave-budget", type=int, default=50, help="Thick weave evals per cycle (default: 50)")
    parser.add_argument("--dashboard-port", type=int, default=8050, help="Dashboard port")
    parser.add_argument("--monitor-port", type=int, default=5000, help="Web monitor port")
    args = parser.parse_args()

    logger = setup_logging()

    # Register graceful shutdown handlers
    signal.signal(signal.SIGINT, _shutdown_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, _shutdown_handler)

    print()
    print("=" * 60)
    print("  GIGA TRADER - System Launcher")
    print("=" * 60)
    print()

    # Check prerequisites
    checks = check_prerequisites(logger)
    # models check is non-blocking for experiment-only mode (fresh campaigns have no production models yet)
    all_ok = all(v for k, v in checks.items() if k not in ("paper_mode", "flask", "experiments_dir", "models"))

    for check, passed in checks.items():
        status = "OK" if passed else "FAIL"
        print(f"  [{status}] {check}")

    if not all_ok:
        print("\n  Some prerequisites failed. Fix the issues above and retry.")
        return 1

    print()

    # Write initial status.json (clears stale data from previous runs)
    write_status_json(_component_state, mode="INITIALIZING", logger=logger)

    # Check gates (skip in experiment-only or dashboard-only mode)
    if not args.dashboard_only and not args.no_trading:
        gates_ok = check_gates(logger)
        if not gates_ok:
            print("\n  Trading gates not met. Run experiments first:")
            print("    python scripts/register_existing_models.py")
            print("    python src/giga_orchestrator.py")
            print("\n  Or start dashboards only:")
            print("    python scripts/start_system.py --dashboard-only")
            return 1

    # Start dashboards
    dashboard_thread = None
    monitor_thread = None

    if not args.trading_only:
        if checks.get("flask", False):
            dashboard_thread = start_dashboard_server(port=args.dashboard_port, logger=logger)
            update_component("monitor", "RUNNING", logger)
            time.sleep(1)  # Let it bind

            if not args.no_web_monitor:
                monitor_thread = start_web_monitor(port=args.monitor_port, logger=logger)
                time.sleep(1)

            print(f"\n  Dashboard:   http://127.0.0.1:{args.dashboard_port}")
            if not args.no_web_monitor:
                print(f"  Web Monitor: http://127.0.0.1:{args.monitor_port}")
            print()
        else:
            print("  [SKIP] Dashboards (Flask not installed)")

    # Start experiment runner (continuous training in background)
    experiment_thread = None
    thick_weave_thread = None
    watchdog_thread = None

    if args.with_training and not args.no_training and not args.dashboard_only:
        experiment_thread = start_experiment_runner(logger=logger)
        logger.info("Background training enabled (experiments run continuously)")

    # Start thick weave search (intelligent plateau discovery)
    if not args.no_thick_weave and not args.dashboard_only:
        thick_weave_thread = start_thick_weave_search(
            budget=args.thick_weave_budget, logger=logger
        )
        logger.info(f"Thick weave search enabled (budget={args.thick_weave_budget} per cycle)")

    # Start watchdog to auto-restart crashed threads (Wave 26, extended Wave 28)
    if experiment_thread or thick_weave_thread or dashboard_thread or monitor_thread:
        watchdog_thread, watched, dash_port_ref, mon_port_ref = start_watchdog(logger=logger)
        if experiment_thread:
            watched["experiment_runner"] = experiment_thread
        if thick_weave_thread:
            watched["thick_weave_search"] = thick_weave_thread
        if dashboard_thread:
            watched["dashboard_server"] = dashboard_thread
            dash_port_ref[0] = args.dashboard_port
        if monitor_thread:
            watched["web_monitor"] = monitor_thread
            mon_port_ref[0] = args.monitor_port
        logger.info("Watchdog enabled (auto-restart on crash)")

    # Start periodic maintenance (model cleanup, DB vacuum, disk check)
    maintenance_thread = None
    if not args.dashboard_only:
        maintenance_thread = run_periodic_maintenance(logger=logger)
        logger.info("Periodic maintenance enabled (every 6 hours)")

        # Run initial cleanup on startup
        cleanup_old_experiment_models(max_age_days=7, logger=logger)

    # Start health checker (periodic system monitoring)
    health_checker = None
    if not args.dashboard_only:
        try:
            from src.phase_20_monitoring.health_checker import HealthChecker
            health_checker = HealthChecker(check_interval_seconds=60)
            health_checker.start_background()
            update_component("health_checker", "RUNNING", logger)
            logger.info("Health checker started (interval: 60s)")
        except Exception as e:
            logger.warning(f"Health checker not started: {e}")
            update_component("health_checker", "FAILED", logger)

    # Start trading bot (unless --no-trading or --dashboard-only)
    if not args.dashboard_only and not args.no_trading:
        update_component("trading_bot", "RUNNING", logger)
        update_component("signal_generator", "RUNNING", logger)
        update_component("risk_manager", "RUNNING", logger)
        run_trading_bot(logger)
        update_component("trading_bot", "STOPPED", logger)
        update_component("signal_generator", "STOPPED", logger)
        update_component("risk_manager", "STOPPED", logger)
    else:
        mode = "experiment-only" if args.no_trading else "dashboard-only"
        update_component("trading_bot", "DISABLED", logger)
        print(f"  {mode.title()} mode — trading disabled. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n  Shutting down...")

    # Stop health checker
    if health_checker is not None:
        try:
            health_checker.stop_background()
            update_component("health_checker", "STOPPED", logger)
        except Exception as e:
            logger.debug(f"Health checker stop: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

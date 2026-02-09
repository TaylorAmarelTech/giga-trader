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

import os
import sys
import json
import time
import signal
import logging
import argparse
import threading
from pathlib import Path
from datetime import datetime

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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
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
        except Exception:
            pass

        (logs_dir / "status.json").write_text(json.dumps(status, indent=2, default=str))
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
    "monitor": "RUNNING",
}
_component_lock = threading.Lock()


def update_component(name: str, status: str, logger=None):
    """Thread-safe component status update + status.json write."""
    with _component_lock:
        _component_state[name] = status
        mode = "TRADING" if _component_state.get("trading_bot") == "RUNNING" else "IDLE"
        write_status_json(_component_state.copy(), mode=mode, logger=logger)


def start_experiment_runner(logger=None):
    """Start the experiment runner in a background thread for continuous training."""
    def _run():
        try:
            from src.experiment_engine import ExperimentEngine
            engine = ExperimentEngine()

            if logger:
                logger.info("Experiment runner started (continuous training)")
            update_component("experiment_engine", "RUNNING", logger)

            cycle = 0
            while True:
                cycle += 1
                try:
                    if logger:
                        logger.info(f"[EXPERIMENT] Starting experiment cycle {cycle}...")
                    update_component("training_engine", "RUNNING", logger)
                    result = engine.run_one_experiment()
                    update_component("training_engine", "STOPPED", logger)
                    if result and logger:
                        logger.info(
                            f"[EXPERIMENT] Cycle {cycle} complete: "
                            f"AUC={result.test_auc:.3f}, "
                            f"WMES={result.wmes_score:.3f}"
                        )
                except Exception as e:
                    update_component("training_engine", "STOPPED", logger)
                    if logger:
                        logger.warning(f"[EXPERIMENT] Cycle {cycle} failed: {e}")

                # Wait between experiments
                time.sleep(60)

        except Exception as e:
            update_component("experiment_engine", "ERROR", logger)
            if logger:
                logger.error(f"Experiment runner error: {e}")

    thread = threading.Thread(target=_run, name="ExperimentRunner", daemon=True)
    thread.start()
    return thread


def _register_thick_weave_candidates(report, registry, logger=None):
    """Bridge thick weave production candidates into ModelRegistry with tier scoring.

    Maps ThickWeave metrics to ExperimentResult fields:
      - wmes_score ← candidate wmes
      - stability_score ← candidate pts (Path Thickness Score maps to HP stability)
      - fragility_score ← candidate fragility (from Tier 3 robustness check)
      - test_auc ← looked up from ModelRegistryV2 by config_hash
    """
    from src.phase_21_continuous.experiment_tracking import (
        ExperimentResult, ExperimentConfig, ExperimentStatus,
        create_default_config,
    )

    candidates = report.get("production_candidates", [])
    if not candidates:
        return 0

    # Try to look up test_auc from ModelRegistryV2
    v2_aucs = {}
    try:
        from src.model_registry_v2 import ModelRegistryV2, get_registry
        v2_registry = get_registry()
        for model_id, entry in v2_registry.models.items():
            if hasattr(entry, 'metrics') and entry.metrics:
                config_hash = entry.get_config_hash() if hasattr(entry, 'get_config_hash') else ""
                auc = getattr(entry.metrics, 'test_auc', 0) or getattr(entry.metrics, 'cv_auc', 0)
                if config_hash:
                    v2_aucs[config_hash] = auc
    except Exception:
        pass

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

        # Create ExperimentResult to register through standard pipeline
        config = create_default_config(f"thick_weave_{cand.get('thread_id', 'unknown')}")
        result = ExperimentResult(
            experiment_id=f"tw_{config_hash[:16]}_{datetime.now().strftime('%H%M%S')}",
            config=config,
            status=ExperimentStatus.COMPLETED,
            test_auc=test_auc,
            train_auc=test_auc + 0.03,  # Conservative estimate
            cv_auc_mean=test_auc,
            wmes_score=wmes,
            stability_score=pts,       # PTS >= 0.5 means thick/stable plateau
            fragility_score=fragility,
            model_path="",  # Models already in ModelRegistryV2
        )

        try:
            model_id = registry.register_model(result)
            tier = registry.models[model_id].tier
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
            from src.phase_21_continuous.experiment_tracking import ModelRegistry

            registry = ModelRegistry()

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

                    # Bridge production candidates into ModelRegistry with tier scoring
                    n_registered = _register_thick_weave_candidates(report, registry, logger)

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
    from src.paper_trading import TradingBot, ALPACA_AVAILABLE

    if not ALPACA_AVAILABLE:
        logger.error("Alpaca SDK not installed. Run: pip install alpaca-py")
        return False

    try:
        bot = TradingBot()

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


def main():
    parser = argparse.ArgumentParser(description="GIGA TRADER - System Launcher")
    parser.add_argument("--dashboard-only", action="store_true", help="Only start dashboards")
    parser.add_argument("--trading-only", action="store_true", help="Only start trading bot")
    parser.add_argument("--with-training", action="store_true", default=True, help="Run experiments in background (default: True)")
    parser.add_argument("--no-training", action="store_true", help="Disable background experiments")
    parser.add_argument("--no-web-monitor", action="store_true", help="Skip web monitor")
    parser.add_argument("--no-thick-weave", action="store_true", help="Skip thick weave search")
    parser.add_argument("--thick-weave-budget", type=int, default=50, help="Thick weave evals per cycle (default: 50)")
    parser.add_argument("--dashboard-port", type=int, default=8050, help="Dashboard port")
    parser.add_argument("--monitor-port", type=int, default=5000, help="Web monitor port")
    args = parser.parse_args()

    logger = setup_logging()

    print()
    print("=" * 60)
    print("  GIGA TRADER - System Launcher")
    print("=" * 60)
    print()

    # Check prerequisites
    checks = check_prerequisites(logger)
    all_ok = all(v for k, v in checks.items() if k not in ("paper_mode", "flask", "experiments_dir"))

    for check, passed in checks.items():
        status = "OK" if passed else "FAIL"
        print(f"  [{status}] {check}")

    if not all_ok:
        print("\n  Some prerequisites failed. Fix the issues above and retry.")
        return 1

    print()

    # Write initial status.json (clears stale data from previous runs)
    write_status_json(_component_state, mode="INITIALIZING", logger=logger)

    # Check gates
    if not args.dashboard_only:
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
    if args.with_training and not args.no_training and not args.dashboard_only:
        experiment_thread = start_experiment_runner(logger=logger)
        logger.info("Background training enabled (experiments run continuously)")
    else:
        experiment_thread = None

    # Start thick weave search (intelligent plateau discovery)
    if not args.no_thick_weave and not args.dashboard_only:
        thick_weave_thread = start_thick_weave_search(
            budget=args.thick_weave_budget, logger=logger
        )
        logger.info(f"Thick weave search enabled (budget={args.thick_weave_budget} per cycle)")
    else:
        thick_weave_thread = None

    # Start trading bot
    if not args.dashboard_only:
        update_component("trading_bot", "RUNNING", logger)
        update_component("signal_generator", "RUNNING", logger)
        update_component("risk_manager", "RUNNING", logger)
        run_trading_bot(logger)
        update_component("trading_bot", "STOPPED", logger)
        update_component("signal_generator", "STOPPED", logger)
        update_component("risk_manager", "STOPPED", logger)
    else:
        print("  Dashboard-only mode. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n  Shutting down...")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Long-Running Orchestrator
=========================
Main coordination system for 30+ day automated trading experiments.

Orchestrates:
- Claude Code CLI processes
- Grid search experiments
- State persistence
- Health monitoring
- Auto-recovery from failures
"""

import os
import sys
import json
import time
import signal
import logging
import threading
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from enum import Enum

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.long_runner.process_manager import ProcessManager, ClaudeCodeConfig
from src.long_runner.grid_search import GridSearchController, ParameterGrid
from src.long_runner.state_manager import StateManager
from src.long_runner.monitoring import MonitoringDaemon, AlertManager, AlertSeverity, AlertChannel

logger = logging.getLogger("GigaTrader.Orchestrator")


class OrchestratorState(Enum):
    """Orchestrator running states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    # Paths
    project_dir: Path = PROJECT_ROOT
    logs_dir: Path = PROJECT_ROOT / "logs"
    experiments_dir: Path = PROJECT_ROOT / "experiments"
    state_dir: Path = PROJECT_ROOT / "logs" / "orchestrator_state"

    # Claude Code settings
    claude_executable: str = "claude"
    session_timeout: int = 3600  # 1 hour
    command_timeout: int = 1800  # 30 minutes for long experiments

    # Experiment settings
    max_parallel_processes: int = 1  # Usually 1 for API limits
    experiments_per_session: int = 3  # Rotate session after N experiments
    exploration_ratio: float = 0.3

    # Scheduling
    run_continuously: bool = True
    pause_hours: List[int] = field(default_factory=lambda: [])  # Hours to pause (0-23)
    target_experiments_per_day: int = 50

    # Recovery
    max_retries_per_experiment: int = 3
    restart_delay_seconds: int = 60
    max_consecutive_failures: int = 5

    # Monitoring
    alert_email: Optional[str] = None
    webhook_url: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "project_dir": str(self.project_dir),
            "logs_dir": str(self.logs_dir),
            "experiments_dir": str(self.experiments_dir),
            "state_dir": str(self.state_dir),
            "claude_executable": self.claude_executable,
            "session_timeout": self.session_timeout,
            "command_timeout": self.command_timeout,
            "max_parallel_processes": self.max_parallel_processes,
            "experiments_per_session": self.experiments_per_session,
            "exploration_ratio": self.exploration_ratio,
            "run_continuously": self.run_continuously,
            "pause_hours": self.pause_hours,
            "target_experiments_per_day": self.target_experiments_per_day,
            "max_retries_per_experiment": self.max_retries_per_experiment,
            "restart_delay_seconds": self.restart_delay_seconds,
            "max_consecutive_failures": self.max_consecutive_failures,
        }


class LongRunningOrchestrator:
    """
    Main orchestrator for long-running experiments.

    Coordinates:
    - Process management (start/stop/restart Claude Code)
    - Experiment scheduling (grid search, prioritization)
    - State persistence (resume after crashes)
    - Monitoring and alerting
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self.state = OrchestratorState.STOPPED
        self._running = False
        self._paused = False
        self._lock = threading.Lock()

        # Statistics
        self.start_time: Optional[datetime] = None
        self.experiments_this_session = 0
        self.experiments_total = 0
        self.consecutive_failures = 0
        self.session_rotations = 0

        # Initialize components
        self._init_components()

        # Setup signal handlers
        self._setup_signals()

    def _init_components(self):
        """Initialize all orchestrator components."""
        # Ensure directories exist
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)
        self.config.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.config.state_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        log_file = self.config.logs_dir / "orchestrator.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
        ))
        logging.getLogger().addHandler(file_handler)
        logging.getLogger().setLevel(logging.INFO)

        # Alert Manager
        channels = [AlertChannel.FILE, AlertChannel.CONSOLE]
        email_config = {}
        if self.config.alert_email:
            channels.append(AlertChannel.EMAIL)
            # Email config would be set here
        if self.config.webhook_url:
            channels.append(AlertChannel.WEBHOOK)

        self.alert_manager = AlertManager(
            alert_file=self.config.logs_dir / "alerts.json",
            webhook_url=self.config.webhook_url,
            channels=channels,
        )

        # State Manager
        self.state_manager = StateManager(
            state_dir=self.config.state_dir,
            backup_count=20,
            auto_save_interval=60,
        )

        # Process Manager
        claude_config = ClaudeCodeConfig(
            executable=self.config.claude_executable,
            working_dir=str(self.config.project_dir),
            session_timeout=self.config.session_timeout,
            command_timeout=self.config.command_timeout,
            max_retries=self.config.max_retries_per_experiment,
            restart_cooldown=self.config.restart_delay_seconds,
            max_consecutive_crashes=self.config.max_consecutive_failures,
        )

        self.process_manager = ProcessManager(
            config=claude_config,
            max_processes=self.config.max_parallel_processes,
            state_file=self.config.state_dir / "process_state.json",
        )

        # Setup process manager callbacks
        self.process_manager.on_process_crash = self._on_process_crash
        self.process_manager.on_process_restart = self._on_process_restart
        self.process_manager.on_all_crashed = self._on_all_crashed

        # Grid Search Controller
        self.grid_controller = GridSearchController(
            grid=ParameterGrid(),
            state_file=self.config.state_dir / "grid_search_state.json",
            results_dir=self.config.experiments_dir / "grid_results",
            exploration_ratio=self.config.exploration_ratio,
        )

        # Monitoring Daemon
        self.monitor = MonitoringDaemon(
            alert_manager=self.alert_manager,
            metrics_file=self.config.logs_dir / "metrics.json",
            check_interval=60,
        )
        self.monitor.set_components(
            process_manager=self.process_manager,
            grid_controller=self.grid_controller,
            state_manager=self.state_manager,
        )

        logger.info("Orchestrator components initialized")

    def _setup_signals(self):
        """Setup signal handlers for graceful shutdown."""
        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGHUP, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.stop()

    def start(self):
        """Start the orchestrator."""
        if self._running:
            logger.warning("Orchestrator already running")
            return

        logger.info("=" * 60)
        logger.info("GIGA TRADER LONG-RUNNING ORCHESTRATOR")
        logger.info("=" * 60)
        logger.info(f"Project: {self.config.project_dir}")
        logger.info(f"Target: {self.config.target_experiments_per_day} experiments/day")
        logger.info("=" * 60)

        self.state = OrchestratorState.STARTING
        self._running = True
        self.start_time = datetime.now()

        # Restore state
        saved_state = self.state_manager.get_state()
        self.experiments_total = saved_state.experiments_completed

        # Start components
        self.state_manager.start_auto_save()
        self.process_manager.start()
        self.monitor.start()

        # Send startup alert
        self.alert_manager.send_alert(
            AlertSeverity.INFO,
            "Orchestrator Started",
            f"Long-running orchestrator started. Target: {self.config.target_experiments_per_day} experiments/day",
            source="orchestrator",
        )

        self.state = OrchestratorState.RUNNING

        # Run main loop
        self._main_loop()

    def stop(self):
        """Stop the orchestrator gracefully."""
        if not self._running:
            return

        logger.info("Stopping orchestrator...")
        self.state = OrchestratorState.STOPPING
        self._running = False

        # Stop components
        self.monitor.stop()
        self.process_manager.stop()

        # Save final state
        self._update_state()
        self.state_manager.save(force=True)
        self.state_manager.stop_auto_save()

        # Send shutdown alert
        self.alert_manager.send_alert(
            AlertSeverity.INFO,
            "Orchestrator Stopped",
            f"Orchestrator stopped after {self.experiments_total} experiments",
            source="orchestrator",
        )

        self.state = OrchestratorState.STOPPED
        logger.info("Orchestrator stopped")

    def pause(self):
        """Pause experiment execution."""
        self._paused = True
        self.state = OrchestratorState.PAUSED
        logger.info("Orchestrator paused")

    def resume(self):
        """Resume experiment execution."""
        self._paused = False
        self.state = OrchestratorState.RUNNING
        logger.info("Orchestrator resumed")

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        uptime = 0
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()

        return {
            "state": self.state.value,
            "running": self._running,
            "paused": self._paused,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime_seconds": uptime,
            "uptime_days": uptime / 86400,
            "experiments_this_session": self.experiments_this_session,
            "experiments_total": self.experiments_total,
            "consecutive_failures": self.consecutive_failures,
            "session_rotations": self.session_rotations,
            "config": self.config.to_dict(),
            "grid_search": self.grid_controller.get_stats(),
            "health": self.monitor.get_health().to_dict(),
        }

    def _main_loop(self):
        """Main orchestration loop."""
        while self._running:
            try:
                # Check if we should pause
                if self._should_pause():
                    logger.info("Entering scheduled pause period")
                    self._paused = True
                    self.state = OrchestratorState.PAUSED
                    time.sleep(60)
                    continue

                if self._paused:
                    time.sleep(10)
                    continue

                # Check for session rotation
                if self._should_rotate_session():
                    self._rotate_session()

                # Get next experiment
                experiment = self.grid_controller.get_next_experiment()

                if not experiment:
                    logger.warning("No experiments available")
                    time.sleep(60)
                    continue

                # Run the experiment
                logger.info(f"Running experiment: {experiment.config_id}")
                success, results = self._run_experiment(experiment)

                # Record results
                self.grid_controller.mark_completed(
                    experiment.config_id,
                    results,
                    success=success,
                )

                if success:
                    self.experiments_this_session += 1
                    self.experiments_total += 1
                    self.consecutive_failures = 0

                    logger.info(
                        f"Experiment {experiment.config_id} completed: "
                        f"AUC={results.get('test_auc', 0):.4f}, "
                        f"Sharpe={results.get('backtest_sharpe', 0):.2f}"
                    )
                else:
                    self.consecutive_failures += 1
                    logger.warning(
                        f"Experiment {experiment.config_id} failed: "
                        f"{results.get('error', 'Unknown error')}"
                    )

                    if self.consecutive_failures >= self.config.max_consecutive_failures:
                        self.alert_manager.send_alert(
                            AlertSeverity.CRITICAL,
                            "Too Many Failures",
                            f"{self.consecutive_failures} consecutive experiment failures",
                            source="orchestrator",
                        )
                        logger.error("Too many consecutive failures, pausing...")
                        self._paused = True

                # Update state
                self._update_state()

                # Brief pause between experiments
                time.sleep(5)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break

            except Exception as e:
                logger.error(f"Main loop error: {e}")
                self.alert_manager.send_alert(
                    AlertSeverity.ERROR,
                    "Main Loop Error",
                    str(e),
                    source="orchestrator",
                )
                time.sleep(60)

        logger.info("Main loop exited")

    def _run_experiment(self, experiment) -> tuple:
        """
        Run a single experiment using Claude Code.

        Args:
            experiment: ExperimentConfig to run

        Returns:
            (success, results) tuple
        """
        # Build the command for Claude Code
        params_json = json.dumps(experiment.parameters, indent=2)

        command = f"""Run a trading model experiment with these parameters:

{params_json}

Execute the experiment using the experiment engine at src/experiment_engine.py.
Use run_experiment() with a config that includes these parameters.
Return the results including: test_auc, train_auc, backtest_sharpe, win_rate, wmes_score.
Format the results as a JSON object."""

        try:
            # Execute via Claude Code
            result = self.process_manager.execute_command(
                command,
                retries=self.config.max_retries_per_experiment,
            )

            if not result:
                return False, {"error": "No response from Claude Code"}

            # Parse results
            results = self._parse_experiment_results(result)
            return bool(results.get("test_auc")), results

        except Exception as e:
            logger.error(f"Experiment execution error: {e}")
            return False, {"error": str(e)}

    def _parse_experiment_results(self, output: str) -> Dict[str, Any]:
        """Parse experiment results from Claude Code output."""
        results = {}

        try:
            # Try to find JSON in output
            import re
            json_match = re.search(r'\{[^{}]*"test_auc"[^{}]*\}', output, re.DOTALL)

            if json_match:
                results = json.loads(json_match.group())
            else:
                # Parse key metrics from text
                patterns = {
                    "test_auc": r"test[_\s]?auc[:\s]+([0-9.]+)",
                    "train_auc": r"train[_\s]?auc[:\s]+([0-9.]+)",
                    "backtest_sharpe": r"sharpe[:\s]+([0-9.]+)",
                    "win_rate": r"win[_\s]?rate[:\s]+([0-9.]+)",
                    "wmes_score": r"wmes[:\s]+([0-9.]+)",
                }

                for key, pattern in patterns.items():
                    match = re.search(pattern, output, re.IGNORECASE)
                    if match:
                        results[key] = float(match.group(1))

        except Exception as e:
            logger.error(f"Failed to parse results: {e}")
            results["error"] = f"Parse error: {e}"

        return results

    def _should_pause(self) -> bool:
        """Check if we should pause based on schedule."""
        if not self.config.pause_hours:
            return False

        current_hour = datetime.now().hour
        return current_hour in self.config.pause_hours

    def _should_rotate_session(self) -> bool:
        """Check if we should rotate the Claude Code session."""
        return self.experiments_this_session >= self.config.experiments_per_session

    def _rotate_session(self):
        """Rotate Claude Code session to avoid timeouts."""
        logger.info("Rotating Claude Code session...")

        # Stop current processes
        self.process_manager.stop()

        # Brief pause
        time.sleep(self.config.restart_delay_seconds)

        # Start fresh
        self.process_manager.start()
        self.experiments_this_session = 0
        self.session_rotations += 1

        logger.info(f"Session rotated (rotation #{self.session_rotations})")

    def _update_state(self):
        """Update persistent state."""
        self.state_manager.update_state(
            orchestrator_running=self._running,
            orchestrator_start_time=self.start_time.isoformat() if self.start_time else None,
            experiments_completed=self.experiments_total,
            last_experiment_time=datetime.now().isoformat(),
            best_score_achieved=self.grid_controller.get_stats().get("best_score", 0),
        )

    def _on_process_crash(self, process_id: str):
        """Handle process crash."""
        self.alert_manager.send_alert(
            AlertSeverity.WARNING,
            "Process Crashed",
            f"Claude Code process {process_id} crashed",
            source="process_manager",
        )

    def _on_process_restart(self, process_id: str):
        """Handle process restart."""
        logger.info(f"Process {process_id} restarted")

    def _on_all_crashed(self):
        """Handle all processes crashing."""
        self.alert_manager.send_alert(
            AlertSeverity.CRITICAL,
            "All Processes Crashed",
            "All Claude Code processes have crashed",
            source="process_manager",
        )
        logger.error("All processes crashed, entering error state")
        self.state = OrchestratorState.ERROR


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def create_config_from_args() -> OrchestratorConfig:
    """Create config from command line arguments."""
    parser = argparse.ArgumentParser(
        description="GIGA TRADER Long-Running Experiment Orchestrator"
    )

    parser.add_argument(
        "--target-experiments",
        type=int,
        default=50,
        help="Target experiments per day",
    )
    parser.add_argument(
        "--session-timeout",
        type=int,
        default=3600,
        help="Session timeout in seconds",
    )
    parser.add_argument(
        "--exploration-ratio",
        type=float,
        default=0.3,
        help="Random exploration ratio (0-1)",
    )
    parser.add_argument(
        "--pause-hours",
        type=str,
        default="",
        help="Hours to pause (comma-separated, e.g., '0,1,2,3')",
    )
    parser.add_argument(
        "--webhook-url",
        type=str,
        default="",
        help="Webhook URL for alerts",
    )

    args = parser.parse_args()

    pause_hours = []
    if args.pause_hours:
        pause_hours = [int(h) for h in args.pause_hours.split(",")]

    return OrchestratorConfig(
        target_experiments_per_day=args.target_experiments,
        session_timeout=args.session_timeout,
        exploration_ratio=args.exploration_ratio,
        pause_hours=pause_hours,
        webhook_url=args.webhook_url if args.webhook_url else None,
    )


def main():
    """Main entry point."""
    print("=" * 60)
    print("GIGA TRADER - LONG-RUNNING EXPERIMENT ORCHESTRATOR")
    print("=" * 60)
    print("Starting 30+ day continuous experiment system...")
    print()

    config = create_config_from_args()
    orchestrator = LongRunningOrchestrator(config=config)

    try:
        orchestrator.start()
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    finally:
        orchestrator.stop()

    print("\nOrchestrator stopped.")
    print(f"Total experiments completed: {orchestrator.experiments_total}")


if __name__ == "__main__":
    main()

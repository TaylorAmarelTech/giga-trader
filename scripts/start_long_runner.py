#!/usr/bin/env python
"""
GIGA TRADER - Long-Running Experiment Launcher
===============================================
Easy launcher for the 30+ day continuous experiment system.

Usage:
    python scripts/start_long_runner.py              # Start in foreground
    python scripts/start_long_runner.py --daemon     # Start as background process
    python scripts/start_long_runner.py --status     # Check status
    python scripts/start_long_runner.py --stop       # Stop running instance
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_pid_file() -> Path:
    """Get path to PID file."""
    return PROJECT_ROOT / "logs" / "orchestrator.pid"


def get_status_file() -> Path:
    """Get path to status file."""
    return PROJECT_ROOT / "logs" / "orchestrator_status.json"


def is_running() -> bool:
    """Check if orchestrator is already running."""
    pid_file = get_pid_file()
    if not pid_file.exists():
        return False

    try:
        with open(pid_file) as f:
            pid = int(f.read().strip())

        # Check if process is alive
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(1, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True

    except (ValueError, OSError, ProcessLookupError):
        # Process doesn't exist
        pid_file.unlink(missing_ok=True)
        return False


def write_pid():
    """Write current PID to file."""
    pid_file = get_pid_file()
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))


def remove_pid():
    """Remove PID file."""
    get_pid_file().unlink(missing_ok=True)


def show_status():
    """Show orchestrator status."""
    print("=" * 60)
    print("GIGA TRADER - ORCHESTRATOR STATUS")
    print("=" * 60)

    if is_running():
        print("[RUNNING] Orchestrator is running")

        # Read status file
        status_file = get_status_file()
        if status_file.exists():
            try:
                with open(status_file) as f:
                    status = json.load(f)

                print(f"\nUptime: {status.get('uptime_days', 0):.2f} days")
                print(f"Experiments completed: {status.get('experiments_total', 0)}")
                print(f"Session experiments: {status.get('experiments_this_session', 0)}")

                grid_stats = status.get('grid_search', {})
                print(f"\nGrid Search:")
                print(f"  Completed: {grid_stats.get('completed', 0)}")
                print(f"  Failed: {grid_stats.get('failed', 0)}")
                print(f"  Success Rate: {grid_stats.get('success_rate', 0):.1%}")
                print(f"  Best Score: {grid_stats.get('best_score', 0):.4f}")

                health = status.get('health', {})
                print(f"\nSystem Health:")
                print(f"  CPU: {health.get('cpu_percent', 0):.1f}%")
                print(f"  Memory: {health.get('memory_percent', 0):.1f}%")
                print(f"  Disk: {health.get('disk_percent', 0):.1f}%")

            except Exception as e:
                print(f"\nCouldn't read status file: {e}")

    else:
        print("[STOPPED] Orchestrator is not running")

    # Show recent logs
    log_file = PROJECT_ROOT / "logs" / "orchestrator.log"
    if log_file.exists():
        print(f"\nRecent logs ({log_file}):")
        print("-" * 40)
        try:
            with open(log_file) as f:
                lines = f.readlines()
                for line in lines[-10:]:
                    print(line.rstrip())
        except:
            pass


def stop_orchestrator():
    """Stop running orchestrator."""
    pid_file = get_pid_file()
    if not pid_file.exists():
        print("Orchestrator is not running")
        return

    try:
        with open(pid_file) as f:
            pid = int(f.read().strip())

        print(f"Stopping orchestrator (PID {pid})...")

        if sys.platform == "win32":
            # Windows
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=False)
        else:
            # Unix
            os.kill(pid, 15)  # SIGTERM

        # Wait for shutdown
        for _ in range(30):
            if not is_running():
                break
            time.sleep(1)

        if is_running():
            print("Force killing...")
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=False)
            else:
                os.kill(pid, 9)  # SIGKILL

        remove_pid()
        print("Orchestrator stopped")

    except Exception as e:
        print(f"Error stopping orchestrator: {e}")


def start_foreground(args):
    """Start orchestrator in foreground."""
    if is_running():
        print("Orchestrator is already running!")
        print("Use --stop to stop it first, or --status to check status")
        return

    write_pid()

    try:
        from src.long_runner.orchestrator import LongRunningOrchestrator, OrchestratorConfig

        config = OrchestratorConfig(
            target_experiments_per_day=args.target_experiments,
            exploration_ratio=args.exploration_ratio,
            session_timeout=args.session_timeout,
        )

        orchestrator = LongRunningOrchestrator(config=config)
        orchestrator.start()

    except KeyboardInterrupt:
        print("\nShutdown requested...")
    finally:
        remove_pid()


def start_daemon(args):
    """Start orchestrator as daemon/background process."""
    if is_running():
        print("Orchestrator is already running!")
        return

    print("Starting orchestrator in background...")

    # Build command
    python = sys.executable
    script = Path(__file__).absolute()

    cmd = [
        python, str(script),
        "--target-experiments", str(args.target_experiments),
        "--exploration-ratio", str(args.exploration_ratio),
        "--session-timeout", str(args.session_timeout),
    ]

    if sys.platform == "win32":
        # Windows: use pythonw and CREATE_NEW_PROCESS_GROUP
        subprocess.Popen(
            cmd,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        # Unix: double fork
        if os.fork() > 0:
            return

        os.setsid()

        if os.fork() > 0:
            sys.exit(0)

        # Redirect standard file descriptors
        sys.stdout.flush()
        sys.stderr.flush()

        with open("/dev/null", "r") as f:
            os.dup2(f.fileno(), sys.stdin.fileno())
        with open("/dev/null", "a+") as f:
            os.dup2(f.fileno(), sys.stdout.fileno())
            os.dup2(f.fileno(), sys.stderr.fileno())

        start_foreground(args)

    print("Orchestrator started in background")
    print("Use --status to check status")
    print("Use --stop to stop it")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GIGA TRADER Long-Running Experiment Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/start_long_runner.py              # Start in foreground
    python scripts/start_long_runner.py --daemon     # Start in background
    python scripts/start_long_runner.py --status     # Check status
    python scripts/start_long_runner.py --stop       # Stop running instance
    python scripts/start_long_runner.py --target-experiments 100

The orchestrator will:
- Run continuous experiments using Claude Code CLI
- Intelligently explore the hyperparameter space
- Auto-restart on failures
- Persist state for recovery
- Monitor system health
- Send alerts on issues
        """,
    )

    # Mode arguments
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--daemon", "-d",
        action="store_true",
        help="Start as background daemon",
    )
    mode_group.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show current status",
    )
    mode_group.add_argument(
        "--stop",
        action="store_true",
        help="Stop running instance",
    )

    # Configuration arguments
    parser.add_argument(
        "--target-experiments",
        type=int,
        default=50,
        help="Target experiments per day (default: 50)",
    )
    parser.add_argument(
        "--exploration-ratio",
        type=float,
        default=0.3,
        help="Random exploration ratio 0-1 (default: 0.3)",
    )
    parser.add_argument(
        "--session-timeout",
        type=int,
        default=3600,
        help="Session timeout in seconds (default: 3600)",
    )

    args = parser.parse_args()

    # Handle modes
    if args.status:
        show_status()
    elif args.stop:
        stop_orchestrator()
    elif args.daemon:
        start_daemon(args)
    else:
        # Default: foreground
        print("=" * 60)
        print("GIGA TRADER - LONG-RUNNING EXPERIMENT SYSTEM")
        print("=" * 60)
        print(f"Target: {args.target_experiments} experiments/day")
        print(f"Exploration ratio: {args.exploration_ratio}")
        print(f"Session timeout: {args.session_timeout}s")
        print()
        print("Press Ctrl+C to stop")
        print("=" * 60)
        print()
        start_foreground(args)


if __name__ == "__main__":
    main()

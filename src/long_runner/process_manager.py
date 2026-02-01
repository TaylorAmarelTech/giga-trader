"""
Process Manager for Claude Code CLI
====================================
Manages Claude Code CLI subprocess lifecycle with:
- Automatic timeout handling
- Process health monitoring
- Auto-restart on failures
- Session rotation (Claude Code has session limits)
"""

import os
import sys
import time
import json
import signal
import logging
import subprocess
import threading
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from queue import Queue, Empty
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GigaTrader.ProcessManager")


class ProcessState(Enum):
    """States a managed process can be in."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    TIMEOUT = "timeout"
    CRASHED = "crashed"
    RESTARTING = "restarting"


@dataclass
class ProcessStats:
    """Statistics for a managed process."""
    process_id: str
    state: ProcessState = ProcessState.STOPPED
    pid: Optional[int] = None
    started_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    total_runtime_seconds: float = 0.0
    restart_count: int = 0
    crash_count: int = 0
    timeout_count: int = 0
    commands_executed: int = 0
    last_error: str = ""

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["state"] = self.state.value
        d["started_at"] = self.started_at.isoformat() if self.started_at else None
        d["last_activity"] = self.last_activity.isoformat() if self.last_activity else None
        return d


@dataclass
class ClaudeCodeConfig:
    """Configuration for Claude Code CLI."""
    # Claude Code executable path
    executable: str = "claude"

    # Working directory
    working_dir: str = ""

    # Session timeout in seconds (Claude Code sessions have limits)
    session_timeout: int = 3600  # 1 hour default

    # Command timeout in seconds
    command_timeout: int = 600  # 10 minutes default

    # Maximum retries per command
    max_retries: int = 3

    # Delay between retries (seconds)
    retry_delay: int = 30

    # Auto-restart on crash
    auto_restart: bool = True

    # Maximum consecutive crashes before giving up
    max_consecutive_crashes: int = 5

    # Cooldown between restarts (seconds)
    restart_cooldown: int = 60

    # Environment variables to pass
    env_vars: Dict[str, str] = field(default_factory=dict)


class ClaudeCodeProcess:
    """
    Wrapper for a Claude Code CLI process.

    Handles:
    - Starting/stopping the process
    - Sending commands
    - Capturing output
    - Timeout detection
    - Error handling
    """

    def __init__(
        self,
        process_id: str,
        config: ClaudeCodeConfig,
        on_output: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_state_change: Optional[Callable[[ProcessState], None]] = None,
    ):
        self.process_id = process_id
        self.config = config
        self.on_output = on_output
        self.on_error = on_error
        self.on_state_change = on_state_change

        self.stats = ProcessStats(process_id=process_id)
        self._process: Optional[subprocess.Popen] = None
        self._output_thread: Optional[threading.Thread] = None
        self._error_thread: Optional[threading.Thread] = None
        self._running = False
        self._output_queue: Queue = Queue()
        self._last_output_time: Optional[datetime] = None
        self._session_start: Optional[datetime] = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        """Start the Claude Code process."""
        with self._lock:
            if self._running:
                logger.warning(f"[{self.process_id}] Process already running")
                return False

            self._set_state(ProcessState.STARTING)

            try:
                # Build environment
                env = os.environ.copy()
                env.update(self.config.env_vars)

                # Start Claude Code in interactive mode
                cmd = [
                    self.config.executable,
                    "--dangerously-skip-permissions",  # For automation
                ]

                logger.info(f"[{self.process_id}] Starting Claude Code: {' '.join(cmd)}")

                self._process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=self.config.working_dir or None,
                    env=env,
                    text=True,
                    bufsize=1,  # Line buffered
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
                )

                self._running = True
                self._session_start = datetime.now()
                self.stats.started_at = self._session_start
                self.stats.pid = self._process.pid
                self._last_output_time = datetime.now()

                # Start output reader threads
                self._output_thread = threading.Thread(
                    target=self._read_output,
                    daemon=True,
                )
                self._output_thread.start()

                self._error_thread = threading.Thread(
                    target=self._read_errors,
                    daemon=True,
                )
                self._error_thread.start()

                self._set_state(ProcessState.RUNNING)
                logger.info(f"[{self.process_id}] Process started with PID {self._process.pid}")
                return True

            except Exception as e:
                logger.error(f"[{self.process_id}] Failed to start: {e}")
                self.stats.last_error = str(e)
                self._set_state(ProcessState.CRASHED)
                self.stats.crash_count += 1
                return False

    def stop(self, timeout: int = 30) -> bool:
        """Stop the Claude Code process gracefully."""
        with self._lock:
            if not self._running or not self._process:
                return True

            logger.info(f"[{self.process_id}] Stopping process...")

            try:
                # Try graceful shutdown first
                if self._process.stdin:
                    try:
                        self._process.stdin.write("/exit\n")
                        self._process.stdin.flush()
                    except (IOError, OSError, BrokenPipeError):
                        pass

                # Wait for graceful exit
                try:
                    self._process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    logger.warning(f"[{self.process_id}] Graceful shutdown timed out, forcing...")
                    if sys.platform == "win32":
                        self._process.terminate()
                    else:
                        self._process.send_signal(signal.SIGTERM)

                    try:
                        self._process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._process.kill()

                self._running = False
                self._update_runtime()
                self._set_state(ProcessState.STOPPED)
                logger.info(f"[{self.process_id}] Process stopped")
                return True

            except Exception as e:
                logger.error(f"[{self.process_id}] Error stopping process: {e}")
                self.stats.last_error = str(e)
                return False

    def send_command(self, command: str, timeout: Optional[int] = None) -> Optional[str]:
        """
        Send a command to Claude Code and wait for response.

        Args:
            command: The command/prompt to send
            timeout: Optional timeout override

        Returns:
            The output from Claude Code, or None on error
        """
        if not self._running or not self._process:
            logger.error(f"[{self.process_id}] Cannot send command - process not running")
            return None

        timeout = timeout or self.config.command_timeout

        try:
            # Clear output queue
            while not self._output_queue.empty():
                try:
                    self._output_queue.get_nowait()
                except Empty:
                    break

            # Send command
            if self._process.stdin:
                self._process.stdin.write(command + "\n")
                self._process.stdin.flush()
                self.stats.commands_executed += 1
                logger.debug(f"[{self.process_id}] Sent command: {command[:100]}...")

            # Collect output until timeout or completion marker
            output_lines = []
            start_time = time.time()

            while time.time() - start_time < timeout:
                try:
                    line = self._output_queue.get(timeout=1.0)
                    output_lines.append(line)
                    self._last_output_time = datetime.now()
                    self.stats.last_activity = self._last_output_time

                    # Check for completion markers
                    if ">" in line or line.strip().endswith(">"):
                        break

                except Empty:
                    # Check if process died
                    if self._process.poll() is not None:
                        logger.error(f"[{self.process_id}] Process died during command")
                        self._set_state(ProcessState.CRASHED)
                        self.stats.crash_count += 1
                        return None

            return "\n".join(output_lines)

        except Exception as e:
            logger.error(f"[{self.process_id}] Error sending command: {e}")
            self.stats.last_error = str(e)
            return None

    def is_healthy(self) -> bool:
        """Check if the process is healthy."""
        if not self._running or not self._process:
            return False

        # Check if process is still alive
        if self._process.poll() is not None:
            return False

        # Check for session timeout
        if self._session_start:
            session_age = (datetime.now() - self._session_start).total_seconds()
            if session_age > self.config.session_timeout:
                logger.warning(f"[{self.process_id}] Session timeout exceeded")
                self._set_state(ProcessState.TIMEOUT)
                self.stats.timeout_count += 1
                return False

        # Check for activity timeout (stuck process)
        if self._last_output_time:
            idle_time = (datetime.now() - self._last_output_time).total_seconds()
            if idle_time > self.config.command_timeout * 2:
                logger.warning(f"[{self.process_id}] Process appears stuck (no output for {idle_time:.0f}s)")
                return False

        return True

    def needs_restart(self) -> bool:
        """Check if the process needs to be restarted."""
        if self.stats.state in [ProcessState.CRASHED, ProcessState.TIMEOUT]:
            return True

        if not self.is_healthy():
            return True

        return False

    def _read_output(self):
        """Background thread to read stdout."""
        try:
            while self._running and self._process and self._process.stdout:
                line = self._process.stdout.readline()
                if not line:
                    break

                line = line.rstrip()
                self._output_queue.put(line)
                self._last_output_time = datetime.now()

                if self.on_output:
                    self.on_output(line)

        except Exception as e:
            logger.error(f"[{self.process_id}] Output reader error: {e}")

    def _read_errors(self):
        """Background thread to read stderr."""
        try:
            while self._running and self._process and self._process.stderr:
                line = self._process.stderr.readline()
                if not line:
                    break

                line = line.rstrip()

                if self.on_error:
                    self.on_error(line)
                else:
                    logger.warning(f"[{self.process_id}] STDERR: {line}")

        except Exception as e:
            logger.error(f"[{self.process_id}] Error reader error: {e}")

    def _set_state(self, state: ProcessState):
        """Update process state and trigger callback."""
        self.stats.state = state
        if self.on_state_change:
            self.on_state_change(state)

    def _update_runtime(self):
        """Update total runtime statistics."""
        if self.stats.started_at:
            runtime = (datetime.now() - self.stats.started_at).total_seconds()
            self.stats.total_runtime_seconds += runtime


class ProcessManager:
    """
    Manages multiple Claude Code processes with:
    - Process pool management
    - Auto-restart and recovery
    - Health monitoring
    - Session rotation
    """

    def __init__(
        self,
        config: ClaudeCodeConfig,
        max_processes: int = 1,
        state_file: Optional[Path] = None,
    ):
        self.config = config
        self.max_processes = max_processes
        self.state_file = state_file

        self.processes: Dict[str, ClaudeCodeProcess] = {}
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._consecutive_crashes = 0

        # Callbacks
        self.on_process_crash: Optional[Callable[[str], None]] = None
        self.on_process_restart: Optional[Callable[[str], None]] = None
        self.on_all_crashed: Optional[Callable[[], None]] = None

    def start(self) -> bool:
        """Start the process manager and monitoring."""
        if self._running:
            return False

        self._running = True

        # Start monitor thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
        )
        self._monitor_thread.start()

        logger.info(f"ProcessManager started with max {self.max_processes} processes")
        return True

    def stop(self):
        """Stop all processes and the manager."""
        self._running = False

        # Stop all processes
        for proc in self.processes.values():
            proc.stop()

        self.processes.clear()
        logger.info("ProcessManager stopped")

    def create_process(self, process_id: Optional[str] = None) -> Optional[ClaudeCodeProcess]:
        """Create a new managed process."""
        with self._lock:
            if len(self.processes) >= self.max_processes:
                logger.warning("Maximum processes reached")
                return None

            process_id = process_id or f"proc_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

            if process_id in self.processes:
                logger.warning(f"Process {process_id} already exists")
                return self.processes[process_id]

            proc = ClaudeCodeProcess(
                process_id=process_id,
                config=self.config,
                on_state_change=lambda state: self._on_process_state_change(process_id, state),
            )

            self.processes[process_id] = proc
            logger.info(f"Created process {process_id}")
            return proc

    def get_process(self, process_id: str) -> Optional[ClaudeCodeProcess]:
        """Get a process by ID."""
        return self.processes.get(process_id)

    def get_healthy_process(self) -> Optional[ClaudeCodeProcess]:
        """Get a healthy running process, or create one if needed."""
        # Try to find a healthy process
        for proc in self.processes.values():
            if proc.is_healthy():
                return proc

        # Try to restart an unhealthy one
        for proc in self.processes.values():
            if proc.needs_restart():
                if self._restart_process(proc.process_id):
                    return proc

        # Create a new one
        proc = self.create_process()
        if proc and proc.start():
            return proc

        return None

    def execute_command(
        self,
        command: str,
        process_id: Optional[str] = None,
        retries: Optional[int] = None,
    ) -> Optional[str]:
        """
        Execute a command on a Claude Code process.

        Args:
            command: The command to execute
            process_id: Optional specific process to use
            retries: Number of retries on failure

        Returns:
            Command output or None on failure
        """
        retries = retries if retries is not None else self.config.max_retries

        for attempt in range(retries + 1):
            # Get a process
            if process_id:
                proc = self.get_process(process_id)
            else:
                proc = self.get_healthy_process()

            if not proc:
                logger.error("No healthy process available")
                time.sleep(self.config.retry_delay)
                continue

            # Execute command
            result = proc.send_command(command)

            if result is not None:
                self._consecutive_crashes = 0
                return result

            # Command failed
            logger.warning(f"Command failed (attempt {attempt + 1}/{retries + 1})")

            if attempt < retries:
                # Try to restart the process
                if not proc.is_healthy():
                    self._restart_process(proc.process_id)
                time.sleep(self.config.retry_delay)

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all processes."""
        return {
            "manager": {
                "running": self._running,
                "max_processes": self.max_processes,
                "active_processes": len(self.processes),
                "consecutive_crashes": self._consecutive_crashes,
            },
            "processes": {
                pid: proc.stats.to_dict()
                for pid, proc in self.processes.items()
            },
        }

    def _restart_process(self, process_id: str) -> bool:
        """Restart a specific process."""
        proc = self.processes.get(process_id)
        if not proc:
            return False

        logger.info(f"Restarting process {process_id}")
        proc.stats.restart_count += 1

        # Stop existing process
        proc.stop()

        # Cooldown
        time.sleep(self.config.restart_cooldown)

        # Restart
        if proc.start():
            if self.on_process_restart:
                self.on_process_restart(process_id)
            return True

        return False

    def _on_process_state_change(self, process_id: str, state: ProcessState):
        """Handle process state changes."""
        if state == ProcessState.CRASHED:
            self._consecutive_crashes += 1

            if self.on_process_crash:
                self.on_process_crash(process_id)

            # Check for too many crashes
            if self._consecutive_crashes >= self.config.max_consecutive_crashes:
                logger.error(f"Too many consecutive crashes ({self._consecutive_crashes})")
                if self.on_all_crashed:
                    self.on_all_crashed()

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                with self._lock:
                    for process_id, proc in list(self.processes.items()):
                        # Check health
                        if not proc.is_healthy() and proc.stats.state == ProcessState.RUNNING:
                            logger.warning(f"Process {process_id} became unhealthy")

                            if self.config.auto_restart:
                                self._restart_process(process_id)

                # Save state
                if self.state_file:
                    self._save_state()

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(5)

    def _save_state(self):
        """Save current state to file."""
        try:
            state = self.get_stats()
            state["timestamp"] = datetime.now().isoformat()

            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save state: {e}")


# =============================================================================
# MAIN ENTRY POINT FOR TESTING
# =============================================================================

def main():
    """Test the process manager."""
    config = ClaudeCodeConfig(
        working_dir=str(Path(__file__).parent.parent.parent),
        session_timeout=3600,
        command_timeout=300,
    )

    manager = ProcessManager(
        config=config,
        max_processes=1,
        state_file=Path(__file__).parent.parent.parent / "logs" / "process_manager_state.json",
    )

    manager.start()

    # Create a process
    proc = manager.create_process("main")
    if proc and proc.start():
        print("Process started successfully")

        # Test command
        result = manager.execute_command("What is 2 + 2?")
        print(f"Result: {result}")

    # Keep running
    try:
        while True:
            stats = manager.get_stats()
            print(f"\nProcess Stats: {json.dumps(stats, indent=2, default=str)}")
            time.sleep(60)
    except KeyboardInterrupt:
        manager.stop()


if __name__ == "__main__":
    main()

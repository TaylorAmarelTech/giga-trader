"""
State Manager
=============
Thread-safe, atomic state persistence for long-running processes.
Adapted from archive/orphaned_2026-02-03/long_runner/state_manager.py.

Features:
- Thread-safe with RLock
- Atomic writes (write to temp, then rename)
- Automatic backups with rotation
- State versioning for migrations
"""

import json
import os
import shutil
import hashlib
import logging
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("STATE_MANAGER")

project_root = Path(__file__).parent.parent.parent


@dataclass
class SystemState:
    """Persistent system state."""
    version: int = 1
    orchestrator_running: bool = False
    mode: str = "IDLE"
    experiments_completed: int = 0
    experiments_total: int = 0
    models_trained: int = 0
    best_auc: float = 0.0
    last_training_time: str = ""
    last_backtest_time: str = ""
    last_trade_time: str = ""
    errors: list = field(default_factory=list)
    custom_data: Dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "SystemState":
        # Handle version migrations
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Write JSON to a file atomically (temp + rename).

    Prevents partial reads by the dashboard or other consumers.
    On Windows, os.replace is atomic at the file-system level.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=str)
        os.replace(tmp_path, path)  # Atomic on NTFS and POSIX
    except Exception:
        # Clean up temp file on failure
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


class StateManager:
    """
    Thread-safe state persistence manager.

    Provides atomic writes (temp → rename) with checksum verification
    and automatic backup rotation.

    Usage:
        sm = StateManager()
        sm.update(experiments_completed=5, best_auc=0.72)
        sm.save()

        # Later...
        sm = StateManager()
        state = sm.load()
        print(state.experiments_completed)  # 5
    """

    def __init__(
        self,
        state_dir: Optional[str] = None,
        max_backups: int = 10,
        auto_save_interval: int = 300,
    ):
        self.state_dir = Path(state_dir) if state_dir else (project_root / "logs")
        self.state_file = self.state_dir / "system_state.json"
        self.backup_dir = self.state_dir / "state_backups"
        self.max_backups = max_backups
        self.auto_save_interval = auto_save_interval

        self._state = SystemState()
        self._lock = threading.RLock()
        self._auto_save_thread: Optional[threading.Thread] = None
        self._running = False

        # Load existing state
        self._load()

    def _load(self):
        """Load state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)

                # Verify checksum if present
                checksum = data.pop("_checksum", None)
                if checksum:
                    computed = self._compute_checksum(data)
                    if computed != checksum:
                        logger.warning("State file checksum mismatch, loading anyway")

                self._state = SystemState.from_dict(data)
                logger.info(f"Loaded state from {self.state_file}")
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
                self._state = SystemState()
        else:
            logger.info("No existing state file, starting fresh")

    def _compute_checksum(self, data: Dict) -> str:
        """Compute MD5 checksum of state data."""
        content = json.dumps(data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    @property
    def state(self) -> SystemState:
        """Get current state (read-only reference)."""
        with self._lock:
            return self._state

    def update(self, **kwargs):
        """Update state fields."""
        with self._lock:
            for key, value in kwargs.items():
                if key == "custom_data":
                    # Merge custom_data rather than replace
                    self._state.custom_data.update(value)
                elif hasattr(self._state, key):
                    setattr(self._state, key, value)
                else:
                    # Store unknown keys in custom_data
                    self._state.custom_data[key] = value

            self._state.updated_at = datetime.now().isoformat()

    def save(self, force: bool = False) -> bool:
        """
        Atomically save state to disk.

        Uses temp file → rename pattern to prevent corruption.
        """
        with self._lock:
            try:
                self.state_dir.mkdir(parents=True, exist_ok=True)

                # Serialize
                data = self._state.to_dict()
                checksum = self._compute_checksum(data)
                data["_checksum"] = checksum

                # Write to temp file
                temp_path = self.state_file.with_suffix(".tmp")
                with open(temp_path, "w") as f:
                    json.dump(data, f, indent=2, default=str)

                # Atomic rename
                if os.name == "nt":
                    # Windows doesn't support atomic rename over existing file
                    if self.state_file.exists():
                        self.state_file.unlink()
                os.rename(str(temp_path), str(self.state_file))

                logger.debug(f"State saved to {self.state_file}")
                return True

            except Exception as e:
                logger.error(f"Failed to save state: {e}")
                return False

    def create_backup(self) -> Optional[str]:
        """Create a timestamped backup of current state."""
        with self._lock:
            try:
                self.backup_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.backup_dir / f"state_{timestamp}.json"

                if self.state_file.exists():
                    shutil.copy2(str(self.state_file), str(backup_path))
                else:
                    # Save current state as backup
                    data = self._state.to_dict()
                    with open(backup_path, "w") as f:
                        json.dump(data, f, indent=2, default=str)

                # Rotate old backups
                self._rotate_backups()

                logger.debug(f"Backup created: {backup_path}")
                return str(backup_path)
            except Exception as e:
                logger.error(f"Failed to create backup: {e}")
                return None

    def _rotate_backups(self):
        """Remove old backups beyond max_backups."""
        if not self.backup_dir.exists():
            return

        backups = sorted(
            self.backup_dir.glob("state_*.json"),
            key=lambda p: p.stat().st_mtime,
        )

        while len(backups) > self.max_backups:
            oldest = backups.pop(0)
            oldest.unlink()
            logger.debug(f"Rotated old backup: {oldest}")

    def start_auto_save(self):
        """Start background auto-save thread."""
        if self._running:
            return

        self._running = True
        self._auto_save_thread = threading.Thread(
            target=self._auto_save_loop, daemon=True
        )
        self._auto_save_thread.start()
        logger.info(f"Auto-save started (interval: {self.auto_save_interval}s)")

    def stop_auto_save(self):
        """Stop background auto-save."""
        self._running = False
        if self._auto_save_thread:
            self._auto_save_thread.join(timeout=10)
        logger.info("Auto-save stopped")

    def _auto_save_loop(self):
        """Background auto-save loop."""
        while self._running:
            time.sleep(self.auto_save_interval)
            if self._running:
                self.save()
                # Create periodic backup (every 10 saves)
                if hasattr(self, "_save_count"):
                    self._save_count += 1
                else:
                    self._save_count = 1
                if self._save_count % 10 == 0:
                    self.create_backup()

"""
State Manager
=============
Handles persistence and recovery for long-running experiments.

Features:
- Atomic state saves with backup
- Corruption detection and recovery
- State versioning
- Automatic compaction
"""

import os
import json
import shutil
import logging
import hashlib
import threading
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from enum import Enum
import gzip

logger = logging.getLogger("GigaTrader.StateManager")


class StateVersion(Enum):
    """State file versions for migration support."""
    V1 = "1.0"
    V2 = "2.0"
    CURRENT = "2.0"


@dataclass
class SystemState:
    """Complete system state for persistence."""
    version: str = StateVersion.CURRENT.value
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())

    # Orchestrator state
    orchestrator_running: bool = False
    orchestrator_start_time: Optional[str] = None
    total_runtime_seconds: float = 0.0

    # Experiment state
    current_experiment_id: Optional[str] = None
    experiments_completed: int = 0
    experiments_failed: int = 0
    last_experiment_time: Optional[str] = None

    # Process state
    active_processes: int = 0
    total_process_restarts: int = 0

    # Grid search state
    grid_search_progress: float = 0.0
    best_score_achieved: float = 0.0
    best_experiment_id: Optional[str] = None

    # Custom data
    custom_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "SystemState":
        return cls(**d)


class StateManager:
    """
    Manages system state persistence with:
    - Atomic writes (write to temp, then rename)
    - Backup management
    - Corruption detection
    - State compression
    """

    def __init__(
        self,
        state_dir: Path,
        backup_count: int = 10,
        auto_save_interval: int = 60,  # seconds
        compress_backups: bool = True,
    ):
        self.state_dir = Path(state_dir)
        self.backup_count = backup_count
        self.auto_save_interval = auto_save_interval
        self.compress_backups = compress_backups

        # Ensure directories exist
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "backups").mkdir(exist_ok=True)

        # State files
        self.state_file = self.state_dir / "system_state.json"
        self.state_temp = self.state_dir / "system_state.tmp"
        self.checksum_file = self.state_dir / "state_checksum.txt"

        # Current state
        self.state: SystemState = SystemState()
        self._lock = threading.RLock()
        self._dirty = False
        self._auto_save_thread: Optional[threading.Thread] = None
        self._running = False

        # Load existing state
        self._load_state()

    def start_auto_save(self):
        """Start automatic state saving."""
        if self._running:
            return

        self._running = True
        self._auto_save_thread = threading.Thread(
            target=self._auto_save_loop,
            daemon=True,
        )
        self._auto_save_thread.start()
        logger.info("Auto-save started")

    def stop_auto_save(self):
        """Stop automatic state saving."""
        self._running = False
        if self._auto_save_thread:
            self._auto_save_thread.join(timeout=5)
        # Final save
        self.save()
        logger.info("Auto-save stopped")

    def get_state(self) -> SystemState:
        """Get the current system state."""
        with self._lock:
            return self.state

    def update_state(self, **kwargs) -> None:
        """Update specific state fields."""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.state, key):
                    setattr(self.state, key, value)
                else:
                    self.state.custom_data[key] = value

            self.state.last_modified = datetime.now().isoformat()
            self._dirty = True

    def set_custom_data(self, key: str, value: Any) -> None:
        """Set custom data in state."""
        with self._lock:
            self.state.custom_data[key] = value
            self.state.last_modified = datetime.now().isoformat()
            self._dirty = True

    def get_custom_data(self, key: str, default: Any = None) -> Any:
        """Get custom data from state."""
        with self._lock:
            return self.state.custom_data.get(key, default)

    def save(self, force: bool = False) -> bool:
        """
        Save current state to disk.

        Args:
            force: Save even if not dirty

        Returns:
            True if saved successfully
        """
        with self._lock:
            if not self._dirty and not force:
                return True

            try:
                # Prepare state
                state_dict = self.state.to_dict()
                state_json = json.dumps(state_dict, indent=2, default=str)

                # Calculate checksum
                checksum = hashlib.sha256(state_json.encode()).hexdigest()

                # Write to temp file
                with open(self.state_temp, "w") as f:
                    f.write(state_json)

                # Verify temp file
                with open(self.state_temp, "r") as f:
                    verify = f.read()
                    verify_checksum = hashlib.sha256(verify.encode()).hexdigest()

                if checksum != verify_checksum:
                    logger.error("State verification failed!")
                    return False

                # Create backup of existing state
                if self.state_file.exists():
                    self._create_backup()

                # Atomic rename
                if os.path.exists(self.state_file):
                    os.remove(self.state_file)
                os.rename(self.state_temp, self.state_file)

                # Save checksum
                with open(self.checksum_file, "w") as f:
                    f.write(checksum)

                self._dirty = False
                logger.debug("State saved successfully")
                return True

            except Exception as e:
                logger.error(f"Failed to save state: {e}")
                return False

    def restore_from_backup(self, backup_index: int = 0) -> bool:
        """
        Restore state from a backup.

        Args:
            backup_index: 0 = most recent, 1 = second most recent, etc.

        Returns:
            True if restored successfully
        """
        backups = self._list_backups()

        if backup_index >= len(backups):
            logger.error(f"Backup index {backup_index} not found")
            return False

        backup_file = backups[backup_index]

        try:
            # Read backup
            if backup_file.suffix == ".gz":
                with gzip.open(backup_file, "rt") as f:
                    state_dict = json.load(f)
            else:
                with open(backup_file, "r") as f:
                    state_dict = json.load(f)

            # Migrate if needed
            state_dict = self._migrate_state(state_dict)

            # Restore
            self.state = SystemState.from_dict(state_dict)
            self._dirty = True
            self.save(force=True)

            logger.info(f"Restored from backup: {backup_file.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return False

    def verify_integrity(self) -> bool:
        """Verify state file integrity."""
        if not self.state_file.exists():
            return False

        if not self.checksum_file.exists():
            return False

        try:
            with open(self.state_file, "r") as f:
                content = f.read()

            with open(self.checksum_file, "r") as f:
                expected_checksum = f.read().strip()

            actual_checksum = hashlib.sha256(content.encode()).hexdigest()
            return actual_checksum == expected_checksum

        except Exception as e:
            logger.error(f"Integrity check failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get state manager statistics."""
        backups = self._list_backups()

        return {
            "state_file": str(self.state_file),
            "state_exists": self.state_file.exists(),
            "integrity_valid": self.verify_integrity(),
            "backup_count": len(backups),
            "auto_save_running": self._running,
            "dirty": self._dirty,
            "last_modified": self.state.last_modified,
            "version": self.state.version,
        }

    def _load_state(self) -> bool:
        """Load state from disk."""
        if not self.state_file.exists():
            logger.info("No existing state file, starting fresh")
            return False

        # Verify integrity
        if not self.verify_integrity():
            logger.warning("State file corrupted, attempting recovery...")
            return self.restore_from_backup()

        try:
            with open(self.state_file, "r") as f:
                state_dict = json.load(f)

            # Migrate if needed
            state_dict = self._migrate_state(state_dict)

            self.state = SystemState.from_dict(state_dict)
            logger.info(f"State loaded: {self.state.experiments_completed} experiments completed")
            return True

        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return self.restore_from_backup()

    def _create_backup(self):
        """Create a backup of current state."""
        if not self.state_file.exists():
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.state_dir / "backups"

        if self.compress_backups:
            backup_file = backup_dir / f"state_{timestamp}.json.gz"
            with open(self.state_file, "rb") as f_in:
                with gzip.open(backup_file, "wb") as f_out:
                    f_out.write(f_in.read())
        else:
            backup_file = backup_dir / f"state_{timestamp}.json"
            shutil.copy2(self.state_file, backup_file)

        # Clean old backups
        self._cleanup_backups()

    def _cleanup_backups(self):
        """Remove old backups beyond the retention limit."""
        backups = self._list_backups()

        while len(backups) > self.backup_count:
            oldest = backups.pop()
            oldest.unlink()
            logger.debug(f"Removed old backup: {oldest.name}")

    def _list_backups(self) -> List[Path]:
        """List backup files sorted by date (newest first)."""
        backup_dir = self.state_dir / "backups"
        backups = list(backup_dir.glob("state_*.json*"))
        backups.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return backups

    def _migrate_state(self, state_dict: Dict) -> Dict:
        """Migrate state from older versions."""
        version = state_dict.get("version", "1.0")

        if version == StateVersion.CURRENT.value:
            return state_dict

        # V1 -> V2 migration
        if version == "1.0":
            state_dict["version"] = "2.0"
            state_dict.setdefault("custom_data", {})
            logger.info("Migrated state from V1 to V2")

        return state_dict

    def _auto_save_loop(self):
        """Background auto-save loop."""
        while self._running:
            try:
                if self._dirty:
                    self.save()
            except Exception as e:
                logger.error(f"Auto-save error: {e}")

            # Sleep in small increments for responsive shutdown
            for _ in range(self.auto_save_interval):
                if not self._running:
                    break
                import time
                time.sleep(1)


# =============================================================================
# MAIN ENTRY POINT FOR TESTING
# =============================================================================

def main():
    """Test the state manager."""
    state_dir = Path(__file__).parent.parent.parent / "logs" / "state"

    manager = StateManager(state_dir=state_dir)
    manager.start_auto_save()

    print(f"Initial stats: {json.dumps(manager.get_stats(), indent=2)}")

    # Update state
    manager.update_state(
        experiments_completed=10,
        best_score_achieved=0.75,
        custom_key="custom_value",
    )

    manager.set_custom_data("test_data", {"nested": "value"})

    # Force save
    manager.save(force=True)

    print(f"\nAfter update: {json.dumps(manager.get_state().to_dict(), indent=2)}")
    print(f"\nFinal stats: {json.dumps(manager.get_stats(), indent=2)}")

    manager.stop_auto_save()


if __name__ == "__main__":
    main()

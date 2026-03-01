"""
Staleness checker for production models.

Ensures models are fresh enough to use for trading decisions.
Checks file modification time, training date metadata, or registry timestamps.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class StalenessChecker:
    """
    Checks model freshness. Refuses to trade with models older than max_age_days.

    Can check:
    1. Model file modification time
    2. Training date stored in model metadata
    3. Registry database entry timestamp
    """

    def __init__(self, max_age_days: int = 30):
        """
        Args:
            max_age_days: Maximum allowed model age in days.
                Models older than this are considered stale.
        """
        self.max_age_days = max_age_days

    def is_stale(
        self,
        model_path: Optional[Path] = None,
        training_date: Optional[str] = None,
        reference_date: Optional[datetime] = None,
    ) -> Tuple[bool, Dict]:
        """
        Check if a model is stale.

        Args:
            model_path: Path to model file (checks file modification time).
            training_date: ISO format date string when model was trained.
            reference_date: Date to compare against (default: now).

        Returns:
            Tuple of (is_stale: bool, info: Dict) where info has:
              - "age_days": int
              - "max_age_days": int
              - "is_stale": bool
              - "source": "file_mtime" or "training_date" or "unknown"
              - "checked_at": str (ISO format)
        """
        if reference_date is None:
            reference_date = datetime.now()

        # Strip tzinfo for consistent comparison
        reference_date = reference_date.replace(tzinfo=None)

        checked_at = reference_date.isoformat()

        # Prefer training_date over file mtime (more accurate)
        if training_date is not None:
            return self._check_training_date(training_date, reference_date, checked_at)

        if model_path is not None:
            return self._check_file_mtime(model_path, reference_date, checked_at)

        # Neither provided
        logger.warning("No model_path or training_date provided; treating as stale")
        return True, {
            "age_days": -1,
            "max_age_days": self.max_age_days,
            "is_stale": True,
            "source": "unknown",
            "checked_at": checked_at,
        }

    def _check_training_date(
        self,
        training_date: str,
        reference_date: datetime,
        checked_at: str,
    ) -> Tuple[bool, Dict]:
        """Check staleness using an ISO-format training date string."""
        try:
            trained_dt = datetime.fromisoformat(training_date)
            trained_dt = trained_dt.replace(tzinfo=None)
        except (ValueError, TypeError) as exc:
            logger.warning("Invalid training_date '%s': %s; treating as stale", training_date, exc)
            return True, {
                "age_days": -1,
                "max_age_days": self.max_age_days,
                "is_stale": True,
                "source": "training_date",
                "checked_at": checked_at,
            }

        age_days = (reference_date - trained_dt).days
        stale = age_days > self.max_age_days

        if stale:
            logger.warning(
                "Model trained on %s is %d days old (max %d) - STALE",
                training_date, age_days, self.max_age_days,
            )
        else:
            logger.info(
                "Model trained on %s is %d days old (max %d) - fresh",
                training_date, age_days, self.max_age_days,
            )

        return stale, {
            "age_days": age_days,
            "max_age_days": self.max_age_days,
            "is_stale": stale,
            "source": "training_date",
            "checked_at": checked_at,
        }

    def _check_file_mtime(
        self,
        model_path: Path,
        reference_date: datetime,
        checked_at: str,
    ) -> Tuple[bool, Dict]:
        """Check staleness using file modification time."""
        model_path = Path(model_path)

        if not model_path.exists():
            logger.warning("Model file not found: %s; treating as stale", model_path)
            return True, {
                "age_days": -1,
                "max_age_days": self.max_age_days,
                "is_stale": True,
                "source": "file_mtime",
                "checked_at": checked_at,
            }

        mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
        mtime = mtime.replace(tzinfo=None)
        age_days = (reference_date - mtime).days
        stale = age_days > self.max_age_days

        if stale:
            logger.warning(
                "Model %s modified %d days ago (max %d) - STALE",
                model_path.name, age_days, self.max_age_days,
            )
        else:
            logger.info(
                "Model %s modified %d days ago (max %d) - fresh",
                model_path.name, age_days, self.max_age_days,
            )

        return stale, {
            "age_days": age_days,
            "max_age_days": self.max_age_days,
            "is_stale": stale,
            "source": "file_mtime",
            "checked_at": checked_at,
        }

    def check_models(
        self,
        model_dir: Path,
        pattern: str = "*.joblib",
    ) -> Dict[str, Dict]:
        """
        Check all models in a directory for staleness.

        Args:
            model_dir: Directory containing model files.
            pattern: Glob pattern for model files.

        Returns:
            Dict mapping filename to staleness info dict.
        """
        model_dir = Path(model_dir)
        results: Dict[str, Dict] = {}

        if not model_dir.exists():
            logger.warning("Model directory does not exist: %s", model_dir)
            return results

        for model_file in sorted(model_dir.glob(pattern)):
            _, info = self.is_stale(model_path=model_file)
            results[model_file.name] = info

        return results

    def get_freshest_model(
        self,
        model_dir: Path,
        pattern: str = "*.joblib",
    ) -> Optional[Tuple[Path, Dict]]:
        """
        Find the freshest (most recently modified) model in a directory.

        Returns:
            Tuple of (model_path, info_dict) or None if no models found.
        """
        model_dir = Path(model_dir)

        if not model_dir.exists():
            logger.warning("Model directory does not exist: %s", model_dir)
            return None

        best_path: Optional[Path] = None
        best_info: Optional[Dict] = None
        best_age: Optional[int] = None

        for model_file in model_dir.glob(pattern):
            _, info = self.is_stale(model_path=model_file)
            age = info["age_days"]

            # Skip entries where age could not be determined
            if age < 0:
                continue

            if best_age is None or age < best_age:
                best_age = age
                best_path = model_file
                best_info = info

        if best_path is None:
            return None

        return best_path, best_info

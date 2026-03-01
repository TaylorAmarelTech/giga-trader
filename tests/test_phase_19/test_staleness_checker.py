"""
Tests for StalenessChecker.

Validates model freshness detection using file modification times,
training date metadata, and directory scanning.
"""

import os
import pytest
from datetime import datetime, timedelta
from pathlib import Path

from src.phase_19_paper_trading.staleness_checker import StalenessChecker


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def checker():
    """Default checker with 30-day max age."""
    return StalenessChecker(max_age_days=30)


@pytest.fixture
def reference_date():
    """Fixed reference date for deterministic tests."""
    return datetime(2026, 2, 23, 12, 0, 0)


@pytest.fixture
def model_dir(tmp_path):
    """Directory with a few fake .joblib model files at known ages."""
    # Create 3 model files with different modification times
    now = datetime.now()

    fresh = tmp_path / "model_fresh.joblib"
    fresh.write_bytes(b"")
    # Set mtime to 1 day ago
    one_day_ago = (now - timedelta(days=1)).timestamp()
    os.utime(fresh, (one_day_ago, one_day_ago))

    medium = tmp_path / "model_medium.joblib"
    medium.write_bytes(b"")
    # Set mtime to 15 days ago
    fifteen_days_ago = (now - timedelta(days=15)).timestamp()
    os.utime(medium, (fifteen_days_ago, fifteen_days_ago))

    stale = tmp_path / "model_stale.joblib"
    stale.write_bytes(b"")
    # Set mtime to 60 days ago
    sixty_days_ago = (now - timedelta(days=60)).timestamp()
    os.utime(stale, (sixty_days_ago, sixty_days_ago))

    return tmp_path


# ─── Single Model Tests ──────────────────────────────────────────────────────

class TestIsStaleFreshModel:

    def test_fresh_model_not_stale(self, checker, reference_date):
        """A model trained 1 day ago should NOT be stale."""
        training_date = (reference_date - timedelta(days=1)).isoformat()
        stale, info = checker.is_stale(
            training_date=training_date,
            reference_date=reference_date,
        )
        assert stale is False
        assert info["is_stale"] is False
        assert info["age_days"] == 1
        assert info["source"] == "training_date"
        assert info["max_age_days"] == 30


class TestIsStaleOldModel:

    def test_stale_model_detected(self, reference_date):
        """A 60-day-old model with max_age=30 should be stale."""
        checker = StalenessChecker(max_age_days=30)
        training_date = (reference_date - timedelta(days=60)).isoformat()
        stale, info = checker.is_stale(
            training_date=training_date,
            reference_date=reference_date,
        )
        assert stale is True
        assert info["is_stale"] is True
        assert info["age_days"] == 60
        assert info["source"] == "training_date"


class TestBoundaryAge:

    def test_exact_max_age_is_not_stale(self, reference_date):
        """A model at exactly max_age_days should NOT be stale (> not >=)."""
        checker = StalenessChecker(max_age_days=30)
        training_date = (reference_date - timedelta(days=30)).isoformat()
        stale, info = checker.is_stale(
            training_date=training_date,
            reference_date=reference_date,
        )
        assert stale is False
        assert info["age_days"] == 30

    def test_one_day_over_max_is_stale(self, reference_date):
        """A model at max_age_days + 1 should be stale."""
        checker = StalenessChecker(max_age_days=30)
        training_date = (reference_date - timedelta(days=31)).isoformat()
        stale, info = checker.is_stale(
            training_date=training_date,
            reference_date=reference_date,
        )
        assert stale is True
        assert info["age_days"] == 31


class TestTrainingDatePriority:

    def test_training_date_preferred_over_file_mtime(self, checker, tmp_path, reference_date):
        """When both model_path and training_date are provided, training_date wins."""
        # Create a file that was modified 2 days ago
        model_file = tmp_path / "model.joblib"
        model_file.write_bytes(b"")
        two_days_ago = (reference_date - timedelta(days=2)).timestamp()
        os.utime(model_file, (two_days_ago, two_days_ago))

        # But training_date says it was trained 50 days ago (stale)
        training_date = (reference_date - timedelta(days=50)).isoformat()

        stale, info = checker.is_stale(
            model_path=model_file,
            training_date=training_date,
            reference_date=reference_date,
        )
        # training_date should take priority: 50 days > 30 day limit
        assert stale is True
        assert info["source"] == "training_date"
        assert info["age_days"] == 50


class TestMissingFile:

    def test_missing_file_is_stale(self, checker):
        """A non-existent model file should return is_stale=True."""
        missing = Path("/nonexistent/path/model.joblib")
        stale, info = checker.is_stale(model_path=missing)
        assert stale is True
        assert info["is_stale"] is True
        assert info["age_days"] == -1
        assert info["source"] == "file_mtime"

    def test_no_inputs_is_stale(self, checker):
        """No model_path or training_date should return is_stale=True."""
        stale, info = checker.is_stale()
        assert stale is True
        assert info["source"] == "unknown"


# ─── Directory Scanning Tests ────────────────────────────────────────────────

class TestCheckModels:

    def test_finds_all_models(self, checker, model_dir):
        """check_models() should return info for every .joblib file."""
        results = checker.check_models(model_dir)
        assert len(results) == 3
        assert "model_fresh.joblib" in results
        assert "model_medium.joblib" in results
        assert "model_stale.joblib" in results

    def test_identifies_stale_and_fresh(self, checker, model_dir):
        """The 60-day-old model should be stale; 1-day-old should be fresh."""
        results = checker.check_models(model_dir)
        assert results["model_fresh.joblib"]["is_stale"] is False
        assert results["model_stale.joblib"]["is_stale"] is True

    def test_empty_directory(self, checker, tmp_path):
        """An empty directory should return an empty dict."""
        results = checker.check_models(tmp_path)
        assert results == {}

    def test_nonexistent_directory(self, checker):
        """A non-existent directory should return an empty dict."""
        results = checker.check_models(Path("/nonexistent/dir"))
        assert results == {}


class TestGetFreshestModel:

    def test_returns_freshest(self, checker, model_dir):
        """get_freshest_model() should return the 1-day-old model."""
        result = checker.get_freshest_model(model_dir)
        assert result is not None
        path, info = result
        assert path.name == "model_fresh.joblib"
        assert info["age_days"] <= 2  # ~1 day, allow small rounding

    def test_empty_directory_returns_none(self, checker, tmp_path):
        """An empty directory should return None."""
        result = checker.get_freshest_model(tmp_path)
        assert result is None

    def test_nonexistent_directory_returns_none(self, checker):
        """A non-existent directory should return None."""
        result = checker.get_freshest_model(Path("/nonexistent/dir"))
        assert result is None

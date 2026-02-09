"""
Test HistoricalConstituentProvider initialization, date queries,
and survivorship bias reporting.
"""

import sys
from pathlib import Path
from datetime import date, datetime
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase_01_data_acquisition.historical_constituents import (
    HistoricalConstituentProvider,
    ConstituentChange,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provider():
    """Create a HistoricalConstituentProvider using built-in data."""
    # Use a non-existent CSV so it falls back to built-in changes
    return HistoricalConstituentProvider(changes_csv="__nonexistent_test__.csv")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_provider_initialization(provider):
    """Provider should initialize with built-in changes."""
    assert isinstance(provider, HistoricalConstituentProvider)
    assert len(provider.changes) > 0


def test_changes_are_constituent_change_objects(provider):
    """Each change should be a ConstituentChange dataclass."""
    for change in provider.changes:
        assert isinstance(change, ConstituentChange)
        assert isinstance(change.date, date)
        assert isinstance(change.ticker, str)
        assert change.action in ("ADD", "REMOVE")


def test_get_constituents_at_date_returns_list(provider):
    """get_constituents_at_date should return a list of ticker strings."""
    # Mock current constituents so we don't hit Wikipedia
    provider._current_constituents = provider._get_top50_fallback()

    result = provider.get_constituents_at_date(date(2024, 1, 15))
    assert isinstance(result, list)
    assert len(result) > 0
    for ticker in result:
        assert isinstance(ticker, str)
        assert len(ticker) >= 1


def test_current_constituents_fallback_has_about_50_tickers(provider):
    """Top-50 fallback should have ~50 tickers."""
    fallback = provider._get_top50_fallback()
    assert isinstance(fallback, list)
    assert len(fallback) == 50
    assert "AAPL" in fallback
    assert "MSFT" in fallback


def test_historical_date_may_differ_from_current(provider):
    """A historical date should potentially have different constituents than current."""
    provider._current_constituents = provider._get_top50_fallback()

    current = set(provider.get_current_constituents())
    # TSLA was added 2020-12-21, so if we query before that...
    historical = set(provider.get_constituents_at_date(date(2019, 6, 15)))

    # They may differ because changes are applied
    # At minimum, both should be non-empty lists
    assert len(current) > 0
    assert len(historical) > 0


def test_get_constituent_changes_between(provider):
    """Should return changes within a date range."""
    changes = provider.get_constituent_changes_between(
        date(2020, 1, 1), date(2020, 12, 31)
    )
    assert isinstance(changes, list)
    for change in changes:
        assert isinstance(change, ConstituentChange)
        assert date(2020, 1, 1) <= change.date <= date(2020, 12, 31)


def test_survivorship_bias_report_generation(provider):
    """get_survivorship_bias_report should return a dict with expected keys."""
    provider._current_constituents = provider._get_top50_fallback()

    report = provider.get_survivorship_bias_report(
        date(2020, 1, 1), date(2024, 12, 31)
    )
    assert isinstance(report, dict)
    expected_keys = [
        "backtest_period",
        "total_changes",
        "additions",
        "removals",
        "start_count",
        "end_count",
        "survivors",
        "survival_rate",
        "removed_tickers",
        "added_tickers",
        "bias_risk",
        "recommendation",
    ]
    for key in expected_keys:
        assert key in report, f"Report should contain '{key}'"

    assert report["bias_risk"] in ("LOW", "MEDIUM", "HIGH")
    assert isinstance(report["survival_rate"], float)
    assert 0 <= report["survival_rate"] <= 1


def test_constituent_change_dataclass():
    """ConstituentChange dataclass should work correctly."""
    change = ConstituentChange(
        date=date(2024, 3, 18),
        ticker="SMCI",
        action="ADD",
        reason="Market cap growth",
        replacement_for="",
    )
    assert change.ticker == "SMCI"
    assert change.action == "ADD"
    assert change.date == date(2024, 3, 18)


def test_caching(provider):
    """Repeated queries for the same date should use cache."""
    provider._current_constituents = provider._get_top50_fallback()

    target = date(2023, 6, 15)
    result1 = provider.get_constituents_at_date(target)
    result2 = provider.get_constituents_at_date(target)

    assert result1 == result2
    assert target.isoformat() in provider._cache


def test_datetime_input_accepted(provider):
    """get_constituents_at_date should accept datetime objects too."""
    provider._current_constituents = provider._get_top50_fallback()

    result = provider.get_constituents_at_date(datetime(2023, 6, 15))
    assert isinstance(result, list)
    assert len(result) > 0

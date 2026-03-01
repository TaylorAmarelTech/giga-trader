"""
Tests for Macro Calendar Trading Gate.
"""

import pytest
from datetime import date, datetime
from unittest.mock import patch

from src.phase_19_paper_trading.macro_calendar_gate import (
    FOMC_DATES,
    MacroCalendarDataProvider,
    _first_friday_of_month,
    evaluate_macro_calendar_gate,
)


# ─── Helper: deterministic provider ─────────────────────────────────────────


def _make_provider_data(target_date: date) -> dict:
    """Use the real provider but with a specific date."""
    provider = MacroCalendarDataProvider()
    return provider.fetch(target_date=target_date)


# ─── Test: first_friday_of_month utility ─────────────────────────────────────


class TestFirstFriday:

    def test_january_2026(self):
        # Jan 1 2026 is Thursday, so first Friday is Jan 2
        ff = _first_friday_of_month(2026, 1)
        assert ff == date(2026, 1, 2)
        assert ff.weekday() == 4  # Friday

    def test_february_2026(self):
        # Feb 1 2026 is Sunday, first Friday is Feb 6
        ff = _first_friday_of_month(2026, 2)
        assert ff == date(2026, 2, 6)
        assert ff.weekday() == 4

    def test_march_2026(self):
        # Mar 1 2026 is Sunday, first Friday is Mar 6
        ff = _first_friday_of_month(2026, 3)
        assert ff == date(2026, 3, 6)
        assert ff.weekday() == 4

    def test_always_friday(self):
        """First Friday of every month in 2026 should be a Friday."""
        for month in range(1, 13):
            ff = _first_friday_of_month(2026, month)
            assert ff.weekday() == 4, f"Month {month}: {ff} is not Friday"

    def test_always_within_first_7_days(self):
        """First Friday should always be day 1-7."""
        for year in (2024, 2025, 2026, 2027):
            for month in range(1, 13):
                ff = _first_friday_of_month(year, month)
                assert 1 <= ff.day <= 7


# ─── Test: MacroCalendarDataProvider ─────────────────────────────────────────


class TestMacroCalendarDataProvider:

    def test_default_constructor(self):
        provider = MacroCalendarDataProvider()
        assert provider._data is None
        assert provider.get_latest() is None

    def test_data_age_large_initially(self):
        provider = MacroCalendarDataProvider()
        assert provider.get_data_age_hours() > 100

    def test_fetch_returns_dict(self):
        provider = MacroCalendarDataProvider()
        data = provider.fetch(target_date=date(2026, 2, 15))
        assert isinstance(data, dict)
        assert "is_fomc_day" in data
        assert "is_nfp_day" in data
        assert "is_cpi_day" in data

    def test_fetch_updates_latest(self):
        provider = MacroCalendarDataProvider()
        provider.fetch(target_date=date(2026, 2, 15))
        assert provider.get_latest() is not None

    def test_data_age_small_after_fetch(self):
        provider = MacroCalendarDataProvider()
        provider.fetch(target_date=date(2026, 2, 15))
        assert provider.get_data_age_hours() < 1.0

    def test_fomc_day_detected(self):
        """A known FOMC date should set is_fomc_day=True."""
        data = _make_provider_data(date(2026, 3, 18))
        assert data["is_fomc_day"] is True
        assert data["has_major_event"] is True

    def test_nfp_day_detected(self):
        """First Friday of a month should be detected as NFP day."""
        ff = _first_friday_of_month(2026, 4)
        data = _make_provider_data(ff)
        assert data["is_nfp_day"] is True
        assert data["has_major_event"] is True

    def test_normal_day(self):
        """A random Tuesday in mid-month should have no events."""
        # 2026-02-17 is a Tuesday, not FOMC/NFP/CPI
        data = _make_provider_data(date(2026, 2, 17))
        assert data["is_fomc_day"] is False
        assert data["is_nfp_day"] is False
        # CPI check is approximate, so just verify FOMC/NFP are off

    def test_day_before_fomc_detected(self):
        """Day before a known FOMC date should set is_day_before_fomc=True."""
        # FOMC is 2026-03-18 (Wednesday), day before is 2026-03-17 (Tuesday)
        data = _make_provider_data(date(2026, 3, 17))
        assert data["is_day_before_fomc"] is True
        assert data["has_upcoming_event"] is True


# ─── Test: evaluate_macro_calendar_gate ──────────────────────────────────────


class TestEvaluateMacroCalendarGate:

    def test_no_data_passes(self):
        result = evaluate_macro_calendar_gate(None, "BUY")
        assert result["action"] == "PASS"
        assert result["confidence_multiplier"] == 1.0
        assert result["position_size_multiplier"] == 1.0

    def test_fomc_day_blocks(self):
        data = {"is_fomc_day": True, "is_nfp_day": False, "is_cpi_day": False,
                "is_day_before_fomc": False, "is_day_before_nfp": False, "is_day_before_cpi": False}
        result = evaluate_macro_calendar_gate(data, "BUY")
        assert result["action"] == "BLOCK"
        assert result["confidence_multiplier"] == 0.0
        assert result["position_size_multiplier"] == 0.0

    def test_fomc_day_blocks_sell_too(self):
        data = {"is_fomc_day": True, "is_nfp_day": False, "is_cpi_day": False,
                "is_day_before_fomc": False, "is_day_before_nfp": False, "is_day_before_cpi": False}
        result = evaluate_macro_calendar_gate(data, "SELL")
        assert result["action"] == "BLOCK"

    def test_nfp_day_blocks(self):
        data = {"is_fomc_day": False, "is_nfp_day": True, "is_cpi_day": False,
                "is_day_before_fomc": False, "is_day_before_nfp": False, "is_day_before_cpi": False}
        result = evaluate_macro_calendar_gate(data, "BUY")
        assert result["action"] == "BLOCK"
        assert "NFP" in result["reason"]

    def test_cpi_day_blocks(self):
        data = {"is_fomc_day": False, "is_nfp_day": False, "is_cpi_day": True,
                "is_day_before_fomc": False, "is_day_before_nfp": False, "is_day_before_cpi": False}
        result = evaluate_macro_calendar_gate(data, "SELL")
        assert result["action"] == "BLOCK"
        assert "CPI" in result["reason"]

    def test_day_before_fomc_reduces(self):
        data = {"is_fomc_day": False, "is_nfp_day": False, "is_cpi_day": False,
                "is_day_before_fomc": True, "is_day_before_nfp": False, "is_day_before_cpi": False}
        result = evaluate_macro_calendar_gate(data, "BUY")
        assert result["action"] == "REDUCE"
        assert result["confidence_multiplier"] == 0.7
        assert result["position_size_multiplier"] == 0.5

    def test_day_before_nfp_reduces(self):
        data = {"is_fomc_day": False, "is_nfp_day": False, "is_cpi_day": False,
                "is_day_before_fomc": False, "is_day_before_nfp": True, "is_day_before_cpi": False}
        result = evaluate_macro_calendar_gate(data, "SELL")
        assert result["action"] == "REDUCE"
        assert result["confidence_multiplier"] == 0.8
        assert result["position_size_multiplier"] == 0.6

    def test_day_before_cpi_reduces(self):
        data = {"is_fomc_day": False, "is_nfp_day": False, "is_cpi_day": False,
                "is_day_before_fomc": False, "is_day_before_nfp": False, "is_day_before_cpi": True}
        result = evaluate_macro_calendar_gate(data, "BUY")
        assert result["action"] == "REDUCE"
        assert result["confidence_multiplier"] == 0.8
        assert result["position_size_multiplier"] == 0.6

    def test_normal_day_passes(self):
        data = {"is_fomc_day": False, "is_nfp_day": False, "is_cpi_day": False,
                "is_day_before_fomc": False, "is_day_before_nfp": False, "is_day_before_cpi": False}
        result = evaluate_macro_calendar_gate(data, "BUY")
        assert result["action"] == "PASS"
        assert result["confidence_multiplier"] == 1.0
        assert result["position_size_multiplier"] == 1.0

    def test_result_has_required_keys(self):
        """All results must have action, confidence_multiplier, position_size_multiplier, reason."""
        data = {"is_fomc_day": True, "is_nfp_day": False, "is_cpi_day": False,
                "is_day_before_fomc": False, "is_day_before_nfp": False, "is_day_before_cpi": False}
        result = evaluate_macro_calendar_gate(data, "BUY")
        assert "action" in result
        assert "confidence_multiplier" in result
        assert "position_size_multiplier" in result
        assert "reason" in result

    def test_fomc_priority_over_day_before(self):
        """If both FOMC day and day-before flags are set, BLOCK takes priority."""
        data = {"is_fomc_day": True, "is_nfp_day": False, "is_cpi_day": False,
                "is_day_before_fomc": True, "is_day_before_nfp": False, "is_day_before_cpi": False}
        result = evaluate_macro_calendar_gate(data, "BUY")
        assert result["action"] == "BLOCK"


# ─── Test: FOMC_DATES set integrity ─────────────────────────────────────────


class TestFOMCDates:

    def test_fomc_dates_not_empty(self):
        assert len(FOMC_DATES) > 0

    def test_fomc_dates_are_weekdays(self):
        """FOMC decisions happen on weekdays."""
        for d in FOMC_DATES:
            assert d.weekday() < 5, f"FOMC date {d} is a weekend"

    def test_fomc_dates_8_per_year(self):
        """There should be 8 FOMC meetings per year."""
        from collections import Counter
        year_counts = Counter(d.year for d in FOMC_DATES)
        for year, count in year_counts.items():
            assert count == 8, f"Year {year} has {count} FOMC dates, expected 8"


# ─── Test: integration with provider + evaluator ─────────────────────────────


class TestIntegration:

    def test_provider_to_evaluator_fomc_day(self):
        """Full flow: provider fetches FOMC day, evaluator blocks."""
        provider = MacroCalendarDataProvider()
        data = provider.fetch(target_date=date(2026, 1, 28))
        result = evaluate_macro_calendar_gate(data, "BUY")
        assert result["action"] == "BLOCK"

    def test_provider_to_evaluator_normal_day(self):
        """Full flow: provider fetches normal day, evaluator passes."""
        provider = MacroCalendarDataProvider()
        # 2026-02-20 is a Friday, not FOMC/NFP (NFP is first Friday = 2026-02-06)
        data = provider.fetch(target_date=date(2026, 2, 20))
        # Not FOMC, not NFP
        assert data["is_fomc_day"] is False
        assert data["is_nfp_day"] is False
        # CPI check is approximate; if it happens to be CPI, that's fine
        if not data["is_cpi_day"] and not data["is_day_before_cpi"]:
            result = evaluate_macro_calendar_gate(data, "BUY")
            assert result["action"] == "PASS"

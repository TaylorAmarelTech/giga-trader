"""
GIGA TRADER - Macro Calendar Trading Gate
==========================================
Blocks or reduces trading around major economic events (FOMC, NFP, CPI).

These events cause large directional moves that invalidate typical ML model
predictions. Trading through them introduces uncompensated event risk.

Gate logic:
  - FOMC/NFP/CPI day -> BLOCK (high volatility event risk)
  - Day before FOMC  -> REDUCE (confidence=0.7, position=0.5)
  - Day before NFP/CPI -> REDUCE (confidence=0.8, position=0.6)
  - Normal day        -> PASS
"""

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional, Set

from src.phase_19_paper_trading.trading_gates import GateDataProvider

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# FOMC DECISION DATES (last day of each scheduled meeting)
# ═══════════════════════════════════════════════════════════════════════════════

FOMC_DATES: Set[date] = {
    # 2024
    date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1), date(2024, 6, 12),
    date(2024, 7, 31), date(2024, 9, 18), date(2024, 11, 7), date(2024, 12, 18),
    # 2025
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7), date(2025, 6, 18),
    date(2025, 7, 30), date(2025, 9, 17), date(2025, 10, 29), date(2025, 12, 17),
    # 2026
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 17),
    date(2026, 7, 29), date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 16),
    # 2027
    date(2027, 1, 27), date(2027, 3, 17), date(2027, 4, 28), date(2027, 6, 16),
    date(2027, 7, 28), date(2027, 9, 15), date(2027, 10, 27), date(2027, 12, 15),
}


def _first_friday_of_month(year: int, month: int) -> date:
    """Compute the first Friday of a given month."""
    first_day = date(year, month, 1)
    # Friday = 4 in weekday()
    days_until_friday = (4 - first_day.weekday()) % 7
    return first_day + timedelta(days=days_until_friday)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PROVIDER
# ═══════════════════════════════════════════════════════════════════════════════


class MacroCalendarDataProvider(GateDataProvider):
    """Provides macro calendar event data based on date lookups.

    No network calls required -- all dates are either hardcoded (FOMC) or
    computed deterministically (NFP first-Friday, CPI mid-month).
    """

    def __init__(self):
        self._data: Optional[Dict[str, Any]] = None
        self._last_fetch: Optional[datetime] = None

    def fetch(self, target_date: Optional[date] = None) -> Optional[Dict[str, Any]]:
        """Check target date against known macro event calendars.

        Parameters
        ----------
        target_date : date, optional
            Date to evaluate. Defaults to ``date.today()``.

        Returns
        -------
        dict
            Calendar event flags for the given date.
        """
        today = target_date if target_date is not None else date.today()

        is_fomc = today in FOMC_DATES
        is_day_before_fomc = any(
            (d - today).days == 1
            for d in FOMC_DATES
            if d >= today
        )

        # NFP: first Friday of month
        first_friday = _first_friday_of_month(today.year, today.month)
        is_nfp = today == first_friday
        # Day before NFP (Thursday before first Friday, if it is a weekday)
        day_before_nfp = first_friday - timedelta(days=1)
        is_day_before_nfp = today == day_before_nfp and today.weekday() < 5

        # CPI: approximately 10th-15th of month (business day check)
        is_cpi = today.day in (10, 11, 12, 13, 14, 15) and today.weekday() < 5
        is_day_before_cpi = today.day in (9, 10, 11, 12, 13, 14) and today.weekday() < 5

        self._data = {
            "date": today.isoformat(),
            "is_fomc_day": is_fomc,
            "is_nfp_day": is_nfp,
            "is_cpi_day": is_cpi,
            "is_day_before_fomc": is_day_before_fomc,
            "is_day_before_nfp": is_day_before_nfp,
            "is_day_before_cpi": is_day_before_cpi,
            "has_major_event": is_fomc or is_nfp or is_cpi,
            "has_upcoming_event": is_day_before_fomc or is_day_before_nfp or is_day_before_cpi,
        }
        self._last_fetch = datetime.now()
        return self._data

    def get_latest(self) -> Optional[Dict[str, Any]]:
        return self._data

    def get_data_age_hours(self) -> float:
        if self._last_fetch is None:
            return 999.0
        return (datetime.now() - self._last_fetch).total_seconds() / 3600


# ═══════════════════════════════════════════════════════════════════════════════
# GATE EVALUATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_macro_calendar_gate(
    data: Optional[Dict[str, Any]],
    signal_type: str,
) -> Dict[str, Any]:
    """
    Evaluate macro calendar gate.

    Logic:
    - FOMC/NFP/CPI day            -> BLOCK
    - Day before FOMC              -> REDUCE (confidence=0.7, position=0.5)
    - Day before NFP/CPI           -> REDUCE (confidence=0.8, position=0.6)
    - Normal                       -> PASS

    Parameters
    ----------
    data : dict or None
        Calendar event flags from MacroCalendarDataProvider.
    signal_type : str
        "BUY", "SELL", or "HOLD".

    Returns
    -------
    dict
        Gate decision with action, confidence_multiplier,
        position_size_multiplier, and reason.
    """
    if data is None:
        return {
            "action": "PASS",
            "confidence_multiplier": 1.0,
            "position_size_multiplier": 1.0,
            "reason": "No calendar data available",
        }

    # BLOCK on event days
    if data.get("is_fomc_day", False):
        return {
            "action": "BLOCK",
            "confidence_multiplier": 0.0,
            "position_size_multiplier": 0.0,
            "reason": "FOMC decision day — high volatility event risk",
        }
    if data.get("is_nfp_day", False):
        return {
            "action": "BLOCK",
            "confidence_multiplier": 0.0,
            "position_size_multiplier": 0.0,
            "reason": "NFP release day — employment data event risk",
        }
    if data.get("is_cpi_day", False):
        return {
            "action": "BLOCK",
            "confidence_multiplier": 0.0,
            "position_size_multiplier": 0.0,
            "reason": "CPI release day — inflation data event risk",
        }

    # REDUCE day before events
    if data.get("is_day_before_fomc", False):
        return {
            "action": "REDUCE",
            "confidence_multiplier": 0.7,
            "position_size_multiplier": 0.5,
            "reason": "Day before FOMC — pre-event caution",
        }
    if data.get("is_day_before_nfp", False):
        return {
            "action": "REDUCE",
            "confidence_multiplier": 0.8,
            "position_size_multiplier": 0.6,
            "reason": "Day before NFP — pre-event caution",
        }
    if data.get("is_day_before_cpi", False):
        return {
            "action": "REDUCE",
            "confidence_multiplier": 0.8,
            "position_size_multiplier": 0.6,
            "reason": "Day before CPI — pre-event caution",
        }

    return {
        "action": "PASS",
        "confidence_multiplier": 1.0,
        "position_size_multiplier": 1.0,
        "reason": "No macro events today",
    }

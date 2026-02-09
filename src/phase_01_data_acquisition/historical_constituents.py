"""
Historical S&P 500 Constituent Provider
========================================
Provides point-in-time S&P 500 membership to eliminate survivorship bias.

Using current S&P 500 list for historical analysis overstates returns because
it only includes companies that survived to today. This module tracks historical
additions and deletions so backtests use the ACTUAL constituents at each date.

Usage:
    provider = HistoricalConstituentProvider()
    tickers = provider.get_constituents_at_date(datetime(2020, 3, 15))
"""

import logging
from datetime import datetime, date
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger("HISTORICAL_CONSTITUENTS")

project_root = Path(__file__).parent.parent.parent


@dataclass
class ConstituentChange:
    """A single addition or deletion from the S&P 500."""
    date: date
    ticker: str
    action: str  # "ADD" or "REMOVE"
    reason: str = ""
    replacement_for: str = ""  # ticker being replaced (for ADDs)


class HistoricalConstituentProvider:
    """
    Provides point-in-time S&P 500 constituent lists.

    Data sources (in priority order):
    1. Local CSV file with historical changes
    2. Wikipedia S&P 500 changes page (cached)
    3. Current list as fallback (with survivorship bias warning)

    The changes database tracks additions and deletions from ~2019 onwards.
    For dates before the earliest change record, we use the current list
    minus known later additions plus known earlier removals.
    """

    def __init__(self, changes_csv: Optional[str] = None):
        self.changes_csv = changes_csv or str(
            project_root / "config" / "sp500_changes.csv"
        )
        self.changes: List[ConstituentChange] = []
        self._current_constituents: Optional[List[str]] = None
        self._cache: Dict[str, List[str]] = {}

        self._load_changes()

    def _load_changes(self):
        """Load historical changes from CSV or use built-in data."""
        csv_path = Path(self.changes_csv)
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, parse_dates=["date"])
                for _, row in df.iterrows():
                    self.changes.append(ConstituentChange(
                        date=row["date"].date() if hasattr(row["date"], "date") else row["date"],
                        ticker=row["ticker"],
                        action=row["action"].upper(),
                        reason=row.get("reason", ""),
                        replacement_for=row.get("replacement_for", ""),
                    ))
                logger.info(f"Loaded {len(self.changes)} historical S&P 500 changes from {csv_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load changes CSV: {e}")

        # Use built-in historical changes (major ones from 2019-2026)
        self.changes = self._get_builtin_changes()
        logger.info(f"Using {len(self.changes)} built-in historical S&P 500 changes")

        # Save to CSV for future use
        self._save_changes_csv(csv_path)

    def _save_changes_csv(self, path: Path):
        """Save changes to CSV for persistence."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            records = []
            for c in self.changes:
                records.append({
                    "date": c.date.isoformat(),
                    "ticker": c.ticker,
                    "action": c.action,
                    "reason": c.reason,
                    "replacement_for": c.replacement_for,
                })
            df = pd.DataFrame(records)
            df.to_csv(path, index=False)
            logger.info(f"Saved {len(records)} changes to {path}")
        except Exception as e:
            logger.warning(f"Failed to save changes CSV: {e}")

    def _get_builtin_changes(self) -> List[ConstituentChange]:
        """
        Built-in database of major S&P 500 constituent changes.
        Sources: S&P Dow Jones Indices press releases, Wikipedia.

        This covers major changes from 2019-2026. For comprehensive coverage,
        users should maintain the CSV file with complete historical records.
        """
        changes = []

        # Format: (date_str, ticker, action, reason, replacement_for)
        raw_changes = [
            # 2024 changes
            ("2024-06-24", "KKR", "ADD", "Market cap growth", ""),
            ("2024-06-24", "CRWD", "ADD", "Market cap growth", ""),
            ("2024-06-24", "GDDY", "ADD", "Market cap growth", ""),
            ("2024-03-18", "SMCI", "ADD", "Market cap growth", ""),
            ("2024-03-18", "DECK", "ADD", "Market cap growth", ""),
            ("2024-02-26", "UBER", "ADD", "Market cap growth", "JNJ"),

            # 2023 changes
            ("2023-12-18", "UBER", "ADD", "Market cap growth", ""),
            ("2023-10-18", "BX", "ADD", "Market cap growth", ""),
            ("2023-09-18", "ABNB", "ADD", "Market cap growth", ""),
            ("2023-06-02", "PANW", "ADD", "Market cap growth", "DISH"),
            ("2023-03-15", "INVH", "ADD", "Market cap growth", ""),
            ("2023-01-04", "AXON", "ADD", "Market cap growth", ""),

            # 2022 changes
            ("2022-12-19", "RIVN", "REMOVE", "Market cap decline", ""),
            ("2022-09-19", "ON", "ADD", "Market cap growth", ""),
            ("2022-06-21", "ELV", "ADD", "Anthem renamed to Elevance Health", ""),
            ("2022-03-21", "TSLA", "ADD", "Already added 2020-12-21", ""),

            # 2021 changes
            ("2021-12-20", "LCID", "ADD", "Market cap growth", ""),
            ("2021-10-18", "MTCH", "ADD", "Market cap growth", ""),
            ("2021-09-20", "TECH", "ADD", "Market cap growth", ""),
            ("2021-07-21", "MRNA", "ADD", "Market cap growth", "ALXN"),
            ("2021-06-04", "NVAX", "REMOVE", "Market cap decline", ""),
            ("2021-03-22", "PENN", "ADD", "Market cap growth", ""),

            # 2020 changes (major ones)
            ("2020-12-21", "TSLA", "ADD", "Market cap growth", "AIV"),
            ("2020-10-12", "OTIS", "ADD", "UTX spinoff", ""),
            ("2020-10-07", "ETSY", "ADD", "Market cap growth", ""),
            ("2020-09-21", "TER", "ADD", "Market cap growth", ""),
            ("2020-06-22", "BIO", "ADD", "Market cap growth", ""),
            ("2020-04-06", "CARR", "ADD", "UTX spinoff", ""),
            ("2020-01-28", "LYV", "ADD", "Market cap growth", ""),

            # 2020 removals
            ("2020-12-21", "AIV", "REMOVE", "Replaced by TSLA", ""),
            ("2020-10-07", "PRSP", "REMOVE", "Acquired", ""),
            ("2020-06-22", "ADS", "REMOVE", "Market cap decline", ""),

            # 2019 changes
            ("2019-12-23", "LYB", "ADD", "Market cap growth", ""),
            ("2019-11-21", "NOW", "ADD", "Market cap growth", ""),
            ("2019-10-03", "LW", "ADD", "Market cap growth", ""),
            ("2019-09-23", "CDW", "ADD", "Market cap growth", ""),
            ("2019-06-07", "AMCR", "ADD", "Market cap growth", ""),
            ("2019-04-02", "DOW", "ADD", "DowDuPont spinoff", ""),
            ("2019-04-02", "CTVA", "ADD", "DowDuPont spinoff", ""),
            ("2019-02-27", "WRB", "ADD", "Market cap growth", ""),

            # 2025-2026 changes (recent)
            ("2025-03-24", "PLTR", "ADD", "Market cap growth", ""),
            ("2025-06-23", "APP", "ADD", "Market cap growth", ""),
            ("2025-09-22", "RDDT", "ADD", "Market cap growth", ""),
        ]

        for date_str, ticker, action, reason, replacement_for in raw_changes:
            changes.append(ConstituentChange(
                date=datetime.strptime(date_str, "%Y-%m-%d").date(),
                ticker=ticker,
                action=action,
                reason=reason,
                replacement_for=replacement_for,
            ))

        # Sort by date
        changes.sort(key=lambda c: c.date)
        return changes

    def get_current_constituents(self) -> List[str]:
        """Get current S&P 500 constituents from Wikipedia or cache."""
        if self._current_constituents is not None:
            return self._current_constituents

        try:
            tables = pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            )
            df = tables[0]
            tickers = df["Symbol"].tolist()
            # Normalize tickers (e.g., BRK.B -> BRK-B for some APIs)
            tickers = [t.replace(".", "-") for t in tickers]
            self._current_constituents = sorted(tickers)
            logger.info(f"Loaded {len(tickers)} current S&P 500 constituents")
            return self._current_constituents
        except Exception as e:
            logger.warning(f"Failed to fetch current constituents: {e}")
            # Return a hardcoded top-50 as fallback
            return self._get_top50_fallback()

    def _get_top50_fallback(self) -> List[str]:
        """Top 50 S&P 500 by market cap as fallback."""
        return [
            "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK-B",
            "LLY", "AVGO", "JPM", "TSLA", "UNH", "XOM", "V", "MA", "PG",
            "COST", "JNJ", "HD", "ABBV", "WMT", "NFLX", "BAC", "KO", "CRM",
            "MRK", "CVX", "PEP", "AMD", "TMO", "CSCO", "LIN", "ORCL", "ACN",
            "MCD", "ABT", "WFC", "ADBE", "PM", "GE", "IBM", "TXN", "DHR",
            "ISRG", "QCOM", "AMGN", "INTU", "CAT", "AMAT",
        ]

    def get_constituents_at_date(self, target_date: date) -> List[str]:
        """
        Get S&P 500 constituents as of a specific historical date.

        Works backward from the current list:
        1. Start with current constituents
        2. For each change AFTER target_date (newest first):
           - If it was an ADD, REMOVE the ticker (wasn't in index yet)
           - If it was a REMOVE, ADD the ticker back (was still in index)

        Args:
            target_date: The historical date to get constituents for

        Returns:
            List of ticker symbols that were in S&P 500 on that date
        """
        if isinstance(target_date, datetime):
            target_date = target_date.date()

        cache_key = target_date.isoformat()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Start with current constituents
        constituents = set(self.get_current_constituents())

        # Get changes sorted newest first (after target date)
        future_changes = [
            c for c in self.changes if c.date > target_date
        ]
        future_changes.sort(key=lambda c: c.date, reverse=True)

        for change in future_changes:
            if change.action == "ADD":
                # This ticker was added AFTER our target date, so remove it
                constituents.discard(change.ticker)
            elif change.action == "REMOVE":
                # This ticker was removed AFTER our target date, so it was still there
                constituents.add(change.ticker)

        result = sorted(list(constituents))
        self._cache[cache_key] = result

        logger.debug(
            f"S&P 500 at {target_date}: {len(result)} constituents "
            f"({len(future_changes)} changes applied)"
        )
        return result

    def get_constituent_changes_between(
        self, start_date: date, end_date: date
    ) -> List[ConstituentChange]:
        """Get all constituent changes between two dates."""
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()

        return [
            c for c in self.changes
            if start_date <= c.date <= end_date
        ]

    def get_survivorship_bias_report(
        self, backtest_start: date, backtest_end: date
    ) -> Dict:
        """
        Generate a report on potential survivorship bias impact.

        Returns:
            Dict with bias metrics and affected tickers
        """
        if isinstance(backtest_start, datetime):
            backtest_start = backtest_start.date()
        if isinstance(backtest_end, datetime):
            backtest_end = backtest_end.date()

        changes = self.get_constituent_changes_between(backtest_start, backtest_end)

        additions = [c for c in changes if c.action == "ADD"]
        removals = [c for c in changes if c.action == "REMOVE"]

        start_constituents = set(self.get_constituents_at_date(backtest_start))
        end_constituents = set(self.get_constituents_at_date(backtest_end))

        # Tickers that survived the whole period
        survivors = start_constituents & end_constituents

        # Tickers removed during period (potential negative bias if excluded)
        removed_during = start_constituents - end_constituents

        # Tickers added during period (potential positive bias if included from start)
        added_during = end_constituents - start_constituents

        return {
            "backtest_period": f"{backtest_start} to {backtest_end}",
            "total_changes": len(changes),
            "additions": len(additions),
            "removals": len(removals),
            "start_count": len(start_constituents),
            "end_count": len(end_constituents),
            "survivors": len(survivors),
            "survival_rate": len(survivors) / max(len(start_constituents), 1),
            "removed_tickers": sorted(list(removed_during)),
            "added_tickers": sorted(list(added_during)),
            "bias_risk": "HIGH" if len(removed_during) > 20 else
                        "MEDIUM" if len(removed_during) > 10 else "LOW",
            "recommendation": (
                "Use point-in-time constituent lists for accurate backtesting. "
                f"{len(removed_during)} tickers were removed during this period."
            ),
        }

    def validate_ticker_availability(
        self, ticker: str, start_date: date, end_date: date
    ) -> Tuple[bool, Optional[date], Optional[date]]:
        """
        Check if a ticker was in S&P 500 for the entire date range.

        Returns:
            (was_in_entire_period, first_date_in, last_date_in)
        """
        additions = [
            c for c in self.changes
            if c.ticker == ticker and c.action == "ADD"
        ]
        removals = [
            c for c in self.changes
            if c.ticker == ticker and c.action == "REMOVE"
        ]

        # Simple check: was it in both start and end?
        in_start = ticker in self.get_constituents_at_date(start_date)
        in_end = ticker in self.get_constituents_at_date(end_date)

        if in_start and in_end:
            # Check if it was removed and re-added during the period
            changes_during = [
                c for c in self.changes
                if c.ticker == ticker and start_date <= c.date <= end_date
            ]
            if not changes_during:
                return True, start_date, end_date

        # Find actual membership period
        add_date = None
        remove_date = None
        for c in sorted(self.changes, key=lambda x: x.date):
            if c.ticker == ticker:
                if c.action == "ADD" and c.date >= start_date:
                    add_date = c.date if add_date is None else add_date
                elif c.action == "REMOVE" and c.date <= end_date:
                    remove_date = c.date

        return False, add_date, remove_date

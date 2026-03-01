"""
GIGA TRADER - CUSUM Event Filter for Training Data Sampling
============================================================
Implements the Cumulative Sum (CUSUM) filter from Lopez de Prado,
"Advances in Financial Machine Learning", Chapter 2.

Instead of training on ALL daily observations (most of which are noise),
the CUSUM filter reduces training data to only dates where significant
cumulative price moves occurred. This focuses the model on genuinely
meaningful events where the signal-to-noise ratio is highest.

Algorithm:
    1. Compute daily returns from close prices (or accept pre-computed)
    2. Initialize S_pos = 0, S_neg = 0 (cumulative sums)
    3. For each day t:
       - S_pos = max(0, S_pos + return_t)   (tracks upward drift)
       - S_neg = min(0, S_neg + return_t)    (tracks downward drift)
       - If S_pos >= threshold: record event, reset S_pos = 0
       - If S_neg <= -threshold: record event, reset S_neg = 0
    4. Return list of event indices/dates
"""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CUSUMFilter:
    """
    CUSUM event filter for training data sampling.

    Monitors cumulative positive and negative returns.
    When either exceeds a threshold, it signals an "event" and resets.
    Only event dates are kept for training.

    This reduces noise in training data by filtering out
    insignificant days where the signal-to-noise ratio is low.

    Parameters
    ----------
    threshold : float
        Cumulative return threshold to trigger an event (default 1%).
        Used for both up and down if symmetric=True.
    symmetric : bool
        If True, use the same threshold for positive and negative CUSUM.
        If False, use up_threshold and down_threshold separately.
    up_threshold : float or None
        Override threshold for positive CUSUM (upward moves).
        Only used when symmetric=False. Defaults to threshold if None.
    down_threshold : float or None
        Override threshold for negative CUSUM (downward moves).
        Only used when symmetric=False. Defaults to threshold if None.
    min_events : int
        Minimum number of events required. If fewer events are detected,
        return all dates unfiltered (safety guard against over-filtering).
    max_filter_ratio : float
        Maximum fraction of dates that can be filtered out (0.0 to 1.0).
        If filtering would remove more than this ratio, the threshold is
        progressively relaxed until enough events pass through.
    """

    def __init__(
        self,
        threshold: float = 0.01,
        symmetric: bool = True,
        up_threshold: Optional[float] = None,
        down_threshold: Optional[float] = None,
        min_events: int = 100,
        max_filter_ratio: float = 0.8,
    ):
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")
        if not 0.0 < max_filter_ratio <= 1.0:
            raise ValueError(
                f"max_filter_ratio must be in (0, 1], got {max_filter_ratio}"
            )
        if min_events < 0:
            raise ValueError(f"min_events must be non-negative, got {min_events}")

        self.threshold = threshold
        self.symmetric = symmetric
        self.min_events = min_events
        self.max_filter_ratio = max_filter_ratio

        # Resolve effective thresholds
        if symmetric:
            self._up_threshold = threshold
            self._down_threshold = threshold
        else:
            self._up_threshold = up_threshold if up_threshold is not None else threshold
            self._down_threshold = (
                down_threshold if down_threshold is not None else threshold
            )

        # Validate asymmetric thresholds
        if self._up_threshold < 0:
            raise ValueError(
                f"up_threshold must be non-negative, got {self._up_threshold}"
            )
        if self._down_threshold < 0:
            raise ValueError(
                f"down_threshold must be non-negative, got {self._down_threshold}"
            )

    def get_event_indices(self, returns: np.ndarray) -> List[int]:
        """
        Run the CUSUM filter and return indices where events occur.

        Parameters
        ----------
        returns : np.ndarray
            1-D array of returns (e.g. daily log or simple returns).

        Returns
        -------
        List[int]
            Sorted list of indices where CUSUM events triggered.
        """
        returns = np.asarray(returns, dtype=np.float64)

        if returns.ndim != 1:
            raise ValueError(
                f"returns must be 1-D, got shape {returns.shape}"
            )

        if len(returns) == 0:
            return []

        # Replace NaN with 0.0 so they don't pollute the cumulative sum
        clean_returns = np.where(np.isnan(returns), 0.0, returns)

        events: List[int] = []
        s_pos = 0.0
        s_neg = 0.0

        for i in range(len(clean_returns)):
            r = clean_returns[i]
            s_pos = max(0.0, s_pos + r)
            s_neg = min(0.0, s_neg + r)

            triggered = False
            if s_pos >= self._up_threshold:
                triggered = True
                s_pos = 0.0
            if s_neg <= -self._down_threshold:
                triggered = True
                s_neg = 0.0

            if triggered:
                events.append(i)

        return events

    def filter_events(self, returns: np.ndarray) -> np.ndarray:
        """
        Return a boolean mask marking event dates.

        Parameters
        ----------
        returns : np.ndarray
            1-D array of returns.

        Returns
        -------
        np.ndarray
            Boolean array of same length as returns. True = event date.
        """
        returns = np.asarray(returns, dtype=np.float64)
        n = len(returns)

        if n == 0:
            return np.array([], dtype=bool)

        # Special case: threshold of exactly 0 means every non-zero return is
        # an event (both CUSUM accumulators trigger immediately). For
        # consistency with the cumulative-sum logic, treat threshold=0 as
        # "accept all observations".
        if self._up_threshold == 0.0 and self._down_threshold == 0.0:
            logger.info(
                "CUSUM threshold=0 -> returning all %d dates as events", n
            )
            return np.ones(n, dtype=bool)

        event_indices = self.get_event_indices(returns)
        n_events = len(event_indices)

        # --- Safety guard: min_events ---
        if n_events < self.min_events:
            logger.warning(
                "CUSUM filter found only %d events (min_events=%d). "
                "Returning all %d dates unfiltered.",
                n_events,
                self.min_events,
                n,
            )
            return np.ones(n, dtype=bool)

        # --- Safety guard: max_filter_ratio ---
        filter_ratio = 1.0 - (n_events / n)
        if filter_ratio > self.max_filter_ratio:
            # Too many dates filtered out. Progressively relax threshold.
            mask = self._relax_threshold(returns)
            return mask

        # Normal case: build boolean mask from event indices
        mask = np.zeros(n, dtype=bool)
        for idx in event_indices:
            mask[idx] = True

        logger.info(
            "CUSUM filter: %d / %d dates selected (%.1f%% kept, threshold=%.4f)",
            n_events,
            n,
            100.0 * n_events / n,
            self.threshold,
        )

        return mask

    def _relax_threshold(self, returns: np.ndarray) -> np.ndarray:
        """
        Progressively halve the threshold until enough events pass.

        If after 20 halvings we still cannot satisfy max_filter_ratio,
        return all dates.
        """
        n = len(returns)
        min_events_needed = int(np.ceil(n * (1.0 - self.max_filter_ratio)))

        current_up = self._up_threshold
        current_down = self._down_threshold

        for attempt in range(20):
            current_up *= 0.5
            current_down *= 0.5

            # Run CUSUM with relaxed thresholds
            relaxed = CUSUMFilter(
                threshold=current_up,
                symmetric=False,
                up_threshold=current_up,
                down_threshold=current_down,
                min_events=0,  # Disable guard to avoid recursion
                max_filter_ratio=1.0,  # Disable guard to avoid recursion
            )
            event_indices = relaxed.get_event_indices(returns)

            if len(event_indices) >= min_events_needed:
                mask = np.zeros(n, dtype=bool)
                for idx in event_indices:
                    mask[idx] = True
                logger.info(
                    "CUSUM threshold relaxed to up=%.6f / down=%.6f "
                    "after %d attempts: %d / %d events (%.1f%% kept)",
                    current_up,
                    current_down,
                    attempt + 1,
                    len(event_indices),
                    n,
                    100.0 * len(event_indices) / n,
                )
                return mask

        # Could not relax enough - return everything
        logger.warning(
            "CUSUM threshold relaxation exhausted after 20 attempts. "
            "Returning all %d dates.",
            n,
        )
        return np.ones(n, dtype=bool)

    def filter_dataframe(
        self,
        df: pd.DataFrame,
        return_col: str = "day_return",
        close_col: str = "close",
    ) -> pd.DataFrame:
        """
        Filter a DataFrame to only CUSUM event dates.

        Tries to use pre-computed returns from ``return_col`` first.
        If that column is missing, computes returns from ``close_col``.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with daily data.
        return_col : str
            Column name containing pre-computed returns.
        close_col : str
            Column name containing close prices (used as fallback).

        Returns
        -------
        pd.DataFrame
            Subset of df containing only event dates.
            The original index is preserved.

        Raises
        ------
        ValueError
            If neither return_col nor close_col exist in df.
        """
        if df.empty:
            logger.info("CUSUM filter_dataframe: empty DataFrame, returning as-is.")
            return df.copy()

        # Resolve returns
        if return_col in df.columns:
            returns = df[return_col].values.astype(np.float64)
        elif close_col in df.columns:
            prices = df[close_col].values.astype(np.float64)
            # First return is NaN (no prior price), replace with 0
            returns = np.empty(len(prices), dtype=np.float64)
            returns[0] = 0.0
            returns[1:] = np.diff(prices) / prices[:-1]
        else:
            raise ValueError(
                f"DataFrame must contain either '{return_col}' or '{close_col}' column. "
                f"Found columns: {list(df.columns)}"
            )

        mask = self.filter_events(returns)
        result = df.loc[mask].copy()

        logger.info(
            "CUSUM filter_dataframe: %d -> %d rows (%.1f%% kept)",
            len(df),
            len(result),
            100.0 * len(result) / max(len(df), 1),
        )

        return result

    def __repr__(self) -> str:
        if self.symmetric:
            thresh_str = f"threshold={self.threshold}"
        else:
            thresh_str = (
                f"up_threshold={self._up_threshold}, "
                f"down_threshold={self._down_threshold}"
            )
        return (
            f"CUSUMFilter({thresh_str}, min_events={self.min_events}, "
            f"max_filter_ratio={self.max_filter_ratio})"
        )

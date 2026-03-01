"""
Information-Driven Bars -- alternative bar aggregation methods.

Time bars sample at fixed intervals, which means busy periods are
undersampled and quiet periods are oversampled. Information bars
instead sample when a threshold amount of market activity occurs.

Types:
  - Dollar bars: New bar when cumulative dollar volume exceeds threshold
  - Volume bars: New bar when cumulative share volume exceeds threshold
  - Tick bars: New bar every N trades/rows

Benefits:
  - More uniform information content per bar
  - Better statistical properties (closer to IID)
  - Naturally adjust for varying market activity

Reference: Lopez de Prado, "Advances in Financial Machine Learning", Ch. 2

Usage:
    from src.phase_02_preprocessing.information_bars import InformationBarGenerator

    gen = InformationBarGenerator(bar_type="dollar")
    bars = gen.generate(df_1min)
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical OHLCV column names (lowercase)
_REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

# Approximate number of trading days per year
_TRADING_DAYS_PER_YEAR = 252

# Valid bar types
_VALID_BAR_TYPES = ("dollar", "volume", "tick")


class InformationBarGenerator:
    """
    Generate information-driven bars from tick/minute OHLCV data.

    Parameters
    ----------
    bar_type : str
        Type of bars: "dollar", "volume", "tick"
    threshold : float or int or None
        Threshold for bar generation. If None, auto-calibrates from data
        to produce approximately the same number of bars as trading days.
    auto_calibrate : bool
        If True and threshold is None, automatically set threshold
        to produce ~1 bar per trading day equivalent.
    """

    def __init__(
        self,
        bar_type: str = "dollar",
        threshold: Optional[float] = None,
        auto_calibrate: bool = True,
    ):
        bar_type = bar_type.lower().strip()
        if bar_type not in _VALID_BAR_TYPES:
            raise ValueError(
                f"Invalid bar_type '{bar_type}'. Must be one of: {_VALID_BAR_TYPES}"
            )
        self.bar_type = bar_type
        self.threshold = threshold
        self.auto_calibrate = auto_calibrate
        self._calibrated_threshold: Optional[float] = None

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate information bars from input OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame
            Input data with columns: open/Open, high/High, low/Low,
            close/Close, volume/Volume. Index should be datetime.

        Returns
        -------
        pd.DataFrame
            Aggregated bars with OHLCV columns plus:
            - bar_duration: number of source rows in this bar
            - bar_dollar_volume: total dollar volume in bar
        """
        if df.empty:
            return self._empty_result(df)

        # Normalise column names to lowercase (work on a copy)
        df = self._normalise_columns(df.copy())

        # Determine threshold
        threshold = self.threshold
        if threshold is None:
            if self.auto_calibrate:
                threshold = self._auto_threshold(df)
                self._calibrated_threshold = threshold
                logger.info(
                    "[INFO_BARS] Auto-calibrated %s threshold: %.2f",
                    self.bar_type,
                    threshold,
                )
            else:
                raise ValueError(
                    "threshold is None and auto_calibrate is False. "
                    "Provide an explicit threshold or set auto_calibrate=True."
                )

        # Dispatch to the appropriate generator
        if self.bar_type == "dollar":
            result = self._generate_dollar_bars(df, threshold)
        elif self.bar_type == "volume":
            result = self._generate_volume_bars(df, threshold)
        else:  # tick
            result = self._generate_tick_bars(df, int(threshold))

        logger.info(
            "[INFO_BARS] Generated %d %s bars from %d source rows",
            len(result),
            self.bar_type,
            len(df),
        )

        return result

    # ------------------------------------------------------------------
    # Auto-calibration
    # ------------------------------------------------------------------

    def _auto_threshold(self, df: pd.DataFrame) -> float:
        """
        Auto-calibrate threshold to produce ~1 bar per trading day.

        For dollar bars: total_dollar_volume / n_trading_days
        For volume bars: total_volume / n_trading_days
        For tick bars: n_rows / n_trading_days
        """
        n_trading_days = self._estimate_trading_days(df)
        if n_trading_days < 1:
            n_trading_days = 1

        if self.bar_type == "dollar":
            dollar_volume = (df["close"] * df["volume"]).sum()
            return float(dollar_volume / n_trading_days)
        elif self.bar_type == "volume":
            total_volume = df["volume"].sum()
            return float(total_volume / n_trading_days)
        else:  # tick
            return float(len(df) / n_trading_days)

    @staticmethod
    def _estimate_trading_days(df: pd.DataFrame) -> int:
        """
        Estimate the number of trading days in the dataset.

        Uses the datetime index if available, otherwise falls back to
        dividing total rows by an assumed intraday bar count.
        """
        idx = df.index
        if isinstance(idx, pd.DatetimeIndex):
            return max(int(idx.normalize().nunique()), 1)

        # Attempt to parse a 'date' column if present
        if "date" in df.columns:
            try:
                return max(int(pd.to_datetime(df["date"]).dt.date.nunique()), 1)
            except Exception:
                pass

        # Fallback: assume ~390 1-min bars per trading day (regular hours)
        return max(len(df) // 390, 1)

    # ------------------------------------------------------------------
    # Bar generators
    # ------------------------------------------------------------------

    def _generate_dollar_bars(
        self, df: pd.DataFrame, threshold: float
    ) -> pd.DataFrame:
        """Generate dollar bars: new bar when cumulative dollar volume >= threshold."""
        dollar_volume = (df["close"].values * df["volume"].values).astype(np.float64)
        boundaries = self._find_boundaries_cumsum(dollar_volume, threshold)
        return self._build_bars(df, boundaries)

    def _generate_volume_bars(
        self, df: pd.DataFrame, threshold: float
    ) -> pd.DataFrame:
        """Generate volume bars: new bar when cumulative volume >= threshold."""
        volume = df["volume"].values.astype(np.float64)
        boundaries = self._find_boundaries_cumsum(volume, threshold)
        return self._build_bars(df, boundaries)

    def _generate_tick_bars(
        self, df: pd.DataFrame, threshold: int
    ) -> pd.DataFrame:
        """Generate tick bars: new bar every N rows."""
        n = len(df)
        if threshold < 1:
            threshold = 1

        boundaries = list(range(0, n, threshold))
        # Ensure the last boundary includes any remaining rows
        if boundaries[-1] != n:
            boundaries.append(n)

        return self._build_bars_from_indices(df, boundaries)

    # ------------------------------------------------------------------
    # Boundary detection
    # ------------------------------------------------------------------

    @staticmethod
    def _find_boundaries_cumsum(
        values: np.ndarray, threshold: float
    ) -> list:
        """
        Find bar boundary indices using cumulative sum threshold crossing.

        Returns a list of indices where each index marks the start of a new
        bar. The last element is len(values) to close the final bar.
        """
        n = len(values)
        if n == 0:
            return [0]

        boundaries = [0]
        cumulative = 0.0

        for i in range(n):
            cumulative += values[i]
            if cumulative >= threshold:
                # This row completes a bar; next bar starts at i+1
                boundaries.append(i + 1)
                cumulative = 0.0

        # Close the final bar
        if boundaries[-1] != n:
            boundaries.append(n)

        return boundaries

    # ------------------------------------------------------------------
    # Bar construction
    # ------------------------------------------------------------------

    def _build_bars(
        self, df: pd.DataFrame, boundaries: list
    ) -> pd.DataFrame:
        """Build bars from boundary indices produced by cumsum detection."""
        return self._build_bars_from_indices(df, boundaries)

    def _build_bars_from_indices(
        self, df: pd.DataFrame, boundaries: list
    ) -> pd.DataFrame:
        """
        Aggregate source rows into bars using boundary indices.

        Parameters
        ----------
        df : pd.DataFrame
            Source data (lowercase columns, datetime index).
        boundaries : list[int]
            Sorted list of row indices marking bar boundaries.
            boundaries[i] to boundaries[i+1]-1 are aggregated into one bar.
        """
        records = []
        has_datetime_index = isinstance(df.index, pd.DatetimeIndex)

        open_vals = df["open"].values
        high_vals = df["high"].values
        low_vals = df["low"].values
        close_vals = df["close"].values
        volume_vals = df["volume"].values.astype(np.float64)
        dollar_vol_vals = (close_vals * volume_vals).astype(np.float64)

        if has_datetime_index:
            index_vals = df.index

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            if start >= end:
                continue

            bar = {
                "open": float(open_vals[start]),
                "high": float(np.max(high_vals[start:end])),
                "low": float(np.min(low_vals[start:end])),
                "close": float(close_vals[end - 1]),
                "volume": float(np.sum(volume_vals[start:end])),
                "bar_duration": end - start,
                "bar_dollar_volume": float(np.sum(dollar_vol_vals[start:end])),
            }

            if has_datetime_index:
                bar["timestamp"] = index_vals[start]

            records.append(bar)

        if not records:
            return self._empty_result(df)

        result = pd.DataFrame(records)

        # Set datetime index if available
        if "timestamp" in result.columns:
            result = result.set_index("timestamp")
            result.index.name = None  # match typical OHLCV style

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise column names to lowercase and verify required columns exist.

        Raises ValueError if any required OHLCV column is missing.
        """
        col_map = {c: c.lower() for c in df.columns}
        df = df.rename(columns=col_map)

        missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required OHLCV columns: {missing}. "
                f"Available: {list(df.columns)}"
            )
        return df

    @staticmethod
    def _empty_result(df: pd.DataFrame) -> pd.DataFrame:
        """Return an empty DataFrame with the expected output columns."""
        cols = ["open", "high", "low", "close", "volume", "bar_duration", "bar_dollar_volume"]
        empty = pd.DataFrame(columns=cols)
        if isinstance(df.index, pd.DatetimeIndex):
            empty.index = pd.DatetimeIndex([], name=None)
        return empty

    @staticmethod
    def _aggregate_bar(group: pd.DataFrame) -> dict:
        """
        Aggregate a group of rows into a single bar.

        open = first open, high = max high, low = min low, close = last close
        volume = sum, dollar_volume = sum(close * volume)
        """
        return {
            "open": float(group["open"].iloc[0]),
            "high": float(group["high"].max()),
            "low": float(group["low"].min()),
            "close": float(group["close"].iloc[-1]),
            "volume": float(group["volume"].sum()),
            "bar_duration": len(group),
            "bar_dollar_volume": float((group["close"] * group["volume"]).sum()),
        }

    @property
    def calibrated_threshold(self) -> Optional[float]:
        """Return the auto-calibrated threshold (None if not yet calibrated)."""
        return self._calibrated_threshold

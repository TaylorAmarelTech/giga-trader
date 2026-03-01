"""
GIGA TRADER - OHLC Data Validator
==================================
Validates and cleans OHLC market data before pipeline ingestion.

Catches corrupt, malformed, or suspicious bars before they enter the
ML feature engineering pipeline.  Works with both daily and intraday
(1-min) DataFrames.

Validation checks (applied in order):
  1. Required columns exist (open, high, low, close, volume)
  2. No negative prices (O, H, L, C must be > 0)
  3. No negative or zero volume
  4. High >= max(Open, Close)  -- auto-fix: set H = max(O, H, C)
  5. Low  <= min(Open, Close)  -- auto-fix: set L = min(O, L, C)
  6. High >= Low               -- auto-fix: swap H and L
  7. No extreme daily moves (|close/prev_close - 1| > threshold)
  8. No duplicate index values (keep first)

Usage:
    from src.phase_02_preprocessing.ohlc_validator import OHLCValidator

    validator = OHLCValidator(max_daily_pct_change=0.50, auto_fix=True)
    clean_df, stats = validator.validate(raw_df)
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical lowercase column names expected by the pipeline
REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


class OHLCValidator:
    """Validates OHLC data integrity. Catches corrupt market data before pipeline ingestion."""

    def __init__(
        self,
        max_daily_pct_change: float = 0.50,
        auto_fix: bool = True,
    ):
        """
        Args:
            max_daily_pct_change: Maximum allowed daily price change as a
                fraction (0.50 = 50%).  Rows exceeding this are dropped.
            auto_fix: If True, attempt to fix minor violations (e.g. swap
                High/Low).  If False, drop violating rows instead.
        """
        if max_daily_pct_change <= 0:
            raise ValueError("max_daily_pct_change must be positive")
        self.max_daily_pct_change = max_daily_pct_change
        self.auto_fix = auto_fix

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Validate and clean OHLC data.

        Args:
            df: DataFrame with at minimum columns: open, high, low, close,
                volume (case-insensitive column matching).

        Returns:
            Tuple of (cleaned_df, stats_dict) where *stats_dict* contains:
              - ``rows_input``:  number of rows in the original frame
              - ``rows_output``: number of rows in the cleaned frame
              - ``rows_dropped``: total rows removed
              - ``rows_fixed``:  total rows repaired in-place
              - ``violations``:  dict mapping violation type to count
        """
        violations: Dict[str, int] = {}
        rows_fixed = 0

        # Work on a deep copy -- never mutate the caller's frame
        df = df.copy(deep=True)

        # --- 0. Handle empty DataFrame ----------------------------------
        if df.empty:
            return df, self._make_stats(0, 0, 0, violations)

        rows_input = len(df)

        # --- 1. Normalise column names to lowercase ---------------------
        df = self._normalise_columns(df)  # may raise ValueError

        # --- 2. Drop rows with NaN or inf in price columns ---------------
        price_cols = ["open", "high", "low", "close"]
        nan_mask = df[price_cols].isna().any(axis=1) | df[price_cols].isin(
            [np.inf, -np.inf]
        ).any(axis=1)
        n_nan = int(nan_mask.sum())
        if n_nan:
            violations["nan_or_inf_prices"] = n_nan
            logger.warning("[OHLC] Dropping %d rows with NaN/inf prices", n_nan)
            df = df.loc[~nan_mask]

        if df.empty:
            return df, self._make_stats(rows_input, 0, rows_fixed, violations)

        # --- 3. Negative prices -----------------------------------------
        neg_mask = (df[price_cols] <= 0).any(axis=1)
        n_neg = int(neg_mask.sum())
        if n_neg:
            violations["negative_prices"] = n_neg
            logger.warning("[OHLC] Dropping %d rows with non-positive prices", n_neg)
            df = df.loc[~neg_mask]

        if df.empty:
            return df, self._make_stats(rows_input, 0, rows_fixed, violations)

        # --- 4. Zero or negative volume ---------------------------------
        vol_mask = df["volume"] <= 0
        n_vol = int(vol_mask.sum())
        if n_vol:
            violations["zero_or_negative_volume"] = n_vol
            logger.warning("[OHLC] Dropping %d rows with volume <= 0", n_vol)
            df = df.loc[~vol_mask]

        if df.empty:
            return df, self._make_stats(rows_input, 0, rows_fixed, violations)

        # --- 5. High >= max(Open, Close) --------------------------------
        oc_max = df[["open", "close"]].max(axis=1)
        high_low_mask = df["high"] < oc_max
        n_high = int(high_low_mask.sum())
        if n_high:
            violations["high_below_oc_max"] = n_high
            if self.auto_fix:
                logger.warning(
                    "[OHLC] Fixing %d rows where High < max(O, C)", n_high
                )
                df.loc[high_low_mask, "high"] = df.loc[
                    high_low_mask, ["open", "high", "close"]
                ].max(axis=1)
                rows_fixed += n_high
            else:
                logger.warning(
                    "[OHLC] Dropping %d rows where High < max(O, C)", n_high
                )
                df = df.loc[~high_low_mask]

        if df.empty:
            return df, self._make_stats(rows_input, 0, rows_fixed, violations)

        # --- 6. Low <= min(Open, Close) ---------------------------------
        oc_min = df[["open", "close"]].min(axis=1)
        low_high_mask = df["low"] > oc_min
        n_low = int(low_high_mask.sum())
        if n_low:
            violations["low_above_oc_min"] = n_low
            if self.auto_fix:
                logger.warning(
                    "[OHLC] Fixing %d rows where Low > min(O, C)", n_low
                )
                df.loc[low_high_mask, "low"] = df.loc[
                    low_high_mask, ["open", "low", "close"]
                ].min(axis=1)
                rows_fixed += n_low
            else:
                logger.warning(
                    "[OHLC] Dropping %d rows where Low > min(O, C)", n_low
                )
                df = df.loc[~low_high_mask]

        if df.empty:
            return df, self._make_stats(rows_input, 0, rows_fixed, violations)

        # --- 7. High >= Low (swap if inverted) --------------------------
        hl_mask = df["high"] < df["low"]
        n_hl = int(hl_mask.sum())
        if n_hl:
            violations["high_below_low"] = n_hl
            if self.auto_fix:
                logger.warning(
                    "[OHLC] Swapping High/Low on %d rows", n_hl
                )
                orig_high = df.loc[hl_mask, "high"].copy()
                df.loc[hl_mask, "high"] = df.loc[hl_mask, "low"]
                df.loc[hl_mask, "low"] = orig_high
                rows_fixed += n_hl
            else:
                logger.warning(
                    "[OHLC] Dropping %d rows where High < Low", n_hl
                )
                df = df.loc[~hl_mask]

        if df.empty:
            return df, self._make_stats(rows_input, 0, rows_fixed, violations)

        # --- 8. Extreme daily moves -------------------------------------
        prev_close = df["close"].shift(1)
        pct_change = (df["close"] / prev_close - 1).abs()
        # First row has no prev_close, never flag it
        extreme_mask = pct_change > self.max_daily_pct_change
        n_extreme = int(extreme_mask.sum())
        if n_extreme:
            violations["extreme_daily_move"] = n_extreme
            logger.warning(
                "[OHLC] Dropping %d rows with daily move > %.0f%%",
                n_extreme,
                self.max_daily_pct_change * 100,
            )
            df = df.loc[~extreme_mask]

        if df.empty:
            return df, self._make_stats(rows_input, 0, rows_fixed, violations)

        # --- 9. Duplicate index values -----------------------------------
        if df.index.duplicated().any():
            n_dup = int(df.index.duplicated().sum())
            violations["duplicate_index"] = n_dup
            logger.warning(
                "[OHLC] Removing %d duplicate index entries (keeping first)", n_dup
            )
            df = df[~df.index.duplicated(keep="first")]

        rows_output = len(df)
        return df, self._make_stats(rows_input, rows_output, rows_fixed, violations)

    def validate_bar(self, row: pd.Series) -> Tuple[bool, List[str]]:
        """
        Validate a single OHLCV bar.

        Args:
            row: A pandas Series with keys open, high, low, close, volume
                 (case-insensitive).

        Returns:
            Tuple of (is_valid, list_of_violation_strings).
        """
        issues: List[str] = []

        # Normalise keys to lowercase for lookup
        data = {str(k).lower(): v for k, v in row.items()}

        for col in REQUIRED_COLUMNS:
            if col not in data:
                issues.append(f"missing_{col}")
                return False, issues

        o, h, l, c, v = (
            data["open"],
            data["high"],
            data["low"],
            data["close"],
            data["volume"],
        )

        # NaN / inf
        for name, val in [("open", o), ("high", h), ("low", l), ("close", c)]:
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                issues.append(f"nan_or_inf_{name}")

        # Positive prices
        for name, val in [("open", o), ("high", h), ("low", l), ("close", c)]:
            try:
                if val <= 0:
                    issues.append(f"non_positive_{name}")
            except TypeError:
                issues.append(f"non_numeric_{name}")

        # Volume
        try:
            if v <= 0:
                issues.append("zero_or_negative_volume")
        except TypeError:
            issues.append("non_numeric_volume")

        # Relationship checks (only if prices are valid numbers)
        if not issues:
            if h < max(o, c):
                issues.append("high_below_oc_max")
            if l > min(o, c):
                issues.append("low_above_oc_min")
            if h < l:
                issues.append("high_below_low")

        is_valid = len(issues) == 0
        return is_valid, issues

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Map columns to lowercase and verify all required columns are present.

        Raises ValueError if any required column is missing.
        """
        col_map = {c: c.lower() for c in df.columns}
        df = df.rename(columns=col_map)

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required OHLCV columns: {missing}. "
                f"Available: {list(df.columns)}"
            )
        return df

    @staticmethod
    def _make_stats(
        rows_input: int,
        rows_output: int,
        rows_fixed: int,
        violations: Dict[str, int],
    ) -> Dict[str, int]:
        return {
            "rows_input": rows_input,
            "rows_output": rows_output,
            "rows_dropped": rows_input - rows_output,
            "rows_fixed": rows_fixed,
            "violations": violations,
        }

"""
GIGA TRADER - Bar Resampler
============================
Converts 1-minute OHLCV bars to any target resolution (2, 3, 5, 10, 15, 30, 60 min).

Resamples using standard OHLCV aggregation rules:
  open=first, high=max, low=min, close=last, volume=sum

Key design:
  - Resamples WITHIN each (date, session) group so no bar spans
    a session boundary (premarket → regular → afterhours)
  - Preserves date, session columns
  - Caches resampled results keyed by resolution for batch training efficiency
  - 1-min passthrough returns a copy (no resampling overhead)

Usage:
    from src.phase_02_preprocessing.bar_resampler import BarResampler

    resampler = BarResampler()
    df_5min = resampler.resample(df_1min, resolution_minutes=5)
    df_15min = resampler.resample(df_1min, resolution_minutes=15)
"""

import logging
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger("BAR_RESAMPLER")


# Resolution string to minutes mapping
RESOLUTION_MAP: Dict[str, int] = {
    "1min": 1,
    "2min": 2,
    "3min": 3,
    "5min": 5,
    "10min": 10,
    "15min": 15,
    "30min": 30,
    "1h": 60,
}


def resolution_to_minutes(resolution: str) -> int:
    """Convert a resolution string like '5min' to integer minutes."""
    if resolution in RESOLUTION_MAP:
        return RESOLUTION_MAP[resolution]
    # Try parsing directly (e.g. "10" → 10)
    try:
        return int(resolution.replace("min", "").replace("m", ""))
    except (ValueError, AttributeError):
        raise ValueError(
            f"Unknown resolution '{resolution}'. "
            f"Valid: {list(RESOLUTION_MAP.keys())}"
        )


class BarResampler:
    """
    Resamples 1-minute OHLCV bars to a target resolution.

    Resamples within each (date, session) group so that no bar spans
    a session boundary. Caches results keyed by resolution to avoid
    redundant computation during grid search / batch training.
    """

    # Standard OHLCV aggregation rules
    OHLCV_AGG = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    # Valid resolutions in minutes
    VALID_RESOLUTIONS = [1, 2, 3, 5, 10, 15, 30, 60]

    def __init__(self):
        self._cache: Dict[int, pd.DataFrame] = {}
        self._source_hash: Optional[str] = None

    def resample(
        self,
        df_1min: pd.DataFrame,
        resolution_minutes: int,
        respect_sessions: bool = True,
    ) -> pd.DataFrame:
        """
        Resample 1-min bars to target resolution.

        Args:
            df_1min: DataFrame with 1-min OHLCV bars.
                     Expected columns: open, high, low, close, volume
                     Optional columns: date, session, timestamp, hour, minute
            resolution_minutes: Target resolution in minutes (2, 3, 5, 10, 15, 30, 60)
            respect_sessions: If True, resample within session boundaries
                             (premarket/regular/afterhours) to prevent cross-session bars

        Returns:
            DataFrame with same column structure but at target resolution
        """
        if resolution_minutes == 1:
            return df_1min.copy()

        if resolution_minutes not in self.VALID_RESOLUTIONS:
            raise ValueError(
                f"Invalid resolution {resolution_minutes}. "
                f"Valid: {self.VALID_RESOLUTIONS}"
            )

        # Check cache
        source_hash = self._compute_hash(df_1min)
        if source_hash == self._source_hash and resolution_minutes in self._cache:
            logger.debug(f"Cache hit for {resolution_minutes}min resolution")
            return self._cache[resolution_minutes].copy()

        # Cache miss - clear stale cache if source data changed
        if source_hash != self._source_hash:
            self._cache.clear()
            self._source_hash = source_hash

        logger.info(
            f"[RESAMPLE] Converting 1-min bars to {resolution_minutes}-min "
            f"({len(df_1min):,} input bars)"
        )

        if respect_sessions and "session" in df_1min.columns:
            result = self._resample_by_session(df_1min, resolution_minutes)
        elif "date" in df_1min.columns:
            result = self._resample_by_date(df_1min, resolution_minutes)
        else:
            result = self._resample_simple(df_1min, resolution_minutes)

        self._cache[resolution_minutes] = result

        logger.info(
            f"[RESAMPLE] Produced {len(result):,} bars at {resolution_minutes}-min "
            f"(compression ratio: {len(df_1min) / max(len(result), 1):.1f}x)"
        )

        return result.copy()

    def _resample_by_session(
        self, df: pd.DataFrame, resolution_minutes: int
    ) -> pd.DataFrame:
        """Resample within each (date, session) group to prevent cross-session bars."""
        chunks: List[pd.DataFrame] = []

        group_cols = ["date", "session"]
        for group_key, group in df.groupby(group_cols, sort=False):
            if len(group) < 2:
                # Single bar - keep as-is
                chunks.append(group.copy())
                continue

            resampled = self._resample_group(group, resolution_minutes)
            if len(resampled) > 0:
                date_val, session_val = group_key
                resampled["date"] = date_val
                resampled["session"] = session_val
                chunks.append(resampled)

        if not chunks:
            return pd.DataFrame(columns=df.columns)

        result = pd.concat(chunks, ignore_index=True)

        # Sort by timestamp if available
        if "timestamp" in result.columns:
            result = result.sort_values("timestamp").reset_index(drop=True)
            # Rebuild hour/minute/time from timestamp
            ts = pd.to_datetime(result["timestamp"])
            result["hour"] = ts.dt.hour
            result["minute"] = ts.dt.minute
            result["time"] = ts.dt.time

        return result

    def _resample_by_date(
        self, df: pd.DataFrame, resolution_minutes: int
    ) -> pd.DataFrame:
        """Resample within each date group (no session info)."""
        chunks: List[pd.DataFrame] = []

        for date_val, group in df.groupby("date", sort=False):
            if len(group) < 2:
                chunks.append(group.copy())
                continue

            resampled = self._resample_group(group, resolution_minutes)
            if len(resampled) > 0:
                resampled["date"] = date_val
                chunks.append(resampled)

        if not chunks:
            return pd.DataFrame(columns=df.columns)

        result = pd.concat(chunks, ignore_index=True)
        if "timestamp" in result.columns:
            result = result.sort_values("timestamp").reset_index(drop=True)
            ts = pd.to_datetime(result["timestamp"])
            result["hour"] = ts.dt.hour
            result["minute"] = ts.dt.minute
            result["time"] = ts.dt.time

        return result

    def _resample_simple(
        self, df: pd.DataFrame, resolution_minutes: int
    ) -> pd.DataFrame:
        """Simple resample - no date/session awareness. Last resort."""
        return self._resample_group(df, resolution_minutes)

    def _resample_group(
        self, group: pd.DataFrame, resolution_minutes: int
    ) -> pd.DataFrame:
        """
        Resample a single group of 1-min bars to target resolution.

        Uses pandas resample with standard OHLCV aggregation.
        """
        df = group.copy()

        # Need a datetime index for pandas resample
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        elif not isinstance(df.index, pd.DatetimeIndex):
            # Cannot resample without datetime - return as-is
            logger.warning("[RESAMPLE] No timestamp column or DatetimeIndex, skipping group")
            return group.copy()

        # Only aggregate OHLCV columns that exist
        agg_dict = {
            col: agg
            for col, agg in self.OHLCV_AGG.items()
            if col in df.columns
        }

        if not agg_dict:
            return pd.DataFrame()

        # Also aggregate 'trade_count' if present
        if "trade_count" in df.columns:
            agg_dict["trade_count"] = "sum"

        # Also aggregate 'vwap' as volume-weighted mean if present
        has_vwap = "vwap" in df.columns and "volume" in df.columns

        resampled = df.resample(f"{resolution_minutes}min").agg(agg_dict)

        # Compute VWAP for resampled bars
        if has_vwap:
            vol_groups = df["volume"].resample(f"{resolution_minutes}min").sum()
            vwap_vol = (df["vwap"] * df["volume"]).resample(f"{resolution_minutes}min").sum()
            resampled["vwap"] = vwap_vol / vol_groups.replace(0, np.nan)

        # Drop bars where close is NaN (no data in that window)
        if "close" in resampled.columns:
            resampled = resampled.dropna(subset=["close"])

        # Reset index to get timestamp back as column
        resampled = resampled.reset_index()
        resampled.rename(columns={"index": "timestamp"}, inplace=True)

        return resampled

    def _compute_hash(self, df: pd.DataFrame) -> str:
        """Compute a lightweight hash for cache invalidation."""
        if len(df) == 0:
            return "empty"
        # Use shape + first/last index as a fast fingerprint
        first = str(df.index[0]) if not isinstance(df.index[0], int) else str(df.iloc[0].get("timestamp", 0))
        last = str(df.index[-1]) if not isinstance(df.index[-1], int) else str(df.iloc[-1].get("timestamp", 0))
        return f"{len(df)}_{first}_{last}"

    def clear_cache(self):
        """Clear the resampled data cache."""
        self._cache.clear()
        self._source_hash = None
        logger.debug("[RESAMPLE] Cache cleared")

    def get_cached_resolutions(self) -> List[int]:
        """Return list of currently cached resolutions."""
        return list(self._cache.keys())

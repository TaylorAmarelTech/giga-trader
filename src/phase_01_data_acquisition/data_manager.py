"""
GIGA TRADER - Data Manager with Caching
========================================
Manages historical data with intelligent caching to avoid re-downloading.

Features:
  - Parquet-based caching (fast read/write)
  - Incremental updates (only download new data)
  - Data quality validation
  - 5-10 year data storage
  - Automatic gap detection and filling

Usage:
    from src.data_manager import DataManager
    dm = DataManager()
    df = dm.get_data(years=5)  # Gets cached or downloads
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import hashlib

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(project_root / ".env")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
DATA_CONFIG = {
    "cache_dir": project_root / "data" / "cache",
    "min_years": 5,
    "max_years": 10,
    "default_symbol": "SPY",
    "chunk_days": 30,  # Download in 30-day chunks
    "metadata_file": "data_metadata.json",

    # Data quality thresholds
    "min_bars_per_day": 200,
    "max_gap_minutes": 15,
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MANAGER
# ═══════════════════════════════════════════════════════════════════════════════
class DataManager:
    """
    Intelligent data manager with caching.

    Stores data in parquet format for fast access.
    Only downloads new data when cache is stale or missing.
    """

    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or DATA_CONFIG["cache_dir"]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.cache_dir / DATA_CONFIG["metadata_file"]
        self.metadata = self._load_metadata()

        # Initialize Alpaca client
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize Alpaca data client."""
        try:
            from alpaca.data.historical import StockHistoricalDataClient

            api_key = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")

            if api_key and secret_key:
                self.client = StockHistoricalDataClient(api_key, secret_key)
                print("[DataManager] Alpaca client initialized")
            else:
                print("[DataManager] WARNING: No Alpaca keys found")
        except ImportError:
            print("[DataManager] WARNING: Alpaca SDK not installed")

    def _load_metadata(self) -> Dict:
        """Load cache metadata."""
        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                return json.load(f)
        return {"symbols": {}}

    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def _get_cache_path(self, symbol: str, timeframe: str = "1min") -> Path:
        """Get cache file path for symbol."""
        return self.cache_dir / f"{symbol}_{timeframe}.parquet"

    def get_cached_data(self, symbol: str = "SPY") -> Optional[pd.DataFrame]:
        """Load cached data if available."""
        cache_path = self._get_cache_path(symbol)

        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)

                # Handle MultiIndex (symbol, timestamp) from Alpaca
                if isinstance(df.index, pd.MultiIndex):
                    # Drop the symbol level, keep just timestamp
                    df = df.droplevel('symbol') if 'symbol' in df.index.names else df

                # Ensure 'date' column exists for compatibility with train_robust_model
                if 'date' not in df.columns:
                    df['date'] = df.index.date

                # Add 'session' column based on time (required by engineer_all_features)
                if 'session' not in df.columns:
                    hours = df.index.hour
                    minutes = df.index.minute
                    time_minutes = hours * 60 + minutes
                    # Market: 9:30-16:00 = 570-960 minutes
                    df['session'] = 'regular'
                    df.loc[time_minutes < 570, 'session'] = 'premarket'
                    df.loc[time_minutes >= 960, 'session'] = 'afterhours'

                # Add 'time' column (required by engineer_all_features)
                if 'time' not in df.columns:
                    df['time'] = df.index.time

                # Add 'timestamp' column (required by engineer_all_features)
                if 'timestamp' not in df.columns:
                    df['timestamp'] = df.index

                # Reset index to avoid ambiguity (timestamp is now a column)
                df = df.reset_index(drop=True)

                print(f"[DataManager] Loaded {len(df):,} bars from cache for {symbol}")
                return df
            except Exception as e:
                print(f"[DataManager] Cache read error: {e}")

        return None

    def get_cache_info(self, symbol: str = "SPY") -> Dict:
        """Get information about cached data."""
        cache_path = self._get_cache_path(symbol)

        info = {
            "exists": cache_path.exists(),
            "path": str(cache_path),
            "size_mb": 0,
            "start_date": None,
            "end_date": None,
            "total_bars": 0,
            "trading_days": 0,
        }

        if cache_path.exists():
            info["size_mb"] = cache_path.stat().st_size / (1024 * 1024)

            df = self.get_cached_data(symbol)
            if df is not None and len(df) > 0:
                # Handle MultiIndex (symbol, timestamp) from Alpaca
                if isinstance(df.index, pd.MultiIndex):
                    # Drop the symbol level, keep just timestamp
                    df = df.droplevel(0) if df.index.names[0] == 'symbol' else df.droplevel(1)

                # Get dates - prefer 'timestamp' column if index is numeric
                if 'timestamp' in df.columns:
                    timestamps = pd.to_datetime(df['timestamp'])
                    info["start_date"] = str(timestamps.min())
                    info["end_date"] = str(timestamps.max())
                    try:
                        info["trading_days"] = timestamps.dt.normalize().nunique()
                    except Exception:
                        info["trading_days"] = len(df) // 390
                elif hasattr(df.index, 'min') and not df.index.dtype == 'int64':
                    info["start_date"] = str(df.index.min())
                    info["end_date"] = str(df.index.max())
                    try:
                        info["trading_days"] = df.index.normalize().nunique()
                    except AttributeError:
                        info["trading_days"] = len(df) // 390
                else:
                    # Fallback: read parquet metadata or estimate
                    info["start_date"] = None
                    info["end_date"] = None
                    info["trading_days"] = len(df) // 390

                info["total_bars"] = len(df)

        return info

    def needs_update(self, symbol: str = "SPY", max_age_hours: int = 24) -> Tuple[bool, str]:
        """Check if cache needs updating."""
        cache_info = self.get_cache_info(symbol)

        if not cache_info["exists"]:
            return True, "No cache exists"

        if cache_info["total_bars"] == 0:
            return True, "Cache is empty"

        # Check if we have enough history
        if cache_info["start_date"]:
            try:
                # Handle edge case where start_date might be numeric index instead of date
                start_str = str(cache_info["start_date"])
                if start_str.isdigit() or not start_str or start_str == "0":
                    # Invalid date - need to refresh
                    return True, "Invalid start_date in cache metadata"
                start = pd.to_datetime(start_str)
                years_of_data = (datetime.now() - start.to_pydatetime().replace(tzinfo=None)).days / 365
                if years_of_data < DATA_CONFIG["min_years"]:
                    return True, f"Only {years_of_data:.1f} years of data, need {DATA_CONFIG['min_years']}"
            except Exception as e:
                # Date parsing failed - cache metadata is corrupted
                return True, f"Invalid start_date format: {e}"

        # Check if data is stale (missing recent data)
        if cache_info["end_date"]:
            try:
                end_str = str(cache_info["end_date"])
                if end_str.isdigit() or not end_str or end_str == "0":
                    return True, "Invalid end_date in cache metadata"
                end = pd.to_datetime(end_str)
                hours_old = (datetime.now() - end.to_pydatetime().replace(tzinfo=None)).total_seconds() / 3600

                # During trading hours, update more frequently
                now = datetime.now()
                is_market_hours = (now.weekday() < 5 and
                                  9 <= now.hour < 17)

                if is_market_hours and hours_old > 1:
                    return True, f"Data is {hours_old:.1f} hours old (market hours)"
                elif hours_old > max_age_hours:
                    return True, f"Data is {hours_old:.1f} hours old"
            except Exception as e:
                return True, f"Invalid end_date format: {e}"

        return False, "Cache is up to date"

    def download_data(
        self,
        symbol: str = "SPY",
        years: int = 5,
        include_extended: bool = True,
        progress_callback=None,
    ) -> pd.DataFrame:
        """
        Download historical data from Alpaca.

        Args:
            symbol: Stock symbol
            years: Years of history to download
            include_extended: Include premarket/afterhours
            progress_callback: Optional callback(current_chunk, total_chunks)

        Returns:
            DataFrame with OHLCV data
        """
        if not self.client:
            raise ValueError("Alpaca client not initialized")

        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        print(f"\n[DataManager] Downloading {years} years of {symbol} data...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)

        # Calculate chunks
        total_days = (end_date - start_date).days
        chunk_size = DATA_CONFIG["chunk_days"]
        n_chunks = (total_days + chunk_size - 1) // chunk_size

        all_bars = []
        current_end = end_date

        for i in range(n_chunks):
            current_start = current_end - timedelta(days=chunk_size)
            if current_start < start_date:
                current_start = start_date

            try:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Minute,
                    start=current_start,
                    end=current_end,
                )

                bars = self.client.get_stock_bars(request)

                if hasattr(bars, 'df') and len(bars.df) > 0:
                    all_bars.append(bars.df)

                if progress_callback:
                    progress_callback(i + 1, n_chunks)
                else:
                    pct = (i + 1) / n_chunks * 100
                    print(f"  Chunk {i+1}/{n_chunks} ({pct:.0f}%) - {current_start.date()} to {current_end.date()}")

            except Exception as e:
                print(f"  ERROR on chunk {i+1}: {e}")

            current_end = current_start

            if current_end <= start_date:
                break

        if not all_bars:
            print("[DataManager] No data downloaded!")
            return pd.DataFrame()

        # Combine all chunks
        df = pd.concat(all_bars)

        # Handle MultiIndex (symbol, timestamp) from Alpaca
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel('symbol') if 'symbol' in df.index.names else df

        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]

        # Clean column names
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [c.lower() for c in df.columns]

        # Add date column for compatibility
        df['date'] = df.index.date

        # Add session column (required by engineer_all_features)
        hours = df.index.hour
        minutes = df.index.minute
        time_minutes = hours * 60 + minutes
        df['session'] = 'regular'
        df.loc[time_minutes < 570, 'session'] = 'premarket'
        df.loc[time_minutes >= 960, 'session'] = 'afterhours'

        # Add 'time' column (required by engineer_all_features)
        df['time'] = df.index.time

        # Add 'timestamp' column (required by engineer_all_features)
        df['timestamp'] = df.index

        print(f"[DataManager] Downloaded {len(df):,} bars")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Reset index to avoid ambiguity (timestamp is now a column)
        df = df.reset_index(drop=True)

        return df

    def update_cache(
        self,
        symbol: str = "SPY",
        years: int = 5,
    ) -> pd.DataFrame:
        """
        Update cache with new data (incremental if possible).
        """
        existing = self.get_cached_data(symbol)

        if existing is not None and len(existing) > 0:
            # Get last date from timestamp column (since index is reset)
            if 'timestamp' in existing.columns:
                last_date = pd.to_datetime(existing['timestamp'].max())
            else:
                # Fallback: try index if it's a datetime
                try:
                    last_date = pd.to_datetime(existing.index.max())
                except Exception:
                    # Can't determine last date, do full download
                    last_date = None

            if last_date is not None:
                days_missing = (datetime.now() - last_date.to_pydatetime().replace(tzinfo=None)).days + 1

                if days_missing < 30:
                    print(f"[DataManager] Incremental update: downloading {days_missing} days")
                    new_data = self.download_data(symbol, years=days_missing/365)

                    if len(new_data) > 0:
                        # Combine and deduplicate using timestamp column
                        combined = pd.concat([existing, new_data], ignore_index=True)
                        if 'timestamp' in combined.columns:
                            combined = combined.sort_values('timestamp')
                            combined = combined.drop_duplicates(subset=['timestamp'], keep='last')
                            combined = combined.reset_index(drop=True)
                        else:
                            combined = combined.sort_index()
                            combined = combined[~combined.index.duplicated(keep='last')]

                        self.save_to_cache(combined, symbol)
                        return combined

                    return existing

        # Full download
        df = self.download_data(symbol, years=years)

        if len(df) > 0:
            self.save_to_cache(df, symbol)

        return df

    def save_to_cache(self, df: pd.DataFrame, symbol: str = "SPY"):
        """Save data to cache."""
        cache_path = self._get_cache_path(symbol)

        df.to_parquet(cache_path, compression='snappy')

        # Update metadata
        self.metadata["symbols"][symbol] = {
            "last_updated": datetime.now().isoformat(),
            "start_date": str(df.index.min()),
            "end_date": str(df.index.max()),
            "total_bars": len(df),
            "file_size_mb": cache_path.stat().st_size / (1024 * 1024),
        }
        self._save_metadata()

        print(f"[DataManager] Saved {len(df):,} bars to {cache_path}")

    def get_data(
        self,
        symbol: str = "SPY",
        years: int = 5,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Get data - from cache if available, otherwise download.

        This is the main entry point for getting data.
        """
        needs_update, reason = self.needs_update(symbol)

        if force_refresh or needs_update:
            print(f"[DataManager] {reason}")
            return self.update_cache(symbol, years=max(years, DATA_CONFIG["min_years"]))

        return self.get_cached_data(symbol)

    def validate_data(self, df: pd.DataFrame) -> Dict:
        """Validate data quality."""
        if df is None or len(df) == 0:
            return {"valid": False, "issues": ["No data"]}

        issues = []

        # Check date range
        days = (df.index.max() - df.index.min()).days
        years = days / 365
        if years < DATA_CONFIG["min_years"]:
            issues.append(f"Only {years:.1f} years of data (need {DATA_CONFIG['min_years']})")

        # Check for gaps
        df_daily = df.resample('D').count()
        low_bar_days = (df_daily['close'] < DATA_CONFIG["min_bars_per_day"]).sum()
        if low_bar_days > 10:
            issues.append(f"{low_bar_days} days with sparse data")

        # Check OHLC consistency
        invalid_ohlc = ((df['high'] < df['low']) |
                       (df['high'] < df['open']) |
                       (df['high'] < df['close']) |
                       (df['low'] > df['open']) |
                       (df['low'] > df['close']))
        if invalid_ohlc.sum() > 0:
            issues.append(f"{invalid_ohlc.sum()} bars with invalid OHLC")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "stats": {
                "total_bars": len(df),
                "years": years,
                "trading_days": df.index.normalize().nunique(),
            }
        }

    def get_data_summary(self) -> Dict:
        """Get summary of all cached data."""
        summary = {
            "cache_dir": str(self.cache_dir),
            "symbols": {},
        }

        for symbol in self.metadata.get("symbols", {}):
            summary["symbols"][symbol] = self.get_cache_info(symbol)

        return summary


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
_data_manager = None

def get_data_manager() -> DataManager:
    """Get singleton data manager."""
    global _data_manager
    if _data_manager is None:
        _data_manager = DataManager()
    return _data_manager


def get_spy_data(years: int = 5, force_refresh: bool = False) -> pd.DataFrame:
    """Convenience function to get SPY data."""
    dm = get_data_manager()
    return dm.get_data("SPY", years=years, force_refresh=force_refresh)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("GIGA TRADER - Data Manager")
    print("=" * 60)

    dm = DataManager()

    # Check cache status
    print("\n[Cache Status]")
    info = dm.get_cache_info("SPY")
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Check if update needed
    needs_update, reason = dm.needs_update("SPY")
    print(f"\n[Update Needed] {needs_update} - {reason}")

    # Get data (cached or download)
    df = dm.get_data("SPY", years=5)

    if df is not None and len(df) > 0:
        # Validate
        validation = dm.validate_data(df)
        print(f"\n[Validation]")
        print(f"  Valid: {validation['valid']}")
        if validation['issues']:
            for issue in validation['issues']:
                print(f"  Issue: {issue}")
        print(f"  Stats: {validation['stats']}")

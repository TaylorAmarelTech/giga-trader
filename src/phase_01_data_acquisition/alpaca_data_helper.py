"""
GIGA TRADER - Alpaca Data Helper
=================================
Helper class for fetching data from Alpaca API.
Replaces yfinance for more reliable and consistent data access.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
load_dotenv(project_root / ".env")


# ═══════════════════════════════════════════════════════════════════════════════
# ALPACA DATA HELPER (replaces yfinance)
# ═══════════════════════════════════════════════════════════════════════════════
class AlpacaDataHelper:
    """
    Helper class for fetching data from Alpaca API.
    Replaces yfinance for more reliable and consistent data access.
    """

    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if AlpacaDataHelper._client is None:
            self._init_client()

    def _init_client(self):
        """Initialize Alpaca data client."""
        try:
            from alpaca.data.historical import StockHistoricalDataClient

            api_key = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")

            if api_key and secret_key:
                AlpacaDataHelper._client = StockHistoricalDataClient(api_key, secret_key)
                print("[AlpacaHelper] Alpaca client initialized")
            else:
                print("[AlpacaHelper] WARNING: No Alpaca keys found in .env")
        except ImportError:
            print("[AlpacaHelper] WARNING: Alpaca SDK not installed (pip install alpaca-py)")

    @property
    def client(self):
        return AlpacaDataHelper._client

    def download_daily_bars(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Download daily OHLCV bars for multiple symbols.

        Args:
            symbols: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with MultiIndex (date, symbol) or columns per symbol
        """
        if not self.client:
            print("[AlpacaHelper] Client not initialized, returning empty DataFrame")
            return pd.DataFrame()

        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            # Download in chunks to avoid rate limits
            all_bars = []
            chunk_size = 20  # Alpaca handles multiple symbols well

            for i in range(0, len(symbols), chunk_size):
                chunk_symbols = symbols[i:i + chunk_size]

                try:
                    request = StockBarsRequest(
                        symbol_or_symbols=chunk_symbols,
                        timeframe=TimeFrame.Day,
                        start=start_date,
                        end=end_date,
                    )

                    bars = self.client.get_stock_bars(request)

                    if hasattr(bars, 'df') and len(bars.df) > 0:
                        all_bars.append(bars.df)

                except Exception as e:
                    print(f"  [AlpacaHelper] Failed to download chunk {chunk_symbols[:3]}...: {e}")
                    continue

            if len(all_bars) == 0:
                return pd.DataFrame()

            # Combine all chunks
            combined = pd.concat(all_bars)

            # Handle MultiIndex (symbol, timestamp)
            if isinstance(combined.index, pd.MultiIndex):
                # Pivot to have symbols as columns with close prices
                close_prices = combined['close'].unstack(level='symbol')
                high_prices = combined['high'].unstack(level='symbol')
                low_prices = combined['low'].unstack(level='symbol')

                # Normalize timestamp index to just dates (no timezone)
                # This ensures compatibility with spy_daily date matching
                close_prices.index = pd.to_datetime(close_prices.index).normalize().tz_localize(None)
                high_prices.index = pd.to_datetime(high_prices.index).normalize().tz_localize(None)
                low_prices.index = pd.to_datetime(low_prices.index).normalize().tz_localize(None)

                # Return close prices with normalized date index
                return {
                    'close': close_prices,
                    'high': high_prices,
                    'low': low_prices,
                }

            return combined

        except Exception as e:
            print(f"[AlpacaHelper] Download error: {e}")
            return pd.DataFrame()

    def download_close_prices(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Download just close prices for multiple symbols.
        Returns DataFrame with date index and symbol columns.
        """
        result = self.download_daily_bars(symbols, start_date, end_date)

        if isinstance(result, dict):
            return result.get('close', pd.DataFrame())
        elif isinstance(result, pd.DataFrame) and not result.empty:
            if 'close' in result.columns:
                return result[['close']]
            return result
        return pd.DataFrame()


# Create singleton instance
_alpaca_helper = None

def get_alpaca_helper() -> AlpacaDataHelper:
    """Get the singleton AlpacaDataHelper instance."""
    global _alpaca_helper
    if _alpaca_helper is None:
        _alpaca_helper = AlpacaDataHelper()
    return _alpaca_helper

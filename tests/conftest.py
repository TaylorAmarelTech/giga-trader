"""
Shared test fixtures for the giga_trader test suite.

All fixtures use synthetic/mock data only -- no API calls.
"""

import os
import sys
import json
import tempfile
from datetime import datetime, timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path so `from src.*` imports work.
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# sample_ohlcv_df -- 500 rows of synthetic OHLCV data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """DataFrame with 500 rows of synthetic OHLCV data."""
    np.random.seed(42)
    n = 500

    # Generate a realistic-looking price series starting around 450
    base_price = 450.0
    daily_returns = np.random.normal(0.0003, 0.012, n)
    close_prices = base_price * np.cumprod(1 + daily_returns)

    # Build OHLC from close with realistic relationships
    open_prices = close_prices * (1 + np.random.normal(0, 0.003, n))
    high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0, 0.005, n)))
    volume = np.random.randint(50_000_000, 200_000_000, n).astype(float)

    # Generate business-day timestamps
    start_date = pd.Timestamp("2023-01-03", tz="US/Eastern")
    timestamps = pd.bdate_range(start=start_date, periods=n, freq="B")

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume,
    })
    return df


# ---------------------------------------------------------------------------
# sample_daily_df -- 250 rows of daily data with some pre-computed features
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_daily_df() -> pd.DataFrame:
    """DataFrame with 250 rows of daily data including pre-computed features."""
    np.random.seed(123)
    n = 250

    base_price = 440.0
    daily_returns = np.random.normal(0.0005, 0.011, n)
    close_prices = base_price * np.cumprod(1 + daily_returns)

    open_prices = close_prices * (1 + np.random.normal(0, 0.002, n))
    high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0, 0.004, n)))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0, 0.004, n)))
    volume = np.random.randint(60_000_000, 180_000_000, n).astype(float)

    start_date = pd.Timestamp("2024-01-02", tz="US/Eastern")
    dates = pd.bdate_range(start=start_date, periods=n, freq="B")

    df = pd.DataFrame({
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume,
    }, index=dates)
    df.index.name = "date"

    # Pre-computed features
    df["day_return"] = df["close"].pct_change()
    df["rsi_14"] = _compute_rsi(df["close"], 14)
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["bb_upper"] = df["sma_20"] + 2 * df["close"].rolling(20).std()
    df["bb_lower"] = df["sma_20"] - 2 * df["close"].rolling(20).std()
    df["atr_14"] = _compute_atr(df, 14)
    df["vol_20d"] = df["day_return"].rolling(20).std()

    return df


# ---------------------------------------------------------------------------
# sample_1min_df -- ~2000 rows of 1-minute bars in market hours
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_1min_df() -> pd.DataFrame:
    """DataFrame with ~2000 rows of 1-minute bars (market hours 9:30-16:00 ET)."""
    np.random.seed(7)

    # 5 trading days x 390 minutes per day = 1950 bars
    n_days = 5
    minutes_per_day = 390  # 9:30 to 16:00

    all_rows = []
    base_price = 450.0

    for day_offset in range(n_days):
        day_date = pd.Timestamp("2025-01-06", tz="US/Eastern") + pd.Timedelta(days=day_offset)
        # Skip weekends
        while day_date.weekday() >= 5:
            day_date += pd.Timedelta(days=1)

        market_open = day_date.replace(hour=9, minute=30, second=0, microsecond=0)

        for minute in range(minutes_per_day):
            ts = market_open + pd.Timedelta(minutes=minute)
            ret = np.random.normal(0, 0.0003)
            base_price *= (1 + ret)

            o = base_price * (1 + np.random.normal(0, 0.0001))
            c = base_price
            h = max(o, c) * (1 + abs(np.random.normal(0, 0.0002)))
            lo = min(o, c) * (1 - abs(np.random.normal(0, 0.0002)))
            v = float(np.random.randint(50_000, 500_000))

            all_rows.append({
                "timestamp": ts,
                "open": o,
                "high": h,
                "low": lo,
                "close": c,
                "volume": v,
            })

    df = pd.DataFrame(all_rows)
    return df


# ---------------------------------------------------------------------------
# tmp_model_dir -- temporary directory for saving/loading test models
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_model_dir(tmp_path):
    """Provide a temporary directory for model save/load tests."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


# ---------------------------------------------------------------------------
# mock_registry_data -- dict mimicking model_registry.json format
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_registry_data() -> dict:
    """Dict fixture mimicking model_registry.json format."""
    return {
        "version": "2.0",
        "updated_at": "2026-01-29T12:00:00",
        "n_models": 2,
        "models": {
            "model_swing_001": {
                "model_id": "model_swing_001",
                "target_type": "swing",
                "status": "production",
                "created_at": "2026-01-28T10:00:00",
                "updated_at": "2026-01-29T12:00:00",
                "tags": ["production", "v1"],
                "metrics": {
                    "cv_auc": 0.769,
                    "test_auc": 0.755,
                    "win_rate": 0.781,
                    "sharpe_ratio": 1.2,
                },
            },
            "model_timing_001": {
                "model_id": "model_timing_001",
                "target_type": "timing",
                "status": "trained",
                "created_at": "2026-01-28T10:30:00",
                "updated_at": "2026-01-29T12:00:00",
                "tags": ["experiment"],
                "metrics": {
                    "cv_auc": 0.706,
                    "test_auc": 0.690,
                    "win_rate": 0.619,
                    "sharpe_ratio": 0.8,
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# resource_config -- system-aware resource configuration
# ---------------------------------------------------------------------------

@pytest.fixture
def resource_config():
    """Provide a ResourceConfig using actual system detection."""
    from src.core.system_resources import create_resource_config
    return create_resource_config()


@pytest.fixture
def resource_test_rows(resource_config):
    """Return resource-appropriate test data sizes.

    LOW: 100 rows, MEDIUM: 300, HIGH: 500, ULTRA: 1000.
    """
    tier_sizes = {"low": 100, "medium": 300, "high": 500, "ultra": 1000}
    return tier_sizes.get(resource_config.tier, 300)


# ---------------------------------------------------------------------------
# Helpers (not fixtures -- internal use only)
# ---------------------------------------------------------------------------

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI for test data generation."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute ATR for test data generation."""
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

"""
GIGA TRADER - Volatility Regime Features (VIX-based)
=====================================================
Volatility regime features using VIX and related indicators.

Provides market regime classification for validation:
- Low vol (VIX < 15): Complacent/trending
- Normal vol (VIX 15-25): Standard conditions
- High vol (VIX 25-35): Elevated uncertainty
- Extreme vol (VIX > 35): Crisis conditions

Features include VIX levels, term structure, and regime transitions.
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

from src.phase_01_data_acquisition.alpaca_data_helper import get_alpaca_helper
from src.core.feature_base import FeatureModuleBase


class VolatilityRegimeFeatures(FeatureModuleBase):
    """
    Volatility regime features using VIX and related indicators.

    Provides market regime classification for validation:
    - Low vol (VIX < 15): Complacent/trending
    - Normal vol (VIX 15-25): Standard conditions
    - High vol (VIX 25-35): Elevated uncertainty
    - Extreme vol (VIX > 35): Crisis conditions

    Features include VIX levels, term structure, and regime transitions.

    FEATURE_NAMES is empty because the exact feature set varies dynamically
    based on which VIX-related instruments have data available.
    """

    # Dynamic feature set — varies by available vol instruments at runtime
    FEATURE_NAMES = []

    # VIX-related ETFs/indicators
    VOL_TICKERS = ["VXX", "UVXY", "SVXY", "VIXY"]

    # VIX regime thresholds
    REGIMES = {
        "low": 15,
        "normal": 25,
        "high": 35,
        "extreme": 50,
    }

    def __init__(self):
        self.data_cache = {}

    def download_vol_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Download volatility-related data via Alpaca."""
        print("\n[VOL] Downloading volatility regime data via Alpaca...")

        try:
            helper = get_alpaca_helper()

            # Download VXX as proxy for VIX (VIX itself not tradeable)
            tickers = ["VXX", "UVXY"]

            data = helper.download_daily_bars(tickers, start_date, end_date)

            if isinstance(data, dict):
                close = data.get("close", pd.DataFrame())
            else:
                print("  [WARN] Unexpected data format from Alpaca")
                return pd.DataFrame()

            if close.empty:
                print("  [WARN] No volatility data returned from Alpaca")
                return pd.DataFrame()

            print(f"  Downloaded {len(close.columns)} vol instruments, {len(close)} days")

            self.data_cache = {"close": close}
            return close

        except Exception as e:
            print(f"  [ERROR] Failed to download volatility data: {e}")
            return pd.DataFrame()

    def create_vol_features(
        self,
        spy_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create volatility regime features."""
        if not self.data_cache:
            print("  [WARN] No vol data cached. Call download_vol_data first.")
            return spy_daily

        close = self.data_cache.get("close", pd.DataFrame())

        if close.empty:
            return spy_daily

        print("\n[VOL] Engineering volatility regime features...")

        features = spy_daily.copy()
        vol_features = pd.DataFrame(index=close.index)

        # Use VXX as VIX proxy (scaled)
        if "VXX" in close.columns:
            vxx = close["VXX"]
            vxx_return = vxx.pct_change()

            # 1. VXX level features
            vol_features["vxx_level"] = vxx
            vol_features["vxx_return"] = vxx_return
            vol_features["vxx_5d_change"] = vxx.pct_change(5)
            vol_features["vxx_20d_change"] = vxx.pct_change(20)

            # 2. VXX moving averages
            vol_features["vxx_sma_5"] = vxx.rolling(5).mean()
            vol_features["vxx_sma_20"] = vxx.rolling(20).mean()
            vol_features["vxx_above_sma20"] = (vxx > vol_features["vxx_sma_20"]).astype(int)

            # 3. VXX percentile rank (rolling 60-day)
            vol_features["vxx_percentile_60d"] = vxx.rolling(60).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            )

            # 4. Volatility spike detection
            vxx_std = vxx_return.rolling(20).std()
            vol_features["vxx_spike"] = (vxx_return > 2 * vxx_std).astype(int)
            vol_features["vxx_crash"] = (vxx_return < -2 * vxx_std).astype(int)

            # 5. Realized volatility (SPY-based, using features if available)
            if "day_return" in features.columns:
                spy_ret = features.set_index("date")["day_return"] if "date" in features.columns else features["day_return"]
                try:
                    # Align to vol_features index
                    aligned_spy = spy_ret.reindex(vol_features.index)
                    vol_features["realized_vol_5d"] = aligned_spy.rolling(5).std() * np.sqrt(252)
                    vol_features["realized_vol_20d"] = aligned_spy.rolling(20).std() * np.sqrt(252)
                except (KeyError, ValueError):
                    pass

            # 6. VXX term structure proxy (VXX vs UVXY ratio)
            if "UVXY" in close.columns:
                vol_features["vol_term_structure"] = close["VXX"] / close["UVXY"]
                vol_features["vol_contango"] = (vol_features["vol_term_structure"] < 1).astype(int)
                vol_features["vol_backwardation"] = (vol_features["vol_term_structure"] > 1).astype(int)

        # 7. Regime classification
        if "vxx_percentile_60d" in vol_features.columns:
            pct = vol_features["vxx_percentile_60d"]
            vol_features["vol_regime_low"] = (pct < 0.25).astype(int)
            vol_features["vol_regime_normal"] = ((pct >= 0.25) & (pct < 0.75)).astype(int)
            vol_features["vol_regime_high"] = ((pct >= 0.75) & (pct < 0.9)).astype(int)
            vol_features["vol_regime_extreme"] = (pct >= 0.9).astype(int)

        # 8. Regime transitions
        if "vol_regime_low" in vol_features.columns:
            # Detect regime changes
            vol_features["vol_regime_change"] = 0
            for regime in ["low", "normal", "high", "extreme"]:
                col = f"vol_regime_{regime}"
                if col in vol_features.columns:
                    vol_features["vol_regime_change"] += vol_features[col].diff().abs()

        # Add date for merge
        vol_features["date"] = pd.to_datetime(vol_features.index.date)

        # Merge with SPY daily
        features = features.merge(
            vol_features.reset_index(drop=True),
            on="date",
            how="left"
        )

        # Fill NaN
        vol_cols = [c for c in features.columns if c.startswith(("vxx_", "vol_", "realized_"))]
        features[vol_cols] = features[vol_cols].fillna(0)

        print(f"  Added {len(vol_cols)} volatility regime features")
        return features

    def analyze_vol_regime(self, features: pd.DataFrame) -> Dict:
        """Analyze current volatility regime."""
        if len(features) == 0:
            return {}

        latest = features.iloc[-1].to_dict()

        signal = {
            "date": latest.get("date"),
            "vxx_level": latest.get("vxx_level", 0),
            "vxx_percentile": latest.get("vxx_percentile_60d", 0.5),
            "realized_vol_20d": latest.get("realized_vol_20d", 0),
            "vol_spike": latest.get("vxx_spike", 0),
        }

        # Regime classification
        pct = signal["vxx_percentile"]
        if pct < 0.25:
            signal["regime"] = "LOW_VOL"
            signal["market_condition"] = "COMPLACENT"
        elif pct < 0.75:
            signal["regime"] = "NORMAL_VOL"
            signal["market_condition"] = "STANDARD"
        elif pct < 0.9:
            signal["regime"] = "HIGH_VOL"
            signal["market_condition"] = "ELEVATED_UNCERTAINTY"
        else:
            signal["regime"] = "EXTREME_VOL"
            signal["market_condition"] = "CRISIS"

        return signal

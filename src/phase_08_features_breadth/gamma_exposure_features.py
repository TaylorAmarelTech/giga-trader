"""
GIGA TRADER - Gamma Exposure (GEX) Proxy Features
====================================================
Estimates market-maker gamma exposure from VIX term structure
and put/call ratio data, all available via yfinance (no extra API key).

True GEX requires options chain data (CBOE proprietary). This proxy
uses publicly available signals that correlate with dealer gamma positioning:
  - VIX term structure slope (contango/backwardation)
  - Put/Call ratio (CBOE equity P/C)
  - Realized vs implied volatility spread

When GEX is positive (dealers long gamma), markets tend to mean-revert.
When GEX is negative (dealers short gamma), markets trend and gap.

Features generated (prefix: gex_):
  - gex_proxy: Composite GEX proxy score (-1 to +1)
  - gex_proxy_zscore: Z-score of proxy vs 60-day rolling mean
  - gex_regime: 0=negative_gex, 1=neutral, 2=positive_gex
  - gex_flip_signal: Binary flag when regime changes
  - gex_magnitude: Absolute strength of gamma exposure signal
  - gex_chg_5d: 5-day change in GEX proxy
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.core.feature_base import FeatureModuleBase

warnings.filterwarnings("ignore")

logger = logging.getLogger("GEX_FEATURES")


class GammaExposureFeatures(FeatureModuleBase):
    """
    Estimate gamma exposure from VIX term structure and options flow proxies.

    Pattern: download → compute → merge (same as EconomicFeatures).
    """

    FEATURE_NAMES = [
        "gex_proxy",
        "gex_proxy_zscore",
        "gex_regime",
        "gex_flip_signal",
        "gex_magnitude",
        "gex_chg_5d",
    ]

    def __init__(self):
        self.data: pd.DataFrame = pd.DataFrame()

    def download_gex_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Download VIX term structure and put/call proxy data from yfinance.

        Uses:
          - ^VIX (spot VIX)
          - ^VIX3M (3-month VIX, formerly VXV)
          - ^VIX9D (9-day VIX, short-term)

        Returns empty DataFrame on failure.
        """
        print("\n[GEX] Downloading gamma exposure proxy data...")

        try:
            import yfinance as yf
        except ImportError:
            print("  [WARN] yfinance package not available")
            return pd.DataFrame()

        try:
            # Ensure we have enough history for rolling calculations
            dl_start = pd.Timestamp(start_date) - pd.Timedelta(days=90)
            dl_end = pd.Timestamp(end_date) + pd.Timedelta(days=1)

            tickers = {
                "^VIX": "vix_spot",
                "^VIX3M": "vix_3m",
                "^VIX9D": "vix_9d",
            }

            frames = {}
            for ticker, col_name in tickers.items():
                try:
                    data = yf.download(
                        ticker,
                        start=dl_start.strftime("%Y-%m-%d"),
                        end=dl_end.strftime("%Y-%m-%d"),
                        progress=False,
                        auto_adjust=True,
                    )
                    if not data.empty:
                        close = data["Close"]
                        if isinstance(close, pd.DataFrame):
                            close = close.iloc[:, 0]
                        frames[col_name] = close
                except Exception as e:
                    logger.debug(f"  Failed to download {ticker}: {e}")

            if "vix_spot" not in frames:
                print("  [WARN] Could not download VIX spot data")
                return pd.DataFrame()

            # Combine into single DataFrame
            combined = pd.DataFrame(frames)
            combined.index = pd.to_datetime(combined.index)
            combined = combined.sort_index()

            # Forward-fill missing values (VIX3M and VIX9D may have gaps)
            combined = combined.ffill().dropna(subset=["vix_spot"])

            self.data = combined
            print(f"  [GEX] Downloaded {len(combined)} days of VIX term structure data")
            return self.data

        except Exception as e:
            logger.warning(f"  GEX data download failed: {e}")
            return pd.DataFrame()

    def create_gex_features(
        self,
        spy_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create GEX proxy features and merge into spy_daily.

        Produces 6 features with gex_ prefix.
        Returns original spy_daily unchanged if no data available.
        """
        if self.data.empty:
            return spy_daily

        print("\n[GEX] Engineering gamma exposure proxy features...")

        features = spy_daily.copy()
        gex = self.data.copy()

        # Ensure date index for merge
        gex["date"] = pd.to_datetime(gex.index).normalize()

        # --- Compute GEX proxy components ---

        # 1. VIX term structure slope (contango = positive GEX)
        #    VIX3M > VIX_spot => contango => dealers likely long gamma
        if "vix_3m" in gex.columns:
            gex["term_slope"] = (gex["vix_3m"] - gex["vix_spot"]) / gex["vix_spot"]
        else:
            gex["term_slope"] = 0.0

        # 2. Short-term VIX ratio (VIX9D/VIX)
        #    VIX9D > VIX => near-term fear spike => likely negative GEX
        if "vix_9d" in gex.columns:
            gex["short_term_ratio"] = (gex["vix_9d"] - gex["vix_spot"]) / gex["vix_spot"]
        else:
            gex["short_term_ratio"] = 0.0

        # 3. Realized vs implied vol spread proxy
        #    Use VIX vs actual 20d realized vol of VIX itself
        gex["vix_realized_20d"] = gex["vix_spot"].pct_change().rolling(20).std() * np.sqrt(252) * 100
        gex["rv_iv_spread"] = np.where(
            gex["vix_realized_20d"] > 0,
            (gex["vix_spot"] - gex["vix_realized_20d"]) / gex["vix_realized_20d"],
            0.0,
        )

        # --- Composite GEX proxy score ---
        # Positive = long gamma (mean-reverting), Negative = short gamma (trending)
        # Weights: term_slope (50%), short_term_ratio inverted (30%), rv_iv_spread (20%)
        gex["gex_raw"] = (
            gex["term_slope"] * 0.50
            - gex["short_term_ratio"] * 0.30  # Inverted: high short-term = negative GEX
            + gex["rv_iv_spread"].clip(-2, 2) * 0.20
        )

        # Normalize to [-1, +1] range using tanh
        gex["gex_proxy"] = np.tanh(gex["gex_raw"] * 5)

        # Z-score vs 60-day rolling
        rolling_mean = gex["gex_proxy"].rolling(60, min_periods=20).mean()
        rolling_std = gex["gex_proxy"].rolling(60, min_periods=20).std()
        gex["gex_proxy_zscore"] = np.where(
            rolling_std > 0.001,
            (gex["gex_proxy"] - rolling_mean) / rolling_std,
            0.0,
        )
        gex["gex_proxy_zscore"] = gex["gex_proxy_zscore"].clip(-3, 3)

        # Regime classification
        gex["gex_regime"] = np.where(
            gex["gex_proxy"] > 0.2, 2,  # Positive GEX (mean-reverting)
            np.where(gex["gex_proxy"] < -0.2, 0, 1)  # Negative GEX (trending) / Neutral
        )

        # Regime flip signal
        gex["gex_flip_signal"] = (gex["gex_regime"] != gex["gex_regime"].shift(1)).astype(float)

        # Magnitude
        gex["gex_magnitude"] = gex["gex_proxy"].abs()

        # 5-day change
        gex["gex_chg_5d"] = gex["gex_proxy"] - gex["gex_proxy"].shift(5)

        # --- Merge into spy_daily ---
        merge_cols = ["date", "gex_proxy", "gex_proxy_zscore", "gex_regime",
                      "gex_flip_signal", "gex_magnitude", "gex_chg_5d"]
        merge_data = gex[merge_cols].copy()

        features = features.merge(merge_data, on="date", how="left")

        # Fill NaN with neutral values
        gex_cols = [c for c in features.columns if c.startswith("gex_")]
        features[gex_cols] = features[gex_cols].fillna(0)

        print(f"  [GEX] Added {len(gex_cols)} gamma exposure proxy features")
        return features

    def analyze_current_gex(self, spy_daily: pd.DataFrame) -> Optional[dict]:
        """Analyze current GEX proxy for dashboard display."""
        if self.data.empty:
            return None

        gex_cols = [c for c in spy_daily.columns if c.startswith("gex_")]
        if not gex_cols or spy_daily.empty:
            return None

        latest = spy_daily.iloc[-1]
        regime_map = {0: "negative_gex", 1: "neutral", 2: "positive_gex"}

        return {
            "gex_proxy": float(latest.get("gex_proxy", 0)),
            "gex_zscore": float(latest.get("gex_proxy_zscore", 0)),
            "gex_regime": regime_map.get(int(latest.get("gex_regime", 1)), "neutral"),
            "market_behavior": "mean_reverting" if latest.get("gex_regime", 1) == 2 else
                              "trending" if latest.get("gex_regime", 1) == 0 else "mixed",
        }

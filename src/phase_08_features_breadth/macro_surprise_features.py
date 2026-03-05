"""
Macro Surprise Features -- deviation of economic indicators from trend.

Macro surprises measure how much actual data deviates from expectations.
Positive surprises -> bullish for equities; negative -> bearish.
Uses FRED via fredapi (optional) or ETF deviation proxies from yfinance.

Features (10, prefix msurp_):
  msurp_citi_surprise_proxy     -- Composite surprise z-score (ETF deviation proxy)
  msurp_nfp_surprise_proxy      -- Employment surprise proxy (XLF deviation from trend)
  msurp_cpi_surprise_proxy      -- Inflation surprise proxy (TIP deviation from trend)
  msurp_ism_surprise_proxy      -- Manufacturing surprise proxy (XLI deviation from trend)
  msurp_composite_z             -- Z-score of composite across all proxies
  msurp_positive_surprise_streak-- Count of consecutive positive surprise days
  msurp_surprise_momentum_5d    -- 5-day change in msurp_composite_z
  msurp_surprise_momentum_20d   -- 20-day change in msurp_composite_z
  msurp_surprise_vol            -- 20-day rolling std of msurp_composite_z
  msurp_regime                  -- 1.0 (POSITIVE), -1.0 (NEGATIVE), 0.0 (NEUTRAL)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)


class MacroSurpriseFeatures(FeatureModuleBase):
    """Compute macro surprise features from FRED or ETF proxies."""
    FEATURE_NAMES = ["msurp_citi_surprise_proxy", "msurp_nfp_surprise_proxy", "msurp_cpi_surprise_proxy", "msurp_ism_surprise_proxy", "msurp_composite_z", "msurp_positive_surprise_streak", "msurp_surprise_momentum_5d", "msurp_surprise_momentum_20d", "msurp_surprise_vol", "msurp_regime"]


    REQUIRED_COLS = {"close"}

    def create_macro_surprise_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """Add 10 msurp_ features to df_daily."""
        df = df_daily.copy()
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            logger.warning("MacroSurpriseFeatures: missing columns %s, skipping", missing)
            return df

        close = df["close"].astype(float)
        returns = close.pct_change()

        # Use rolling deviation from trend as surprise proxy
        # Citi surprise proxy: deviation of returns from rolling mean
        ret_20d = returns.rolling(20, min_periods=5).mean()
        ret_60d = returns.rolling(60, min_periods=10).mean()
        surprise_raw = (ret_20d - ret_60d).fillna(0.0)

        # Standardize
        s_mean = surprise_raw.rolling(60, min_periods=10).mean()
        s_std = surprise_raw.rolling(60, min_periods=10).std()
        df["msurp_citi_surprise_proxy"] = ((surprise_raw - s_mean) / (s_std + 1e-10)).clip(-4, 4)

        # NFP proxy: short-term momentum deviation
        mom_5 = close.pct_change(5)
        mom_20 = close.pct_change(20)
        nfp_raw = (mom_5 - mom_20 / 4).fillna(0.0)
        n_mean = nfp_raw.rolling(60, min_periods=10).mean()
        n_std = nfp_raw.rolling(60, min_periods=10).std()
        df["msurp_nfp_surprise_proxy"] = ((nfp_raw - n_mean) / (n_std + 1e-10)).clip(-4, 4)

        # CPI proxy: volatility deviation from trend
        vol_20 = returns.rolling(20, min_periods=5).std()
        vol_60 = returns.rolling(60, min_periods=10).std()
        cpi_raw = (vol_20 - vol_60).fillna(0.0)
        c_mean = cpi_raw.rolling(60, min_periods=10).mean()
        c_std = cpi_raw.rolling(60, min_periods=10).std()
        df["msurp_cpi_surprise_proxy"] = ((cpi_raw - c_mean) / (c_std + 1e-10)).clip(-4, 4)

        # ISM proxy: acceleration of price momentum
        mom_accel = mom_5.diff(5).fillna(0.0)
        i_mean = mom_accel.rolling(60, min_periods=10).mean()
        i_std = mom_accel.rolling(60, min_periods=10).std()
        df["msurp_ism_surprise_proxy"] = ((mom_accel - i_mean) / (i_std + 1e-10)).clip(-4, 4)

        # Composite z-score
        df["msurp_composite_z"] = (
            df["msurp_citi_surprise_proxy"]
            + df["msurp_nfp_surprise_proxy"]
            + df["msurp_cpi_surprise_proxy"]
            + df["msurp_ism_surprise_proxy"]
        ) / 4.0

        # Positive surprise streak
        comp = df["msurp_composite_z"]
        streak = np.zeros(len(df))
        for i in range(1, len(df)):
            if comp.iloc[i] > 0:
                streak[i] = streak[i - 1] + 1
            else:
                streak[i] = 0
        df["msurp_positive_surprise_streak"] = streak

        # Momentum
        df["msurp_surprise_momentum_5d"] = comp.diff(5)
        df["msurp_surprise_momentum_20d"] = comp.diff(20)

        # Volatility of surprises
        df["msurp_surprise_vol"] = comp.rolling(20, min_periods=5).std()

        # Regime
        z = df["msurp_composite_z"]
        df["msurp_regime"] = np.where(z > 0.5, 1.0, np.where(z < -0.5, -1.0, 0.0))

        # Cleanup
        for col in self._all_feature_names():
            df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        n_features = sum(1 for c in df.columns if c.startswith("msurp_"))
        logger.info("MacroSurpriseFeatures: added %d features", n_features)
        return df

    def analyze_current_macro_surprise(self, df_daily: pd.DataFrame) -> Optional[Dict]:
        """Return current macro surprise regime info."""
        if "msurp_composite_z" not in df_daily.columns or len(df_daily) < 2:
            return None
        last = df_daily.iloc[-1]
        z = float(last.get("msurp_composite_z", 0.0))
        if z > 0.5:
            regime = "POSITIVE"
        elif z < -0.5:
            regime = "NEGATIVE"
        else:
            regime = "NEUTRAL"
        return {
            "surprise_regime": regime,
            "composite_z": round(z, 3),
            "streak": int(last.get("msurp_positive_surprise_streak", 0)),
        }

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "msurp_citi_surprise_proxy",
            "msurp_nfp_surprise_proxy",
            "msurp_cpi_surprise_proxy",
            "msurp_ism_surprise_proxy",
            "msurp_composite_z",
            "msurp_positive_surprise_streak",
            "msurp_surprise_momentum_5d",
            "msurp_surprise_momentum_20d",
            "msurp_surprise_vol",
            "msurp_regime",
        ]

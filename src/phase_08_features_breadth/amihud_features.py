"""
Amihud Illiquidity Features — price impact per unit of volume.

Amihud (2002) illiquidity ratio: |return| / dollar_volume.
High illiquidity predicts higher volatility and larger price moves.

Features (4, prefix liq_):
  liq_amihud_raw     — Raw daily Amihud ratio
  liq_amihud_20d     — 20-day rolling mean of Amihud ratio
  liq_amihud_z       — 60-day z-score of Amihud ratio
  liq_amihud_regime  — Regime indicator (1=illiquid, -1=liquid, 0=normal)
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AmihudFeatures:
    """Compute Amihud illiquidity ratio features from daily OHLCV data."""

    REQUIRED_COLS = {"close"}

    def create_amihud_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Add Amihud illiquidity features to df_daily.

        Parameters
        ----------
        df_daily : pd.DataFrame
            Must have 'close' column. 'volume' column optional (features default to 0.0).

        Returns
        -------
        pd.DataFrame
            Original df_daily with 4 new liq_ columns added.
        """
        df = df_daily.copy()

        if "close" not in df.columns:
            logger.warning("AmihudFeatures: 'close' column missing, skipping")
            return df

        # Need volume for Amihud ratio
        has_volume = "volume" in df.columns and df["volume"].sum() > 0

        if not has_volume:
            logger.info("AmihudFeatures: no volume data, defaulting to 0.0")
            for col in ["liq_amihud_raw", "liq_amihud_20d", "liq_amihud_z", "liq_amihud_regime"]:
                df[col] = 0.0
            return df

        # --- Compute Amihud ratio ---
        returns = df["close"].pct_change().abs()
        dollar_volume = df["close"] * df["volume"]

        # Amihud = |return| / dollar_volume (scaled by 1e6 for readability)
        raw = (returns / (dollar_volume + 1e-10)) * 1e6
        raw = raw.clip(upper=raw.quantile(0.99))  # clip extreme outliers

        df["liq_amihud_raw"] = raw

        # 20-day rolling mean
        df["liq_amihud_20d"] = raw.rolling(20, min_periods=5).mean()

        # 60-day z-score
        rolling_mean = raw.rolling(60, min_periods=20).mean()
        rolling_std = raw.rolling(60, min_periods=20).std()
        df["liq_amihud_z"] = ((raw - rolling_mean) / (rolling_std + 1e-10)).clip(-4, 4)

        # Regime: illiquid (z > 1.5), liquid (z < -1.5), normal
        z = df["liq_amihud_z"]
        df["liq_amihud_regime"] = np.where(z > 1.5, 1.0, np.where(z < -1.5, -1.0, 0.0))

        # Cleanup
        for col in ["liq_amihud_raw", "liq_amihud_20d", "liq_amihud_z", "liq_amihud_regime"]:
            df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        n_features = sum(1 for c in df.columns if c.startswith("liq_"))
        logger.info(f"AmihudFeatures: added {n_features} features")
        return df

    def analyze_current_liquidity(self, df_daily: pd.DataFrame) -> Optional[Dict]:
        """Analyze current liquidity regime for dashboard display."""
        if "liq_amihud_z" not in df_daily.columns or len(df_daily) < 2:
            return None

        last = df_daily.iloc[-1]
        z = last.get("liq_amihud_z", 0.0)

        if z > 1.5:
            regime = "ILLIQUID"
        elif z < -1.5:
            regime = "LIQUID"
        else:
            regime = "NORMAL"

        return {
            "liquidity_regime": regime,
            "amihud_z": round(float(z), 3),
            "amihud_raw": round(float(last.get("liq_amihud_raw", 0.0)), 6),
        }

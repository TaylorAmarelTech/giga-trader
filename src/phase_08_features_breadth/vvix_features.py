"""
Wave M2: VVIX Features — Volatility of Volatility.

Downloads ^VVIX (VIX of VIX) from yfinance and derives:
  - VVIX level z-score
  - VVIX/VIX ratio (vol-of-vol premium)
  - VVIX momentum signals
  - VVIX-VIX divergence
  - VVIX regime classification

Prefix: vvix_
Default: ON
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from src.core.feature_base import FeatureModuleBase


class VVIXFeatures(FeatureModuleBase):
    """VVIX (VIX of VIX) feature engineering."""
    FEATURE_NAMES = ["vvix_level_z", "vvix_vix_ratio", "vvix_momentum_5d", "vvix_momentum_20d", "vvix_percentile", "vvix_vix_divergence", "vvix_acceleration", "vvix_regime"]


    def __init__(self):
        self._data: Optional[pd.DataFrame] = None
        self._data_source: str = "none"

    # ------------------------------------------------------------------
    def download_vvix_data(
        self, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Download ^VVIX and ^VIX from yfinance."""
        try:
            import yfinance as yf

            # Try VVIX first
            for ticker in ["^VVIX"]:
                try:
                    raw = yf.download(ticker, start=start, end=end, progress=False)
                    if raw is not None and len(raw) > 10:
                        df = pd.DataFrame(index=raw.index)
                        if isinstance(raw.columns, pd.MultiIndex):
                            df["vvix_close"] = raw[("Close", ticker)].values
                        else:
                            df["vvix_close"] = raw["Close"].values
                        self._data = df
                        self._data_source = "yfinance"
                        print(f"[VVIX] Downloaded {len(df)} days of ^VVIX data")
                        return df
                except Exception:
                    pass

            print("[VVIX] Could not download ^VVIX — using proxy")
            self._data_source = "proxy"
            return None

        except ImportError:
            print("[VVIX] yfinance not installed — using proxy")
            self._data_source = "proxy"
            return None
        except Exception as e:
            print(f"[VVIX] Download error: {e} — using proxy")
            self._data_source = "proxy"
            return None

    # ------------------------------------------------------------------
    def create_vvix_features(
        self, df_daily: pd.DataFrame
    ) -> pd.DataFrame:
        """Create 8 vvix_ features from VVIX data or proxy."""
        df = df_daily.copy()

        if self._data is not None and self._data_source == "yfinance":
            df = self._create_from_vvix(df)
        else:
            df = self._create_proxy(df)

        for feat in self._all_feature_names():
            if feat not in df.columns:
                df[feat] = 0.0
            df[feat] = (
                df[feat]
                .replace([np.inf, -np.inf], 0.0)
                .fillna(0.0)
                .astype(np.float64)
            )

        return df

    # ------------------------------------------------------------------
    def _create_from_vvix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from actual VVIX data."""
        vvix = self._data["vvix_close"].reindex(df.index, method="ffill").fillna(method="bfill")

        # 1. VVIX level z-score (60-day)
        mu = vvix.rolling(60, min_periods=20).mean()
        std = vvix.rolling(60, min_periods=20).std().replace(0, 1)
        df["vvix_level_z"] = ((vvix - mu) / std).clip(-4, 4)

        # 2. VVIX/VIX ratio — need VIX from df or download
        if "vix_close" in df.columns:
            vix_vals = df["vix_close"]
        elif "close" in df.columns:
            # Estimate VIX from realized vol
            ret = df["close"].pct_change().fillna(0.0)
            vix_vals = ret.rolling(20, min_periods=5).std() * np.sqrt(252) * 100
        else:
            vix_vals = pd.Series(20.0, index=df.index)
        vix_safe = vix_vals.replace(0, 20.0).fillna(20.0)
        df["vvix_vix_ratio"] = (vvix / vix_safe).fillna(1.0)

        # 3. VVIX 5-day momentum
        df["vvix_momentum_5d"] = vvix.pct_change(5).fillna(0.0)

        # 4. VVIX 20-day momentum
        df["vvix_momentum_20d"] = vvix.pct_change(20).fillna(0.0)

        # 5. VVIX percentile (252-day)
        df["vvix_percentile"] = vvix.rolling(252, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        ).fillna(0.5)

        # 6. VVIX-VIX divergence (z-score of ratio change)
        ratio = df["vvix_vix_ratio"]
        ratio_mu = ratio.rolling(60, min_periods=20).mean()
        ratio_std = ratio.rolling(60, min_periods=20).std().replace(0, 1)
        df["vvix_vix_divergence"] = ((ratio - ratio_mu) / ratio_std).clip(-4, 4)

        # 7. VVIX acceleration (momentum of momentum)
        mom5 = vvix.pct_change(5).fillna(0.0)
        df["vvix_acceleration"] = mom5.diff(5).fillna(0.0)

        # 8. VVIX regime (high/low/normal vol-of-vol)
        z = df["vvix_level_z"]
        df["vvix_regime"] = np.where(
            z > 1.0, 1.0,
            np.where(z < -1.0, -1.0, 0.0)
        )

        return df

    # ------------------------------------------------------------------
    def _create_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy VVIX from realized vol-of-vol when VVIX unavailable."""
        close = df["close"] if "close" in df.columns else pd.Series(100.0, index=df.index)
        ret = close.pct_change().fillna(0.0)

        # Realized vol (VIX proxy)
        rvol = ret.rolling(20, min_periods=5).std() * np.sqrt(252)
        # Vol-of-vol (VVIX proxy)
        vov = rvol.rolling(20, min_periods=5).std() * np.sqrt(252)

        mu = vov.rolling(60, min_periods=20).mean()
        std = vov.rolling(60, min_periods=20).std().replace(0, 1)
        df["vvix_level_z"] = ((vov - mu) / std).clip(-4, 4).fillna(0.0)

        rvol_safe = rvol.replace(0, 0.15).fillna(0.15)
        df["vvix_vix_ratio"] = (vov / rvol_safe).fillna(1.0)
        df["vvix_momentum_5d"] = vov.pct_change(5).fillna(0.0)
        df["vvix_momentum_20d"] = vov.pct_change(20).fillna(0.0)
        df["vvix_percentile"] = 0.5
        df["vvix_vix_divergence"] = 0.0
        df["vvix_acceleration"] = vov.pct_change(5).diff(5).fillna(0.0)
        z = df["vvix_level_z"]
        df["vvix_regime"] = np.where(z > 1.0, 1.0, np.where(z < -1.0, -1.0, 0.0))

        return df

    # ------------------------------------------------------------------
    def analyze_current_vvix(
        self, df_daily: pd.DataFrame
    ) -> Optional[Dict]:
        """Analyze current VVIX conditions."""
        if df_daily.empty:
            return None
        feats = self.create_vvix_features(df_daily)
        last = feats.iloc[-1]
        return {
            feat: float(last.get(feat, 0.0))
            for feat in self._all_feature_names()
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "vvix_level_z",
            "vvix_vix_ratio",
            "vvix_momentum_5d",
            "vvix_momentum_20d",
            "vvix_percentile",
            "vvix_vix_divergence",
            "vvix_acceleration",
            "vvix_regime",
        ]

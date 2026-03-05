"""
Wave M5: Money Market Features — Short-term funding rate signals.

Downloads FRED series for overnight and short-term rates:
  SOFR     - Secured Overnight Financing Rate (daily)
  OBFR     - Overnight Bank Funding Rate (daily)
  DPRIME   - Bank Prime Loan Rate (irregular)
  DTBSPCKM - 3-Month Nonfinancial Commercial Paper Rate (monthly)

Prefix: mmkt_
Default: ON
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from src.core.feature_base import FeatureModuleBase


class MoneyMarketFeatures(FeatureModuleBase):
    """Short-term funding market feature engineering."""
    FEATURE_NAMES = ["mmkt_sofr_z", "mmkt_sofr_change", "mmkt_sofr_obfr_spread", "mmkt_obfr_change", "mmkt_prime_change", "mmkt_funding_stress", "mmkt_rate_regime", "mmkt_cp_spread_z"]


    FRED_SERIES = {
        "SOFR": "sofr",
        "OBFR": "obfr",
        "DPRIME": "prime",
        "DTBSPCKM": "cp_rate",
    }

    def __init__(self):
        self._data: Optional[pd.DataFrame] = None
        self._data_source: str = "none"

    # ------------------------------------------------------------------
    def download_money_market_data(
        self, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Download money market rates from FRED."""
        try:
            from fredapi import Fred

            api_key = os.environ.get("FRED_API_KEY", "")
            if not api_key:
                print("[MMKT] No FRED_API_KEY — using proxy")
                self._data_source = "proxy"
                return None

            fred = Fred(api_key=api_key)
            frames: Dict[str, pd.Series] = {}
            for series_id, col_name in self.FRED_SERIES.items():
                try:
                    s = fred.get_series(series_id, start, end)
                    if s is not None and len(s) > 0:
                        frames[col_name] = s
                except Exception:
                    pass

            if not frames:
                self._data_source = "proxy"
                return None

            df = pd.DataFrame(frames)
            bdays = pd.date_range(start, end, freq="B")
            df = df.reindex(bdays).ffill().bfill()
            self._data = df
            self._data_source = "fred"
            print(f"[MMKT] Downloaded {len(frames)}/{len(self.FRED_SERIES)} money market rates")
            return df

        except ImportError:
            print("[MMKT] fredapi not installed — using proxy")
            self._data_source = "proxy"
            return None
        except Exception as e:
            print(f"[MMKT] Download error: {e} — using proxy")
            self._data_source = "proxy"
            return None

    # ------------------------------------------------------------------
    def create_money_market_features(
        self, df_daily: pd.DataFrame
    ) -> pd.DataFrame:
        """Create 8 mmkt_ features."""
        df = df_daily.copy()

        if self._data is not None and self._data_source == "fred":
            df = self._create_from_fred(df)
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
    def _create_from_fred(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from actual FRED money market data."""
        data = self._data.reindex(df.index, method="ffill").ffill().bfill()

        # 1. SOFR level z-score (60d)
        if "sofr" in data.columns:
            sofr = data["sofr"]
            mu = sofr.rolling(60, min_periods=20).mean()
            std = sofr.rolling(60, min_periods=20).std().replace(0, 1)
            df["mmkt_sofr_z"] = ((sofr - mu) / std).clip(-4, 4)
        else:
            df["mmkt_sofr_z"] = 0.0

        # 2. SOFR 5-day change
        if "sofr" in data.columns:
            df["mmkt_sofr_change"] = data["sofr"].diff(5).fillna(0.0) / 100
        else:
            df["mmkt_sofr_change"] = 0.0

        # 3. SOFR-OBFR spread (funding stress)
        if "sofr" in data.columns and "obfr" in data.columns:
            spread = data["sofr"] - data["obfr"]
            df["mmkt_sofr_obfr_spread"] = spread.fillna(0.0) / 100
        else:
            df["mmkt_sofr_obfr_spread"] = 0.0

        # 4. OBFR change
        if "obfr" in data.columns:
            df["mmkt_obfr_change"] = data["obfr"].diff(5).fillna(0.0) / 100
        else:
            df["mmkt_obfr_change"] = 0.0

        # 5. Prime rate change (signals Fed moves)
        if "prime" in data.columns:
            df["mmkt_prime_change"] = data["prime"].diff(21).fillna(0.0) / 100
        else:
            df["mmkt_prime_change"] = 0.0

        # 6. Funding stress indicator (|SOFR-OBFR| z-score)
        if "sofr" in data.columns and "obfr" in data.columns:
            spread = (data["sofr"] - data["obfr"]).abs()
            mu = spread.rolling(60, min_periods=20).mean()
            std = spread.rolling(60, min_periods=20).std().replace(0, 1)
            df["mmkt_funding_stress"] = ((spread - mu) / std).clip(-4, 4)
        else:
            df["mmkt_funding_stress"] = 0.0

        # 7. Rate regime (rising/falling/stable)
        if "sofr" in data.columns:
            sofr_chg = data["sofr"].diff(20).fillna(0.0)
            df["mmkt_rate_regime"] = np.where(
                sofr_chg > 0.1, 1.0,  # Rising
                np.where(sofr_chg < -0.1, -1.0, 0.0)  # Falling / Stable
            )
        else:
            df["mmkt_rate_regime"] = 0.0

        # 8. CP-rate spread (credit risk in short-term markets)
        if "cp_rate" in data.columns and "sofr" in data.columns:
            cp_spread = data["cp_rate"] - data["sofr"]
            mu = cp_spread.rolling(60, min_periods=20).mean()
            std = cp_spread.rolling(60, min_periods=20).std().replace(0, 1)
            df["mmkt_cp_spread_z"] = ((cp_spread - mu) / std).clip(-4, 4)
        elif "cp_rate" in data.columns:
            cp = data["cp_rate"]
            mu = cp.rolling(60, min_periods=20).mean()
            std = cp.rolling(60, min_periods=20).std().replace(0, 1)
            df["mmkt_cp_spread_z"] = ((cp - mu) / std).clip(-4, 4)
        else:
            df["mmkt_cp_spread_z"] = 0.0

        return df

    # ------------------------------------------------------------------
    def _create_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy money market features from SPY when FRED unavailable."""
        close = df["close"] if "close" in df.columns else pd.Series(100.0, index=df.index)
        ret = close.pct_change().fillna(0.0)

        # Use vol-based proxies for funding market stress
        vol = ret.rolling(20, min_periods=5).std()
        mu = vol.rolling(60, min_periods=20).mean()
        std = vol.rolling(60, min_periods=20).std().replace(0, 1)

        df["mmkt_sofr_z"] = 0.0
        df["mmkt_sofr_change"] = 0.0
        df["mmkt_sofr_obfr_spread"] = 0.0
        df["mmkt_obfr_change"] = 0.0
        df["mmkt_prime_change"] = 0.0
        df["mmkt_funding_stress"] = ((vol - mu) / std).clip(-4, 4).fillna(0.0)
        df["mmkt_rate_regime"] = 0.0
        df["mmkt_cp_spread_z"] = 0.0

        return df

    # ------------------------------------------------------------------
    def analyze_current_money_market(
        self, df_daily: pd.DataFrame
    ) -> Optional[Dict]:
        """Analyze current money market conditions."""
        if df_daily.empty:
            return None
        feats = self.create_money_market_features(df_daily)
        last = feats.iloc[-1]
        return {
            feat: float(last.get(feat, 0.0))
            for feat in self._all_feature_names()
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "mmkt_sofr_z",
            "mmkt_sofr_change",
            "mmkt_sofr_obfr_spread",
            "mmkt_obfr_change",
            "mmkt_prime_change",
            "mmkt_funding_stress",
            "mmkt_rate_regime",
            "mmkt_cp_spread_z",
        ]

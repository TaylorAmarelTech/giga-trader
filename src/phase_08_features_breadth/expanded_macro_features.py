"""
Wave M1: Expanded Macro Features — FRED economic indicators not yet covered.

Adds 8 monthly FRED series (forward-filled to daily):
  PCEPILFE  - Core PCE Price Index (Fed's preferred inflation gauge)
  TCU       - Capacity Utilization
  PAYEMS    - Total Nonfarm Payrolls
  JTSJOL    - JOLTS Job Openings
  AWHMAN    - Avg Weekly Hours, Manufacturing
  CSUSHPISA - Case-Shiller National Home Price Index
  HOUST1F   - Single-Family Housing Starts
  NEWORDER  - Manufacturers' New Orders: Durable Goods

Prefix: xmacro_
Default: ON
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class ExpandedMacroFeatures:
    """Expanded FRED macro indicators beyond core economic_features."""

    FRED_SERIES = {
        "PCEPILFE": "core_pce",
        "TCU": "capacity_util",
        "PAYEMS": "nonfarm_payrolls",
        "JTSJOL": "job_openings",
        "AWHMAN": "avg_hours_mfg",
        "CSUSHPISA": "case_shiller",
        "HOUST1F": "housing_starts_1f",
        "NEWORDER": "new_orders_durable",
    }

    def __init__(self):
        self._data: Optional[pd.DataFrame] = None
        self._data_source: str = "none"

    # ------------------------------------------------------------------
    def download_macro_data(
        self, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Download monthly FRED series, forward-fill to daily."""
        try:
            from fredapi import Fred

            api_key = os.environ.get("FRED_API_KEY", "")
            if not api_key:
                print("[XMACRO] No FRED_API_KEY — using proxy")
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
            print(f"[XMACRO] Downloaded {len(frames)}/{len(self.FRED_SERIES)} FRED series")
            return df

        except ImportError:
            print("[XMACRO] fredapi not installed — using proxy")
            self._data_source = "proxy"
            return None
        except Exception as e:
            print(f"[XMACRO] Download error: {e} — using proxy")
            self._data_source = "proxy"
            return None

    # ------------------------------------------------------------------
    def create_expanded_macro_features(
        self, df_daily: pd.DataFrame
    ) -> pd.DataFrame:
        """Create 8 xmacro_ features from FRED data or proxy."""
        df = df_daily.copy()
        n = len(df)

        if self._data is not None and self._data_source == "fred":
            df = self._create_from_fred(df)
        else:
            df = self._create_proxy(df)

        # Ensure all features exist
        for feat in self._all_feature_names():
            if feat not in df.columns:
                df[feat] = 0.0

        # Cleanup
        for feat in self._all_feature_names():
            df[feat] = (
                df[feat]
                .replace([np.inf, -np.inf], 0.0)
                .fillna(0.0)
                .astype(np.float64)
            )

        return df

    # ------------------------------------------------------------------
    def _create_from_fred(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from actual FRED data."""
        data = self._data
        idx = df.index

        # Align FRED data to daily index
        aligned = data.reindex(idx, method="ffill").fillna(method="bfill")

        # M1: Core PCE month-over-month change (annualized)
        if "core_pce" in aligned.columns:
            pce = aligned["core_pce"]
            df["xmacro_core_pce_chg"] = pce.pct_change(21).fillna(0.0)
        else:
            df["xmacro_core_pce_chg"] = 0.0

        # M2: Capacity utilization z-score
        if "capacity_util" in aligned.columns:
            tcu = aligned["capacity_util"]
            mu = tcu.rolling(252, min_periods=60).mean()
            std = tcu.rolling(252, min_periods=60).std().replace(0, 1)
            df["xmacro_capacity_util_z"] = ((tcu - mu) / std).clip(-4, 4)
        else:
            df["xmacro_capacity_util_z"] = 0.0

        # M3: Nonfarm payrolls month-over-month change
        if "nonfarm_payrolls" in aligned.columns:
            pay = aligned["nonfarm_payrolls"]
            df["xmacro_payrolls_chg"] = pay.pct_change(21).fillna(0.0)
        else:
            df["xmacro_payrolls_chg"] = 0.0

        # M4: Job openings change (labor market tightness)
        if "job_openings" in aligned.columns:
            jo = aligned["job_openings"]
            df["xmacro_job_openings_chg"] = jo.pct_change(21).fillna(0.0)
        else:
            df["xmacro_job_openings_chg"] = 0.0

        # M5: Avg weekly hours change (leading indicator)
        if "avg_hours_mfg" in aligned.columns:
            awh = aligned["avg_hours_mfg"]
            mu = awh.rolling(252, min_periods=60).mean()
            std = awh.rolling(252, min_periods=60).std().replace(0, 1)
            df["xmacro_avg_hours_z"] = ((awh - mu) / std).clip(-4, 4)
        else:
            df["xmacro_avg_hours_z"] = 0.0

        # M6: Case-Shiller home price momentum
        if "case_shiller" in aligned.columns:
            cs = aligned["case_shiller"]
            df["xmacro_home_price_mom"] = cs.pct_change(63).fillna(0.0)
        else:
            df["xmacro_home_price_mom"] = 0.0

        # M7: Housing starts change
        if "housing_starts_1f" in aligned.columns:
            hs = aligned["housing_starts_1f"]
            df["xmacro_housing_starts_chg"] = hs.pct_change(21).fillna(0.0)
        else:
            df["xmacro_housing_starts_chg"] = 0.0

        # M8: New orders momentum (manufacturing demand)
        if "new_orders_durable" in aligned.columns:
            no = aligned["new_orders_durable"]
            df["xmacro_new_orders_mom"] = no.pct_change(21).fillna(0.0)
        else:
            df["xmacro_new_orders_mom"] = 0.0

        return df

    # ------------------------------------------------------------------
    def _create_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy features from SPY price/volume when FRED unavailable."""
        close = df["close"] if "close" in df.columns else pd.Series(0.0, index=df.index)
        volume = df["volume"] if "volume" in df.columns else pd.Series(0.0, index=df.index)
        ret = close.pct_change().fillna(0.0)

        # Proxy: use rolling price/volume statistics as macro proxies
        df["xmacro_core_pce_chg"] = ret.rolling(21, min_periods=1).mean()
        vol_20 = ret.rolling(20, min_periods=1).std()
        mu_vol = vol_20.rolling(252, min_periods=60).mean()
        std_vol = vol_20.rolling(252, min_periods=60).std().replace(0, 1)
        df["xmacro_capacity_util_z"] = ((vol_20 - mu_vol) / std_vol).clip(-4, 4)
        df["xmacro_payrolls_chg"] = ret.rolling(21, min_periods=1).sum()
        vol_chg = volume.pct_change(21).fillna(0.0)
        df["xmacro_job_openings_chg"] = vol_chg * 0.1
        df["xmacro_avg_hours_z"] = ret.rolling(63, min_periods=1).mean() * 10
        df["xmacro_home_price_mom"] = ret.rolling(63, min_periods=1).sum()
        df["xmacro_housing_starts_chg"] = ret.rolling(42, min_periods=1).sum()
        df["xmacro_new_orders_mom"] = ret.rolling(21, min_periods=1).sum() * 2

        return df

    # ------------------------------------------------------------------
    def analyze_current_macro(
        self, df_daily: pd.DataFrame
    ) -> Optional[Dict]:
        """Analyze current macro conditions."""
        if df_daily.empty:
            return None
        feats = self.create_expanded_macro_features(df_daily)
        last = feats.iloc[-1]
        return {
            feat: float(last.get(feat, 0.0))
            for feat in self._all_feature_names()
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "xmacro_core_pce_chg",
            "xmacro_capacity_util_z",
            "xmacro_payrolls_chg",
            "xmacro_job_openings_chg",
            "xmacro_avg_hours_z",
            "xmacro_home_price_mom",
            "xmacro_housing_starts_chg",
            "xmacro_new_orders_mom",
        ]

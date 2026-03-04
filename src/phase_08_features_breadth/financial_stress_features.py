"""
Wave M6: Financial Stress Features — Stress/conditions indices.

Downloads FRED financial stress indicators:
  STLFSI4      - St. Louis Fed Financial Stress Index (weekly)
  NFCI         - Chicago Fed National Financial Conditions Index (weekly)
  DTWEXBGS     - Trade Weighted U.S. Dollar Index: Broad, Goods (daily)
  BAMLC0A4CBBB - ICE BofA BBB U.S. Corporate OAS (daily)

Prefix: fstress_
Default: ON
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class FinancialStressFeatures:
    """Financial stress and conditions index features."""

    FRED_SERIES = {
        "STLFSI4": "stlfsi",
        "NFCI": "nfci",
        "DTWEXBGS": "dollar_tw",
        "BAMLC0A4CBBB": "bbb_oas",
    }

    # Fallback series if primary unavailable
    FALLBACK_SERIES = {
        "STLFSI4": ["STLFSI2", "STLFSI"],
        "NFCI": [],
        "DTWEXBGS": ["DTWEXAFEGS"],
        "BAMLC0A4CBBB": ["BAMLH0A0HYM2"],
    }

    def __init__(self):
        self._data: Optional[pd.DataFrame] = None
        self._data_source: str = "none"

    # ------------------------------------------------------------------
    def download_stress_data(
        self, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Download financial stress indices from FRED."""
        try:
            from fredapi import Fred

            api_key = os.environ.get("FRED_API_KEY", "")
            if not api_key:
                print("[FSTRESS] No FRED_API_KEY — using proxy")
                self._data_source = "proxy"
                return None

            fred = Fred(api_key=api_key)
            frames: Dict[str, pd.Series] = {}

            for series_id, col_name in self.FRED_SERIES.items():
                downloaded = False
                # Try primary, then fallbacks
                candidates = [series_id] + self.FALLBACK_SERIES.get(series_id, [])
                for candidate in candidates:
                    try:
                        s = fred.get_series(candidate, start, end)
                        if s is not None and len(s) > 0:
                            frames[col_name] = s
                            downloaded = True
                            break
                    except Exception:
                        pass
                if not downloaded:
                    pass  # Will use proxy for this series

            if not frames:
                self._data_source = "proxy"
                return None

            df = pd.DataFrame(frames)
            bdays = pd.date_range(start, end, freq="B")
            df = df.reindex(bdays).ffill().bfill()
            self._data = df
            self._data_source = "fred"
            print(f"[FSTRESS] Downloaded {len(frames)}/{len(self.FRED_SERIES)} stress indices")
            return df

        except ImportError:
            print("[FSTRESS] fredapi not installed — using proxy")
            self._data_source = "proxy"
            return None
        except Exception as e:
            print(f"[FSTRESS] Download error: {e} — using proxy")
            self._data_source = "proxy"
            return None

    # ------------------------------------------------------------------
    def create_financial_stress_features(
        self, df_daily: pd.DataFrame
    ) -> pd.DataFrame:
        """Create 8 fstress_ features."""
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
        """Create features from actual FRED stress data."""
        data = self._data.reindex(df.index, method="ffill").ffill().bfill()

        # 1. STLFSI level (already a z-score-like index, >0 = stress)
        if "stlfsi" in data.columns:
            df["fstress_stlfsi_level"] = data["stlfsi"]
        else:
            df["fstress_stlfsi_level"] = 0.0

        # 2. STLFSI momentum (5-week change)
        if "stlfsi" in data.columns:
            df["fstress_stlfsi_momentum"] = data["stlfsi"].diff(5).fillna(0.0)
        else:
            df["fstress_stlfsi_momentum"] = 0.0

        # 3. NFCI level (>0 = tighter conditions)
        if "nfci" in data.columns:
            df["fstress_nfci_level"] = data["nfci"]
        else:
            df["fstress_nfci_level"] = 0.0

        # 4. NFCI momentum
        if "nfci" in data.columns:
            df["fstress_nfci_momentum"] = data["nfci"].diff(5).fillna(0.0)
        else:
            df["fstress_nfci_momentum"] = 0.0

        # 5. STLFSI-NFCI divergence
        if "stlfsi" in data.columns and "nfci" in data.columns:
            df["fstress_index_divergence"] = data["stlfsi"] - data["nfci"]
        else:
            df["fstress_index_divergence"] = 0.0

        # 6. Trade-weighted dollar change (20d)
        if "dollar_tw" in data.columns:
            df["fstress_dollar_tw_chg"] = data["dollar_tw"].pct_change(20).fillna(0.0)
        else:
            df["fstress_dollar_tw_chg"] = 0.0

        # 7. Stress composite (average of STLFSI + NFCI + BBB OAS z-score)
        components = []
        if "stlfsi" in data.columns:
            components.append(data["stlfsi"])
        if "nfci" in data.columns:
            components.append(data["nfci"])
        if "bbb_oas" in data.columns:
            bbb = data["bbb_oas"]
            mu = bbb.rolling(252, min_periods=60).mean()
            std = bbb.rolling(252, min_periods=60).std().replace(0, 1)
            components.append(((bbb - mu) / std).clip(-4, 4))

        if components:
            composite = pd.concat(components, axis=1).mean(axis=1)
            df["fstress_composite"] = composite
        else:
            df["fstress_composite"] = 0.0

        # 8. Stress regime
        comp = df["fstress_composite"]
        df["fstress_regime"] = np.where(
            comp > 1.0, 1.0,  # High stress
            np.where(comp < -0.5, -1.0, 0.0)  # Low stress / Normal
        )

        return df

    # ------------------------------------------------------------------
    def _create_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy stress features from SPY when FRED unavailable."""
        close = df["close"] if "close" in df.columns else pd.Series(100.0, index=df.index)
        ret = close.pct_change().fillna(0.0)

        # Use VIX-like proxy from realized vol
        vol = ret.rolling(20, min_periods=5).std() * np.sqrt(252)
        mu = vol.rolling(252, min_periods=60).mean()
        std = vol.rolling(252, min_periods=60).std().replace(0, 1)
        vol_z = ((vol - mu) / std).clip(-4, 4).fillna(0.0)

        df["fstress_stlfsi_level"] = vol_z * 0.5
        df["fstress_stlfsi_momentum"] = vol_z.diff(5).fillna(0.0) * 0.3
        df["fstress_nfci_level"] = vol_z * 0.3
        df["fstress_nfci_momentum"] = vol_z.diff(5).fillna(0.0) * 0.2
        df["fstress_index_divergence"] = 0.0
        df["fstress_dollar_tw_chg"] = 0.0
        df["fstress_composite"] = vol_z * 0.4
        df["fstress_regime"] = np.where(
            vol_z > 1.5, 1.0, np.where(vol_z < -0.5, -1.0, 0.0)
        )

        return df

    # ------------------------------------------------------------------
    def analyze_current_stress(
        self, df_daily: pd.DataFrame
    ) -> Optional[Dict]:
        """Analyze current financial stress conditions."""
        if df_daily.empty:
            return None
        feats = self.create_financial_stress_features(df_daily)
        last = feats.iloc[-1]
        return {
            feat: float(last.get(feat, 0.0))
            for feat in self._all_feature_names()
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "fstress_stlfsi_level",
            "fstress_stlfsi_momentum",
            "fstress_nfci_level",
            "fstress_nfci_momentum",
            "fstress_index_divergence",
            "fstress_dollar_tw_chg",
            "fstress_composite",
            "fstress_regime",
        ]

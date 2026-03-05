"""
Wave M8: Retail Sentiment Features — Retail flow proxy signals.

Uses leveraged/inverse ETF flows as retail sentiment proxies:
  TQQQ  - ProShares UltraPro QQQ (3x leveraged NASDAQ)
  SQQQ  - ProShares UltraPro Short QQQ (3x inverse NASDAQ)
  ARKK  - ARK Innovation ETF (retail favorite)
  UVXY  - ProShares Ultra VIX Short-Term (2x VIX)

Prefix: rflow_
Default: ON
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from src.core.feature_base import FeatureModuleBase


RETAIL_ETFS = {
    "TQQQ": "tqqq",
    "SQQQ": "sqqq",
    "ARKK": "arkk",
    "UVXY": "uvxy",
}


class RetailSentimentFeatures(FeatureModuleBase):
    """Retail sentiment proxy features from leveraged ETF flows."""
    FEATURE_NAMES = ["rflow_bull_bear_ratio", "rflow_arkk_momentum", "rflow_uvxy_z", "rflow_leveraged_dispersion", "rflow_panic_proxy", "rflow_euphoria_proxy", "rflow_retail_inst_divergence", "rflow_regime"]


    def __init__(self):
        self._data: Optional[pd.DataFrame] = None
        self._vol_data: Optional[pd.DataFrame] = None
        self._data_source: str = "none"

    # ------------------------------------------------------------------
    def download_retail_data(
        self, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Download retail-favored ETFs from yfinance (close + volume)."""
        try:
            import yfinance as yf

            tickers = " ".join(RETAIL_ETFS.keys())
            raw = yf.download(tickers, start=start, end=end, progress=False)
            if raw is None or len(raw) < 10:
                self._data_source = "proxy"
                return None

            close_df = pd.DataFrame(index=raw.index)
            vol_df = pd.DataFrame(index=raw.index)

            for ticker, col in RETAIL_ETFS.items():
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        close_df[col] = raw[("Close", ticker)].values
                        vol_df[col] = raw[("Volume", ticker)].values
                    else:
                        close_df[col] = raw["Close"].values
                        vol_df[col] = raw["Volume"].values
                except Exception:
                    pass

            if close_df.empty or len(close_df.columns) < 2:
                self._data_source = "proxy"
                return None

            self._data = close_df
            self._vol_data = vol_df
            self._data_source = "yfinance"
            print(f"[RFLOW] Downloaded {len(close_df.columns)}/{len(RETAIL_ETFS)} retail ETFs, {len(close_df)} days")
            return close_df

        except ImportError:
            self._data_source = "proxy"
            return None
        except Exception as e:
            print(f"[RFLOW] Download error: {e} — using proxy")
            self._data_source = "proxy"
            return None

    # ------------------------------------------------------------------
    def create_retail_sentiment_features(
        self, df_daily: pd.DataFrame
    ) -> pd.DataFrame:
        """Create 8 rflow_ features."""
        df = df_daily.copy()

        if self._data is not None and self._data_source == "yfinance":
            df = self._create_from_retail(df)
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
    def _create_from_retail(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from actual retail ETF data."""
        close = self._data.reindex(df.index, method="ffill").ffill().bfill()
        vol = self._vol_data.reindex(df.index, method="ffill").ffill().bfill() if self._vol_data is not None else None

        # 1. TQQQ/SQQQ volume ratio — retail bull/bear positioning
        if vol is not None and "tqqq" in vol.columns and "sqqq" in vol.columns:
            tqqq_v = vol["tqqq"].replace(0, 1)
            sqqq_v = vol["sqqq"].replace(0, 1)
            ratio = tqqq_v / sqqq_v
            mu = ratio.rolling(20, min_periods=5).mean()
            std = ratio.rolling(20, min_periods=5).std().replace(0, 1)
            df["rflow_bull_bear_ratio"] = ((ratio - mu) / std).clip(-4, 4)
        else:
            df["rflow_bull_bear_ratio"] = 0.0

        # 2. ARKK momentum (innovation sentiment)
        if "arkk" in close.columns:
            df["rflow_arkk_momentum"] = close["arkk"].pct_change(20).fillna(0.0)
        else:
            df["rflow_arkk_momentum"] = 0.0

        # 3. UVXY level z-score (fear proxy)
        if "uvxy" in close.columns:
            uvxy = close["uvxy"]
            mu = uvxy.rolling(60, min_periods=20).mean()
            std = uvxy.rolling(60, min_periods=20).std().replace(0, 1)
            df["rflow_uvxy_z"] = ((uvxy - mu) / std).clip(-4, 4)
        else:
            df["rflow_uvxy_z"] = 0.0

        # 4. Leveraged fund dispersion (|TQQQ ret| + |SQQQ ret| cross-section vol)
        if "tqqq" in close.columns and "sqqq" in close.columns:
            tqqq_ret = close["tqqq"].pct_change().fillna(0.0)
            sqqq_ret = close["sqqq"].pct_change().fillna(0.0)
            dispersion = (tqqq_ret.abs() + sqqq_ret.abs()).rolling(10, min_periods=3).mean()
            df["rflow_leveraged_dispersion"] = dispersion
        else:
            df["rflow_leveraged_dispersion"] = 0.0

        # 5. Retail panic proxy (high SQQQ volume + high UVXY volume)
        if vol is not None:
            panic_components = []
            if "sqqq" in vol.columns:
                sqqq_v = vol["sqqq"]
                sqqq_z = (sqqq_v - sqqq_v.rolling(20, min_periods=5).mean()) / sqqq_v.rolling(20, min_periods=5).std().replace(0, 1)
                panic_components.append(sqqq_z.clip(-4, 4))
            if "uvxy" in vol.columns:
                uvxy_v = vol["uvxy"]
                uvxy_z = (uvxy_v - uvxy_v.rolling(20, min_periods=5).mean()) / uvxy_v.rolling(20, min_periods=5).std().replace(0, 1)
                panic_components.append(uvxy_z.clip(-4, 4))
            if panic_components:
                df["rflow_panic_proxy"] = pd.concat(panic_components, axis=1).mean(axis=1).fillna(0.0)
            else:
                df["rflow_panic_proxy"] = 0.0
        else:
            df["rflow_panic_proxy"] = 0.0

        # 6. Retail euphoria proxy (high TQQQ volume + high ARKK volume)
        if vol is not None:
            euphoria_components = []
            if "tqqq" in vol.columns:
                tqqq_v = vol["tqqq"]
                tqqq_z = (tqqq_v - tqqq_v.rolling(20, min_periods=5).mean()) / tqqq_v.rolling(20, min_periods=5).std().replace(0, 1)
                euphoria_components.append(tqqq_z.clip(-4, 4))
            if "arkk" in vol.columns:
                arkk_v = vol["arkk"]
                arkk_z = (arkk_v - arkk_v.rolling(20, min_periods=5).mean()) / arkk_v.rolling(20, min_periods=5).std().replace(0, 1)
                euphoria_components.append(arkk_z.clip(-4, 4))
            if euphoria_components:
                df["rflow_euphoria_proxy"] = pd.concat(euphoria_components, axis=1).mean(axis=1).fillna(0.0)
            else:
                df["rflow_euphoria_proxy"] = 0.0
        else:
            df["rflow_euphoria_proxy"] = 0.0

        # 7. Retail-institutional divergence (ARKK vs SPY relative performance)
        spy_ret = df["close"].pct_change(20).fillna(0.0) if "close" in df.columns else pd.Series(0.0, index=df.index)
        if "arkk" in close.columns:
            arkk_ret = close["arkk"].pct_change(20).fillna(0.0)
            df["rflow_retail_inst_divergence"] = arkk_ret - spy_ret
        else:
            df["rflow_retail_inst_divergence"] = 0.0

        # 8. Retail sentiment regime
        panic = df["rflow_panic_proxy"]
        euphoria = df["rflow_euphoria_proxy"]
        net = euphoria - panic
        df["rflow_regime"] = np.where(
            net > 1.0, 1.0,    # Retail euphoria
            np.where(net < -1.0, -1.0, 0.0)  # Retail panic
        )

        return df

    # ------------------------------------------------------------------
    def _create_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy retail sentiment from SPY volume/price when ETFs unavailable."""
        close = df["close"] if "close" in df.columns else pd.Series(100.0, index=df.index)
        volume = df["volume"] if "volume" in df.columns else pd.Series(1e6, index=df.index)
        ret = close.pct_change().fillna(0.0)

        vol_z = (volume - volume.rolling(20, min_periods=5).mean()) / volume.rolling(20, min_periods=5).std().replace(0, 1)
        vol_z = vol_z.clip(-4, 4).fillna(0.0)

        df["rflow_bull_bear_ratio"] = 0.0
        df["rflow_arkk_momentum"] = ret.rolling(20, min_periods=5).sum()
        df["rflow_uvxy_z"] = 0.0
        df["rflow_leveraged_dispersion"] = ret.abs().rolling(10, min_periods=3).mean()
        down_vol = vol_z * (ret < 0).astype(float)
        df["rflow_panic_proxy"] = down_vol.rolling(5, min_periods=1).mean()
        up_vol = vol_z * (ret > 0).astype(float)
        df["rflow_euphoria_proxy"] = up_vol.rolling(5, min_periods=1).mean()
        df["rflow_retail_inst_divergence"] = 0.0
        df["rflow_regime"] = 0.0

        return df

    # ------------------------------------------------------------------
    def analyze_current_retail(
        self, df_daily: pd.DataFrame
    ) -> Optional[Dict]:
        """Analyze current retail sentiment conditions."""
        if df_daily.empty:
            return None
        feats = self.create_retail_sentiment_features(df_daily)
        last = feats.iloc[-1]
        return {
            feat: float(last.get(feat, 0.0))
            for feat in self._all_feature_names()
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "rflow_bull_bear_ratio",
            "rflow_arkk_momentum",
            "rflow_uvxy_z",
            "rflow_leveraged_dispersion",
            "rflow_panic_proxy",
            "rflow_euphoria_proxy",
            "rflow_retail_inst_divergence",
            "rflow_regime",
        ]

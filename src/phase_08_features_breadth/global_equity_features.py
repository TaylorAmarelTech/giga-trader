"""
Wave M7: Global Equity Features — International equity ETF signals.

Downloads major international equity ETFs from yfinance:
  EFA  - iShares MSCI EAFE (Developed ex-US)
  VWO  - Vanguard FTSE Emerging Markets
  FXI  - iShares China Large-Cap
  EWJ  - iShares MSCI Japan
  EWZ  - iShares MSCI Brazil

Prefix: gleq_
Default: ON
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


GLOBAL_ETFS = {
    "EFA": "efa",   # Developed markets ex-US
    "VWO": "vwo",   # Emerging markets
    "FXI": "fxi",   # China
    "EWJ": "ewj",   # Japan
    "EWZ": "ewz",   # Brazil
}


class GlobalEquityFeatures:
    """International equity ETF signal features."""

    def __init__(self):
        self._data: Optional[pd.DataFrame] = None
        self._data_source: str = "none"

    # ------------------------------------------------------------------
    def download_global_data(
        self, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Download international equity ETFs from yfinance."""
        try:
            import yfinance as yf

            tickers = " ".join(GLOBAL_ETFS.keys())
            raw = yf.download(tickers, start=start, end=end, progress=False)
            if raw is None or len(raw) < 10:
                self._data_source = "proxy"
                return None

            df = pd.DataFrame(index=raw.index)
            for ticker, col in GLOBAL_ETFS.items():
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        df[col] = raw[("Close", ticker)].values
                    else:
                        df[col] = raw["Close"].values
                except Exception:
                    pass

            if df.empty or len(df.columns) < 2:
                self._data_source = "proxy"
                return None

            self._data = df
            self._data_source = "yfinance"
            print(f"[GLEQ] Downloaded {len(df.columns)}/{len(GLOBAL_ETFS)} global ETFs, {len(df)} days")
            return df

        except ImportError:
            self._data_source = "proxy"
            return None
        except Exception as e:
            print(f"[GLEQ] Download error: {e} — using proxy")
            self._data_source = "proxy"
            return None

    # ------------------------------------------------------------------
    def create_global_equity_features(
        self, df_daily: pd.DataFrame
    ) -> pd.DataFrame:
        """Create 8 gleq_ features."""
        df = df_daily.copy()

        if self._data is not None and self._data_source == "yfinance":
            df = self._create_from_global(df)
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
    def _create_from_global(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from actual global ETF data."""
        glob = self._data.reindex(df.index, method="ffill").ffill().bfill()
        spy_ret = df["close"].pct_change().fillna(0.0) if "close" in df.columns else pd.Series(0.0, index=df.index)

        # 20-day returns for each
        rets_20d = glob.pct_change(20).fillna(0.0)
        rets_1d = glob.pct_change().fillna(0.0)
        available = list(glob.columns)

        # 1. EFA-SPY spread (developed markets vs US divergence)
        if "efa" in rets_20d.columns:
            spy_20d = spy_ret.rolling(20, min_periods=5).sum()
            df["gleq_dm_us_spread"] = rets_20d["efa"] - spy_20d
        else:
            df["gleq_dm_us_spread"] = 0.0

        # 2. EM-DM spread (emerging vs developed risk appetite)
        if "vwo" in rets_20d.columns and "efa" in rets_20d.columns:
            df["gleq_em_dm_spread"] = rets_20d["vwo"] - rets_20d["efa"]
        else:
            df["gleq_em_dm_spread"] = 0.0

        # 3. China momentum (FXI)
        if "fxi" in rets_20d.columns:
            df["gleq_china_momentum"] = rets_20d["fxi"]
        else:
            df["gleq_china_momentum"] = 0.0

        # 4. Japan momentum (EWJ)
        if "ewj" in rets_20d.columns:
            df["gleq_japan_momentum"] = rets_20d["ewj"]
        else:
            df["gleq_japan_momentum"] = 0.0

        # 5. Global breadth (% of global ETFs with positive 20d return)
        if len(available) >= 2:
            pos = (rets_20d[available] > 0).sum(axis=1)
            df["gleq_global_breadth"] = pos / len(available)
        else:
            df["gleq_global_breadth"] = 0.5

        # 6. Global-US divergence (average global 20d vs SPY 20d)
        if len(available) >= 2:
            global_avg = rets_20d[available].mean(axis=1)
            spy_20d = spy_ret.rolling(20, min_periods=5).sum()
            df["gleq_global_us_divergence"] = global_avg - spy_20d
        else:
            df["gleq_global_us_divergence"] = 0.0

        # 7. Contagion risk (average correlation of global ETFs with SPY)
        if len(available) >= 2:
            corrs = []
            for col in available:
                if col in rets_1d.columns:
                    c = spy_ret.rolling(60, min_periods=20).corr(rets_1d[col])
                    corrs.append(c)
            if corrs:
                avg_corr = pd.concat(corrs, axis=1).mean(axis=1)
                df["gleq_contagion_risk"] = avg_corr.fillna(0.5)
            else:
                df["gleq_contagion_risk"] = 0.5
        else:
            df["gleq_contagion_risk"] = 0.5

        # 8. Global regime (risk-on / risk-off / neutral)
        breadth = df["gleq_global_breadth"]
        df["gleq_regime"] = np.where(
            breadth > 0.7, 1.0,  # Global risk-on
            np.where(breadth < 0.3, -1.0, 0.0)  # Global risk-off
        )

        return df

    # ------------------------------------------------------------------
    def _create_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy global features from SPY when international data unavailable."""
        close = df["close"] if "close" in df.columns else pd.Series(100.0, index=df.index)
        ret = close.pct_change().fillna(0.0)

        df["gleq_dm_us_spread"] = 0.0
        df["gleq_em_dm_spread"] = 0.0
        df["gleq_china_momentum"] = ret.rolling(20, min_periods=5).sum() * 0.8
        df["gleq_japan_momentum"] = ret.rolling(20, min_periods=5).sum() * 0.9
        df["gleq_global_breadth"] = 0.5
        df["gleq_global_us_divergence"] = 0.0
        df["gleq_contagion_risk"] = 0.5
        df["gleq_regime"] = 0.0

        return df

    # ------------------------------------------------------------------
    def analyze_current_global(
        self, df_daily: pd.DataFrame
    ) -> Optional[Dict]:
        """Analyze current global equity conditions."""
        if df_daily.empty:
            return None
        feats = self.create_global_equity_features(df_daily)
        last = feats.iloc[-1]
        return {
            feat: float(last.get(feat, 0.0))
            for feat in self._all_feature_names()
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "gleq_dm_us_spread",
            "gleq_em_dm_spread",
            "gleq_china_momentum",
            "gleq_japan_momentum",
            "gleq_global_breadth",
            "gleq_global_us_divergence",
            "gleq_contagion_risk",
            "gleq_regime",
        ]

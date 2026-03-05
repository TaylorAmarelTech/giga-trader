"""
Wave M3: Sector Rotation Features — momentum-rank rotation signals.

Uses the 11 SPDR sector ETFs already downloaded by sector_breadth to derive
rotation-specific signals (momentum ranks, leader/laggard spreads, rotation
speed) that go beyond basic breadth metrics.

Prefix: secrot_
Default: ON
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from src.core.feature_base import FeatureModuleBase


SECTOR_ETFS = [
    "XLK", "XLF", "XLE", "XLV", "XLI",
    "XLC", "XLRE", "XLU", "XLP", "XLY", "XLB",
]

CYCLICAL = {"XLK", "XLF", "XLE", "XLI", "XLY", "XLB", "XLC"}
DEFENSIVE = {"XLV", "XLU", "XLP", "XLRE"}


class SectorRotationFeatures(FeatureModuleBase):
    """Sector ETF rotation & momentum-rank signals."""
    FEATURE_NAMES = ["secrot_rotation_speed", "secrot_leader_momentum", "secrot_laggard_momentum", "secrot_leader_laggard_spread", "secrot_momentum_dispersion", "secrot_top3_concentration", "secrot_cyclical_vs_defensive", "secrot_regime"]


    def __init__(self):
        self._data: Optional[pd.DataFrame] = None
        self._data_source: str = "none"

    # ------------------------------------------------------------------
    def download_sector_data(
        self, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Download 11 SPDR sector ETFs from yfinance."""
        try:
            import yfinance as yf

            tickers = " ".join(SECTOR_ETFS)
            raw = yf.download(tickers, start=start, end=end, progress=False)
            if raw is None or len(raw) < 10:
                self._data_source = "proxy"
                return None

            df = pd.DataFrame(index=raw.index)
            for t in SECTOR_ETFS:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        df[t] = raw[("Close", t)].values
                    else:
                        df[t] = raw["Close"].values
                except Exception:
                    pass

            if df.empty or len(df.columns) < 3:
                self._data_source = "proxy"
                return None

            self._data = df
            self._data_source = "yfinance"
            print(f"[SECROT] Downloaded {len(df.columns)} sector ETFs, {len(df)} days")
            return df

        except ImportError:
            self._data_source = "proxy"
            return None
        except Exception as e:
            print(f"[SECROT] Download error: {e} — using proxy")
            self._data_source = "proxy"
            return None

    # ------------------------------------------------------------------
    def create_sector_rotation_features(
        self, df_daily: pd.DataFrame
    ) -> pd.DataFrame:
        """Create 8 secrot_ features."""
        df = df_daily.copy()

        if self._data is not None and self._data_source == "yfinance":
            df = self._create_from_sectors(df)
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
    def _create_from_sectors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from actual sector ETF data."""
        sector_data = self._data.reindex(df.index, method="ffill").ffill().bfill()

        # 20-day returns for each sector
        returns_20d = sector_data.pct_change(20).fillna(0.0)
        available = [c for c in SECTOR_ETFS if c in returns_20d.columns]

        if len(available) < 3:
            return self._create_proxy(df)

        ret_df = returns_20d[available]

        # 1. Rotation speed: average rank change over 5 days
        ranks = ret_df.rank(axis=1, ascending=False)
        rank_chg = ranks.diff(5).abs()
        df["secrot_rotation_speed"] = rank_chg.mean(axis=1).fillna(0.0)

        # 2. Leader momentum (top 3 average 20d return)
        top3 = ret_df.apply(lambda row: row.nlargest(3).mean(), axis=1)
        df["secrot_leader_momentum"] = top3

        # 3. Laggard momentum (bottom 3 average 20d return)
        bot3 = ret_df.apply(lambda row: row.nsmallest(3).mean(), axis=1)
        df["secrot_laggard_momentum"] = bot3

        # 4. Leader-laggard spread
        df["secrot_leader_laggard_spread"] = top3 - bot3

        # 5. Sector momentum dispersion (cross-sectional std)
        df["secrot_momentum_dispersion"] = ret_df.std(axis=1).fillna(0.0)

        # 6. Top-3 concentration (what fraction of total positive momentum)
        total_abs = ret_df.abs().sum(axis=1).replace(0, 1)
        df["secrot_top3_concentration"] = top3.abs() * 3 / total_abs

        # 7. Cyclical vs defensive rotation
        cyc_cols = [c for c in available if c in CYCLICAL]
        def_cols = [c for c in available if c in DEFENSIVE]
        if cyc_cols and def_cols:
            cyc_avg = ret_df[cyc_cols].mean(axis=1)
            def_avg = ret_df[def_cols].mean(axis=1)
            df["secrot_cyclical_vs_defensive"] = cyc_avg - def_avg
        else:
            df["secrot_cyclical_vs_defensive"] = 0.0

        # 8. Rotation regime
        speed = df["secrot_rotation_speed"]
        mu = speed.rolling(60, min_periods=20).mean()
        std = speed.rolling(60, min_periods=20).std().replace(0, 1)
        z = ((speed - mu) / std).fillna(0.0)
        df["secrot_regime"] = np.where(
            z > 1.0, 1.0,  # High rotation
            np.where(z < -1.0, -1.0, 0.0)  # Low rotation / trending
        )

        return df

    # ------------------------------------------------------------------
    def _create_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy rotation features from SPY when sector data unavailable."""
        close = df["close"] if "close" in df.columns else pd.Series(100.0, index=df.index)
        ret = close.pct_change().fillna(0.0)

        df["secrot_rotation_speed"] = ret.rolling(20, min_periods=5).std() * 10
        df["secrot_leader_momentum"] = ret.rolling(20, min_periods=5).mean() + ret.rolling(20, min_periods=5).std()
        df["secrot_laggard_momentum"] = ret.rolling(20, min_periods=5).mean() - ret.rolling(20, min_periods=5).std()
        df["secrot_leader_laggard_spread"] = ret.rolling(20, min_periods=5).std() * 2
        df["secrot_momentum_dispersion"] = ret.rolling(20, min_periods=5).std()
        df["secrot_top3_concentration"] = 0.5
        df["secrot_cyclical_vs_defensive"] = 0.0
        df["secrot_regime"] = 0.0

        return df

    # ------------------------------------------------------------------
    def analyze_current_rotation(
        self, df_daily: pd.DataFrame
    ) -> Optional[Dict]:
        """Analyze current sector rotation state."""
        if df_daily.empty:
            return None
        feats = self.create_sector_rotation_features(df_daily)
        last = feats.iloc[-1]
        return {
            feat: float(last.get(feat, 0.0))
            for feat in self._all_feature_names()
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "secrot_rotation_speed",
            "secrot_leader_momentum",
            "secrot_laggard_momentum",
            "secrot_leader_laggard_spread",
            "secrot_momentum_dispersion",
            "secrot_top3_concentration",
            "secrot_cyclical_vs_defensive",
            "secrot_regime",
        ]

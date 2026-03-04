"""
Wave M4: FX Carry Features — Major currency pair signals.

Downloads EUR/USD, USD/JPY, GBP/USD, AUD/USD from yfinance and derives:
  - Currency momentum signals
  - JPY carry trade proxy
  - FX dispersion (cross-currency volatility)
  - FX-equity correlation
  - Dollar strength composite
  - FX risk-on/off regime

Prefix: fxc_
Default: ON
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


FX_TICKERS = {
    "EURUSD=X": "eur_usd",
    "JPY=X": "usd_jpy",
    "GBPUSD=X": "gbp_usd",
    "AUDUSD=X": "aud_usd",
}


class FXCarryFeatures:
    """FX carry & currency signal feature engineering."""

    def __init__(self):
        self._data: Optional[pd.DataFrame] = None
        self._data_source: str = "none"

    # ------------------------------------------------------------------
    def download_fx_data(
        self, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Download major FX pairs from yfinance."""
        try:
            import yfinance as yf

            frames: Dict[str, pd.Series] = {}
            for ticker, col_name in FX_TICKERS.items():
                try:
                    raw = yf.download(ticker, start=start, end=end, progress=False)
                    if raw is not None and len(raw) > 10:
                        if isinstance(raw.columns, pd.MultiIndex):
                            frames[col_name] = raw[("Close", ticker)]
                        else:
                            frames[col_name] = raw["Close"]
                except Exception:
                    pass

            if not frames:
                self._data_source = "proxy"
                return None

            df = pd.DataFrame(frames)
            self._data = df
            self._data_source = "yfinance"
            print(f"[FXC] Downloaded {len(frames)}/{len(FX_TICKERS)} FX pairs, {len(df)} days")
            return df

        except ImportError:
            self._data_source = "proxy"
            return None
        except Exception as e:
            print(f"[FXC] Download error: {e} — using proxy")
            self._data_source = "proxy"
            return None

    # ------------------------------------------------------------------
    def create_fx_carry_features(
        self, df_daily: pd.DataFrame
    ) -> pd.DataFrame:
        """Create 8 fxc_ features."""
        df = df_daily.copy()

        if self._data is not None and self._data_source == "yfinance":
            df = self._create_from_fx(df)
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
    def _create_from_fx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from actual FX data."""
        fx = self._data.reindex(df.index, method="ffill").ffill().bfill()

        # 1. EUR/USD momentum (20d) — dollar weakness proxy
        if "eur_usd" in fx.columns:
            df["fxc_eur_momentum"] = fx["eur_usd"].pct_change(20).fillna(0.0)
        else:
            df["fxc_eur_momentum"] = 0.0

        # 2. JPY carry proxy — when USD/JPY rises, carry trade risk-on
        if "usd_jpy" in fx.columns:
            jpy_ret = fx["usd_jpy"].pct_change(20).fillna(0.0)
            df["fxc_jpy_carry_signal"] = jpy_ret  # positive = risk-on
        else:
            df["fxc_jpy_carry_signal"] = 0.0

        # 3. GBP momentum
        if "gbp_usd" in fx.columns:
            df["fxc_gbp_momentum"] = fx["gbp_usd"].pct_change(20).fillna(0.0)
        else:
            df["fxc_gbp_momentum"] = 0.0

        # 4. AUD risk-on signal — AUD/USD tracks risk appetite
        if "aud_usd" in fx.columns:
            aud_ret = fx["aud_usd"].pct_change(10).fillna(0.0)
            df["fxc_aud_risk_signal"] = aud_ret
        else:
            df["fxc_aud_risk_signal"] = 0.0

        # 5. FX dispersion (cross-currency volatility)
        rets = pd.DataFrame()
        for col in ["eur_usd", "gbp_usd", "aud_usd"]:
            if col in fx.columns:
                rets[col] = fx[col].pct_change().fillna(0.0)
        if len(rets.columns) >= 2:
            df["fxc_dispersion"] = rets.rolling(20, min_periods=5).std().mean(axis=1).fillna(0.0)
        else:
            df["fxc_dispersion"] = 0.0

        # 6. FX-equity correlation (EUR/USD vs SPY)
        spy_ret = df["close"].pct_change().fillna(0.0) if "close" in df.columns else pd.Series(0.0, index=df.index)
        if "eur_usd" in fx.columns:
            eur_ret = fx["eur_usd"].pct_change().fillna(0.0)
            df["fxc_equity_corr"] = spy_ret.rolling(20, min_periods=10).corr(eur_ret).fillna(0.0)
        else:
            df["fxc_equity_corr"] = 0.0

        # 7. Dollar strength composite (inverse of average major pair returns)
        pair_rets = []
        for col in ["eur_usd", "gbp_usd", "aud_usd"]:
            if col in fx.columns:
                pair_rets.append(fx[col].pct_change(20).fillna(0.0))
        if pair_rets:
            avg_pair_ret = pd.concat(pair_rets, axis=1).mean(axis=1)
            df["fxc_dollar_composite"] = -avg_pair_ret  # negative = strong dollar
        else:
            df["fxc_dollar_composite"] = 0.0

        # 8. FX regime (risk-on / risk-off / neutral)
        if "aud_usd" in fx.columns and "usd_jpy" in fx.columns:
            aud_z = fx["aud_usd"].pct_change(20).fillna(0.0)
            jpy_z = fx["usd_jpy"].pct_change(20).fillna(0.0)
            combined = (aud_z + jpy_z) / 2
            mu = combined.rolling(60, min_periods=20).mean()
            std = combined.rolling(60, min_periods=20).std().replace(0, 1)
            z = ((combined - mu) / std).fillna(0.0)
            df["fxc_regime"] = np.where(z > 1.0, 1.0, np.where(z < -1.0, -1.0, 0.0))
        else:
            df["fxc_regime"] = 0.0

        return df

    # ------------------------------------------------------------------
    def _create_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy FX features from DXY/SPY when currency data unavailable."""
        close = df["close"] if "close" in df.columns else pd.Series(100.0, index=df.index)
        ret = close.pct_change().fillna(0.0)

        # Use SPY inverse as rough dollar proxy
        df["fxc_eur_momentum"] = -ret.rolling(20, min_periods=5).sum() * 0.5
        df["fxc_jpy_carry_signal"] = ret.rolling(20, min_periods=5).sum() * 0.3
        df["fxc_gbp_momentum"] = -ret.rolling(20, min_periods=5).sum() * 0.4
        df["fxc_aud_risk_signal"] = ret.rolling(10, min_periods=5).sum() * 0.5
        df["fxc_dispersion"] = ret.rolling(20, min_periods=5).std() * 0.5
        df["fxc_equity_corr"] = 0.0
        df["fxc_dollar_composite"] = ret.rolling(20, min_periods=5).sum() * 0.3
        df["fxc_regime"] = 0.0

        return df

    # ------------------------------------------------------------------
    def analyze_current_fx(
        self, df_daily: pd.DataFrame
    ) -> Optional[Dict]:
        """Analyze current FX conditions."""
        if df_daily.empty:
            return None
        feats = self.create_fx_carry_features(df_daily)
        last = feats.iloc[-1]
        return {
            feat: float(last.get(feat, 0.0))
            for feat in self._all_feature_names()
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "fxc_eur_momentum",
            "fxc_jpy_carry_signal",
            "fxc_gbp_momentum",
            "fxc_aud_risk_signal",
            "fxc_dispersion",
            "fxc_equity_corr",
            "fxc_dollar_composite",
            "fxc_regime",
        ]

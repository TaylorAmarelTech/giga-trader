"""
GIGA TRADER - Options-Derived Features (IV Rank, SKEW, Vol-of-Vol)
====================================================================
Downloads CBOE VIX and SKEW Index data via yfinance and engineers
features that capture implied volatility context and tail risk.

Distinct from GammaExposureFeatures which uses VIX *term structure*
(contango/backwardation) to proxy dealer gamma positioning. This module
uses VIX *absolute level* context (rank, percentile) plus the CBOE SKEW
Index (tail risk) plus interaction signals.

Data sources (all via yfinance, free, no API key):
  - ^VIX: CBOE Volatility Index (absolute level context)
  - ^SKEW: CBOE Skew Index (tail risk — probability of outlier moves)
  - SPY close: for realized volatility calculation

Features generated (prefix: opt_):
  - opt_iv_rank: VIX percentile rank in 252-day range
  - opt_iv_percentile: % of 252 days VIX was lower
  - opt_iv_zscore: VIX z-score vs 60-day rolling
  - opt_iv_chg_1d: 1-day VIX change
  - opt_iv_chg_5d: 5-day VIX change
  - opt_iv_mean_revert: Mean reversion signal (distance from 60d mean)
  - opt_skew_raw: Normalized SKEW: (SKEW - 100) / 30
  - opt_skew_zscore: SKEW z-score vs 60-day rolling
  - opt_skew_chg_5d: 5-day SKEW change
  - opt_skew_regime: 0=normal, 1=elevated, 2=extreme tails
  - opt_fear_composite: VIX_zscore * SKEW_zscore interaction
  - opt_complacency: Binary contrarian sell signal (low VIX + low SKEW)
  - opt_tail_risk: Binary fat-tail indicator (high SKEW)
  - opt_vol_of_vol: 20-day rolling std of VIX changes
  - opt_vix_rv_spread: VIX minus 20-day realized vol
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.core.feature_base import FeatureModuleBase

warnings.filterwarnings("ignore")

logger = logging.getLogger("OPTIONS_FEATURES")


class OptionsFeatures(FeatureModuleBase):
    """
    Download CBOE VIX/SKEW data and create options-derived features.

    Pattern: download -> compute -> merge (same as other feature classes).
    """

    FEATURE_NAMES = [
        "opt_iv_rank",
        "opt_iv_percentile",
        "opt_iv_zscore",
        "opt_iv_chg_1d",
        "opt_iv_chg_5d",
        "opt_iv_mean_revert",
        "opt_skew_raw",
        "opt_skew_zscore",
        "opt_skew_chg_5d",
        "opt_skew_regime",
        "opt_fear_composite",
        "opt_complacency",
        "opt_tail_risk",
        "opt_vol_of_vol",
        "opt_vix_rv_spread",
    ]

    def __init__(self):
        self.data: pd.DataFrame = pd.DataFrame()

    def download_options_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Download VIX and SKEW Index data from yfinance.

        Uses:
          - ^VIX (CBOE Volatility Index)
          - ^SKEW (CBOE Skew Index — measures tail risk)

        Returns empty DataFrame on failure.
        """
        print("\n[OPTIONS] Downloading VIX and SKEW data...")

        try:
            import yfinance as yf
        except ImportError:
            print("  [WARN] yfinance package not available")
            return pd.DataFrame()

        try:
            # Extra lookback for rolling calculations (252d rank + 90d buffer)
            dl_start = pd.Timestamp(start_date) - pd.Timedelta(days=400)
            dl_end = pd.Timestamp(end_date) + pd.Timedelta(days=1)

            tickers = {
                "^VIX": "vix_spot",
                "^SKEW": "skew",
            }

            frames = {}
            for ticker, col_name in tickers.items():
                try:
                    data = yf.download(
                        ticker,
                        start=dl_start.strftime("%Y-%m-%d"),
                        end=dl_end.strftime("%Y-%m-%d"),
                        progress=False,
                        auto_adjust=True,
                    )
                    if not data.empty:
                        close = data["Close"]
                        if isinstance(close, pd.DataFrame):
                            close = close.iloc[:, 0]
                        frames[col_name] = close
                except Exception as e:
                    logger.debug(f"  Failed to download {ticker}: {e}")

            if "vix_spot" not in frames:
                print("  [WARN] Could not download VIX data")
                return pd.DataFrame()

            # Combine into single DataFrame
            combined = pd.DataFrame(frames)
            combined.index = pd.to_datetime(combined.index)
            combined = combined.sort_index()

            # Forward-fill SKEW gaps (may have fewer trading days than VIX)
            combined = combined.ffill().dropna(subset=["vix_spot"])

            self.data = combined
            skew_status = "with SKEW" if "skew" in combined.columns else "VIX only"
            print(f"  [OPTIONS] Downloaded {len(combined)} days ({skew_status})")
            return self.data

        except Exception as e:
            logger.warning(f"  Options data download failed: {e}")
            return pd.DataFrame()

    def create_options_features(
        self,
        spy_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create 15 options-derived features and merge into spy_daily.

        Returns original spy_daily unchanged if no data available.
        """
        if self.data.empty:
            return spy_daily

        print("\n[OPTIONS] Engineering options IV/SKEW features...")

        features = spy_daily.copy()
        opt = self.data.copy()

        # Ensure date column for merge
        opt["date"] = pd.to_datetime(opt.index).normalize()

        vix = opt["vix_spot"]
        has_skew = "skew" in opt.columns

        # ─── VIX Level Features (distinct from GEX structure features) ────

        # 1. IV Rank: where VIX sits in its 252-day range [0, 1]
        vix_252_high = vix.rolling(252, min_periods=60).max()
        vix_252_low = vix.rolling(252, min_periods=60).min()
        vix_range = vix_252_high - vix_252_low
        opt["opt_iv_rank"] = np.where(
            vix_range > 0.01,
            (vix - vix_252_low) / vix_range,
            0.5,
        )
        opt["opt_iv_rank"] = opt["opt_iv_rank"].clip(0, 1)

        # 2. IV Percentile: % of last 252 days VIX was lower
        opt["opt_iv_percentile"] = vix.rolling(252, min_periods=60).apply(
            lambda x: (x.iloc[:-1] < x.iloc[-1]).mean() if len(x) > 1 else 0.5,
            raw=False,
        )

        # 3. VIX z-score vs 60-day rolling
        vix_mean_60 = vix.rolling(60, min_periods=20).mean()
        vix_std_60 = vix.rolling(60, min_periods=20).std()
        opt["opt_iv_zscore"] = np.where(
            vix_std_60 > 0.01,
            (vix - vix_mean_60) / vix_std_60,
            0.0,
        )
        opt["opt_iv_zscore"] = opt["opt_iv_zscore"].clip(-3, 3)

        # 4. 1-day VIX change
        opt["opt_iv_chg_1d"] = vix.diff(1)

        # 5. 5-day VIX change
        opt["opt_iv_chg_5d"] = vix.diff(5)

        # 6. Mean reversion signal: positive = VIX below mean (expect reversion up)
        opt["opt_iv_mean_revert"] = np.where(
            vix_std_60 > 0.01,
            (vix_mean_60 - vix) / vix_std_60,
            0.0,
        )
        opt["opt_iv_mean_revert"] = opt["opt_iv_mean_revert"].clip(-3, 3)

        # ─── SKEW Features (entirely new data source) ────────────────────

        if has_skew:
            skew = opt["skew"]

            # 7. Normalized SKEW: center at 100, scale by ~30
            opt["opt_skew_raw"] = (skew - 100) / 30

            # 8. SKEW z-score vs 60-day rolling
            skew_mean_60 = skew.rolling(60, min_periods=20).mean()
            skew_std_60 = skew.rolling(60, min_periods=20).std()
            opt["opt_skew_zscore"] = np.where(
                skew_std_60 > 0.01,
                (skew - skew_mean_60) / skew_std_60,
                0.0,
            )
            opt["opt_skew_zscore"] = opt["opt_skew_zscore"].clip(-3, 3)

            # 9. 5-day SKEW change
            opt["opt_skew_chg_5d"] = skew.diff(5)

            # 10. SKEW regime: tail risk classification
            opt["opt_skew_regime"] = np.where(
                skew > 135, 2,  # Extreme tails
                np.where(skew > 115, 1, 0),  # Elevated / Normal
            )
        else:
            opt["opt_skew_raw"] = 0.0
            opt["opt_skew_zscore"] = 0.0
            opt["opt_skew_chg_5d"] = 0.0
            opt["opt_skew_regime"] = 0.0

        # ─── Interaction / Composite Features ─────────────────────────────

        # 11. Fear composite: VIX_zscore * SKEW_zscore
        #     High positive = panic (high VIX + high SKEW)
        #     High negative = mixed signals
        #     Low = complacency
        skew_z = opt["opt_skew_zscore"]
        opt["opt_fear_composite"] = opt["opt_iv_zscore"] * skew_z

        # 12. Complacency signal: VIX < 20th percentile AND SKEW < 20th percentile
        vix_p20 = vix.rolling(252, min_periods=60).quantile(0.20)
        if has_skew:
            skew_p20 = opt["skew"].rolling(252, min_periods=60).quantile(0.20)
            opt["opt_complacency"] = np.where(
                (vix < vix_p20) & (opt["skew"] < skew_p20), 1.0, 0.0
            )
        else:
            opt["opt_complacency"] = np.where(vix < vix_p20, 1.0, 0.0)

        # 13. Tail risk signal: SKEW > 80th percentile
        if has_skew:
            skew_p80 = opt["skew"].rolling(252, min_periods=60).quantile(0.80)
            opt["opt_tail_risk"] = np.where(opt["skew"] > skew_p80, 1.0, 0.0)
        else:
            opt["opt_tail_risk"] = 0.0

        # 14. Vol-of-vol: 20-day rolling std of daily VIX changes
        vix_daily_chg = vix.diff(1)
        opt["opt_vol_of_vol"] = vix_daily_chg.rolling(20, min_periods=10).std()

        # 15. VIX vs Realized Vol spread
        #     Positive = vol overpriced (premium for sellers)
        #     Compute 20-day realized vol from spy_daily close if available
        if "close" in features.columns and len(features) > 20:
            spy_returns = features["close"].pct_change()
            realized_vol = spy_returns.rolling(20, min_periods=10).std() * np.sqrt(252) * 100
            # Align by date for merge
            rv_df = pd.DataFrame({
                "date": features["date"].values,
                "realized_vol_20d": realized_vol.values,
            })
            opt = opt.merge(rv_df, on="date", how="left")
            opt["realized_vol_20d"] = opt["realized_vol_20d"].ffill().fillna(0)
            opt["opt_vix_rv_spread"] = np.where(
                opt["realized_vol_20d"] > 0,
                opt["vix_spot"] - opt["realized_vol_20d"],
                0.0,
            )
        else:
            opt["opt_vix_rv_spread"] = 0.0

        # ─── Merge into spy_daily ─────────────────────────────────────────

        opt_feature_cols = [
            "opt_iv_rank", "opt_iv_percentile", "opt_iv_zscore",
            "opt_iv_chg_1d", "opt_iv_chg_5d", "opt_iv_mean_revert",
            "opt_skew_raw", "opt_skew_zscore", "opt_skew_chg_5d",
            "opt_skew_regime", "opt_fear_composite", "opt_complacency",
            "opt_tail_risk", "opt_vol_of_vol", "opt_vix_rv_spread",
        ]

        merge_cols = ["date"] + [c for c in opt_feature_cols if c in opt.columns]
        merge_data = opt[merge_cols].copy()

        features = features.merge(merge_data, on="date", how="left")

        # Fill NaN with 0
        for col in opt_feature_cols:
            if col in features.columns:
                features[col] = features[col].fillna(0)

        print(f"  [OPTIONS] Added {sum(1 for c in features.columns if c.startswith('opt_'))} "
              f"options IV/SKEW features")
        return features

    def analyze_current_options(self, spy_daily: pd.DataFrame) -> Optional[dict]:
        """Analyze current options IV/SKEW conditions for dashboard display."""
        if self.data.empty:
            return None

        opt_cols = [c for c in spy_daily.columns if c.startswith("opt_")]
        if not opt_cols or spy_daily.empty:
            return None

        latest = spy_daily.iloc[-1]
        regime_map = {0: "normal", 1: "elevated", 2: "extreme"}

        iv_rank = float(latest.get("opt_iv_rank", 0))
        complacency = bool(latest.get("opt_complacency", 0))
        tail_risk = bool(latest.get("opt_tail_risk", 0))

        if iv_rank > 0.8:
            iv_state = "high_fear"
        elif iv_rank < 0.2:
            iv_state = "low_complacency"
        else:
            iv_state = "normal"

        return {
            "iv_rank": iv_rank * 100,  # Display as percentage
            "iv_percentile": float(latest.get("opt_iv_percentile", 0)) * 100,
            "iv_zscore": float(latest.get("opt_iv_zscore", 0)),
            "skew_regime": regime_map.get(int(latest.get("opt_skew_regime", 0)), "normal"),
            "fear_composite": float(latest.get("opt_fear_composite", 0)),
            "complacency_signal": complacency,
            "tail_risk_signal": tail_risk,
            "iv_state": iv_state,
            "vol_of_vol": float(latest.get("opt_vol_of_vol", 0)),
        }

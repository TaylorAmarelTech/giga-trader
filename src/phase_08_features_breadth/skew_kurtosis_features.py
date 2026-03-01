"""
Skew-Kurtosis Features -- higher-order moment features from return distributions.

Skewness measures asymmetry of returns; kurtosis measures tail heaviness.
Negative skew implies larger left tails (crash risk); excess kurtosis
implies fatter tails than normal.

All computations use rolling windows over close-price returns with
manual moment formulas (no scipy dependency).

Features (6, prefix skku_):
  skku_skew_5d        -- Rolling 5-day realized skewness of returns
  skku_skew_20d       -- Rolling 20-day realized skewness of returns
  skku_skew_60d       -- Rolling 60-day realized skewness of returns
  skku_kurtosis_20d   -- Rolling 20-day excess kurtosis (kurtosis - 3)
  skku_tail_asymmetry -- Rolling 60d: (count(ret > 2*std) - count(ret < -2*std)) / n
  skku_skew_z         -- Rolling 60d z-score of skku_skew_20d, clipped [-4, 4]
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SkewKurtosisFeatures:
    """Compute higher-order moment features from daily close prices."""

    REQUIRED_COLS = {"close"}

    def create_skew_kurtosis_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Add 6 skku_ features to df_daily.

        Parameters
        ----------
        df_daily : pd.DataFrame
            Must have 'close' column.

        Returns
        -------
        pd.DataFrame
            Original df_daily with 6 new skku_ columns added.
        """
        df = df_daily.copy()

        if "close" not in df.columns:
            logger.warning("SkewKurtosisFeatures: 'close' column missing, skipping")
            return df

        returns = df["close"].pct_change().values.astype(float)

        # --- skku_skew_5d: rolling 5-day skewness ---
        df["skku_skew_5d"] = self._rolling_skewness(returns, window=5)

        # --- skku_skew_20d: rolling 20-day skewness ---
        df["skku_skew_20d"] = self._rolling_skewness(returns, window=20)

        # --- skku_skew_60d: rolling 60-day skewness ---
        df["skku_skew_60d"] = self._rolling_skewness(returns, window=60)

        # --- skku_kurtosis_20d: rolling 20-day excess kurtosis ---
        df["skku_kurtosis_20d"] = self._rolling_excess_kurtosis(returns, window=20)

        # --- skku_tail_asymmetry: rolling 60d tail count asymmetry ---
        df["skku_tail_asymmetry"] = self._rolling_tail_asymmetry(returns, window=60)

        # --- skku_skew_z: rolling 60d z-score of skku_skew_20d ---
        skew_20 = df["skku_skew_20d"]
        rmean = skew_20.rolling(60, min_periods=20).mean()
        rstd = skew_20.rolling(60, min_periods=20).std()
        df["skku_skew_z"] = ((skew_20 - rmean) / (rstd + 1e-10)).clip(-4, 4)

        # Cleanup: fill NaN with 0.0, remove infinities
        for col in self._all_feature_names():
            df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        n_features = sum(1 for c in df.columns if c.startswith("skku_"))
        logger.info(f"SkewKurtosisFeatures: added {n_features} features")
        return df

    def analyze_current_skew_kurtosis(
        self, df_daily: pd.DataFrame
    ) -> Optional[Dict]:
        """Analyze current skew/kurtosis regime for dashboard display."""
        if "skku_skew_20d" not in df_daily.columns or len(df_daily) < 2:
            return None

        last = df_daily.iloc[-1]
        skew_20 = float(last.get("skku_skew_20d", 0.0))
        kurt_20 = float(last.get("skku_kurtosis_20d", 0.0))

        if skew_20 < -0.5:
            regime = "NEGATIVE_SKEW"
        elif skew_20 > 0.5:
            regime = "POSITIVE_SKEW"
        else:
            regime = "NORMAL"

        return {
            "skew_regime": regime,
            "skew_20d": round(skew_20, 4),
            "kurtosis_20d": round(kurt_20, 4),
        }

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "skku_skew_5d",
            "skku_skew_20d",
            "skku_skew_60d",
            "skku_kurtosis_20d",
            "skku_tail_asymmetry",
            "skku_skew_z",
        ]

    # ---- Internal helpers ------------------------------------------------

    @staticmethod
    def _rolling_skewness(returns: np.ndarray, window: int) -> np.ndarray:
        """
        Rolling skewness using manual formula:
        skew = (1/n) * sum((x - mean)^3) / ((1/n) * sum((x - mean)^2))^1.5
        """
        n = len(returns)
        result = np.full(n, np.nan)
        min_periods = max(3, window // 2)

        for i in range(min_periods, n):
            start = max(0, i - window + 1)
            w = returns[start : i + 1]
            valid = w[~np.isnan(w)]
            if len(valid) < 3:
                continue

            m = np.mean(valid)
            diffs = valid - m
            m2 = np.mean(diffs ** 2)
            if m2 < 1e-20:
                result[i] = 0.0
                continue

            m3 = np.mean(diffs ** 3)
            result[i] = m3 / (m2 ** 1.5)

        return result

    @staticmethod
    def _rolling_excess_kurtosis(returns: np.ndarray, window: int) -> np.ndarray:
        """
        Rolling excess kurtosis using manual formula:
        kurt = (1/n) * sum((x - mean)^4) / ((1/n) * sum((x - mean)^2))^2 - 3
        """
        n = len(returns)
        result = np.full(n, np.nan)
        min_periods = max(4, window // 2)

        for i in range(min_periods, n):
            start = max(0, i - window + 1)
            w = returns[start : i + 1]
            valid = w[~np.isnan(w)]
            if len(valid) < 4:
                continue

            m = np.mean(valid)
            diffs = valid - m
            m2 = np.mean(diffs ** 2)
            if m2 < 1e-20:
                result[i] = 0.0
                continue

            m4 = np.mean(diffs ** 4)
            result[i] = (m4 / (m2 ** 2)) - 3.0

        return result

    @staticmethod
    def _rolling_tail_asymmetry(returns: np.ndarray, window: int) -> np.ndarray:
        """
        Rolling tail asymmetry over *window* days:
        count(returns > 2*std) - count(returns < -2*std), divided by count.
        """
        n = len(returns)
        result = np.full(n, np.nan)
        min_periods = max(10, window // 2)

        for i in range(min_periods, n):
            start = max(0, i - window + 1)
            w = returns[start : i + 1]
            valid = w[~np.isnan(w)]
            if len(valid) < 10:
                continue

            std = np.std(valid, ddof=1)
            if std < 1e-15:
                result[i] = 0.0
                continue

            m = np.mean(valid)
            upper = np.sum(valid > m + 2 * std)
            lower = np.sum(valid < m - 2 * std)
            result[i] = (upper - lower) / len(valid)

        return result

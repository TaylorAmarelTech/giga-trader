"""
Normalized Mutual Information Features — market efficiency via information theory.

NMI between returns and their lagged versions measures how predictable
(inefficient) the market is at different horizons.  High NMI = autocorrelation
structure exists; low NMI = market is informationally efficient.

Features (3, prefix nmi_):
  nmi_lag1_50d   — Rolling 50-day NMI between ret_t and ret_{t-1}
  nmi_lag5_50d   — Rolling 50-day NMI between ret_t and ret_{t-5}
  nmi_efficiency — 1 - max(lag1, lag5); higher = more efficient market
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

logger = logging.getLogger(__name__)


def _rolling_nmi(returns: np.ndarray, lag: int, window: int = 50, n_bins: int = 10) -> np.ndarray:
    """Compute rolling NMI between returns and their lagged version."""
    n = len(returns)
    result = np.full(n, 0.0)

    for i in range(window + lag, n):
        # Get the window of returns
        ret_window = returns[i - window:i]
        lag_window = returns[i - window - lag:i - lag]

        # Discretize into bins
        ret_bins = np.digitize(
            ret_window,
            np.linspace(ret_window.min() - 1e-10, ret_window.max() + 1e-10, n_bins + 1),
        )
        lag_bins = np.digitize(
            lag_window,
            np.linspace(lag_window.min() - 1e-10, lag_window.max() + 1e-10, n_bins + 1),
        )

        # Compute NMI
        result[i] = normalized_mutual_info_score(ret_bins, lag_bins)

    return result


class NMIFeatures:
    """Compute Normalized Mutual Information features from daily OHLCV data."""

    REQUIRED_COLS = {"close"}

    def create_nmi_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Add NMI market-efficiency features to df_daily.

        Parameters
        ----------
        df_daily : pd.DataFrame
            Must have 'close' column.

        Returns
        -------
        pd.DataFrame
            Original df_daily with 3 new nmi_ columns added.
        """
        df = df_daily.copy()

        if "close" not in df.columns:
            logger.warning("NMIFeatures: 'close' column missing, skipping")
            return df

        # Need enough data points for rolling window + lag
        n = len(df)
        if n < 56:  # 50 window + 5 lag + 1
            logger.info("NMIFeatures: not enough data (%d rows), defaulting to 0.0", n)
            for col in self._all_feature_names():
                df[col] = 0.0
            return df

        # Compute returns
        returns = df["close"].pct_change().fillna(0.0).values

        # Rolling NMI for lag 1 and lag 5
        nmi_lag1 = _rolling_nmi(returns, lag=1, window=50, n_bins=10)
        nmi_lag5 = _rolling_nmi(returns, lag=5, window=50, n_bins=10)

        df["nmi_lag1_50d"] = nmi_lag1
        df["nmi_lag5_50d"] = nmi_lag5

        # Efficiency = 1 - max(lag1, lag5); higher means more efficient
        df["nmi_efficiency"] = 1.0 - np.maximum(nmi_lag1, nmi_lag5)

        # Cleanup: fill NaN, remove infinities, clip to [0, 1]
        for col in self._all_feature_names():
            df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0).clip(0.0, 1.0)

        n_features = sum(1 for c in df.columns if c.startswith("nmi_"))
        logger.info("NMIFeatures: added %d features", n_features)
        return df

    def analyze_current_efficiency(self, df_daily: pd.DataFrame) -> Optional[Dict]:
        """Analyze current market efficiency regime for dashboard display."""
        if "nmi_efficiency" not in df_daily.columns or len(df_daily) < 2:
            return None

        last = df_daily.iloc[-1]
        efficiency = float(last.get("nmi_efficiency", 0.0))
        lag1 = float(last.get("nmi_lag1_50d", 0.0))
        lag5 = float(last.get("nmi_lag5_50d", 0.0))

        if efficiency >= 0.85:
            regime = "EFFICIENT"
        elif efficiency <= 0.60:
            regime = "PREDICTABLE"
        else:
            regime = "MODERATE"

        return {
            "efficiency_regime": regime,
            "nmi_efficiency": round(efficiency, 4),
            "nmi_lag1_50d": round(lag1, 4),
            "nmi_lag5_50d": round(lag5, 4),
        }

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "nmi_lag1_50d",
            "nmi_lag5_50d",
            "nmi_efficiency",
        ]

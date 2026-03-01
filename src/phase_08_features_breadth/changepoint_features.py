"""
Changepoint Detection Features — Bayesian Online Changepoint Detection.

Simplified BOCPD (Adams & MacKay 2007) using Gaussian conjugate prior,
pure numpy implementation.

Features (3, prefix cpd_):
  cpd_run_length  — Estimated number of days since last changepoint (MAP run length)
  cpd_prob_change — Probability of a changepoint at this timestep
  cpd_regime_id   — Integer regime label (increments on changepoint, mod 10)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ChangepointFeatures:
    """Compute Bayesian Online Changepoint Detection features from daily data."""

    REQUIRED_COLS = {"close"}

    def create_changepoint_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Add changepoint detection features to df_daily.

        Parameters
        ----------
        df_daily : pd.DataFrame
            Must have 'close' column.

        Returns
        -------
        pd.DataFrame
            Original df_daily with 3 new cpd_ columns added.
        """
        df = df_daily.copy()

        if "close" not in df.columns:
            logger.warning("ChangepointFeatures: 'close' column missing, skipping")
            return df

        # Compute returns from close prices
        close = df["close"].values.astype(float)
        returns = np.empty(len(close))
        returns[0] = 0.0
        returns[1:] = np.diff(close) / (np.abs(close[:-1]) + 1e-10)

        # Run BOCPD
        run_length, prob_change, regime_id = self._bocpd(returns)

        df["cpd_run_length"] = run_length
        df["cpd_prob_change"] = prob_change
        df["cpd_regime_id"] = regime_id

        # Cleanup: NaN -> 0.0, no inf
        for col in self._all_feature_names():
            df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        n_features = sum(1 for c in df.columns if c.startswith("cpd_"))
        logger.info(f"ChangepointFeatures: added {n_features} features")
        return df

    def analyze_current_changepoint(self, df_daily: pd.DataFrame) -> Optional[Dict]:
        """Analyze current changepoint state for dashboard display."""
        if "cpd_run_length" not in df_daily.columns or len(df_daily) < 2:
            return None

        last = df_daily.iloc[-1]

        return {
            "run_length": int(last.get("cpd_run_length", 0)),
            "prob_change": round(float(last.get("cpd_prob_change", 0.0)), 4),
            "regime_id": int(last.get("cpd_regime_id", 0)),
        }

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "cpd_run_length",
            "cpd_prob_change",
            "cpd_regime_id",
        ]

    # ─── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _bocpd(returns, hazard_lambda=250):
        """Simplified BOCPD using Gaussian conjugate prior."""
        n = len(returns)
        run_length = np.zeros(n)
        prob_change = np.zeros(n)
        regime_id = np.zeros(n)

        if n < 10:
            return run_length, prob_change, regime_id

        # Hazard function: constant rate 1/hazard_lambda
        hazard = 1.0 / hazard_lambda

        current_run = 0
        current_regime = 0
        # Online statistics for the current run
        run_sum = 0.0
        run_sumsq = 0.0

        for i in range(n):
            x = returns[i]
            current_run += 1
            run_sum += x
            run_sumsq += x * x

            # Compute predictive probability under current run
            if current_run > 1:
                run_mean = run_sum / current_run
                run_var = max(run_sumsq / current_run - run_mean**2, 1e-10)
                # How surprising is this observation?
                z = abs(x - run_mean) / (np.sqrt(run_var) + 1e-10)
                # Change probability based on surprise + hazard
                prob_change[i] = min(1.0, hazard + 0.3 * max(0, z - 2.0))
            else:
                prob_change[i] = hazard

            # Detect changepoint
            if prob_change[i] > 0.5 and current_run > 10:
                current_regime = (current_regime + 1) % 10
                current_run = 1
                run_sum = x
                run_sumsq = x * x

            run_length[i] = min(current_run, 252)
            regime_id[i] = current_regime

        return run_length, prob_change, regime_id

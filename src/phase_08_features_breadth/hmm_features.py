"""
HMM-style Regime Features -- rolling regime detection via quantile-based classification.

Uses a pure-numpy rolling regime detector (no hmmlearn dependency) that classifies
market regimes into bear (0), neutral (1), and bull (2) states based on return
distributions within a sliding window.

Features (5, prefix hmm_):
  hmm_state            -- Current most likely regime state (0=bear, 1=neutral, 2=bull)
  hmm_bull_prob        -- Probability of being in the bull (highest-mean) state
  hmm_bear_prob        -- Probability of being in the bear (lowest-mean) state
  hmm_transition_prob  -- Probability of state transition (1.0 if state changed, else 0.0)
  hmm_regime_duration  -- Rolling count of consecutive days in the same state
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)


class HMMFeatures(FeatureModuleBase):
    """Compute HMM-style regime detection features from daily OHLCV data."""
    FEATURE_NAMES = ["hmm_state", "hmm_bull_prob", "hmm_bear_prob", "hmm_transition_prob", "hmm_regime_duration"]


    REQUIRED_COLS = {"close"}

    def create_hmm_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Add HMM-style regime features to df_daily.

        Parameters
        ----------
        df_daily : pd.DataFrame
            Must have 'close' column.

        Returns
        -------
        pd.DataFrame
            Original df_daily with 5 new hmm_ columns added.
        """
        df = df_daily.copy()

        if "close" not in df.columns:
            logger.warning("HMMFeatures: 'close' column missing, skipping")
            return df

        close = df["close"].values.astype(float)
        n = len(close)

        if n < 2:
            for col in self._all_feature_names():
                df[col] = _default_for(col)
            return df

        # Compute returns
        returns = np.empty(n)
        returns[0] = 0.0
        returns[1:] = close[1:] / close[:-1] - 1.0

        # Run rolling regime detection
        states, bull_prob, bear_prob, trans_prob, duration = _rolling_regime_detect(
            returns
        )

        df["hmm_state"] = states
        df["hmm_bull_prob"] = bull_prob
        df["hmm_bear_prob"] = bear_prob
        df["hmm_transition_prob"] = trans_prob
        df["hmm_regime_duration"] = duration

        # Cleanup: NaN -> sensible defaults, inf -> defaults
        for col in self._all_feature_names():
            default = _default_for(col)
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(default)

        n_features = sum(1 for c in df.columns if c.startswith("hmm_"))
        logger.info(f"HMMFeatures: added {n_features} features")
        return df

    def analyze_current_hmm(self, df_daily: pd.DataFrame) -> Optional[Dict]:
        """Analyze current HMM regime for dashboard display."""
        if "hmm_state" not in df_daily.columns or len(df_daily) < 2:
            return None

        last = df_daily.iloc[-1]
        state = float(last.get("hmm_state", 1.0))
        bull_p = float(last.get("hmm_bull_prob", 0.33))
        bear_p = float(last.get("hmm_bear_prob", 0.33))
        duration = float(last.get("hmm_regime_duration", 1.0))

        if state == 2.0:
            regime = "BULL"
        elif state == 0.0:
            regime = "BEAR"
        else:
            regime = "NEUTRAL"

        return {
            "hmm_regime": regime,
            "hmm_bull_prob": round(bull_p, 3),
            "hmm_bear_prob": round(bear_p, 3),
            "hmm_regime_duration": round(duration),
        }

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "hmm_state",
            "hmm_bull_prob",
            "hmm_bear_prob",
            "hmm_transition_prob",
            "hmm_regime_duration",
        ]


# --- Defaults per feature -----------------------------------------------------------


def _default_for(col: str) -> float:
    """Return the sensible default value for a given hmm_ feature."""
    defaults = {
        "hmm_state": 1.0,          # neutral
        "hmm_bull_prob": 0.33,
        "hmm_bear_prob": 0.33,
        "hmm_transition_prob": 0.0,
        "hmm_regime_duration": 1.0,
    }
    return defaults.get(col, 0.0)


# --- Internal helpers ----------------------------------------------------------------


def _rolling_regime_detect(
    returns: np.ndarray, n_states: int = 3, window: int = 120
) -> tuple:
    """
    Simple rolling regime detection using quantile-based classification on
    return distributions.  Simulates a 3-state Gaussian HMM without the
    hmmlearn dependency.

    Parameters
    ----------
    returns : np.ndarray
        Array of daily returns (length n).
    n_states : int
        Number of regime states (always 3: bear/neutral/bull).
    window : int
        Lookback window for computing quantile thresholds.

    Returns
    -------
    tuple of (states, bull_prob, bear_prob, trans_prob, duration)
        Each is a np.ndarray of length len(returns).
    """
    n = len(returns)
    states = np.full(n, 1.0)        # default = neutral state
    bull_prob = np.full(n, 0.33)
    bear_prob = np.full(n, 0.33)
    trans_prob = np.full(n, 0.0)
    duration = np.ones(n)

    if n < window:
        return states, bull_prob, bear_prob, trans_prob, duration

    for i in range(window, n):
        w = returns[i - window : i]

        # Simple 3-regime classification based on return quantiles
        q33 = np.percentile(w, 33)
        q67 = np.percentile(w, 67)

        recent = returns[max(i - 20, 0) : i]
        recent_mean = np.mean(recent)

        if recent_mean > q67:
            states[i] = 2  # Bull
        elif recent_mean < q33:
            states[i] = 0  # Bear
        else:
            states[i] = 1  # Neutral

        # Compute state probabilities via softmax of inverse distance to
        # each regime's centroid within the window
        bear_mask = w <= q33
        neutral_mask = (w > q33) & (w <= q67)
        bull_mask = w > q67

        bear_center = np.mean(w[bear_mask]) if np.sum(bear_mask) > 0 else q33
        neutral_center = (
            np.mean(w[neutral_mask]) if np.sum(neutral_mask) > 0 else (q33 + q67) / 2
        )
        bull_center = np.mean(w[bull_mask]) if np.sum(bull_mask) > 0 else q67

        distances = np.array([
            abs(recent_mean - bear_center),
            abs(recent_mean - neutral_center),
            abs(recent_mean - bull_center),
        ])
        distances = np.maximum(distances, 1e-10)
        inv_distances = 1.0 / distances
        probs = inv_distances / inv_distances.sum()

        bear_prob[i] = probs[0]
        bull_prob[i] = probs[2]

        # Transition probability: did state change from yesterday?
        if i > window and states[i] != states[i - 1]:
            trans_prob[i] = 1.0

        # Duration: consecutive days in same state
        if i > window and states[i] == states[i - 1]:
            duration[i] = duration[i - 1] + 1
        else:
            duration[i] = 1

    return states, bull_prob, bear_prob, trans_prob, duration

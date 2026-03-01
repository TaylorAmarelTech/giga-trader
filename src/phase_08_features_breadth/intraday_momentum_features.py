"""
GIGA TRADER - Intraday Momentum Features
==========================================
Derive intraday momentum signals from daily OHLC data.

Since we work with daily bars (not tick or minute data), these features
approximate intraday dynamics using open, high, low, and close prices.
The key insight is that OHLC bars encode substantial intraday structure:

  - Where price opened relative to yesterday's close  → pre-market/open momentum
  - Where price closed relative to the day's range    → late-session direction
  - Correlation between intraday first/second half    → reversal vs continuation

No external API keys required. All features are derived purely from
standard daily OHLCV data already present in the pipeline.

Features generated (prefix: imom_):
  - imom_first_30min: Overnight gap as proxy for open-to-first-30min return.
                      Computed as (open - prev_close) / prev_close.
                      Captures pre-market and opening momentum.
  - imom_last_60min:  Close location within the day's range, as proxy for
                      late-session buying/selling pressure.
                      Computed as (close - midpoint) / (half_range + eps),
                      where midpoint = (high + low) / 2 and half_range = (high - low) / 2.
                      Ranges in [-1, 1]; +1 = closed at the high, -1 = closed at the low.
  - imom_midday_reversal: 20-day rolling correlation between the first-half
                      proxy ((high+low)/2 - open) and the second-half proxy
                      (close - (high+low)/2).  Negative values indicate a
                      systematic open-surge → close-fade reversal pattern.
  - imom_overnight_gap_impact: Fraction of the daily return explained by the
                      overnight gap.  (open - prev_close) / (close - prev_close),
                      clipped to [-2, 2].  Values near 1.0 mean the gap already
                      "explained" all of the day's move; values near 0 mean the
                      session added most of the return itself.
"""

import logging
import warnings
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger("INTRADAY_MOMENTUM")

# Features produced by this module
IMOM_FEATURES: List[str] = [
    "imom_first_30min",
    "imom_last_60min",
    "imom_midday_reversal",
    "imom_overnight_gap_impact",
]

# Required OHLC columns
REQUIRED_COLUMNS = {"close", "open", "high", "low"}


class IntradayMomentumFeatures:
    """
    Create intraday momentum features from daily OHLC data.

    Uses only standard open/high/low/close columns — no external data
    sources or API keys required.

    Pattern: download (no-op) → compute → merge  (same as other phase_08 modules).
    """

    def __init__(self, correlation_window: int = 20):
        """
        Parameters
        ----------
        correlation_window : int
            Rolling window length (trading days) used for the midday
            reversal correlation feature.  Default is 20.
        """
        if correlation_window < 2:
            raise ValueError("correlation_window must be >= 2")
        self.correlation_window = correlation_window

    # ─── Download (no-op) ────────────────────────────────────────────────────

    def download_intraday_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        No external download required — features are derived from the
        daily OHLC data already present in the pipeline.

        Returns an empty DataFrame (conforming to the standard module API).
        """
        logger.debug(
            "[INTRADAY_MOMENTUM] No download required — using existing OHLC data"
        )
        return pd.DataFrame()

    # ─── Feature Engineering ─────────────────────────────────────────────────

    def create_intraday_momentum_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create intraday momentum features and append them to ``df``.

        Parameters
        ----------
        df : pd.DataFrame
            Daily OHLCV DataFrame.  Must contain ``close``, ``open``,
            ``high``, ``low`` columns.  Rows must be ordered chronologically
            (oldest first).

        Returns
        -------
        pd.DataFrame
            Original DataFrame with four additional ``imom_*`` columns.
            Returns the original DataFrame unchanged if required columns
            are missing.
        """
        # ── Validate required columns ──────────────────────────────────────
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            logger.warning(
                f"[INTRADAY_MOMENTUM] Missing required columns {missing}; "
                "returning df unchanged."
            )
            return df

        print("\n[INTRADAY_MOMENTUM] Engineering intraday momentum features...")

        result = df.copy()
        n = len(result)

        o = result["open"].values.astype(float)
        h = result["high"].values.astype(float)
        lo = result["low"].values.astype(float)
        c = result["close"].values.astype(float)

        prev_close = np.empty(n)
        prev_close[:] = np.nan
        prev_close[1:] = c[:-1]

        eps = 1e-10

        # ── Feature 1: imom_first_30min ────────────────────────────────────
        # Overnight gap: (open - prev_close) / prev_close
        # Proxy for pre-market + opening momentum.
        first_30 = np.where(
            ~np.isnan(prev_close) & (np.abs(prev_close) > eps),
            (o - prev_close) / (prev_close + eps),
            0.0,
        )
        # Clip to prevent extreme values from fat-tailed gap distributions
        first_30 = np.clip(first_30, -0.10, 0.10)

        # ── Feature 2: imom_last_60min ─────────────────────────────────────
        # Close location within the day's range, normalized to [-1, +1].
        #   midpoint  = (high + low) / 2
        #   half_range = (high - low) / 2
        #   last_60   = (close - midpoint) / (half_range + eps)
        # Result = +1 when close == high, -1 when close == low.
        midpoint = (h + lo) / 2.0
        half_range = (h - lo) / 2.0
        last_60 = (c - midpoint) / (half_range + eps)
        # Guarantee bounded output despite floating-point edge cases
        last_60 = np.clip(last_60, -1.0, 1.0)

        # ── Feature 3: imom_midday_reversal ────────────────────────────────
        # Rolling correlation between:
        #   first_half_proxy  = midpoint - open  (direction from open to VWAP-like mid)
        #   second_half_proxy = close - midpoint  (direction from mid to close)
        # Negative correlation → reversal pattern (gaps up then fades, etc.)
        first_half = midpoint - o     # morning direction
        second_half = c - midpoint    # afternoon direction

        s1 = pd.Series(first_half)
        s2 = pd.Series(second_half)
        min_p = min(self.correlation_window, max(2, self.correlation_window // 4))
        midday_rev = s1.rolling(self.correlation_window, min_periods=min_p).corr(s2)
        midday_rev = midday_rev.fillna(0.0).values
        # Correlation is already in [-1, 1]; clip defensively
        midday_rev = np.clip(midday_rev, -1.0, 1.0)

        # ── Feature 4: imom_overnight_gap_impact ───────────────────────────
        # Fraction of daily return explained by the overnight gap.
        # = (open - prev_close) / (close - prev_close)
        # Clipped to [-2, 2].
        daily_ret = np.where(
            ~np.isnan(prev_close),
            c - prev_close,
            np.nan,
        )
        gap = np.where(
            ~np.isnan(prev_close),
            o - prev_close,
            np.nan,
        )
        # Avoid division by zero: if daily_ret ≈ 0, impact is undefined → 0
        gap_impact = np.where(
            np.abs(daily_ret) > eps,
            gap / daily_ret,
            0.0,
        )
        gap_impact = np.where(np.isnan(gap_impact), 0.0, gap_impact)
        gap_impact = np.clip(gap_impact, -2.0, 2.0)

        # ── Assign features ────────────────────────────────────────────────
        result["imom_first_30min"] = first_30
        result["imom_last_60min"] = last_60
        result["imom_midday_reversal"] = midday_rev
        result["imom_overnight_gap_impact"] = gap_impact

        # Fill any residual NaN with 0
        imom_cols = IMOM_FEATURES
        result[imom_cols] = result[imom_cols].fillna(0.0)

        print(f"  [INTRADAY_MOMENTUM] Added {len(imom_cols)} intraday momentum features")
        return result

    # ─── Analysis ────────────────────────────────────────────────────────────

    def analyze_current_momentum(
        self,
        df: pd.DataFrame,
    ) -> Optional[Dict]:
        """
        Summarize the current intraday momentum state for dashboard display.

        Requires that ``create_intraday_momentum_features`` has already been
        called on ``df`` (i.e. the ``imom_*`` columns must be present).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that already contains the ``imom_*`` feature columns.

        Returns
        -------
        dict or None
            Dictionary with current values and a ``momentum_regime`` key
            (one of ``"REVERSAL"``, ``"CONTINUATION"``, or ``"MIXED"``).
            Returns ``None`` if the required feature columns are absent.
        """
        present = [col for col in IMOM_FEATURES if col in df.columns]
        if not present:
            return None

        if df.empty:
            return None

        last = df.iloc[-1]

        result: Dict = {}

        for col in IMOM_FEATURES:
            if col in df.columns:
                result[col] = float(last[col])

        # Determine momentum regime based on available features
        reversal_score = 0.0
        continuation_score = 0.0
        n_signals = 0

        # midday_reversal: negative → reversal, positive → continuation
        if "imom_midday_reversal" in df.columns:
            rev = float(last["imom_midday_reversal"])
            if rev < -0.2:
                reversal_score += 1.0
            elif rev > 0.2:
                continuation_score += 1.0
            n_signals += 1

        # gap_impact near 1 → gap already captured return → reversal during day
        # gap_impact near 0 → intraday session drove return → continuation
        if "imom_overnight_gap_impact" in df.columns:
            gi = float(last["imom_overnight_gap_impact"])
            if gi > 0.7:
                reversal_score += 1.0
            elif gi < 0.3:
                continuation_score += 1.0
            n_signals += 1

        # last_60min: high positive → late buying (continuation)
        # highly negative → late selling (reversal of daytime gains)
        if "imom_last_60min" in df.columns:
            l60 = float(last["imom_last_60min"])
            if l60 > 0.3:
                continuation_score += 1.0
            elif l60 < -0.3:
                reversal_score += 1.0
            n_signals += 1

        if n_signals == 0:
            regime = "MIXED"
        elif reversal_score > continuation_score:
            regime = "REVERSAL"
        elif continuation_score > reversal_score:
            regime = "CONTINUATION"
        else:
            regime = "MIXED"

        result["momentum_regime"] = regime
        result["n_signals"] = n_signals

        return result

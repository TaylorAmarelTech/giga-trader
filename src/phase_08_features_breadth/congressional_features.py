"""
GIGA TRADER - Congressional Trading Features
=============================================
Proxy "smart money" features derived from price momentum and volume patterns.

Since reliable real-time congressional trading disclosure APIs are unavailable
(QuiverQuant, Unusual Whales, etc. require paid subscriptions), this module
constructs a **proxy signal** based on the observation that informed buyers
tend to accumulate on high-volume up days and reduce exposure on high-volume
down days.

The proxy logic:
  - "Smart money buy day":  volume > volume_threshold * 20d_MA  AND  return > 0
  - "Smart money sell day": volume > volume_threshold * 20d_MA  AND  return < 0

This captures the same statistical footprint as informed institutional flow
without requiring external data.

Features generated (prefix: congress_):
  - congress_net_buys_30d:  Net buy ratio over rolling 30 days
                            (buy_days - sell_days) / 30
  - congress_volume_z:      60-day z-score of the net_buys_30d signal
  - congress_buy_ratio:     High-volume up days / total high-volume days (30d)
  - congress_sentiment:     Combined signal: 0.5 * net_buys_norm + 0.5 * buy_ratio
                            Clipped to [-1, 1]
"""

import logging
import warnings
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger("CONGRESSIONAL_FEATURES")

# Columns required by create_congressional_features()
REQUIRED_COLUMNS = {"close", "volume"}


class CongressionalFeatures:
    """
    Proxy "smart money" features based on volume-filtered up/down days.

    The download step is a no-op (proxy-based, no API required).
    All computations rely solely on the close and volume columns already
    present in the main SPY daily DataFrame.

    Parameters
    ----------
    window : int
        Rolling window length in trading days (default 30).
    volume_threshold : float
        Multiplier over the 20-day volume MA that defines a "high-volume" day.
        Default 1.2 (i.e. >120% of the 20-day average volume).
    """

    FEATURE_COLS = [
        "congress_net_buys_30d",
        "congress_volume_z",
        "congress_buy_ratio",
        "congress_sentiment",
    ]

    def __init__(
        self,
        window: int = 30,
        volume_threshold: float = 1.2,
    ) -> None:
        self.window = window
        self.volume_threshold = volume_threshold
        # Kept for API symmetry with other feature classes; always empty.
        self.data: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    # Download (no-op — proxy only)
    # ------------------------------------------------------------------

    def download_congressional_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        No-op stub for API symmetry.

        Congressional trading data is approximated from price/volume; no
        external download is needed.  Always returns an empty DataFrame.
        """
        logger.info(
            "[CONGRESSIONAL] Proxy mode — no external download required. "
            "Features derived from close/volume."
        )
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Feature creation
    # ------------------------------------------------------------------

    def create_congressional_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute proxy congressional / smart-money features and append to df.

        Returns df unchanged if required columns (close, volume) are absent.

        Parameters
        ----------
        df : pd.DataFrame
            SPY daily DataFrame.  Must contain at least 'close' and 'volume'.

        Returns
        -------
        pd.DataFrame
            Copy of df with four new congress_ columns appended and NaN
            values filled with 0.
        """
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            logger.warning(
                "[CONGRESSIONAL] Missing required columns %s — skipping.", missing
            )
            return df

        print("\n[CONGRESSIONAL] Engineering proxy smart-money features...")

        out = df.copy()

        # --- Pre-compute building blocks -----------------------------------

        # Daily return (forward-filling handles any missing close gracefully)
        close = out["close"].astype(float)
        volume = out["volume"].astype(float)

        day_return = close.pct_change()

        # 20-day volume moving average
        vol_ma_20 = volume.rolling(20, min_periods=1).mean()

        # High-volume flag: True when volume exceeds threshold * 20d MA
        high_vol = volume > (self.volume_threshold * vol_ma_20)

        # Smart-money buy day:  high-volume AND positive return
        hv_buy = (high_vol & (day_return > 0)).astype(float)

        # Smart-money sell day: high-volume AND negative return
        hv_sell = (high_vol & (day_return < 0)).astype(float)

        # --- Feature 1: congress_net_buys_30d ------------------------------
        # (buy_days - sell_days) / window  over rolling window
        rolling_buys = hv_buy.rolling(self.window, min_periods=1).sum()
        rolling_sells = hv_sell.rolling(self.window, min_periods=1).sum()
        net_buys = (rolling_buys - rolling_sells) / self.window
        out["congress_net_buys_30d"] = net_buys.fillna(0.0)

        # --- Feature 2: congress_volume_z ----------------------------------
        # 60-day rolling z-score of the net_buys signal
        nb_mean = net_buys.rolling(60, min_periods=10).mean()
        nb_std = net_buys.rolling(60, min_periods=10).std()
        volume_z = (net_buys - nb_mean) / (nb_std + 1e-10)
        out["congress_volume_z"] = volume_z.fillna(0.0)

        # --- Feature 3: congress_buy_ratio ---------------------------------
        # hv_buy days / total hv days (rolling window); 0 if no high-vol days
        total_hv = (hv_buy + hv_sell).rolling(self.window, min_periods=1).sum()
        buy_ratio = rolling_buys / (total_hv + 1e-10)
        # When there are no high-volume days at all, ratio is 0 (not 0/eps)
        buy_ratio = buy_ratio.where(total_hv > 0, other=0.0)
        out["congress_buy_ratio"] = buy_ratio.fillna(0.0)

        # --- Feature 4: congress_sentiment ---------------------------------
        # net_buys is already in [-1, 1] range (bounded by ±1 by definition)
        # buy_ratio is in [0, 1]; normalise to [-1, 1] for symmetric blending
        buy_ratio_norm = buy_ratio * 2.0 - 1.0
        sentiment = 0.5 * net_buys + 0.5 * buy_ratio_norm
        sentiment = sentiment.clip(-1.0, 1.0)
        out["congress_sentiment"] = sentiment.fillna(0.0)

        print(
            f"  [CONGRESSIONAL] Added {len(self.FEATURE_COLS)} proxy "
            f"smart-money features (window={self.window}d, "
            f"vol_threshold={self.volume_threshold}x)"
        )
        return out

    # ------------------------------------------------------------------
    # Dashboard / analysis helper
    # ------------------------------------------------------------------

    def analyze_current_congressional(
        self,
        df: pd.DataFrame,
    ) -> Optional[Dict]:
        """
        Summarise the most recent congressional proxy conditions.

        Parameters
        ----------
        df : pd.DataFrame
            SPY daily DataFrame (with or without pre-computed congress_ cols).

        Returns
        -------
        dict or None
            Dict with keys: congressional_regime, net_buys, buy_ratio,
            sentiment, date.  Returns None if required columns are absent
            or df is empty.
        """
        if df is None or df.empty:
            return None

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            return None

        # Compute features if not already present
        if "congress_net_buys_30d" not in df.columns:
            df = self.create_congressional_features(df)

        if df.empty or "congress_net_buys_30d" not in df.columns:
            return None

        latest = df.iloc[-1]

        net_buys = float(latest.get("congress_net_buys_30d", 0.0))
        buy_ratio = float(latest.get("congress_buy_ratio", 0.0))
        sentiment = float(latest.get("congress_sentiment", 0.0))

        # Regime classification using sentiment composite
        if sentiment >= 0.2:
            regime = "BUYING"
        elif sentiment <= -0.2:
            regime = "SELLING"
        else:
            regime = "NEUTRAL"

        result: Dict = {
            "congressional_regime": regime,
            "net_buys": round(net_buys, 4),
            "buy_ratio": round(buy_ratio, 4),
            "sentiment": round(sentiment, 4),
        }

        # Include date if available
        if "date" in df.columns:
            result["date"] = str(latest["date"])

        return result

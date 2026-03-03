"""
GIGA TRADER - Regime-Aware Stop Loss & Feature Alignment Checker
=================================================================
Dynamic stop-loss and take-profit levels conditioned on ATR and VIX,
replacing static percentage stops.

Also includes FeatureAlignmentChecker to detect when live features
silently misalign with training features.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeAwareStopLoss:
    """ATR/VIX-conditioned dynamic stop-loss and take-profit.

    Instead of static 1% stops, this adapts to current volatility:
    - Low VIX (<15): tighter stops (0.8x ATR mult) — less noise
    - Normal VIX (15-25): standard stops (1.0x)
    - High VIX (25-35): wider stops (1.5x) — more noise to absorb
    - Extreme VIX (>35): widest stops (2.0x) — crisis mode

    Parameters
    ----------
    atr_period : int
        Period for ATR calculation (default 14).
    atr_multiplier_stop : float
        Base ATR multiplier for stop loss (default 1.5).
    atr_multiplier_tp : float
        Base ATR multiplier for take profit (default 2.5).
    vix_scaling : bool
        Whether to scale by VIX regime (default True).
    min_stop_pct : float
        Minimum stop loss percentage (default 0.001 = 0.1%).
    max_stop_pct : float
        Maximum stop loss percentage (default 0.05 = 5%).
    """

    VIX_REGIMES = {
        "LOW_VOL": {"max_vix": 15.0, "scale": 0.8},
        "NORMAL": {"max_vix": 25.0, "scale": 1.0},
        "HIGH_VOL": {"max_vix": 35.0, "scale": 1.5},
        "EXTREME": {"max_vix": float("inf"), "scale": 2.0},
    }

    def __init__(
        self,
        atr_period: int = 14,
        atr_multiplier_stop: float = 1.5,
        atr_multiplier_tp: float = 2.5,
        vix_scaling: bool = True,
        min_stop_pct: float = 0.001,
        max_stop_pct: float = 0.05,
    ):
        self.atr_period = atr_period
        self.atr_multiplier_stop = atr_multiplier_stop
        self.atr_multiplier_tp = atr_multiplier_tp
        self.vix_scaling = vix_scaling
        self.min_stop_pct = min_stop_pct
        self.max_stop_pct = max_stop_pct
        self._atr: Optional[float] = None
        self._atr_series: Optional[pd.Series] = None

    def fit(self, df_daily: pd.DataFrame) -> "RegimeAwareStopLoss":
        """Compute ATR from OHLCV data.

        Parameters
        ----------
        df_daily : pd.DataFrame
            Must contain 'high', 'low', 'close' columns.
        """
        required = {"high", "low", "close"}
        missing = required - set(df_daily.columns)
        if missing:
            logger.warning(f"RegimeAwareStopLoss: missing columns {missing}, using close-only proxy")
            if "close" in df_daily.columns:
                close = df_daily["close"].values
                # Proxy ATR from close-to-close range
                returns = np.abs(np.diff(close) / close[:-1])
                if len(returns) >= self.atr_period:
                    self._atr = float(np.mean(returns[-self.atr_period:]) * close[-1])
                else:
                    self._atr = float(np.std(returns) * close[-1]) if len(returns) > 0 else 1.0
            return self

        high = df_daily["high"].values.astype(float)
        low = df_daily["low"].values.astype(float)
        close = df_daily["close"].values.astype(float)

        # True Range
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # ATR as EMA of TR
        atr_series = np.zeros(len(tr))
        atr_series[: self.atr_period] = np.mean(tr[: self.atr_period])
        alpha = 2.0 / (self.atr_period + 1)
        for i in range(self.atr_period, len(tr)):
            atr_series[i] = alpha * tr[i] + (1 - alpha) * atr_series[i - 1]

        self._atr = float(atr_series[-1]) if len(atr_series) > 0 else 1.0
        self._atr_series = pd.Series(atr_series, index=df_daily.index)
        return self

    def _get_regime(self, vix_estimate: float) -> str:
        """Determine VIX regime."""
        for regime, info in self.VIX_REGIMES.items():
            if vix_estimate < info["max_vix"]:
                return regime
        return "EXTREME"

    def _get_vix_scale(self, vix_estimate: float) -> float:
        """Get VIX scaling factor."""
        regime = self._get_regime(vix_estimate)
        return self.VIX_REGIMES[regime]["scale"]

    def compute_levels(
        self,
        current_price: float,
        direction: str = "LONG",
        vix_estimate: float = 20.0,
    ) -> Dict:
        """Compute dynamic stop-loss and take-profit levels.

        Parameters
        ----------
        current_price : float
            Current asset price.
        direction : str
            "LONG" or "SHORT".
        vix_estimate : float
            Estimated or actual VIX level.

        Returns
        -------
        dict with stop_loss, take_profit, stop_pct, tp_pct, atr, regime.
        """
        atr = self._atr if self._atr is not None else current_price * 0.01
        regime = self._get_regime(vix_estimate)
        vix_scale = self._get_vix_scale(vix_estimate) if self.vix_scaling else 1.0

        stop_distance = atr * self.atr_multiplier_stop * vix_scale
        tp_distance = atr * self.atr_multiplier_tp * vix_scale

        stop_pct = stop_distance / current_price
        tp_pct = tp_distance / current_price

        # Clamp stop percentage
        stop_pct = max(self.min_stop_pct, min(self.max_stop_pct, stop_pct))
        stop_distance = current_price * stop_pct

        if direction.upper() == "LONG":
            stop_loss = current_price - stop_distance
            take_profit = current_price + tp_distance
        else:
            stop_loss = current_price + stop_distance
            take_profit = current_price - tp_distance

        return {
            "stop_loss": round(stop_loss, 4),
            "take_profit": round(take_profit, 4),
            "stop_pct": round(stop_pct, 6),
            "tp_pct": round(tp_pct, 6),
            "atr": round(atr, 4),
            "regime": regime,
            "vix_scale": round(vix_scale, 2),
        }


class FeatureAlignmentChecker:
    """Check that live inference features match training features.

    Alerts when features silently misalign — e.g., a feature module fails
    and its columns get zero-filled without warning.

    Parameters
    ----------
    expected_features : list of str
        Feature names from training.
    tolerance_missing_pct : float
        Fraction of features that can be missing before CRITICAL (default 0.10).
    """

    def __init__(
        self,
        expected_features: List[str],
        tolerance_missing_pct: float = 0.10,
    ):
        self.expected_features = list(expected_features)
        self.tolerance_missing_pct = tolerance_missing_pct

    def check(self, df: pd.DataFrame) -> Dict:
        """Check feature alignment.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with current features.

        Returns
        -------
        dict with aligned, missing, extra, missing_pct, severity, warning.
        """
        current_features = set(df.columns)
        expected_set = set(self.expected_features)

        missing = sorted(expected_set - current_features)
        extra = sorted(current_features - expected_set)
        n_expected = len(self.expected_features) if self.expected_features else 1
        missing_pct = len(missing) / n_expected

        # Determine severity
        if missing_pct == 0:
            severity = "OK"
            warning = None
        elif missing_pct <= 0.05:
            severity = "OK"
            warning = f"{len(missing)} features missing ({missing_pct:.1%}), within tolerance"
        elif missing_pct <= self.tolerance_missing_pct:
            severity = "WARNING"
            warning = f"{len(missing)} features missing ({missing_pct:.1%}): {missing[:5]}..."
        else:
            severity = "CRITICAL"
            warning = (
                f"{len(missing)} features missing ({missing_pct:.1%}), "
                f"exceeds {self.tolerance_missing_pct:.0%} tolerance: {missing[:10]}..."
            )

        if severity != "OK":
            log_fn = logger.warning if severity == "WARNING" else logger.error
            log_fn(f"FeatureAlignment [{severity}]: {warning}")

        # Check for zero-filled features (columns that exist but are all zero)
        zero_filled = []
        for feat in self.expected_features:
            if feat in df.columns:
                col = df[feat]
                if col.dtype in (np.float64, np.float32, np.int64, np.int32, float, int):
                    if (col == 0).all() and len(col) > 10:
                        zero_filled.append(feat)

        if zero_filled:
            logger.warning(
                f"FeatureAlignment: {len(zero_filled)} features are all-zero "
                f"(possibly failed silently): {zero_filled[:5]}..."
            )

        return {
            "aligned": len(missing) == 0,
            "missing": missing,
            "extra": extra,
            "missing_pct": round(missing_pct, 4),
            "zero_filled": zero_filled,
            "severity": severity,
            "warning": warning,
        }

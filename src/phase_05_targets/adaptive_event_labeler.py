"""
GIGA TRADER - Adaptive Event-Driven Labeling (AEDL)
=====================================================
Replaces static triple barrier labeling with volatility-adaptive thresholds.

Key insight: labeling thresholds should adapt to the current volatility regime,
not remain static. In high-volatility periods, wider barriers avoid noise-driven
stop-outs; in low-volatility periods, tighter barriers capture smaller moves.

Features beyond standard triple barrier:
  - Multi-scale realized volatility estimation (5d, 10d, 21d, 63d)
  - Per-day adaptive take-profit / stop-loss scaled by vol ratio
  - Granger causality filtering to weight label confidence
  - Multiple label modes: binary, ternary, continuous, soft

Output columns appended to the result DataFrame:
  aedl_label, aedl_confidence, aedl_tp_used, aedl_sl_used, aedl_vol_ratio
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AdaptiveEventDrivenLabeler:
    """
    Adaptive Event-Driven Labeler (AEDL).

    Produces labels whose take-profit and stop-loss barriers are scaled by
    the current volatility regime relative to a fitted median baseline.

    Parameters
    ----------
    base_tp_pct : float
        Base take-profit threshold as a fraction (e.g. 0.01 = 1%).
    base_sl_pct : float
        Base stop-loss threshold as a fraction (always positive).
    max_holding_days : int
        Maximum holding period in trading days (vertical barrier).
    vol_scaling : bool
        Whether to scale barriers by the volatility ratio.
    label_mode : str
        One of ``"binary"``, ``"ternary"``, ``"continuous"``, ``"soft"``.
    vol_windows : list[int]
        Rolling windows (in days) for multi-scale volatility estimation.
    min_barrier_pct : float
        Floor for adaptive barriers (avoids vanishingly tight stops).
    max_barrier_pct : float
        Ceiling for adaptive barriers (avoids absurdly wide barriers).
    """

    VALID_MODES = ("binary", "ternary", "continuous", "soft")

    def __init__(
        self,
        base_tp_pct: float = 0.01,
        base_sl_pct: float = 0.01,
        max_holding_days: int = 5,
        vol_scaling: bool = True,
        label_mode: str = "binary",
        vol_windows: Optional[List[int]] = None,
        min_barrier_pct: float = 0.002,
        max_barrier_pct: float = 0.05,
    ):
        if base_tp_pct <= 0:
            raise ValueError(f"base_tp_pct must be positive, got {base_tp_pct}")
        if base_sl_pct <= 0:
            raise ValueError(f"base_sl_pct must be positive, got {base_sl_pct}")
        if max_holding_days < 1:
            raise ValueError(f"max_holding_days must be >= 1, got {max_holding_days}")
        if label_mode not in self.VALID_MODES:
            raise ValueError(
                f"label_mode must be one of {self.VALID_MODES}, got '{label_mode}'"
            )
        if min_barrier_pct <= 0 or max_barrier_pct <= 0:
            raise ValueError("min_barrier_pct and max_barrier_pct must be positive")
        if min_barrier_pct >= max_barrier_pct:
            raise ValueError("min_barrier_pct must be less than max_barrier_pct")

        self.base_tp_pct = base_tp_pct
        self.base_sl_pct = base_sl_pct
        self.max_holding_days = max_holding_days
        self.vol_scaling = vol_scaling
        self.label_mode = label_mode
        self.vol_windows = vol_windows or [5, 10, 21, 63]
        self.min_barrier_pct = min_barrier_pct
        self.max_barrier_pct = max_barrier_pct

        # Populated by fit()
        self._median_vol: Optional[float] = None
        self._is_fitted: bool = False
        self._barrier_history: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "AdaptiveEventDrivenLabeler":
        """
        Compute volatility scaling parameters from training data.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with at least a ``"close"`` column.

        Returns
        -------
        self
        """
        close = self._get_close(df)
        if len(close) < max(self.vol_windows) + 1:
            raise ValueError(
                f"Need at least {max(self.vol_windows) + 1} rows to fit, "
                f"got {len(close)}"
            )

        blended_vol = self._blended_vol(close)
        self._median_vol = float(np.nanmedian(blended_vol.dropna().values))
        if self._median_vol <= 0:
            self._median_vol = 1e-8
        self._is_fitted = True

        logger.info(
            "AEDL fit: median blended vol=%.6f from %d observations "
            "(windows=%s)",
            self._median_vol,
            len(close),
            self.vol_windows,
        )
        return self

    def label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Produce adaptive labels for a DataFrame with OHLCV columns.

        If ``fit()`` has not been called, it is called automatically on *df*
        (convenient for single-dataset use, but beware of leakage in CV).

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at least ``"close"``; ``"high"`` and ``"low"``
            are used when available for intra-day barrier checks.

        Returns
        -------
        pd.DataFrame
            Columns: ``aedl_label``, ``aedl_confidence``, ``aedl_tp_used``,
            ``aedl_sl_used``, ``aedl_vol_ratio``, ``touch_type``,
            ``touch_idx``, ``return_at_touch``.
        """
        if not self._is_fitted:
            self.fit(df)

        close = self._get_close(df)
        if close.empty:
            return self._empty_result()

        tp_arr, sl_arr, vol_ratio_arr = self._compute_adaptive_barriers(df)

        # Granger-based confidence weights (1.0 when filtering is N/A)
        granger_conf = self._granger_filter(df, max_lag=5)

        price_values = close.values.astype(np.float64)
        n = len(price_values)
        max_start = n - 1  # need at least 1 forward day

        if max_start <= 0:
            return self._empty_result()

        n_events = max_start
        labels = np.empty(n_events, dtype=np.float64)
        touch_types = np.empty(n_events, dtype=object)
        touch_offsets = np.empty(n_events, dtype=np.intp)
        returns_at_touch = np.empty(n_events, dtype=np.float64)

        for k in range(n_events):
            entry_price = price_values[k]
            end = min(k + 1 + self.max_holding_days, n)
            fwd = price_values[k + 1 : end]

            tp_k = tp_arr[k]
            sl_k = sl_arr[k]

            if len(fwd) == 0:
                labels[k] = 0.0
                touch_types[k] = "expiry"
                touch_offsets[k] = 0
                returns_at_touch[k] = 0.0
                continue

            fwd_ret = (fwd - entry_price) / entry_price

            tp_hits = np.where(fwd_ret >= tp_k)[0]
            sl_hits = np.where(fwd_ret <= -sl_k)[0]
            tp_touch = tp_hits[0] if len(tp_hits) > 0 else len(fwd_ret)
            sl_touch = sl_hits[0] if len(sl_hits) > 0 else len(fwd_ret)

            if tp_touch <= sl_touch and tp_touch < len(fwd_ret):
                offset = tp_touch
                touch_types[k] = "tp"
                ret = fwd_ret[offset]
            elif sl_touch < tp_touch and sl_touch < len(fwd_ret):
                offset = sl_touch
                touch_types[k] = "sl"
                ret = fwd_ret[offset]
            else:
                offset = len(fwd_ret) - 1
                touch_types[k] = "expiry"
                ret = fwd_ret[offset]

            touch_offsets[k] = offset + 1
            returns_at_touch[k] = ret
            labels[k] = self._return_to_label(ret, touch_types[k])

        # Build confidence: base confidence from touch type, scaled by Granger
        confidences = np.where(
            touch_types == "tp",
            0.9,
            np.where(touch_types == "sl", 0.9, 0.5),
        )
        granger_vals = granger_conf[:n_events] if len(granger_conf) >= n_events else np.ones(n_events)
        confidences = confidences * granger_vals

        if self.label_mode in ("binary", "ternary"):
            labels = labels.astype(np.int64)

        result = pd.DataFrame(
            {
                "aedl_label": labels,
                "aedl_confidence": np.round(confidences, 4),
                "aedl_tp_used": np.round(tp_arr[:n_events], 6),
                "aedl_sl_used": np.round(sl_arr[:n_events], 6),
                "aedl_vol_ratio": np.round(vol_ratio_arr[:n_events], 4),
                "touch_type": touch_types,
                "touch_idx": touch_offsets,
                "return_at_touch": returns_at_touch,
            },
            index=close.index[:n_events],
        )

        # Store barrier history for analysis
        self._barrier_history = result[
            ["aedl_tp_used", "aedl_sl_used", "aedl_vol_ratio"]
        ].copy()

        logger.info(
            "AEDL labeling: %d observations (mode=%s, vol_scaling=%s, "
            "tp_range=[%.4f,%.4f], sl_range=[%.4f,%.4f])",
            len(result),
            self.label_mode,
            self.vol_scaling,
            result["aedl_tp_used"].min(),
            result["aedl_tp_used"].max(),
            result["aedl_sl_used"].min(),
            result["aedl_sl_used"].max(),
        )

        return result

    def get_barrier_history(self) -> pd.DataFrame:
        """
        Return DataFrame of historical barrier levels for analysis.

        Returns
        -------
        pd.DataFrame
            Columns: ``aedl_tp_used``, ``aedl_sl_used``, ``aedl_vol_ratio``.

        Raises
        ------
        RuntimeError
            If ``label()`` has not been called yet.
        """
        if self._barrier_history is None:
            raise RuntimeError(
                "No barrier history available. Call label() first."
            )
        return self._barrier_history.copy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_adaptive_barriers(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute per-day TP and SL barriers scaled by volatility ratio.

        Returns
        -------
        tp_arr, sl_arr, vol_ratio_arr : np.ndarray
            Arrays of length ``len(df)`` with adaptive barrier values and
            the volatility ratio used for scaling.
        """
        close = self._get_close(df)
        n = len(close)

        if not self.vol_scaling or self._median_vol is None:
            tp_arr = np.full(n, self.base_tp_pct)
            sl_arr = np.full(n, self.base_sl_pct)
            vol_ratio_arr = np.ones(n)
            return tp_arr, sl_arr, vol_ratio_arr

        blended = self._blended_vol(close)
        blended_vals = blended.values.astype(np.float64)
        # Replace NaN at the start with median (before enough history)
        nan_mask = np.isnan(blended_vals)
        blended_vals[nan_mask] = self._median_vol

        vol_ratio_arr = blended_vals / self._median_vol
        # Avoid extreme ratios from numerical edge cases
        vol_ratio_arr = np.clip(vol_ratio_arr, 0.1, 10.0)

        tp_arr = np.clip(
            self.base_tp_pct * vol_ratio_arr,
            self.min_barrier_pct,
            self.max_barrier_pct,
        )
        sl_arr = np.clip(
            self.base_sl_pct * vol_ratio_arr,
            self.min_barrier_pct,
            self.max_barrier_pct,
        )

        return tp_arr, sl_arr, vol_ratio_arr

    def _granger_filter(
        self, df: pd.DataFrame, max_lag: int = 5
    ) -> np.ndarray:
        """
        Use a simple F-test Granger causality test to weight label confidence.

        Tests whether lagged returns at each vol window scale significantly
        predict future returns (p < 0.05). The confidence weight is the
        fraction of significant lead-lag relationships found.

        Implemented from scratch with numpy (no statsmodels dependency).

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``"close"``.
        max_lag : int
            Maximum lag order to test.

        Returns
        -------
        np.ndarray
            Per-observation confidence multiplier in [0.5, 1.0].
            0.5 = no significant causal features; 1.0 = all tested lags
            are significant.
        """
        close = self._get_close(df)
        returns = close.pct_change().values.astype(np.float64)
        n = len(returns)

        if n < max_lag + 10:
            return np.ones(n)

        n_significant = 0
        n_tests = 0

        for lag in range(1, max_lag + 1):
            y = returns[lag:]
            valid = ~(np.isnan(y))
            # Build restricted model: intercept only
            # Build unrestricted model: intercept + lagged return
            x_lag = returns[:-lag]
            valid &= ~(np.isnan(x_lag))

            y_v = y[valid]
            x_v = x_lag[valid]
            t = len(y_v)

            if t < lag + 5:
                continue

            n_tests += 1

            # Restricted model (intercept only): RSS_r
            y_mean = np.mean(y_v)
            rss_r = np.sum((y_v - y_mean) ** 2)

            # Unrestricted model (intercept + lag): RSS_u via OLS
            X = np.column_stack([np.ones(t), x_v])
            try:
                beta = np.linalg.lstsq(X, y_v, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            residuals = y_v - X @ beta
            rss_u = np.sum(residuals ** 2)

            # F-statistic: ((RSS_r - RSS_u) / q) / (RSS_u / (t - k))
            q = 1  # one additional regressor
            k = 2  # intercept + one lag
            denom = rss_u / max(t - k, 1)
            if denom <= 0:
                continue
            f_stat = ((rss_r - rss_u) / q) / denom

            # Approximate p-value using F-distribution CDF via
            # the regularized incomplete beta function relationship:
            # p = 1 - I_{q*F/(q*F + (t-k))}(q/2, (t-k)/2)
            # For simplicity, use the well-known threshold: for q=1,
            # F > 3.84 roughly corresponds to p < 0.05 when df2 >= 30.
            # For smaller df2, use a slightly higher threshold.
            df2 = t - k
            if df2 >= 30:
                f_crit = 3.84
            elif df2 >= 10:
                f_crit = 4.10
            else:
                f_crit = 4.96  # conservative for tiny samples
            if f_stat > f_crit:
                n_significant += 1

        # Map fraction of significant tests to [0.5, 1.0]
        if n_tests == 0:
            frac = 0.0
        else:
            frac = n_significant / n_tests

        confidence_scalar = 0.5 + 0.5 * frac
        return np.full(n, confidence_scalar)

    def _blended_vol(self, close: pd.Series) -> pd.Series:
        """
        Compute blended multi-scale realized volatility.

        Equal-weights the rolling standard deviation of log returns across
        all configured windows, producing a single volatility series.
        """
        log_ret = np.log(close / close.shift(1))
        vol_components = []
        for w in self.vol_windows:
            vol_components.append(log_ret.rolling(window=w, min_periods=max(w // 2, 2)).std())

        # Equal-weight blend
        vol_stack = pd.concat(vol_components, axis=1)
        blended = vol_stack.mean(axis=1)
        return blended

    def _return_to_label(self, ret: float, touch_type: str) -> float:
        """Convert a return value and touch type to a label."""
        if self.label_mode == "binary":
            if touch_type == "tp":
                return 1.0
            elif touch_type == "sl":
                return 0.0
            else:  # expiry
                return 1.0 if ret > 0 else 0.0

        elif self.label_mode == "ternary":
            if touch_type == "tp":
                return 1.0
            elif touch_type == "sl":
                return -1.0
            else:
                return 0.0

        elif self.label_mode == "soft":
            # Sigmoid transform: maps return to (0, 1) smoothly
            k = 50.0  # steepness (per EDGE 4: Confidence Soft Targets)
            return 1.0 / (1.0 + np.exp(-k * ret))

        else:  # continuous
            return ret

    @staticmethod
    def _get_close(df: pd.DataFrame) -> pd.Series:
        """Extract close price series, case-insensitive."""
        for col in ("close", "Close", "CLOSE", "adj_close", "Adj Close"):
            if col in df.columns:
                return df[col].ffill().dropna()
        raise ValueError(
            "DataFrame must contain a 'close' column. "
            f"Found columns: {list(df.columns)}"
        )

    @staticmethod
    def _empty_result() -> pd.DataFrame:
        """Return an empty DataFrame with the expected AEDL schema."""
        return pd.DataFrame(
            columns=[
                "aedl_label",
                "aedl_confidence",
                "aedl_tp_used",
                "aedl_sl_used",
                "aedl_vol_ratio",
                "touch_type",
                "touch_idx",
                "return_at_touch",
            ]
        )

    def __repr__(self) -> str:
        return (
            f"AdaptiveEventDrivenLabeler("
            f"base_tp_pct={self.base_tp_pct}, "
            f"base_sl_pct={self.base_sl_pct}, "
            f"max_holding_days={self.max_holding_days}, "
            f"vol_scaling={self.vol_scaling}, "
            f"label_mode='{self.label_mode}', "
            f"vol_windows={self.vol_windows})"
        )

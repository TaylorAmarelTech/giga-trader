"""
GIGA TRADER - Triple Barrier Labeling
=======================================
Implements the triple barrier method from Lopez de Prado,
"Advances in Financial Machine Learning", Chapter 3.

For each observation, three barriers are defined:
  1. Upper barrier: Take-profit (price rises tp_pct from entry)
  2. Lower barrier: Stop-loss (price drops sl_pct from entry)
  3. Vertical barrier: Time expiry (max_holding_days reached)

The label is determined by which barrier is touched first.

This is a superior alternative to fixed-horizon labeling because:
  - It captures the path-dependency of returns
  - It naturally handles different volatility regimes
  - It creates labels that are more aligned with actual trading
  - It avoids labeling choppy, inconclusive periods as strong signals
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TripleBarrierLabeler:
    """
    Triple barrier labeling (Lopez de Prado, AFML Ch. 3).

    For each observation, three barriers are defined:
      1. Upper barrier: Take-profit (price rises tp_pct from entry)
      2. Lower barrier: Stop-loss (price drops sl_pct from entry)
      3. Vertical barrier: Time expiry (max_holding_days reached)

    The label is determined by which barrier is touched first.

    Parameters
    ----------
    tp_pct : float
        Take-profit threshold as fraction (0.01 = 1%).
    sl_pct : float
        Stop-loss threshold as fraction (0.01 = 1%). Always positive;
        internally used as a negative return threshold (-sl_pct).
    max_holding_days : int
        Maximum holding period in trading days.
    label_mode : str
        How to generate labels:
        - "binary": 1 (hit TP first or positive at expiry),
                    0 (hit SL first or negative/zero at expiry)
        - "ternary": 1 (TP), -1 (SL), 0 (time expiry)
        - "continuous": Actual return at the first touch point
    """

    VALID_MODES = ("binary", "ternary", "continuous")

    def __init__(
        self,
        tp_pct: float = 0.01,
        sl_pct: float = 0.01,
        max_holding_days: int = 5,
        label_mode: str = "binary",
    ):
        if tp_pct <= 0:
            raise ValueError(f"tp_pct must be positive, got {tp_pct}")
        if sl_pct <= 0:
            raise ValueError(f"sl_pct must be positive, got {sl_pct}")
        if max_holding_days < 1:
            raise ValueError(
                f"max_holding_days must be >= 1, got {max_holding_days}"
            )
        if label_mode not in self.VALID_MODES:
            raise ValueError(
                f"label_mode must be one of {self.VALID_MODES}, got '{label_mode}'"
            )

        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.max_holding_days = max_holding_days
        self.label_mode = label_mode

    def label(
        self,
        prices: pd.Series,
        events: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        """
        Generate triple barrier labels for a price series.

        Parameters
        ----------
        prices : pd.Series
            Close price series with DatetimeIndex or integer index.
        events : pd.DatetimeIndex or Index, optional
            Specific dates/indices to label. If None, label all dates
            except the last max_holding_days which lack a full forward window.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed like events (or prices minus tail), with columns:
            - "label": int or float depending on label_mode
            - "touch_type": str ("tp", "sl", "expiry")
            - "touch_idx": int (offset from entry where barrier was touched)
            - "return_at_touch": float (actual return at touch point)
            - "entry_price": float
            - "touch_price": float

        Algorithm
        ---------
        For each entry point i:
          1. Forward window: prices[i+1 : i+1+max_holding_days]
          2. Returns: (prices_forward - prices[i]) / prices[i]
          3. TP touch: first index where return >= tp_pct
          4. SL touch: first index where return <= -sl_pct
          5. Touch = min(TP touch, SL touch, max_holding_days)
          6. Label based on which barrier was touched first
        """
        if prices.empty:
            return self._empty_result()

        # Clean NaN prices: forward-fill then drop any remaining leading NaNs
        prices = prices.ffill().dropna()
        if prices.empty:
            return self._empty_result()

        price_values = prices.values.astype(np.float64)
        n = len(price_values)

        # Determine which indices to label
        if events is not None:
            # Map event dates/labels to integer positions in the price series
            if hasattr(prices.index, 'get_indexer'):
                event_positions = prices.index.get_indexer(events)
                # Filter out events that didn't match (-1) and those too close to end
                valid_mask = (event_positions >= 0) & (
                    event_positions < n - 1
                )
                event_positions = event_positions[valid_mask]
                event_indices = events[valid_mask]
            else:
                event_positions = np.array(
                    [i for i in range(n) if prices.index[i] in events],
                    dtype=np.intp,
                )
                event_indices = prices.index[event_positions]
        else:
            # Label all dates except those without enough forward data
            # We need at least 1 day forward; max_holding_days is the ideal window
            max_start = n - 1  # Must have at least 1 forward day
            event_positions = np.arange(max_start, dtype=np.intp)
            event_indices = prices.index[:max_start]

        if len(event_positions) == 0:
            return self._empty_result()

        # Vectorized barrier computation
        labels, touch_types, touch_offsets, returns_at_touch, entry_prices, touch_prices = (
            self._compute_barriers_vectorized(
                price_values, event_positions, self.tp_pct, self.sl_pct,
                self.max_holding_days, self.label_mode,
            )
        )

        result = pd.DataFrame(
            {
                "label": labels,
                "touch_type": touch_types,
                "touch_idx": touch_offsets,
                "return_at_touch": returns_at_touch,
                "entry_price": entry_prices,
                "touch_price": touch_prices,
            },
            index=event_indices,
        )

        logger.info(
            "Triple barrier labeling: %d observations labeled "
            "(tp=%.4f, sl=%.4f, hold=%d, mode=%s)",
            len(result),
            self.tp_pct,
            self.sl_pct,
            self.max_holding_days,
            self.label_mode,
        )

        return result

    @staticmethod
    def _compute_barriers_vectorized(
        price_values: np.ndarray,
        event_positions: np.ndarray,
        tp_pct: float,
        sl_pct: float,
        max_holding_days: int,
        label_mode: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute barrier touches for all events using vectorized operations.

        Uses numpy broadcasting to compute forward returns for all events
        simultaneously, then finds first exceedances.

        Returns
        -------
        Tuple of (labels, touch_types, touch_offsets, returns_at_touch,
                  entry_prices, touch_prices)
        """
        n_prices = len(price_values)
        n_events = len(event_positions)

        labels = np.empty(n_events, dtype=np.float64)
        touch_offsets = np.empty(n_events, dtype=np.intp)
        returns_at_touch = np.empty(n_events, dtype=np.float64)
        entry_prices = np.empty(n_events, dtype=np.float64)
        touch_prices = np.empty(n_events, dtype=np.float64)
        touch_types = np.empty(n_events, dtype=object)

        for k in range(n_events):
            pos = event_positions[k]
            entry_price = price_values[pos]
            entry_prices[k] = entry_price

            # Forward window: from pos+1 up to pos+1+max_holding_days
            end = min(pos + 1 + max_holding_days, n_prices)
            forward_prices = price_values[pos + 1: end]

            if len(forward_prices) == 0:
                # No forward data at all; treat as expiry with zero return
                touch_types[k] = "expiry"
                touch_offsets[k] = 0
                returns_at_touch[k] = 0.0
                touch_prices[k] = entry_price
                if label_mode == "binary":
                    labels[k] = 0
                elif label_mode == "ternary":
                    labels[k] = 0
                else:
                    labels[k] = 0.0
                continue

            # Compute returns relative to entry
            forward_returns = (forward_prices - entry_price) / entry_price

            # Find first TP touch (return >= tp_pct)
            tp_hits = np.where(forward_returns >= tp_pct)[0]
            tp_touch = tp_hits[0] if len(tp_hits) > 0 else len(forward_returns)

            # Find first SL touch (return <= -sl_pct)
            sl_hits = np.where(forward_returns <= -sl_pct)[0]
            sl_touch = sl_hits[0] if len(sl_hits) > 0 else len(forward_returns)

            # Determine which barrier was touched first
            expiry_idx = len(forward_returns) - 1  # Last available index

            if tp_touch <= sl_touch and tp_touch < len(forward_returns):
                # TP hit first
                touch_types[k] = "tp"
                offset = tp_touch
                touch_offsets[k] = offset + 1  # 1-based offset from entry
                returns_at_touch[k] = forward_returns[offset]
                touch_prices[k] = forward_prices[offset]
                if label_mode == "binary":
                    labels[k] = 1
                elif label_mode == "ternary":
                    labels[k] = 1
                else:
                    labels[k] = forward_returns[offset]

            elif sl_touch < tp_touch and sl_touch < len(forward_returns):
                # SL hit first
                touch_types[k] = "sl"
                offset = sl_touch
                touch_offsets[k] = offset + 1
                returns_at_touch[k] = forward_returns[offset]
                touch_prices[k] = forward_prices[offset]
                if label_mode == "binary":
                    labels[k] = 0
                elif label_mode == "ternary":
                    labels[k] = -1
                else:
                    labels[k] = forward_returns[offset]

            else:
                # Time expiry: neither TP nor SL touched
                touch_types[k] = "expiry"
                touch_offsets[k] = expiry_idx + 1
                returns_at_touch[k] = forward_returns[expiry_idx]
                touch_prices[k] = forward_prices[expiry_idx]
                if label_mode == "binary":
                    labels[k] = 1 if forward_returns[expiry_idx] > 0 else 0
                elif label_mode == "ternary":
                    labels[k] = 0
                else:
                    labels[k] = forward_returns[expiry_idx]

        # Cast label dtype based on mode
        if label_mode in ("binary", "ternary"):
            labels = labels.astype(np.int64)

        return labels, touch_types, touch_offsets, returns_at_touch, entry_prices, touch_prices

    def label_with_volatility(
        self,
        prices: pd.Series,
        vol_series: pd.Series,
        vol_multiplier: float = 1.0,
    ) -> pd.DataFrame:
        """
        Dynamic barriers scaled by recent volatility.

        The base tp_pct and sl_pct are multiplied by
        ``vol_series * vol_multiplier`` for each observation, making barriers
        wider in high-volatility regimes and tighter in low-volatility regimes.

        Parameters
        ----------
        prices : pd.Series
            Close price series.
        vol_series : pd.Series
            Volatility series aligned with prices (e.g., rolling std of returns).
            Must have the same index as prices.
        vol_multiplier : float
            Scaling factor applied to vol_series before multiplying barriers.
            Default 1.0 (barriers = base * vol).

        Returns
        -------
        pd.DataFrame
            Same format as ``label()``.
        """
        if prices.empty:
            return self._empty_result()

        prices = prices.ffill().dropna()
        if prices.empty:
            return self._empty_result()

        # Align vol_series to prices index; fill missing with median
        vol_aligned = vol_series.reindex(prices.index)
        median_vol = vol_aligned.median()
        if pd.isna(median_vol) or median_vol == 0:
            median_vol = 1.0
        vol_aligned = vol_aligned.fillna(median_vol)

        # Ensure vol values are positive
        vol_values = np.maximum(vol_aligned.values.astype(np.float64), 1e-8)

        price_values = prices.values.astype(np.float64)
        n = len(price_values)

        # Label all dates except those without forward data
        max_start = n - 1
        event_positions = np.arange(max_start, dtype=np.intp)
        event_indices = prices.index[:max_start]

        if len(event_positions) == 0:
            return self._empty_result()

        # Per-event dynamic barriers
        n_events = len(event_positions)
        labels = np.empty(n_events, dtype=np.float64)
        touch_offsets = np.empty(n_events, dtype=np.intp)
        returns_at_touch = np.empty(n_events, dtype=np.float64)
        entry_prices = np.empty(n_events, dtype=np.float64)
        touch_prices = np.empty(n_events, dtype=np.float64)
        touch_types = np.empty(n_events, dtype=object)

        for k in range(n_events):
            pos = event_positions[k]
            entry_price = price_values[pos]
            entry_prices[k] = entry_price

            # Dynamic barriers for this observation
            vol_scale = vol_values[pos] * vol_multiplier
            dynamic_tp = self.tp_pct * vol_scale
            dynamic_sl = self.sl_pct * vol_scale

            # Forward window
            end = min(pos + 1 + self.max_holding_days, n)
            forward_prices = price_values[pos + 1: end]

            if len(forward_prices) == 0:
                touch_types[k] = "expiry"
                touch_offsets[k] = 0
                returns_at_touch[k] = 0.0
                touch_prices[k] = entry_price
                if self.label_mode == "binary":
                    labels[k] = 0
                elif self.label_mode == "ternary":
                    labels[k] = 0
                else:
                    labels[k] = 0.0
                continue

            forward_returns = (forward_prices - entry_price) / entry_price

            tp_hits = np.where(forward_returns >= dynamic_tp)[0]
            tp_touch = tp_hits[0] if len(tp_hits) > 0 else len(forward_returns)

            sl_hits = np.where(forward_returns <= -dynamic_sl)[0]
            sl_touch = sl_hits[0] if len(sl_hits) > 0 else len(forward_returns)

            expiry_idx = len(forward_returns) - 1

            if tp_touch <= sl_touch and tp_touch < len(forward_returns):
                touch_types[k] = "tp"
                offset = tp_touch
                touch_offsets[k] = offset + 1
                returns_at_touch[k] = forward_returns[offset]
                touch_prices[k] = forward_prices[offset]
                if self.label_mode == "binary":
                    labels[k] = 1
                elif self.label_mode == "ternary":
                    labels[k] = 1
                else:
                    labels[k] = forward_returns[offset]

            elif sl_touch < tp_touch and sl_touch < len(forward_returns):
                touch_types[k] = "sl"
                offset = sl_touch
                touch_offsets[k] = offset + 1
                returns_at_touch[k] = forward_returns[offset]
                touch_prices[k] = forward_prices[offset]
                if self.label_mode == "binary":
                    labels[k] = 0
                elif self.label_mode == "ternary":
                    labels[k] = -1
                else:
                    labels[k] = forward_returns[offset]

            else:
                touch_types[k] = "expiry"
                touch_offsets[k] = expiry_idx + 1
                returns_at_touch[k] = forward_returns[expiry_idx]
                touch_prices[k] = forward_prices[expiry_idx]
                if self.label_mode == "binary":
                    labels[k] = 1 if forward_returns[expiry_idx] > 0 else 0
                elif self.label_mode == "ternary":
                    labels[k] = 0
                else:
                    labels[k] = forward_returns[expiry_idx]

        if self.label_mode in ("binary", "ternary"):
            labels = labels.astype(np.int64)

        result = pd.DataFrame(
            {
                "label": labels,
                "touch_type": touch_types,
                "touch_idx": touch_offsets,
                "return_at_touch": returns_at_touch,
                "entry_price": entry_prices,
                "touch_price": touch_prices,
            },
            index=event_indices,
        )

        logger.info(
            "Triple barrier labeling (vol-scaled): %d observations labeled "
            "(base tp=%.4f, base sl=%.4f, hold=%d, vol_mult=%.2f, mode=%s)",
            len(result),
            self.tp_pct,
            self.sl_pct,
            self.max_holding_days,
            vol_multiplier,
            self.label_mode,
        )

        return result

    def get_stats(self, labels_df: pd.DataFrame) -> Dict:
        """
        Compute statistics about the labeling.

        Parameters
        ----------
        labels_df : pd.DataFrame
            Output from ``label()`` or ``label_with_volatility()``.

        Returns
        -------
        dict
            Statistics including counts, percentages, and average returns
            for each touch type.
        """
        if labels_df.empty:
            return {
                "n_tp": 0, "n_sl": 0, "n_expiry": 0,
                "pct_tp": 0.0, "pct_sl": 0.0, "pct_expiry": 0.0,
                "avg_return_tp": 0.0, "avg_return_sl": 0.0,
                "avg_return_expiry": 0.0, "avg_touch_idx": 0.0,
            }

        n = len(labels_df)
        touch_counts = labels_df["touch_type"].value_counts()

        n_tp = int(touch_counts.get("tp", 0))
        n_sl = int(touch_counts.get("sl", 0))
        n_expiry = int(touch_counts.get("expiry", 0))

        def _avg_return_for(ttype: str) -> float:
            subset = labels_df.loc[labels_df["touch_type"] == ttype, "return_at_touch"]
            return float(subset.mean()) if len(subset) > 0 else 0.0

        return {
            "n_tp": n_tp,
            "n_sl": n_sl,
            "n_expiry": n_expiry,
            "pct_tp": n_tp / n if n > 0 else 0.0,
            "pct_sl": n_sl / n if n > 0 else 0.0,
            "pct_expiry": n_expiry / n if n > 0 else 0.0,
            "avg_return_tp": _avg_return_for("tp"),
            "avg_return_sl": _avg_return_for("sl"),
            "avg_return_expiry": _avg_return_for("expiry"),
            "avg_touch_idx": float(labels_df["touch_idx"].mean()),
        }

    @staticmethod
    def _empty_result() -> pd.DataFrame:
        """Return an empty DataFrame with the expected schema."""
        return pd.DataFrame(
            columns=[
                "label", "touch_type", "touch_idx",
                "return_at_touch", "entry_price", "touch_price",
            ]
        )

    def __repr__(self) -> str:
        return (
            f"TripleBarrierLabeler(tp_pct={self.tp_pct}, sl_pct={self.sl_pct}, "
            f"max_holding_days={self.max_holding_days}, label_mode='{self.label_mode}')"
        )

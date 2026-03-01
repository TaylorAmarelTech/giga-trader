"""
GIGA TRADER - Dynamic Kelly Criterion Position Sizer
=====================================================
Kelly criterion conditioned on VIX regime for adaptive position sizing.

The Kelly criterion computes the optimal bet fraction given an edge:
    f* = p - q/b
where p = win probability, q = 1-p, b = win/loss ratio.

Raw Kelly is too aggressive for real trading (maximizes long-run growth
but with extreme drawdowns). This module applies:
  1. Fractional Kelly (never full Kelly -- default half or less)
  2. VIX-conditioned scaling (reduce fraction as volatility rises)
  3. Position bounds clipping (always within [min_position, max_position])

VIX scaling:
  - Low VIX (<15):      0.50x Kelly (calm market, higher confidence)
  - Normal VIX (15-25): 0.35x Kelly (standard conditions)
  - High VIX (25-35):   0.20x Kelly (elevated uncertainty)
  - Extreme VIX (>35):  0.10x Kelly (crisis mode)
"""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DynamicKellySizer:
    """Kelly criterion position sizer conditioned on VIX regime.

    Parameters
    ----------
    vix_bins : list of float, optional
        VIX thresholds separating regimes (default [15, 25, 35]).
    vix_scales : list of float, optional
        Kelly fraction multiplier for each regime (default [0.50, 0.35, 0.20, 0.10]).
        Must have len(vix_bins) + 1 entries.
    min_position : float
        Minimum position size as fraction of portfolio (default 0.01).
    max_position : float
        Maximum position size as fraction of portfolio (default 0.25).
    """

    def __init__(
        self,
        vix_bins: Optional[List[float]] = None,
        vix_scales: Optional[List[float]] = None,
        min_position: float = 0.01,
        max_position: float = 0.25,
    ):
        self.vix_bins = vix_bins or [15.0, 25.0, 35.0]
        self.vix_scales = vix_scales or [0.50, 0.35, 0.20, 0.10]
        self.min_position = min_position
        self.max_position = max_position

        if len(self.vix_scales) != len(self.vix_bins) + 1:
            raise ValueError(
                f"vix_scales must have len(vix_bins)+1 entries, "
                f"got {len(self.vix_scales)} scales for {len(self.vix_bins)} bins"
            )

        self._fitted = False
        self._estimated_edge: Optional[float] = None
        self._estimated_odds: Optional[float] = None

    def fit(
        self,
        returns: np.ndarray,
        win_rate: Optional[float] = None,
    ) -> "DynamicKellySizer":
        """Estimate Kelly parameters from historical returns.

        Parameters
        ----------
        returns : np.ndarray
            1-D array of historical daily returns.
        win_rate : float, optional
            Override the empirical win rate. If None, computed from returns.

        Returns
        -------
        self
            For method chaining.
        """
        returns = np.asarray(returns, dtype=float).ravel()
        returns = returns[~np.isnan(returns)]

        if len(returns) < 20:
            logger.warning(
                "DynamicKellySizer.fit: fewer than 20 returns — "
                "using neutral estimates."
            )
            self._estimated_edge = 0.0
            self._estimated_odds = 1.0
            self._fitted = True
            return self

        wins = returns[returns > 0]
        losses = returns[returns <= 0]

        if win_rate is None:
            win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.5

        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.001
        avg_loss = float(np.abs(np.mean(losses))) if len(losses) > 0 else 0.001

        self._estimated_edge = win_rate
        self._estimated_odds = avg_win / (avg_loss + 1e-10)
        self._fitted = True

        logger.info(
            "DynamicKellySizer fitted: win_rate=%.3f, odds=%.3f, "
            "n_returns=%d",
            win_rate,
            self._estimated_odds,
            len(returns),
        )
        return self

    def _kelly_fraction(self, win_prob: float, win_loss_ratio: float) -> float:
        """Compute raw Kelly fraction.

        f* = p - q/b

        where p = win probability, q = 1 - p, b = win/loss ratio.

        Returns
        -------
        float
            Raw Kelly fraction, floored at 0.0 (never go negative / short).
        """
        q = 1.0 - win_prob
        if win_loss_ratio <= 0:
            return 0.0
        kelly = win_prob - q / win_loss_ratio
        return max(0.0, kelly)

    def _vix_scale(self, vix_level: float) -> float:
        """Get VIX-conditioned scale factor.

        Parameters
        ----------
        vix_level : float
            Current VIX level.

        Returns
        -------
        float
            Fractional Kelly multiplier for the current VIX regime.
        """
        for i, threshold in enumerate(self.vix_bins):
            if vix_level < threshold:
                return self.vix_scales[i]
        return self.vix_scales[-1]  # Above all thresholds

    def size(
        self,
        win_probability: float,
        vix_level: float,
        win_loss_ratio: float = 1.5,
    ) -> float:
        """Compute VIX-conditioned Kelly position size.

        Parameters
        ----------
        win_probability : float
            Estimated probability of a winning trade (0 to 1).
        vix_level : float
            Current VIX level.
        win_loss_ratio : float
            Average win / average loss ratio (default 1.5).

        Returns
        -------
        float
            Position size in [min_position, max_position].
        """
        raw_kelly = self._kelly_fraction(win_probability, win_loss_ratio)
        scale = self._vix_scale(vix_level)
        position = raw_kelly * scale
        return float(np.clip(position, self.min_position, self.max_position))

    @property
    def estimated_edge(self) -> Optional[float]:
        """Win rate estimated during fit(), or None if not fitted."""
        return self._estimated_edge if self._fitted else None

    @property
    def estimated_odds(self) -> Optional[float]:
        """Win/loss ratio estimated during fit(), or None if not fitted."""
        return self._estimated_odds if self._fitted else None

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        edge_str = f", edge={self._estimated_edge:.3f}" if self._fitted else ""
        odds_str = f", odds={self._estimated_odds:.3f}" if self._fitted else ""
        return (
            f"DynamicKellySizer(vix_bins={self.vix_bins}, "
            f"min={self.min_position}, max={self.max_position}"
            f"{edge_str}{odds_str}, {status})"
        )

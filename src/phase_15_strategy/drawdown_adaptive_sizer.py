"""
GIGA TRADER - Drawdown-Adaptive Position Sizer
================================================
Reduces position size as drawdown deepens, using a power-law decay formula.

When an account is in drawdown, reducing position sizes protects against
further losses while still allowing recovery. The decay is quadratic
(power=2.0) so positions shrink more aggressively as drawdown increases.

Formula:
  position = base * (1 - drawdown/max_drawdown)^power

At max drawdown, position approaches min_position.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DrawdownAdaptiveSizer:
    """Position sizer that reduces size as drawdown deepens.

    Parameters
    ----------
    max_drawdown : float
        Maximum drawdown threshold (default 0.10 = 10%). At this level,
        position approaches min_position.
    power : float
        Decay exponent (default 2.0 = quadratic). Higher values mean
        faster reduction.
    min_position : float
        Minimum position size as fraction of portfolio (default 0.02).
    max_position : float
        Maximum position size as fraction of portfolio (default 0.25).
    """

    def __init__(
        self,
        max_drawdown: float = 0.10,
        power: float = 2.0,
        min_position: float = 0.02,
        max_position: float = 0.25,
    ):
        if max_drawdown <= 0.0:
            raise ValueError(f"max_drawdown must be > 0, got {max_drawdown}")
        if power <= 0.0:
            raise ValueError(f"power must be > 0, got {power}")
        if not 0.0 < min_position <= max_position <= 1.0:
            raise ValueError(
                f"Need 0 < min_position <= max_position <= 1, "
                f"got min_position={min_position}, max_position={max_position}"
            )

        self.max_drawdown = max_drawdown
        self.power = power
        self.min_position = min_position
        self.max_position = max_position

        self._fitted = False
        self._peak: Optional[float] = None
        self._current_drawdown: Optional[float] = None

    def fit(self, equity_curve: np.ndarray) -> "DrawdownAdaptiveSizer":
        """Compute current drawdown from equity curve.

        Parameters
        ----------
        equity_curve : np.ndarray
            1-D array of portfolio values (most recent last).

        Returns
        -------
        self
            For method chaining.
        """
        equity = np.asarray(equity_curve, dtype=float).ravel()
        equity = equity[~np.isnan(equity)]

        if len(equity) < 2:
            logger.warning("DrawdownAdaptiveSizer.fit: fewer than 2 values")
            self._fitted = False
            return self

        self._peak = float(np.max(equity))
        current = float(equity[-1])

        if self._peak > 0:
            self._current_drawdown = max(0.0, (self._peak - current) / self._peak)
        else:
            self._current_drawdown = 0.0

        self._fitted = True
        logger.info(
            "DrawdownAdaptiveSizer fitted: peak=%.2f, current=%.2f, drawdown=%.4f",
            self._peak, current, self._current_drawdown,
        )
        return self

    def size(self, base_position: float, current_drawdown: Optional[float] = None) -> float:
        """Compute drawdown-adjusted position size.

        Parameters
        ----------
        base_position : float
            Desired position size before drawdown adjustment.
        current_drawdown : float, optional
            Current drawdown as a fraction (0.0 = no drawdown, 0.10 = 10%).
            If None, uses the value computed during fit().

        Returns
        -------
        float
            Adjusted position size in [min_position, max_position].
        """
        if current_drawdown is None:
            if self._current_drawdown is not None:
                current_drawdown = self._current_drawdown
            else:
                current_drawdown = 0.0

        # Clamp drawdown to [0, max_drawdown]
        dd_clamped = min(max(current_drawdown, 0.0), self.max_drawdown)

        # Power-law decay
        ratio = 1.0 - dd_clamped / self.max_drawdown
        scale = ratio ** self.power
        position = base_position * scale

        return float(np.clip(position, self.min_position, self.max_position))

    @property
    def current_drawdown(self) -> Optional[float]:
        """Current drawdown estimated during fit(), or None if not fitted."""
        return self._current_drawdown if self._fitted else None

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        dd_str = f", dd={self._current_drawdown:.4f}" if self._fitted else ""
        return (
            f"DrawdownAdaptiveSizer(max_dd={self.max_drawdown}, power={self.power}, "
            f"min={self.min_position}, max={self.max_position}{dd_str}, {status})"
        )

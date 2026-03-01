"""
GIGA TRADER - Volatility Regime Trading Gate
==============================================
Blocks or reduces trading in extreme VIX conditions.

VIX is the market's implied volatility gauge. Extreme readings indicate
dislocated markets where ML model predictions lose reliability.

Gate logic:
  - VIX > 35               -> BLOCK (extreme fear, chaotic price action)
  - VIX 28-35 + BUY        -> REDUCE (0.6x confidence, 0.5x position)
  - VIX 28-35 + SELL       -> BOOST  (1.15x confidence, contrarian)
  - VIX < 28 + backwardation -> REDUCE (0.7x confidence, 0.8x position)
  - Normal                  -> PASS
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from src.phase_19_paper_trading.trading_gates import GateDataProvider

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PROVIDER
# ═══════════════════════════════════════════════════════════════════════════════


class VolRegimeDataProvider(GateDataProvider):
    """Fetches current VIX and VIX3M levels via yfinance.

    VIX3M is the 3-month implied volatility index. When VIX > VIX3M
    (backwardation), it signals near-term stress exceeding longer-term
    expectations -- a warning sign even if absolute VIX is not extreme.
    """

    def __init__(self):
        self._data: Optional[Dict[str, Any]] = None
        self._last_fetch: Optional[datetime] = None

    def fetch(self) -> Optional[Dict[str, Any]]:
        """Download current VIX and VIX3M levels.

        Returns
        -------
        dict or None
            VIX data dict on success, None on failure.
        """
        try:
            import yfinance as yf
            vix = yf.download("^VIX", period="5d", progress=False)
            vix3m_data = yf.download("^VIX3M", period="5d", progress=False)

            if len(vix) == 0:
                logger.warning("VolRegimeGate: VIX download returned no data")
                return None

            vix_close = vix["Close"]
            if hasattr(vix_close, "iloc") and hasattr(vix_close, "ndim"):
                if vix_close.ndim > 1:
                    vix_close = vix_close.iloc[:, 0]
            vix_level = float(vix_close.iloc[-1])

            vix3m_level = None
            if len(vix3m_data) > 0:
                vix3m_close = vix3m_data["Close"]
                if hasattr(vix3m_close, "ndim") and vix3m_close.ndim > 1:
                    vix3m_close = vix3m_close.iloc[:, 0]
                vix3m_level = float(vix3m_close.iloc[-1])

            vix_ratio = vix_level / vix3m_level if vix3m_level and vix3m_level > 0 else 1.0

            self._data = {
                "vix_level": vix_level,
                "vix3m_level": vix3m_level,
                "vix_ratio": vix_ratio,
                "is_backwardation": vix_ratio > 1.0,
                "timestamp": datetime.now().isoformat(),
            }
            self._last_fetch = datetime.now()
            return self._data

        except Exception as e:
            logger.warning(f"VolRegimeGate: failed to fetch VIX data: {e}")
            return None

    def get_latest(self) -> Optional[Dict[str, Any]]:
        return self._data

    def get_data_age_hours(self) -> float:
        if self._last_fetch is None:
            return 999.0
        return (datetime.now() - self._last_fetch).total_seconds() / 3600


# ═══════════════════════════════════════════════════════════════════════════════
# GATE EVALUATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_vol_regime_gate(
    data: Optional[Dict[str, Any]],
    signal_type: str,
    block_vix_threshold: float = 35.0,
    reduce_vix_threshold: float = 28.0,
    backwardation_reduce: float = 0.7,
) -> Dict[str, Any]:
    """
    Evaluate volatility regime gate.

    Parameters
    ----------
    data : dict or None
        VIX data from VolRegimeDataProvider.
    signal_type : str
        "BUY", "SELL", or "HOLD".
    block_vix_threshold : float
        VIX level above which all trades are blocked (default 35).
    reduce_vix_threshold : float
        VIX level above which buy confidence is reduced (default 28).
    backwardation_reduce : float
        Confidence multiplier when VIX term structure is in backwardation
        but VIX is below reduce_vix_threshold (default 0.7).

    Returns
    -------
    dict
        Gate decision with action, confidence_multiplier,
        position_size_multiplier, and reason.
    """
    if data is None:
        return {
            "action": "PASS",
            "confidence_multiplier": 1.0,
            "position_size_multiplier": 1.0,
            "reason": "No VIX data available",
        }

    vix = data.get("vix_level", 20.0)
    is_back = data.get("is_backwardation", False)

    # Extreme VIX -> BLOCK everything
    if vix > block_vix_threshold:
        return {
            "action": "BLOCK",
            "confidence_multiplier": 0.0,
            "position_size_multiplier": 0.0,
            "reason": f"VIX at {vix:.1f} — extreme volatility, blocking all trades",
        }

    # Elevated VIX -> direction-dependent
    if vix > reduce_vix_threshold:
        if signal_type == "BUY":
            return {
                "action": "REDUCE",
                "confidence_multiplier": 0.6,
                "position_size_multiplier": 0.5,
                "reason": f"VIX at {vix:.1f} — elevated vol, reducing buy confidence",
            }
        elif signal_type == "SELL":
            return {
                "action": "BOOST",
                "confidence_multiplier": 1.15,
                "position_size_multiplier": 1.0,
                "reason": f"VIX at {vix:.1f} — elevated vol, boosting sell (contrarian)",
            }

    # Backwardation with moderate VIX -> caution
    if is_back and vix <= reduce_vix_threshold:
        return {
            "action": "REDUCE",
            "confidence_multiplier": backwardation_reduce,
            "position_size_multiplier": 0.8,
            "reason": "VIX term structure in backwardation — stress signal",
        }

    # Normal conditions
    return {
        "action": "PASS",
        "confidence_multiplier": 1.0,
        "position_size_multiplier": 1.0,
        "reason": f"VIX at {vix:.1f} — normal conditions",
    }

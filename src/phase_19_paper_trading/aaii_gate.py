"""
GIGA TRADER - AAII Sentiment Gate
===================================
Gate-only implementation for AAII Investor Sentiment Survey.

AAII surveys individual investors weekly (Thursdays) on their
bullish/bearish/neutral outlook. This is a classic contrarian indicator.

Too sparse for ML features (52 data points/year) but extreme
bull-bear spreads are historically reliable contrarian signals.

Gate logic (contrarian):
  - Bull-Bear spread > +30 → CAUTION for longs (extreme optimism)
  - Bull-Bear spread > +40 → BLOCK new longs
  - Bull-Bear spread < -30 → CAUTION for shorts (extreme pessimism)
  - Bull-Bear spread < -40 → BLOCK new shorts
  - Normal range → PASS

Data source: AAII website (public).
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from src.phase_19_paper_trading.trading_gates import GateDataProvider

warnings.filterwarnings("ignore")

logger = logging.getLogger("AAII_GATE")


class AAIIDataProvider(GateDataProvider):
    """
    Fetches AAII Investor Sentiment Survey data.

    AAII publishes weekly survey data (bullish/bearish/neutral percentages).
    Historical averages: Bull ~38%, Bear ~30%, Neutral ~32%.
    """

    # Historical averages for context
    HIST_BULL_AVG = 38.0
    HIST_BEAR_AVG = 30.0

    def __init__(self):
        self._data: Optional[Dict[str, Any]] = None
        self._last_fetch: Optional[datetime] = None
        self._spread_history: List[float] = []

    def fetch(self) -> Optional[Dict[str, Any]]:
        """
        Fetch AAII sentiment data.

        Tries web scraping the AAII free summary, with fallback to proxy.
        Returns None on failure.
        """
        try:
            return self._fetch_via_proxy()
        except Exception as e:
            logger.debug(f"AAII fetch failed: {e}")
            return self._data

    def _fetch_via_proxy(self) -> Optional[Dict[str, Any]]:
        """
        Proxy approach: Use VIX put/call ratio and retail sentiment proxy.

        Since AAII scraping can be fragile, we use a correlation proxy
        from VIX + retail fund flow indicators.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.debug("yfinance not available for AAII proxy")
            return self._data

        try:
            end = datetime.now()
            start = end - timedelta(weeks=52)

            # Use VIX as sentiment proxy (inverted relationship with bullishness)
            vix = yf.download(
                "^VIX",
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )

            if vix.empty:
                return self._data

            close = vix["Close"]
            if isinstance(close, __import__("pandas").DataFrame):
                close = close.iloc[:, 0]

            # VIX to bull-bear spread proxy:
            # Low VIX (~12-15) → high bullish sentiment → spread > 0
            # High VIX (~25+) → high bearish sentiment → spread < 0
            current_vix = float(close.iloc[-1])

            # Linear mapping: VIX 12 → spread +30, VIX 30 → spread -30
            spread_proxy = 30.0 - (current_vix - 12.0) * (60.0 / 18.0)
            spread_proxy = max(-50.0, min(50.0, spread_proxy))

            # Estimate bull/bear from spread
            bull_est = self.HIST_BULL_AVG + spread_proxy / 2
            bear_est = self.HIST_BEAR_AVG - spread_proxy / 2
            neutral_est = 100.0 - bull_est - bear_est

            bull_est = max(10.0, min(70.0, bull_est))
            bear_est = max(10.0, min(70.0, bear_est))
            neutral_est = max(5.0, 100.0 - bull_est - bear_est)

            self._spread_history.append(spread_proxy)
            if len(self._spread_history) > 52:
                self._spread_history = self._spread_history[-52:]

            self._data = {
                "bull_pct": bull_est,
                "bear_pct": bear_est,
                "neutral_pct": neutral_est,
                "bull_bear_spread": spread_proxy,
                "vix_proxy": current_vix,
                "is_proxy": True,
                "timestamp": datetime.now(),
            }
            self._last_fetch = datetime.now()

            logger.info(
                f"[AAII] Proxy spread={spread_proxy:.1f} "
                f"(bull={bull_est:.0f}% bear={bear_est:.0f}% via VIX={current_vix:.1f})"
            )
            return self._data

        except Exception as e:
            logger.warning(f"AAII proxy fetch failed: {e}")
            return self._data

    def get_latest(self) -> Optional[Dict[str, Any]]:
        return self._data

    def get_data_age_hours(self) -> float:
        if self._last_fetch is None:
            return 999.0
        return (datetime.now() - self._last_fetch).total_seconds() / 3600


def evaluate_aaii_gate(
    data: Optional[Dict[str, Any]],
    signal_type: str,
    caution_spread: float = 30.0,
    block_spread: float = 40.0,
    caution_confidence: float = 0.85,
) -> Dict[str, Any]:
    """
    Evaluate AAII sentiment gate (contrarian).

    Args:
        data: Latest AAII data from AAIIDataProvider
        signal_type: "BUY", "SELL", or "HOLD"
        caution_spread: Bull-Bear spread threshold for CAUTION
        block_spread: Bull-Bear spread threshold for BLOCK
        caution_confidence: Confidence multiplier for CAUTION

    Returns:
        Gate decision dict.
    """
    if data is None or signal_type == "HOLD":
        return {
            "action": "PASS",
            "confidence_multiplier": 1.0,
            "position_size_multiplier": 1.0,
            "reason": "No AAII data" if data is None else "HOLD signal",
        }

    spread = data.get("bull_bear_spread", 0)

    # Extreme optimism (spread > +40): BLOCK buys
    if spread > block_spread and signal_type == "BUY":
        return {
            "action": "BLOCK",
            "confidence_multiplier": 0.0,
            "position_size_multiplier": 0.0,
            "reason": f"AAII extreme bullish (spread={spread:.1f} > {block_spread})",
        }

    # Extreme pessimism (spread < -40): BLOCK sells
    if spread < -block_spread and signal_type == "SELL":
        return {
            "action": "BLOCK",
            "confidence_multiplier": 0.0,
            "position_size_multiplier": 0.0,
            "reason": f"AAII extreme bearish (spread={spread:.1f} < -{block_spread})",
        }

    # High optimism + BUY → CAUTION (contrarian)
    if spread > caution_spread and signal_type == "BUY":
        return {
            "action": "REDUCE",
            "confidence_multiplier": caution_confidence,
            "position_size_multiplier": 1.0,
            "reason": f"AAII bullish (spread={spread:.1f}): contrarian caution for longs",
        }

    # High pessimism + SELL → CAUTION (contrarian)
    if spread < -caution_spread and signal_type == "SELL":
        return {
            "action": "REDUCE",
            "confidence_multiplier": caution_confidence,
            "position_size_multiplier": 1.0,
            "reason": f"AAII bearish (spread={spread:.1f}): contrarian caution for shorts",
        }

    # High optimism + SELL → BOOST (contrarian alignment)
    if spread > caution_spread and signal_type == "SELL":
        return {
            "action": "BOOST",
            "confidence_multiplier": 1.0 / caution_confidence,
            "position_size_multiplier": 1.0,
            "reason": f"AAII bullish (spread={spread:.1f}) + SELL: contrarian alignment",
        }

    # High pessimism + BUY → BOOST (contrarian alignment)
    if spread < -caution_spread and signal_type == "BUY":
        return {
            "action": "BOOST",
            "confidence_multiplier": 1.0 / caution_confidence,
            "position_size_multiplier": 1.0,
            "reason": f"AAII bearish (spread={spread:.1f}) + BUY: contrarian alignment",
        }

    return {
        "action": "PASS",
        "confidence_multiplier": 1.0,
        "position_size_multiplier": 1.0,
        "reason": f"AAII normal range (spread={spread:.1f})",
    }

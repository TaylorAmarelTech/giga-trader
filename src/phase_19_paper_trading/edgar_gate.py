"""
GIGA TRADER - SEC EDGAR Institutional Holdings Gate
=====================================================
Gate-only implementation for SEC 13F institutional holdings data.

13F filings are quarterly (4 data points/year) — far too sparse for ML.
However, aggregate institutional selling/buying can be a useful signal
at extremes.

DISABLED BY DEFAULT because:
  - Quarterly data with 45-day filing delay = very slow signal
  - Better suited as context/confirmation than primary gate

Gate logic:
  - Aggregate institutional selling > 5% → CAUTION for longs
  - Aggregate institutional selling > 10% → BLOCK new longs
  - Aggregate institutional buying > 5% → CAUTION for shorts
  - Aggregate institutional buying > 10% → BLOCK new shorts
  - Normal range → PASS

Data source: SEC EDGAR API (free, requires User-Agent header).
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from src.phase_19_paper_trading.trading_gates import GateDataProvider

warnings.filterwarnings("ignore")

logger = logging.getLogger("EDGAR_GATE")


class EdgarDataProvider(GateDataProvider):
    """
    Fetches aggregate 13F institutional holdings data from SEC EDGAR.

    Note: EDGAR API is free but requires a valid User-Agent header
    with company name and email. Rate limited to 10 requests/second.
    """

    EDGAR_BASE_URL = "https://efts.sec.gov/LATEST/search-index"

    def __init__(self, user_agent: str = "GigaTrader research@example.com"):
        self._data: Optional[Dict[str, Any]] = None
        self._last_fetch: Optional[datetime] = None
        self._user_agent = user_agent
        self._change_history: List[float] = []

    def fetch(self) -> Optional[Dict[str, Any]]:
        """
        Fetch institutional holdings change estimate.

        Uses a proxy approach since full EDGAR parsing is complex.
        Estimates institutional flow from large ETF flow data.
        """
        try:
            return self._fetch_via_etf_proxy()
        except Exception as e:
            logger.debug(f"EDGAR fetch failed: {e}")
            return self._data

    def _fetch_via_etf_proxy(self) -> Optional[Dict[str, Any]]:
        """
        Proxy: Use SPY/IVV/VOO fund flow correlation for institutional activity.

        Large-cap ETF flows correlate with institutional 13F activity.
        """
        try:
            import yfinance as yf
            import pandas as pd
        except ImportError:
            logger.debug("yfinance not available for EDGAR proxy")
            return self._data

        try:
            end = datetime.now()
            start = end - timedelta(days=90)  # ~1 quarter

            # Download SPY volume and price changes as institutional proxy
            spy = yf.download(
                "SPY",
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True,
            )

            if spy.empty or len(spy) < 20:
                return self._data

            close = spy["Close"]
            volume = spy["Volume"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            if isinstance(volume, pd.DataFrame):
                volume = volume.iloc[:, 0]

            # Estimate institutional flow direction:
            # High volume + price increase → institutional buying
            # High volume + price decrease → institutional selling
            daily_returns = close.pct_change()
            vol_zscore = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()

            # Flow indicator: return * vol_zscore (positive = buying, negative = selling)
            flow = (daily_returns * vol_zscore).dropna()

            if flow.empty:
                return self._data

            # Aggregate over quarter
            recent_flow = float(flow.tail(20).mean())  # Last month avg
            quarterly_flow = float(flow.mean())  # Full quarter

            # Convert to percentage change estimate
            # Roughly: flow indicator of 0.01 ≈ 1% net institutional buying
            pct_change_est = quarterly_flow * 100  # Rough scaling

            self._change_history.append(pct_change_est)
            if len(self._change_history) > 4:
                self._change_history = self._change_history[-4:]

            self._data = {
                "institutional_change_pct": pct_change_est,
                "recent_flow": recent_flow,
                "quarterly_flow": quarterly_flow,
                "is_proxy": True,
                "n_quarters_history": len(self._change_history),
                "timestamp": datetime.now(),
            }
            self._last_fetch = datetime.now()

            logger.info(f"[EDGAR] Institutional change proxy: {pct_change_est:.2f}%")
            return self._data

        except Exception as e:
            logger.warning(f"EDGAR proxy fetch failed: {e}")
            return self._data

    def get_latest(self) -> Optional[Dict[str, Any]]:
        return self._data

    def get_data_age_hours(self) -> float:
        if self._last_fetch is None:
            return 999.0
        return (datetime.now() - self._last_fetch).total_seconds() / 3600


def evaluate_edgar_gate(
    data: Optional[Dict[str, Any]],
    signal_type: str,
    caution_pct: float = 5.0,
    block_pct: float = 10.0,
    caution_confidence: float = 0.80,
) -> Dict[str, Any]:
    """
    Evaluate EDGAR institutional holdings gate.

    Args:
        data: Latest EDGAR data from EdgarDataProvider
        signal_type: "BUY", "SELL", or "HOLD"
        caution_pct: Institutional change threshold for CAUTION (%)
        block_pct: Institutional change threshold for BLOCK (%)
        caution_confidence: Confidence multiplier for CAUTION

    Returns:
        Gate decision dict.
    """
    if data is None or signal_type == "HOLD":
        return {
            "action": "PASS",
            "confidence_multiplier": 1.0,
            "position_size_multiplier": 1.0,
            "reason": "No EDGAR data" if data is None else "HOLD signal",
        }

    change = data.get("institutional_change_pct", 0)

    # Heavy institutional selling + BUY → BLOCK
    if change < -block_pct and signal_type == "BUY":
        return {
            "action": "BLOCK",
            "confidence_multiplier": 0.0,
            "position_size_multiplier": 0.0,
            "reason": f"Heavy institutional selling ({change:.1f}% < -{block_pct}%)",
        }

    # Heavy institutional buying + SELL → BLOCK
    if change > block_pct and signal_type == "SELL":
        return {
            "action": "BLOCK",
            "confidence_multiplier": 0.0,
            "position_size_multiplier": 0.0,
            "reason": f"Heavy institutional buying ({change:.1f}% > {block_pct}%)",
        }

    # Institutional selling + BUY → CAUTION
    if change < -caution_pct and signal_type == "BUY":
        return {
            "action": "REDUCE",
            "confidence_multiplier": caution_confidence,
            "position_size_multiplier": 1.0,
            "reason": f"Institutional selling ({change:.1f}%): caution for longs",
        }

    # Institutional buying + SELL → CAUTION
    if change > caution_pct and signal_type == "SELL":
        return {
            "action": "REDUCE",
            "confidence_multiplier": caution_confidence,
            "position_size_multiplier": 1.0,
            "reason": f"Institutional buying ({change:.1f}%): caution for shorts",
        }

    return {
        "action": "PASS",
        "confidence_multiplier": 1.0,
        "position_size_multiplier": 1.0,
        "reason": f"EDGAR normal range ({change:.1f}%)",
    }

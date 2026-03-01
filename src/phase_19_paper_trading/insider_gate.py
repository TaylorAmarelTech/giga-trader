"""
GIGA TRADER - Finnhub Insider Sentiment Gate
===============================================
Gate-only implementation for insider trading sentiment from Finnhub.

Insider filings are too sporadic for ML features (~20-50 per year for SPY
components), but extreme net buying/selling provides useful contrarian signals.

Requires: FINNHUB_API_KEY environment variable.
Silent PASS if API key missing or data unavailable.

Gate logic (contrarian):
  - Net insider BUYING in top 10% historically → ALLOW (bullish insiders)
  - Net insider SELLING in top 10% historically → CAUTION (-50% position size)
  - Normal range → PASS (no effect)

Data source: Finnhub insider transactions API
API endpoint: https://finnhub.io/api/v1/stock/insider-transactions
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from src.phase_19_paper_trading.trading_gates import GateDataProvider

logger = logging.getLogger("INSIDER_GATE")


class InsiderSentimentProvider(GateDataProvider):
    """
    Fetches insider transaction data from Finnhub for SPY components.

    Aggregates net insider buying/selling across top SPY components
    over a rolling window (default 90 days).
    """

    # Top SPY components to track insider activity
    SPY_TICKERS = [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA",
        "BRK.B", "UNH", "XOM", "JPM", "JNJ", "V", "PG", "MA",
        "HD", "CVX", "MRK", "ABBV", "LLY",
    ]

    def __init__(self, lookback_days: int = 90, tickers: Optional[List[str]] = None):
        self._data: Optional[Dict[str, Any]] = None
        self._last_fetch: Optional[datetime] = None
        self._lookback_days = lookback_days
        self._tickers = tickers or self.SPY_TICKERS
        self._api_key: Optional[str] = None
        # Historical percentile tracking for z-score
        self._net_buy_history: List[float] = []

    def _get_api_key(self) -> Optional[str]:
        """Get Finnhub API key from environment."""
        if self._api_key:
            return self._api_key
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        self._api_key = os.environ.get("FINNHUB_API_KEY")
        return self._api_key

    def fetch(self) -> Optional[Dict[str, Any]]:
        """
        Fetch insider transactions from Finnhub for SPY components.

        Returns aggregated net buying metrics or None on failure.
        """
        api_key = self._get_api_key()
        if not api_key:
            logger.debug("FINNHUB_API_KEY not set - insider gate disabled")
            return self._data

        try:
            import requests
        except ImportError:
            return self._data

        try:
            from_date = (datetime.now() - timedelta(days=self._lookback_days)).strftime("%Y-%m-%d")
            to_date = datetime.now().strftime("%Y-%m-%d")

            total_buy_value = 0.0
            total_sell_value = 0.0
            n_buys = 0
            n_sells = 0
            n_tickers_with_data = 0

            for ticker in self._tickers[:10]:  # Limit to 10 to respect rate limits
                try:
                    url = "https://finnhub.io/api/v1/stock/insider-transactions"
                    params = {
                        "symbol": ticker,
                        "from": from_date,
                        "to": to_date,
                        "token": api_key,
                    }
                    resp = requests.get(url, params=params, timeout=15)

                    if resp.status_code == 429:
                        logger.debug("Finnhub rate limit - stopping early")
                        break

                    if resp.status_code != 200:
                        continue

                    data = resp.json()
                    transactions = data.get("data", [])

                    if not transactions:
                        continue

                    n_tickers_with_data += 1

                    for txn in transactions:
                        change = txn.get("change", 0) or 0
                        price = txn.get("transactionPrice", 0) or 0
                        value = abs(change * price)

                        if change > 0:  # Buy
                            total_buy_value += value
                            n_buys += 1
                        elif change < 0:  # Sell
                            total_sell_value += value
                            n_sells += 1

                except Exception as e:
                    logger.debug(f"  Insider data failed for {ticker}: {e}")
                    continue

            if n_tickers_with_data == 0:
                return self._data

            # Net buying ratio: (buys - sells) / (buys + sells)
            total = total_buy_value + total_sell_value
            net_buy_ratio = (total_buy_value - total_sell_value) / total if total > 0 else 0.0

            # Track history for percentile calculation
            self._net_buy_history.append(net_buy_ratio)
            if len(self._net_buy_history) > 52:  # ~1 year of weekly data
                self._net_buy_history = self._net_buy_history[-52:]

            # Calculate percentile
            if len(self._net_buy_history) >= 5:
                percentile = (
                    sum(1 for x in self._net_buy_history if x <= net_buy_ratio)
                    / len(self._net_buy_history)
                )
            else:
                percentile = 0.5  # Not enough history

            self._data = {
                "net_buy_ratio": net_buy_ratio,
                "percentile": percentile,
                "total_buy_value": total_buy_value,
                "total_sell_value": total_sell_value,
                "n_buys": n_buys,
                "n_sells": n_sells,
                "n_tickers": n_tickers_with_data,
                "timestamp": datetime.now(),
            }
            self._last_fetch = datetime.now()
            logger.info(
                f"[INSIDER] Net buy ratio={net_buy_ratio:.3f}, "
                f"percentile={percentile:.2f}, {n_tickers_with_data} tickers"
            )
            return self._data

        except Exception as e:
            logger.warning(f"Insider data fetch failed: {e}")
            return self._data

    def get_latest(self) -> Optional[Dict[str, Any]]:
        return self._data

    def get_data_age_hours(self) -> float:
        if self._last_fetch is None:
            return 999.0
        return (datetime.now() - self._last_fetch).total_seconds() / 3600


def evaluate_insider_gate(
    data: Optional[Dict[str, Any]],
    signal_type: str,
    caution_percentile: float = 0.90,
    caution_position_reduce: float = 0.50,
) -> Dict[str, Any]:
    """
    Evaluate insider sentiment gate.

    Args:
        data: Latest insider data from InsiderSentimentProvider
        signal_type: "BUY", "SELL", or "HOLD"
        caution_percentile: Threshold for extreme readings (default top/bottom 10%)
        caution_position_reduce: Position size multiplier for caution (default 50%)

    Returns:
        Gate decision dict compatible with TradingGates system.
    """
    if data is None or signal_type == "HOLD":
        return {
            "action": "PASS",
            "confidence_multiplier": 1.0,
            "position_size_multiplier": 1.0,
            "reason": "No insider data available" if data is None else "HOLD signal",
        }

    net_buy_ratio = data.get("net_buy_ratio", 0)
    percentile = data.get("percentile", 0.5)

    # Net insider buying in top percentile → bullish signal
    if percentile >= caution_percentile:
        if signal_type == "BUY":
            return {
                "action": "PASS",  # Allow normal execution (insiders agree)
                "confidence_multiplier": 1.0,
                "position_size_multiplier": 1.0,
                "reason": f"Net buying top {100*(1-caution_percentile):.0f}% (ratio={net_buy_ratio:.3f})",
            }
        elif signal_type == "SELL":
            return {
                "action": "REDUCE",
                "confidence_multiplier": 1.0,
                "position_size_multiplier": caution_position_reduce,
                "reason": f"Net buying top {100*(1-caution_percentile):.0f}% vs SELL (ratio={net_buy_ratio:.3f})",
            }

    # Net insider selling in bottom percentile → bearish signal
    if percentile <= (1 - caution_percentile):
        if signal_type == "SELL":
            return {
                "action": "PASS",  # Allow normal execution (insiders agree)
                "confidence_multiplier": 1.0,
                "position_size_multiplier": 1.0,
                "reason": f"Net selling bottom {100*(1-caution_percentile):.0f}% (ratio={net_buy_ratio:.3f})",
            }
        elif signal_type == "BUY":
            return {
                "action": "REDUCE",
                "confidence_multiplier": 1.0,
                "position_size_multiplier": caution_position_reduce,
                "reason": f"Net selling bottom {100*(1-caution_percentile):.0f}% vs BUY (ratio={net_buy_ratio:.3f})",
            }

    # Normal range → no effect
    return {
        "action": "PASS",
        "confidence_multiplier": 1.0,
        "position_size_multiplier": 1.0,
        "reason": f"Normal range (percentile={percentile:.2f})",
    }

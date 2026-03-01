"""
Tests for InsiderSentimentProvider and insider gate evaluation.
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.phase_19_paper_trading.insider_gate import (
    InsiderSentimentProvider,
    evaluate_insider_gate,
)


# ─── Provider Tests ──────────────────────────────────────────────────────────

class TestInsiderProvider:

    def test_default_constructor(self):
        provider = InsiderSentimentProvider()
        assert provider._data is None
        assert provider._last_fetch is None
        assert len(provider._tickers) == 20

    def test_custom_tickers(self):
        provider = InsiderSentimentProvider(tickers=["AAPL", "MSFT"])
        assert len(provider._tickers) == 2

    def test_get_latest_none_initially(self):
        provider = InsiderSentimentProvider()
        assert provider.get_latest() is None

    def test_data_age_large_when_no_data(self):
        provider = InsiderSentimentProvider()
        assert provider.get_data_age_hours() > 100

    def test_no_api_key_returns_cached(self):
        provider = InsiderSentimentProvider()
        provider._api_key = None
        with patch.dict("os.environ", {}, clear=True):
            result = provider.fetch()
            assert result is None  # No cached data, no key

    @patch("requests.get")
    def test_fetch_with_data(self, mock_get):
        provider = InsiderSentimentProvider(tickers=["AAPL"])
        provider._api_key = "test_key"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"change": 1000, "transactionPrice": 150.0},
                {"change": -500, "transactionPrice": 150.0},
                {"change": 2000, "transactionPrice": 150.0},
            ]
        }
        mock_get.return_value = mock_resp

        result = provider.fetch()
        assert result is not None
        assert "net_buy_ratio" in result
        assert "percentile" in result
        assert "n_buys" in result
        assert "n_sells" in result

    @patch("requests.get")
    def test_fetch_net_buy_positive(self, mock_get):
        """More buying than selling → positive net_buy_ratio."""
        provider = InsiderSentimentProvider(tickers=["AAPL"])
        provider._api_key = "test_key"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"change": 5000, "transactionPrice": 150.0},
                {"change": -100, "transactionPrice": 150.0},
            ]
        }
        mock_get.return_value = mock_resp

        result = provider.fetch()
        assert result["net_buy_ratio"] > 0

    @patch("requests.get")
    def test_fetch_rate_limit(self, mock_get):
        provider = InsiderSentimentProvider(tickers=["AAPL"])
        provider._api_key = "test_key"

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_get.return_value = mock_resp

        result = provider.fetch()
        # Should return None (no cached data)
        assert result is None

    @patch("requests.get")
    def test_data_age_after_fetch(self, mock_get):
        provider = InsiderSentimentProvider(tickers=["AAPL"])
        provider._api_key = "test_key"

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [{"change": 100, "transactionPrice": 150.0}]
        }
        mock_get.return_value = mock_resp

        provider.fetch()
        assert provider.get_data_age_hours() < 0.01  # Just fetched


# ─── Gate Evaluation Tests ───────────────────────────────────────────────────

class TestInsiderGateEvaluation:

    def test_no_data_passes(self):
        result = evaluate_insider_gate(None, "BUY")
        assert result["action"] == "PASS"
        assert result["confidence_multiplier"] == 1.0

    def test_hold_passes(self):
        data = {"net_buy_ratio": 0.5, "percentile": 0.95}
        result = evaluate_insider_gate(data, "HOLD")
        assert result["action"] == "PASS"

    def test_strong_buying_buy_passes(self):
        """Net buying top 10% + BUY → PASS (insiders agree)."""
        data = {"net_buy_ratio": 0.8, "percentile": 0.95}
        result = evaluate_insider_gate(data, "BUY")
        assert result["action"] == "PASS"

    def test_strong_buying_sell_reduces(self):
        """Net buying top 10% + SELL → REDUCE position."""
        data = {"net_buy_ratio": 0.8, "percentile": 0.95}
        result = evaluate_insider_gate(data, "SELL")
        assert result["action"] == "REDUCE"
        assert result["position_size_multiplier"] < 1.0

    def test_strong_selling_sell_passes(self):
        """Net selling bottom 10% + SELL → PASS (insiders agree)."""
        data = {"net_buy_ratio": -0.8, "percentile": 0.05}
        result = evaluate_insider_gate(data, "SELL")
        assert result["action"] == "PASS"

    def test_strong_selling_buy_reduces(self):
        """Net selling bottom 10% + BUY → REDUCE position."""
        data = {"net_buy_ratio": -0.8, "percentile": 0.05}
        result = evaluate_insider_gate(data, "BUY")
        assert result["action"] == "REDUCE"
        assert result["position_size_multiplier"] < 1.0

    def test_normal_range_passes(self):
        """Normal insider activity → PASS."""
        data = {"net_buy_ratio": 0.1, "percentile": 0.50}
        result = evaluate_insider_gate(data, "BUY")
        assert result["action"] == "PASS"
        assert result["confidence_multiplier"] == 1.0
        assert result["position_size_multiplier"] == 1.0

    def test_custom_thresholds(self):
        """Custom percentile threshold should be respected."""
        data = {"net_buy_ratio": 0.6, "percentile": 0.85}
        # Default 0.90 would PASS, but 0.80 should trigger
        result = evaluate_insider_gate(data, "SELL", caution_percentile=0.80)
        assert result["action"] == "REDUCE"

    def test_custom_position_reduce(self):
        """Custom position reduce factor."""
        data = {"net_buy_ratio": -0.8, "percentile": 0.05}
        result = evaluate_insider_gate(
            data, "BUY", caution_position_reduce=0.30
        )
        assert result["position_size_multiplier"] == 0.30

"""
Tests for AAII Sentiment gate evaluation logic.
"""

import pytest
from datetime import datetime

from src.phase_19_paper_trading.aaii_gate import AAIIDataProvider, evaluate_aaii_gate


class TestAAIIProvider:

    def test_default_constructor(self):
        provider = AAIIDataProvider()
        assert provider._data is None

    def test_get_latest_none(self):
        provider = AAIIDataProvider()
        assert provider.get_latest() is None

    def test_data_age_large_initially(self):
        provider = AAIIDataProvider()
        assert provider.get_data_age_hours() > 100


class TestAAIIGateEvaluation:

    def test_no_data_passes(self):
        result = evaluate_aaii_gate(None, "BUY")
        assert result["action"] == "PASS"

    def test_hold_passes(self):
        data = {"bull_bear_spread": 45.0}
        result = evaluate_aaii_gate(data, "HOLD")
        assert result["action"] == "PASS"

    def test_extreme_bullish_buy_blocked(self):
        """Spread > +40 + BUY → BLOCK."""
        data = {"bull_bear_spread": 45.0}
        result = evaluate_aaii_gate(data, "BUY")
        assert result["action"] == "BLOCK"

    def test_extreme_bearish_sell_blocked(self):
        """Spread < -40 + SELL → BLOCK."""
        data = {"bull_bear_spread": -45.0}
        result = evaluate_aaii_gate(data, "SELL")
        assert result["action"] == "BLOCK"

    def test_bullish_buy_caution(self):
        """Spread > +30 + BUY → REDUCE."""
        data = {"bull_bear_spread": 35.0}
        result = evaluate_aaii_gate(data, "BUY")
        assert result["action"] == "REDUCE"
        assert result["confidence_multiplier"] < 1.0

    def test_bearish_sell_caution(self):
        """Spread < -30 + SELL → REDUCE."""
        data = {"bull_bear_spread": -35.0}
        result = evaluate_aaii_gate(data, "SELL")
        assert result["action"] == "REDUCE"

    def test_bullish_sell_boost(self):
        """Spread > +30 + SELL → BOOST (contrarian alignment)."""
        data = {"bull_bear_spread": 35.0}
        result = evaluate_aaii_gate(data, "SELL")
        assert result["action"] == "BOOST"
        assert result["confidence_multiplier"] > 1.0

    def test_bearish_buy_boost(self):
        """Spread < -30 + BUY → BOOST (contrarian alignment)."""
        data = {"bull_bear_spread": -35.0}
        result = evaluate_aaii_gate(data, "BUY")
        assert result["action"] == "BOOST"

    def test_normal_range_passes(self):
        data = {"bull_bear_spread": 10.0}
        result = evaluate_aaii_gate(data, "BUY")
        assert result["action"] == "PASS"
        assert result["confidence_multiplier"] == 1.0

    def test_custom_thresholds(self):
        data = {"bull_bear_spread": 25.0}
        result = evaluate_aaii_gate(data, "BUY", caution_spread=20.0)
        assert result["action"] == "REDUCE"

    def test_custom_block_threshold(self):
        data = {"bull_bear_spread": 35.0}
        result = evaluate_aaii_gate(data, "BUY", block_spread=30.0)
        assert result["action"] == "BLOCK"

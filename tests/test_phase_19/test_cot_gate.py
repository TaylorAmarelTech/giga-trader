"""
Tests for CFTC COT gate evaluation logic.
"""

import pytest
from datetime import datetime

from src.phase_19_paper_trading.cot_gate import COTDataProvider, evaluate_cot_gate


class TestCOTProvider:

    def test_default_constructor(self):
        provider = COTDataProvider()
        assert provider._data is None
        assert provider._last_fetch is None

    def test_get_latest_none(self):
        provider = COTDataProvider()
        assert provider.get_latest() is None

    def test_data_age_large_initially(self):
        provider = COTDataProvider()
        assert provider.get_data_age_hours() > 100


class TestCOTGateEvaluation:

    def test_no_data_passes(self):
        result = evaluate_cot_gate(None, "BUY")
        assert result["action"] == "PASS"

    def test_hold_passes(self):
        data = {"net_speculator_zscore": 3.5}
        result = evaluate_cot_gate(data, "HOLD")
        assert result["action"] == "PASS"

    def test_extreme_long_buy_blocked(self):
        """z > +3 + BUY → BLOCK."""
        data = {"net_speculator_zscore": 3.5}
        result = evaluate_cot_gate(data, "BUY")
        assert result["action"] == "BLOCK"

    def test_extreme_short_sell_blocked(self):
        """z < -3 + SELL → BLOCK."""
        data = {"net_speculator_zscore": -3.5}
        result = evaluate_cot_gate(data, "SELL")
        assert result["action"] == "BLOCK"

    def test_crowded_long_buy_caution(self):
        """z > +2 + BUY → REDUCE."""
        data = {"net_speculator_zscore": 2.5}
        result = evaluate_cot_gate(data, "BUY")
        assert result["action"] == "REDUCE"
        assert result["confidence_multiplier"] < 1.0

    def test_crowded_short_sell_caution(self):
        """z < -2 + SELL → REDUCE."""
        data = {"net_speculator_zscore": -2.5}
        result = evaluate_cot_gate(data, "SELL")
        assert result["action"] == "REDUCE"

    def test_crowded_long_sell_boost(self):
        """z > +2 + SELL → BOOST (contrarian alignment)."""
        data = {"net_speculator_zscore": 2.5}
        result = evaluate_cot_gate(data, "SELL")
        assert result["action"] == "BOOST"
        assert result["confidence_multiplier"] > 1.0

    def test_crowded_short_buy_boost(self):
        """z < -2 + BUY → BOOST (contrarian alignment)."""
        data = {"net_speculator_zscore": -2.5}
        result = evaluate_cot_gate(data, "BUY")
        assert result["action"] == "BOOST"

    def test_normal_range_passes(self):
        data = {"net_speculator_zscore": 0.5}
        result = evaluate_cot_gate(data, "BUY")
        assert result["action"] == "PASS"
        assert result["confidence_multiplier"] == 1.0

    def test_custom_thresholds(self):
        data = {"net_speculator_zscore": 1.5}
        result = evaluate_cot_gate(data, "BUY", caution_zscore=1.0)
        assert result["action"] == "REDUCE"

    def test_block_threshold_custom(self):
        data = {"net_speculator_zscore": 2.5}
        result = evaluate_cot_gate(data, "BUY", block_zscore=2.0)
        assert result["action"] == "BLOCK"

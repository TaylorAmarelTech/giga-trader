"""
Tests for SEC EDGAR institutional holdings gate evaluation logic.
"""

import pytest
from datetime import datetime

from src.phase_19_paper_trading.edgar_gate import EdgarDataProvider, evaluate_edgar_gate


class TestEdgarProvider:

    def test_default_constructor(self):
        provider = EdgarDataProvider()
        assert provider._data is None

    def test_get_latest_none(self):
        provider = EdgarDataProvider()
        assert provider.get_latest() is None

    def test_data_age_large_initially(self):
        provider = EdgarDataProvider()
        assert provider.get_data_age_hours() > 100


class TestEdgarGateEvaluation:

    def test_no_data_passes(self):
        result = evaluate_edgar_gate(None, "BUY")
        assert result["action"] == "PASS"

    def test_hold_passes(self):
        data = {"institutional_change_pct": -15.0}
        result = evaluate_edgar_gate(data, "HOLD")
        assert result["action"] == "PASS"

    def test_heavy_selling_buy_blocked(self):
        """Selling > 10% + BUY → BLOCK."""
        data = {"institutional_change_pct": -12.0}
        result = evaluate_edgar_gate(data, "BUY")
        assert result["action"] == "BLOCK"

    def test_heavy_buying_sell_blocked(self):
        """Buying > 10% + SELL → BLOCK."""
        data = {"institutional_change_pct": 12.0}
        result = evaluate_edgar_gate(data, "SELL")
        assert result["action"] == "BLOCK"

    def test_moderate_selling_buy_caution(self):
        """Selling > 5% + BUY → REDUCE."""
        data = {"institutional_change_pct": -7.0}
        result = evaluate_edgar_gate(data, "BUY")
        assert result["action"] == "REDUCE"
        assert result["confidence_multiplier"] < 1.0

    def test_moderate_buying_sell_caution(self):
        """Buying > 5% + SELL → REDUCE."""
        data = {"institutional_change_pct": 7.0}
        result = evaluate_edgar_gate(data, "SELL")
        assert result["action"] == "REDUCE"

    def test_selling_sell_passes(self):
        """Selling + SELL → PASS (aligned)."""
        data = {"institutional_change_pct": -7.0}
        result = evaluate_edgar_gate(data, "SELL")
        assert result["action"] == "PASS"

    def test_buying_buy_passes(self):
        """Buying + BUY → PASS (aligned)."""
        data = {"institutional_change_pct": 7.0}
        result = evaluate_edgar_gate(data, "BUY")
        assert result["action"] == "PASS"

    def test_normal_range_passes(self):
        data = {"institutional_change_pct": 2.0}
        result = evaluate_edgar_gate(data, "BUY")
        assert result["action"] == "PASS"
        assert result["confidence_multiplier"] == 1.0

    def test_custom_caution_threshold(self):
        data = {"institutional_change_pct": -4.0}
        result = evaluate_edgar_gate(data, "BUY", caution_pct=3.0)
        assert result["action"] == "REDUCE"

"""
Tests for Volatility Regime Trading Gate.

All tests use mocked data dicts -- no yfinance calls.
"""

import pytest
from datetime import datetime

from src.phase_19_paper_trading.vol_regime_gate import (
    VolRegimeDataProvider,
    evaluate_vol_regime_gate,
)


# ─── Test: VolRegimeDataProvider ─────────────────────────────────────────────


class TestVolRegimeDataProvider:

    def test_default_constructor(self):
        provider = VolRegimeDataProvider()
        assert provider._data is None
        assert provider.get_latest() is None

    def test_data_age_large_initially(self):
        provider = VolRegimeDataProvider()
        assert provider.get_data_age_hours() > 100

    def test_get_latest_returns_none_before_fetch(self):
        provider = VolRegimeDataProvider()
        assert provider.get_latest() is None


# ─── Test: evaluate_vol_regime_gate — BLOCK scenarios ────────────────────────


class TestVolRegimeBlock:

    def test_vix_40_blocks(self):
        """VIX=40 should BLOCK all trades."""
        data = {"vix_level": 40.0, "is_backwardation": False}
        result = evaluate_vol_regime_gate(data, "BUY")
        assert result["action"] == "BLOCK"
        assert result["confidence_multiplier"] == 0.0
        assert result["position_size_multiplier"] == 0.0

    def test_vix_36_blocks(self):
        """VIX=36 (just above default threshold 35) should BLOCK."""
        data = {"vix_level": 36.0, "is_backwardation": False}
        result = evaluate_vol_regime_gate(data, "SELL")
        assert result["action"] == "BLOCK"

    def test_vix_100_blocks(self):
        """Extremely high VIX should BLOCK."""
        data = {"vix_level": 100.0, "is_backwardation": True}
        result = evaluate_vol_regime_gate(data, "BUY")
        assert result["action"] == "BLOCK"

    def test_custom_block_threshold(self):
        """VIX=30 blocks when threshold is lowered to 25."""
        data = {"vix_level": 30.0, "is_backwardation": False}
        result = evaluate_vol_regime_gate(data, "BUY", block_vix_threshold=25.0)
        assert result["action"] == "BLOCK"


# ─── Test: evaluate_vol_regime_gate — REDUCE scenarios ───────────────────────


class TestVolRegimeReduce:

    def test_vix_30_buy_reduces(self):
        """VIX=30 + BUY should REDUCE."""
        data = {"vix_level": 30.0, "is_backwardation": False}
        result = evaluate_vol_regime_gate(data, "BUY")
        assert result["action"] == "REDUCE"
        assert result["confidence_multiplier"] == 0.6
        assert result["position_size_multiplier"] == 0.5

    def test_vix_29_buy_reduces(self):
        """VIX=29 (above 28 reduce threshold) + BUY should REDUCE."""
        data = {"vix_level": 29.0, "is_backwardation": False}
        result = evaluate_vol_regime_gate(data, "BUY")
        assert result["action"] == "REDUCE"

    def test_vix_34_buy_reduces(self):
        """VIX=34 (just below block threshold) + BUY should REDUCE."""
        data = {"vix_level": 34.0, "is_backwardation": False}
        result = evaluate_vol_regime_gate(data, "BUY")
        assert result["action"] == "REDUCE"


# ─── Test: evaluate_vol_regime_gate — BOOST scenarios ────────────────────────


class TestVolRegimeBoost:

    def test_vix_30_sell_boosts(self):
        """VIX=30 + SELL should BOOST (contrarian)."""
        data = {"vix_level": 30.0, "is_backwardation": False}
        result = evaluate_vol_regime_gate(data, "SELL")
        assert result["action"] == "BOOST"
        assert result["confidence_multiplier"] == 1.15
        assert result["position_size_multiplier"] == 1.0

    def test_vix_34_sell_boosts(self):
        """VIX=34 + SELL should BOOST."""
        data = {"vix_level": 34.0, "is_backwardation": False}
        result = evaluate_vol_regime_gate(data, "SELL")
        assert result["action"] == "BOOST"


# ─── Test: evaluate_vol_regime_gate — Backwardation ──────────────────────────


class TestVolRegimeBackwardation:

    def test_backwardation_with_low_vix_reduces(self):
        """VIX=22 + backwardation should REDUCE (stress signal)."""
        data = {"vix_level": 22.0, "is_backwardation": True}
        result = evaluate_vol_regime_gate(data, "BUY")
        assert result["action"] == "REDUCE"
        assert result["confidence_multiplier"] == 0.7
        assert result["position_size_multiplier"] == 0.8

    def test_backwardation_custom_reduce(self):
        """Custom backwardation_reduce multiplier should be used."""
        data = {"vix_level": 20.0, "is_backwardation": True}
        result = evaluate_vol_regime_gate(data, "BUY", backwardation_reduce=0.5)
        assert result["action"] == "REDUCE"
        assert result["confidence_multiplier"] == 0.5

    def test_no_backwardation_normal_passes(self):
        """VIX=22 without backwardation should PASS."""
        data = {"vix_level": 22.0, "is_backwardation": False}
        result = evaluate_vol_regime_gate(data, "BUY")
        assert result["action"] == "PASS"

    def test_backwardation_sell_reduces(self):
        """Backwardation reduces for SELL signals too."""
        data = {"vix_level": 20.0, "is_backwardation": True}
        result = evaluate_vol_regime_gate(data, "SELL")
        assert result["action"] == "REDUCE"


# ─── Test: evaluate_vol_regime_gate — PASS scenarios ─────────────────────────


class TestVolRegimePass:

    def test_normal_vix_buy_passes(self):
        """VIX=18 + no backwardation should PASS for BUY."""
        data = {"vix_level": 18.0, "is_backwardation": False}
        result = evaluate_vol_regime_gate(data, "BUY")
        assert result["action"] == "PASS"
        assert result["confidence_multiplier"] == 1.0
        assert result["position_size_multiplier"] == 1.0

    def test_normal_vix_sell_passes(self):
        """VIX=15 + no backwardation should PASS for SELL."""
        data = {"vix_level": 15.0, "is_backwardation": False}
        result = evaluate_vol_regime_gate(data, "SELL")
        assert result["action"] == "PASS"

    def test_low_vix_passes(self):
        """VIX=10 should PASS."""
        data = {"vix_level": 10.0, "is_backwardation": False}
        result = evaluate_vol_regime_gate(data, "BUY")
        assert result["action"] == "PASS"

    def test_no_data_passes(self):
        """None data should PASS (safe default)."""
        result = evaluate_vol_regime_gate(None, "BUY")
        assert result["action"] == "PASS"
        assert result["confidence_multiplier"] == 1.0

    def test_hold_signal_at_high_vix(self):
        """HOLD signal with elevated VIX should not trigger BUY/SELL logic."""
        data = {"vix_level": 30.0, "is_backwardation": False}
        result = evaluate_vol_regime_gate(data, "HOLD")
        # HOLD is not BUY or SELL, so it falls through to PASS
        assert result["action"] == "PASS"


# ─── Test: result structure ──────────────────────────────────────────────────


class TestResultStructure:

    def test_result_has_required_keys(self):
        """All results must have action, confidence_multiplier, position_size_multiplier, reason."""
        data = {"vix_level": 40.0, "is_backwardation": True}
        result = evaluate_vol_regime_gate(data, "BUY")
        assert "action" in result
        assert "confidence_multiplier" in result
        assert "position_size_multiplier" in result
        assert "reason" in result

    def test_reason_contains_vix_level(self):
        """Reason string should mention VIX level when data is available."""
        data = {"vix_level": 30.0, "is_backwardation": False}
        result = evaluate_vol_regime_gate(data, "BUY")
        assert "30.0" in result["reason"]

    def test_block_reason_mentions_extreme(self):
        data = {"vix_level": 50.0, "is_backwardation": False}
        result = evaluate_vol_regime_gate(data, "BUY")
        assert "extreme" in result["reason"].lower()


# ─── Test: threshold boundary behavior ──────────────────────────────────────


class TestBoundaryBehavior:

    def test_exactly_at_block_threshold_passes(self):
        """VIX exactly at block threshold (35.0) should NOT block (uses >)."""
        data = {"vix_level": 35.0, "is_backwardation": False}
        result = evaluate_vol_regime_gate(data, "BUY")
        # VIX=35 is > reduce_threshold=28, so it should REDUCE for BUY
        assert result["action"] == "REDUCE"

    def test_exactly_at_reduce_threshold_passes(self):
        """VIX exactly at reduce threshold (28.0) should NOT reduce (uses >)."""
        data = {"vix_level": 28.0, "is_backwardation": False}
        result = evaluate_vol_regime_gate(data, "BUY")
        assert result["action"] == "PASS"

    def test_just_above_reduce_threshold(self):
        """VIX=28.01 should trigger REDUCE for BUY."""
        data = {"vix_level": 28.01, "is_backwardation": False}
        result = evaluate_vol_regime_gate(data, "BUY")
        assert result["action"] == "REDUCE"

    def test_just_above_block_threshold(self):
        """VIX=35.01 should trigger BLOCK."""
        data = {"vix_level": 35.01, "is_backwardation": False}
        result = evaluate_vol_regime_gate(data, "BUY")
        assert result["action"] == "BLOCK"

"""
Tests for TradingGates rule-based gate system.

Validates gate evaluation logic using MockGateDataProvider
without requiring live API calls.
"""

import pytest
from datetime import datetime, timedelta

from src.phase_19_paper_trading.trading_gates import (
    TradingGates,
    TradingGatesConfig,
    GateDecision,
    GateResult,
    MockGateDataProvider,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def default_config():
    return TradingGatesConfig()


@pytest.fixture
def fear_greed_provider():
    """Provider with normal Fear & Greed score."""
    return MockGateDataProvider({"score": 50, "timestamp": datetime.now()})


@pytest.fixture
def extreme_fear_provider():
    """Provider with extreme fear score."""
    return MockGateDataProvider({"score": 10, "timestamp": datetime.now()})


@pytest.fixture
def extreme_greed_provider():
    """Provider with extreme greed score."""
    return MockGateDataProvider({"score": 90, "timestamp": datetime.now()})


# ─── Basic Tests ─────────────────────────────────────────────────────────────

class TestGatesBasic:

    def test_gates_disabled_passes_everything(self):
        config = TradingGatesConfig(gates_enabled=False)
        gates = TradingGates(config=config)
        result = gates.evaluate("BUY")
        assert not result.is_blocked
        assert result.confidence_multiplier == 1.0
        assert result.position_size_multiplier == 1.0

    def test_hold_signal_passes(self, default_config, fear_greed_provider):
        gates = TradingGates(
            config=default_config,
            data_providers={"fear_greed": fear_greed_provider},
        )
        result = gates.evaluate("HOLD")
        assert not result.is_blocked
        assert result.confidence_multiplier == 1.0

    def test_no_providers_passes(self, default_config):
        gates = TradingGates(config=default_config, data_providers={})
        result = gates.evaluate("BUY")
        assert not result.is_blocked

    def test_result_has_timestamp(self, default_config, fear_greed_provider):
        gates = TradingGates(
            config=default_config,
            data_providers={"fear_greed": fear_greed_provider},
        )
        result = gates.evaluate("BUY")
        assert isinstance(result.timestamp, datetime)


# ─── Fear & Greed Gate Tests ─────────────────────────────────────────────────

class TestFearGreedGate:

    def test_normal_range_no_effect(self, default_config, fear_greed_provider):
        gates = TradingGates(
            config=default_config,
            data_providers={"fear_greed": fear_greed_provider},
        )
        result = gates.evaluate("BUY")
        assert result.confidence_multiplier == 1.0
        assert not result.is_blocked

    def test_extreme_fear_boosts_buy(self, default_config, extreme_fear_provider):
        gates = TradingGates(
            config=default_config,
            data_providers={"fear_greed": extreme_fear_provider},
        )
        result = gates.evaluate("BUY")
        assert result.confidence_multiplier > 1.0
        assert not result.is_blocked

    def test_extreme_fear_reduces_sell(self, default_config, extreme_fear_provider):
        gates = TradingGates(
            config=default_config,
            data_providers={"fear_greed": extreme_fear_provider},
        )
        result = gates.evaluate("SELL")
        assert result.confidence_multiplier < 1.0

    def test_extreme_greed_reduces_buy(self, default_config, extreme_greed_provider):
        gates = TradingGates(
            config=default_config,
            data_providers={"fear_greed": extreme_greed_provider},
        )
        result = gates.evaluate("BUY")
        assert result.confidence_multiplier < 1.0

    def test_extreme_greed_boosts_sell(self, default_config, extreme_greed_provider):
        gates = TradingGates(
            config=default_config,
            data_providers={"fear_greed": extreme_greed_provider},
        )
        result = gates.evaluate("SELL")
        assert result.confidence_multiplier > 1.0

    def test_custom_thresholds(self):
        """Custom thresholds should be respected."""
        config = TradingGatesConfig(
            fear_greed_extreme_fear=30,
            fear_greed_extreme_greed=70,
        )
        provider = MockGateDataProvider({"score": 25, "timestamp": datetime.now()})
        gates = TradingGates(config=config, data_providers={"fear_greed": provider})
        result = gates.evaluate("BUY")
        # 25 < 30 = extreme fear → boost BUY
        assert result.confidence_multiplier > 1.0

    def test_disabled_gate_no_effect(self):
        config = TradingGatesConfig(fear_greed_enabled=False)
        provider = MockGateDataProvider({"score": 5, "timestamp": datetime.now()})
        gates = TradingGates(config=config, data_providers={"fear_greed": provider})
        result = gates.evaluate("BUY")
        assert result.confidence_multiplier == 1.0


# ─── Stale Data Tests ────────────────────────────────────────────────────────

class TestStaleData:

    def test_stale_data_pass_action(self):
        """Stale data with PASS action should be ignored."""
        config = TradingGatesConfig(
            fear_greed_max_staleness_hours=1,
            stale_data_action="PASS",
        )
        # Old timestamp
        old_ts = datetime.now() - timedelta(hours=5)
        provider = MockGateDataProvider({"score": 10, "timestamp": old_ts})
        provider._last_fetch = old_ts

        gates = TradingGates(config=config, data_providers={"fear_greed": provider})
        result = gates.evaluate("BUY")
        # Stale + PASS = gate ignored, multiplier stays 1.0
        assert result.confidence_multiplier == 1.0

    def test_no_data_passes(self):
        """No data at all should PASS."""
        config = TradingGatesConfig()
        provider = MockGateDataProvider(None)
        gates = TradingGates(config=config, data_providers={"fear_greed": provider})
        result = gates.evaluate("BUY")
        assert result.confidence_multiplier == 1.0


# ─── Multiplier Tests ────────────────────────────────────────────────────────

class TestMultipliers:

    def test_multiplier_clamping_lower(self):
        """Multipliers should not go below minimum."""
        config = TradingGatesConfig(
            min_confidence_multiplier=0.3,
            fear_greed_confidence_reduce=0.1,  # Very aggressive
        )
        provider = MockGateDataProvider({"score": 90, "timestamp": datetime.now()})
        gates = TradingGates(config=config, data_providers={"fear_greed": provider})
        result = gates.evaluate("BUY")
        assert result.confidence_multiplier >= 0.3

    def test_multiplier_clamping_upper(self):
        """Multipliers should not exceed maximum."""
        config = TradingGatesConfig(
            max_confidence_multiplier=1.5,
            fear_greed_confidence_boost=3.0,  # Very aggressive
        )
        provider = MockGateDataProvider({"score": 5, "timestamp": datetime.now()})
        gates = TradingGates(config=config, data_providers={"fear_greed": provider})
        result = gates.evaluate("BUY")
        assert result.confidence_multiplier <= 1.5


# ─── Serialization Tests ─────────────────────────────────────────────────────

class TestSerialization:

    def test_gate_result_to_dict(self, default_config, fear_greed_provider):
        gates = TradingGates(
            config=default_config,
            data_providers={"fear_greed": fear_greed_provider},
        )
        result = gates.evaluate("BUY")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "timestamp" in d
        assert "is_blocked" in d
        assert "confidence_multiplier" in d
        assert "decisions" in d

    def test_gate_result_json_serializable(self, default_config, fear_greed_provider):
        """to_dict should produce JSON-serializable output."""
        import json
        gates = TradingGates(
            config=default_config,
            data_providers={"fear_greed": fear_greed_provider},
        )
        result = gates.evaluate("BUY")
        d = result.to_dict()
        serialized = json.dumps(d)
        assert len(serialized) > 0


# ─── Status Tests ────────────────────────────────────────────────────────────

class TestStatus:

    def test_get_status(self, default_config, fear_greed_provider):
        gates = TradingGates(
            config=default_config,
            data_providers={"fear_greed": fear_greed_provider},
        )
        status = gates.get_status()
        assert "enabled" in status
        assert "gates" in status
        assert "fear_greed" in status["gates"]

    def test_update_data(self, default_config, fear_greed_provider):
        gates = TradingGates(
            config=default_config,
            data_providers={"fear_greed": fear_greed_provider},
        )
        results = gates.update_data()
        assert "fear_greed" in results

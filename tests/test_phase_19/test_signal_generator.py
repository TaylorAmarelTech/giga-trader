"""Tests for SignalGenerator in phase_19_paper_trading."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from src.phase_19_paper_trading.alpaca_client import (
    SignalType,
    TradingSignal,
    TRADING_CONFIG,
    DynamicThresholds,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_model(proba_up: float = 0.7):
    """Return a mock sklearn model whose predict_proba returns [[1-p, p]]."""
    m = MagicMock()
    m.predict_proba.return_value = np.array([[1.0 - proba_up, proba_up]])
    return m


def _make_1min_df(n_bars: int = 400) -> pd.DataFrame:
    """Tiny synthetic 1-min DataFrame (enough for feature engineering)."""
    idx = pd.date_range("2026-01-02 09:30", periods=n_bars, freq="1min", tz="America/New_York")
    rng = np.random.RandomState(42)
    close = 450.0 + rng.randn(n_bars).cumsum() * 0.05
    return pd.DataFrame({
        "timestamp": idx,
        "open": close + rng.randn(n_bars) * 0.01,
        "high": close + abs(rng.randn(n_bars) * 0.05),
        "low": close - abs(rng.randn(n_bars) * 0.05),
        "close": close,
        "volume": rng.randint(100, 5000, n_bars),
    })


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def _patch_load(monkeypatch):
    """Prevent SignalGenerator.__init__ from touching the filesystem."""
    monkeypatch.setattr(
        "src.phase_19_paper_trading.signal_generator.DYNAMIC_SELECTOR_AVAILABLE",
        False,
    )


@pytest.fixture
def gen(_patch_load):
    """Return a SignalGenerator with mocked models injected."""
    with patch.object(
        __import__("src.phase_19_paper_trading.signal_generator", fromlist=["SignalGenerator"]).SignalGenerator,
        "_load_models",
    ):
        from src.phase_19_paper_trading.signal_generator import SignalGenerator
        sg = SignalGenerator.__new__(SignalGenerator)
        # Manually initialise the attributes that __init__ would set
        sg.model_dir = Path("models/production")
        sg.models = {}
        sg.scaler = None
        sg.dim_state = None
        sg.feature_cols = None
        sg.use_leak_proof = False
        sg.model_config = None
        sg.cascade_blend_weight = 0.0
        sg.temporal_cascade = None
        sg.calibrator = None
        sg.drift_monitor = None
        sg._drift_monitor_fitted = False
        sg.dynamic_weighter = None
        sg.thompson_selector = None
        sg._signal_count = 0
        sg._last_feature_quality = 1.0
        sg._prep_cache_key = None
        sg._prep_cache_result = None
        sg.use_dynamic_selector = False
        sg.dynamic_selector = None
        return sg


@pytest.fixture
def df_1min():
    return _make_1min_df()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInstantiation:
    """SignalGenerator can be constructed with mocked internals."""

    def test_generator_created(self, gen):
        assert gen is not None
        assert gen.models == {}

    def test_default_attributes(self, gen):
        assert gen.use_leak_proof is False
        assert gen.use_dynamic_selector is False
        assert gen._signal_count == 0
        assert gen._last_feature_quality == 1.0


class TestThresholdLogic:
    """Signal type is decided by swing_proba vs entry_threshold and timing_proba."""

    def _run_signal(self, gen, swing_p, timing_p, df_1min):
        """Inject mock models and call generate_signal."""
        gen.models["swing_pipeline"] = _mock_model(swing_p)
        gen.models["timing_pipeline"] = _mock_model(timing_p)
        gen.use_leak_proof = True
        gen.feature_cols = [f"f{i}" for i in range(10)]
        X = np.random.randn(1, 10)
        with patch.object(gen, "prepare_features", return_value=X):
            return gen.generate_signal(df_1min, current_price=450.0)

    def test_buy_signal(self, gen, df_1min):
        sig = self._run_signal(gen, swing_p=0.80, timing_p=0.65, df_1min=df_1min)
        assert sig.signal_type == SignalType.BUY

    def test_sell_signal(self, gen, df_1min):
        # swing_proba <= 1 - entry_threshold => SELL
        sig = self._run_signal(gen, swing_p=0.20, timing_p=0.65, df_1min=df_1min)
        assert sig.signal_type == SignalType.SELL

    def test_hold_low_timing(self, gen, df_1min):
        # High swing_proba but timing < 0.5 => HOLD
        sig = self._run_signal(gen, swing_p=0.80, timing_p=0.40, df_1min=df_1min)
        assert sig.signal_type == SignalType.HOLD

    def test_hold_neutral_swing(self, gen, df_1min):
        # swing_proba near 0.5 => HOLD
        sig = self._run_signal(gen, swing_p=0.50, timing_p=0.60, df_1min=df_1min)
        assert sig.signal_type == SignalType.HOLD


class TestEnsemble:
    """Legacy L2+GB ensemble weighting."""

    def test_ensemble_average_no_disagreement(self, gen, df_1min):
        gen.models["swing_l2"] = _mock_model(0.70)
        gen.models["swing_gb"] = _mock_model(0.72)
        gen.models["timing_l2"] = _mock_model(0.60)
        gen.models["timing_gb"] = _mock_model(0.62)
        gen.use_leak_proof = False
        X = np.random.randn(1, 10)
        with patch.object(gen, "prepare_features", return_value=X):
            sig = gen.generate_signal(df_1min, current_price=450.0)
        # Disagreement < 0.2 => simple average => ~0.71 => above default 0.65 threshold
        assert sig.signal_type == SignalType.BUY
        assert 0.70 <= sig.probability <= 0.72

    def test_ensemble_high_disagreement(self, gen, df_1min):
        gen.models["swing_l2"] = _mock_model(0.90)
        gen.models["swing_gb"] = _mock_model(0.55)
        gen.models["timing_l2"] = _mock_model(0.60)
        gen.models["timing_gb"] = _mock_model(0.60)
        gen.use_leak_proof = False
        X = np.random.randn(1, 10)
        with patch.object(gen, "prepare_features", return_value=X):
            sig = gen.generate_signal(df_1min, current_price=450.0)
        # Disagreement 0.35 > 0.2 => confidence_penalty applied
        assert sig.confidence < 1.0


class TestConfidenceCalculation:
    """Confidence = |swing_proba - 0.5| * 2 * confidence_penalty."""

    def test_max_confidence(self, gen, df_1min):
        gen.models["swing_pipeline"] = _mock_model(1.0)
        gen.models["timing_pipeline"] = _mock_model(0.8)
        gen.use_leak_proof = True
        gen.feature_cols = [f"f{i}" for i in range(5)]
        X = np.random.randn(1, 5)
        with patch.object(gen, "prepare_features", return_value=X):
            sig = gen.generate_signal(df_1min, current_price=450.0)
        # base_confidence = |1.0 - 0.5| * 2 = 1.0; penalty = 1.0 (leak-proof)
        assert sig.confidence == pytest.approx(1.0)

    def test_near_half_confidence(self, gen, df_1min):
        gen.models["swing_pipeline"] = _mock_model(0.55)
        gen.models["timing_pipeline"] = _mock_model(0.60)
        gen.use_leak_proof = True
        gen.feature_cols = [f"f{i}" for i in range(5)]
        X = np.random.randn(1, 5)
        with patch.object(gen, "prepare_features", return_value=X):
            sig = gen.generate_signal(df_1min, current_price=450.0)
        # base_confidence = |0.55 - 0.5| * 2 = 0.10
        assert sig.confidence == pytest.approx(0.10)


class TestErrorHandling:
    """Graceful degradation when inputs are bad or models missing."""

    def test_empty_features_returns_hold(self, gen, df_1min):
        gen.models["swing_pipeline"] = _mock_model(0.80)
        gen.models["timing_pipeline"] = _mock_model(0.60)
        gen.use_leak_proof = True
        with patch.object(gen, "prepare_features", return_value=np.array([])):
            sig = gen.generate_signal(df_1min, current_price=450.0)
        assert sig.signal_type == SignalType.HOLD

    def test_no_swing_model_returns_hold(self, gen, df_1min):
        gen.models.clear()
        sig = gen.generate_signal(df_1min, current_price=450.0)
        assert sig.signal_type == SignalType.HOLD

    def test_no_timing_model_returns_hold(self, gen, df_1min):
        gen.models["swing_pipeline"] = _mock_model(0.80)
        gen.use_leak_proof = True
        gen.feature_cols = [f"f{i}" for i in range(5)]
        X = np.random.randn(1, 5)
        with patch.object(gen, "prepare_features", return_value=X):
            sig = gen.generate_signal(df_1min, current_price=450.0)
        assert sig.signal_type == SignalType.HOLD

    def test_model_exception_returns_hold(self, gen, df_1min):
        bad_model = MagicMock()
        bad_model.predict_proba.side_effect = RuntimeError("corrupt model")
        gen.models["swing_pipeline"] = bad_model
        gen.models["timing_pipeline"] = _mock_model(0.6)
        gen.use_leak_proof = True
        gen.feature_cols = [f"f{i}" for i in range(5)]
        X = np.random.randn(1, 5)
        with patch.object(gen, "prepare_features", return_value=X):
            sig = gen.generate_signal(df_1min, current_price=450.0)
        assert sig.signal_type == SignalType.HOLD

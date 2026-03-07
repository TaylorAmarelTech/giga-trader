"""Tests for AdaptiveEventDrivenLabeler (AEDL)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_05_targets.adaptive_event_labeler import AdaptiveEventDrivenLabeler


def _make_ohlcv(n=300, seed=42):
    """Generate synthetic OHLCV daily data."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    opn = close + rng.randn(n) * 0.3
    vol = rng.randint(1_000_000, 10_000_000, n).astype(float)
    df = pd.DataFrame({
        "date": dates,
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    })
    df["daily_return"] = df["close"].pct_change().fillna(0)
    return df


class TestAEDLInit:
    def test_default_construction(self):
        labeler = AdaptiveEventDrivenLabeler()
        assert labeler.base_tp_pct > 0
        assert labeler.base_sl_pct > 0

    def test_custom_params(self):
        labeler = AdaptiveEventDrivenLabeler(
            base_tp_pct=0.02, base_sl_pct=0.015,
            max_holding_days=10, label_mode="ternary",
        )
        assert labeler.base_tp_pct == 0.02
        assert labeler.label_mode == "ternary"

    def test_invalid_label_mode(self):
        with pytest.raises(ValueError, match="label_mode"):
            AdaptiveEventDrivenLabeler(label_mode="invalid")

    def test_invalid_tp_pct(self):
        with pytest.raises(ValueError):
            AdaptiveEventDrivenLabeler(base_tp_pct=-0.01)

    def test_invalid_sl_pct(self):
        with pytest.raises(ValueError):
            AdaptiveEventDrivenLabeler(base_sl_pct=0)


class TestAEDLFitLabel:
    def test_fit_returns_self(self):
        df = _make_ohlcv()
        labeler = AdaptiveEventDrivenLabeler()
        result = labeler.fit(df)
        assert result is labeler

    def test_label_produces_expected_columns(self):
        df = _make_ohlcv()
        labeler = AdaptiveEventDrivenLabeler()
        labeler.fit(df)
        result = labeler.label(df)
        for col in ["aedl_label", "aedl_confidence", "aedl_tp_used",
                     "aedl_sl_used", "aedl_vol_ratio"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_binary_labels_are_0_or_1(self):
        df = _make_ohlcv()
        labeler = AdaptiveEventDrivenLabeler(label_mode="binary")
        labeler.fit(df)
        result = labeler.label(df)
        labels = result["aedl_label"].dropna()
        assert set(labels.unique()).issubset({0, 1, 0.0, 1.0})

    def test_ternary_labels(self):
        df = _make_ohlcv()
        labeler = AdaptiveEventDrivenLabeler(label_mode="ternary")
        labeler.fit(df)
        result = labeler.label(df)
        labels = result["aedl_label"].dropna()
        assert set(labels.unique()).issubset({-1, 0, 1, -1.0, 0.0, 1.0})

    def test_soft_labels_in_01_range(self):
        df = _make_ohlcv()
        labeler = AdaptiveEventDrivenLabeler(label_mode="soft")
        labeler.fit(df)
        result = labeler.label(df)
        labels = result["aedl_label"].dropna()
        assert labels.min() >= 0.0
        assert labels.max() <= 1.0

    def test_continuous_labels(self):
        df = _make_ohlcv()
        labeler = AdaptiveEventDrivenLabeler(label_mode="continuous")
        labeler.fit(df)
        result = labeler.label(df)
        labels = result["aedl_label"].dropna()
        # Continuous labels are raw returns — can be positive or negative
        assert len(labels) > 0

    def test_confidence_in_valid_range(self):
        df = _make_ohlcv()
        labeler = AdaptiveEventDrivenLabeler()
        labeler.fit(df)
        result = labeler.label(df)
        conf = result["aedl_confidence"].dropna()
        assert conf.min() >= 0.0
        assert conf.max() <= 1.0

    def test_adaptive_barriers_vary_with_vol(self):
        df = _make_ohlcv()
        labeler = AdaptiveEventDrivenLabeler()
        labeler.fit(df)
        result = labeler.label(df)
        tp_used = result["aedl_tp_used"].dropna()
        # Barriers should not all be identical (they adapt to vol)
        assert tp_used.std() > 0, "Barriers should vary with volatility"

    def test_barrier_clamping(self):
        df = _make_ohlcv()
        labeler = AdaptiveEventDrivenLabeler(
            min_barrier_pct=0.005, max_barrier_pct=0.02,
        )
        labeler.fit(df)
        result = labeler.label(df)
        tp_used = result["aedl_tp_used"].dropna()
        assert tp_used.min() >= 0.005 - 1e-9
        assert tp_used.max() <= 0.02 + 1e-9


class TestAEDLEdgeCases:
    def test_short_dataframe_raises(self):
        df = _make_ohlcv(n=10)
        labeler = AdaptiveEventDrivenLabeler()
        with pytest.raises(ValueError, match="at least"):
            labeler.fit(df)

    def test_barrier_history(self):
        df = _make_ohlcv()
        labeler = AdaptiveEventDrivenLabeler()
        labeler.fit(df)
        labeler.label(df)
        history = labeler.get_barrier_history()
        assert isinstance(history, pd.DataFrame)
        assert len(history) > 0

    def test_vol_scaling_multiplier(self):
        df = _make_ohlcv()
        labeler = AdaptiveEventDrivenLabeler(vol_scaling=2.0)
        labeler.fit(df)
        result = labeler.label(df)
        tp_high = result["aedl_tp_used"].median()

        labeler2 = AdaptiveEventDrivenLabeler(vol_scaling=0.5)
        labeler2.fit(df)
        result2 = labeler2.label(df)
        tp_low = result2["aedl_tp_used"].median()

        assert tp_high >= tp_low, "Higher vol_scaling should produce wider or equal barriers"

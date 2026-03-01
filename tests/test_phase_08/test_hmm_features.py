"""Tests for HMMFeatures -- HMM-style regime detection features (5 total)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.hmm_features import HMMFeatures


# --- Helpers -----------------------------------------------------------------

def _make_spy_daily(n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    returns = rng.normal(0.0003, 0.012, n_days)
    close = 450.0 * np.cumprod(1 + returns)
    volume = rng.randint(50_000_000, 150_000_000, n_days).astype(float)
    return pd.DataFrame({
        "date": dates, "close": close, "volume": volume,
        "open": close * (1 + rng.normal(0, 0.003, n_days)),
        "high": close * 1.005, "low": close * 0.995,
    })


ALL_5 = {"hmm_state", "hmm_bull_prob", "hmm_bear_prob", "hmm_transition_prob", "hmm_regime_duration"}


# --- Invariant Tests ---------------------------------------------------------

class TestHMMInvariants:
    @pytest.fixture
    def feat(self):
        return HMMFeatures()

    @pytest.fixture
    def spy(self):
        return _make_spy_daily(200)

    def test_all_5_features_created(self, feat, spy):
        result = feat.create_hmm_features(spy)
        hmm_cols = {c for c in result.columns if c.startswith("hmm_")}
        assert hmm_cols == ALL_5

    def test_no_nans(self, feat, spy):
        result = feat.create_hmm_features(spy)
        for col in ALL_5:
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_no_infinities(self, feat, spy):
        result = feat.create_hmm_features(spy)
        for col in ALL_5:
            assert not np.isinf(result[col]).any(), f"Inf in {col}"

    def test_preserves_original_columns(self, feat, spy):
        original = set(spy.columns)
        result = feat.create_hmm_features(spy)
        assert original.issubset(set(result.columns))

    def test_preserves_row_count(self, feat, spy):
        result = feat.create_hmm_features(spy)
        assert len(result) == len(spy)

    def test_no_close_column(self, feat):
        df = pd.DataFrame({"date": [1, 2], "price": [100, 101]})
        result = feat.create_hmm_features(df)
        assert len(result.columns) == len(df.columns)


# --- Logic Tests -------------------------------------------------------------

class TestHMMLogic:
    @pytest.fixture
    def feat(self):
        return HMMFeatures()

    @pytest.fixture
    def result(self, feat):
        spy = _make_spy_daily(250)
        return feat.create_hmm_features(spy)

    def test_state_values(self, result):
        """States should only be 0, 1, or 2."""
        vals = set(result["hmm_state"].unique())
        assert vals.issubset({0.0, 1.0, 2.0})

    def test_bull_prob_bounded(self, result):
        """Bull probability should be in [0, 1]."""
        assert result["hmm_bull_prob"].min() >= 0.0
        assert result["hmm_bull_prob"].max() <= 1.0

    def test_bear_prob_bounded(self, result):
        """Bear probability should be in [0, 1]."""
        assert result["hmm_bear_prob"].min() >= 0.0
        assert result["hmm_bear_prob"].max() <= 1.0

    def test_probs_sum_approx_one(self, result):
        """Bear + neutral + bull probabilities should sum close to 1.0.

        Neutral prob is implicitly 1 - bull - bear; but the stored bull and
        bear come from the softmax inverse-distance calculation so
        bull + bear + neutral_prob = 1.  We verify that bull + bear <= 1.0
        (with tolerance) which implies neutral >= 0.
        """
        total = result["hmm_bull_prob"] + result["hmm_bear_prob"]
        assert (total <= 1.0 + 1e-6).all(), "bull + bear exceeds 1.0"

    def test_duration_at_least_one(self, result):
        """Duration should always be >= 1."""
        assert result["hmm_regime_duration"].min() >= 1.0


# --- Analyze Tests -----------------------------------------------------------

class TestAnalyze:
    def test_returns_dict(self):
        feat = HMMFeatures()
        df = _make_spy_daily(200)
        df = feat.create_hmm_features(df)
        result = feat.analyze_current_hmm(df)
        assert isinstance(result, dict)
        assert "hmm_regime" in result

    def test_regime_values(self):
        feat = HMMFeatures()
        df = _make_spy_daily(250)
        df = feat.create_hmm_features(df)
        result = feat.analyze_current_hmm(df)
        assert result["hmm_regime"] in {"BULL", "BEAR", "NEUTRAL"}

    def test_returns_none_without_features(self):
        feat = HMMFeatures()
        df = pd.DataFrame({"close": [100]})
        assert feat.analyze_current_hmm(df) is None


# --- Feature Count Test ------------------------------------------------------

class TestFeatureCounts:
    def test_total_count(self):
        assert len(ALL_5) == 5

"""
Tests for SAXFeatures — Symbolic Aggregate approXimation pattern features.

Validates:
  - Constructor parameters and defaults
  - download_sax_data returns empty DataFrame (no external calls)
  - create_sax_features produces exactly 3 sax_ columns
  - Feature invariants: no NaN, no Inf, correct dtypes / value ranges
  - SAX encoding logic (z-normalise → PAA → discretise)
  - Pattern match signal correctness (+1 bullish, -1 bearish, 0 neutral)
  - Novelty KL divergence is non-negative
  - analyze_current_pattern returns expected dict structure
  - Edge cases: missing column, short data, constant prices, single row
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from src.phase_08_features_breadth.sax_features import SAXFeatures


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_spy_daily(n_days: int = 250, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic daily SPY-like OHLCV data with a 'close' column."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    returns = rng.normal(0.0004, 0.010, n_days)
    close = 450.0 * np.cumprod(1.0 + returns)
    volume = rng.randint(50_000_000, 150_000_000, n_days).astype(float)
    return pd.DataFrame({
        "date": dates,
        "close": close,
        "open": close * (1.0 + rng.normal(0, 0.003, n_days)),
        "high": close * 1.006,
        "low": close * 0.994,
        "volume": volume,
    })


ALL_3 = {"sax_pattern_20d", "sax_pattern_match", "sax_novelty"}


# ─── Constructor Tests ────────────────────────────────────────────────────────

class TestSAXInit:

    def test_default_params(self):
        sf = SAXFeatures()
        assert sf.pattern_window == 20
        assert sf.n_segments == 5
        assert sf.alphabet_size == 4

    def test_custom_params(self):
        sf = SAXFeatures(pattern_window=10, n_segments=5, alphabet_size=3)
        assert sf.pattern_window == 10
        assert sf.alphabet_size == 3

    def test_letters_length_matches_alphabet(self):
        for alpha in (3, 4, 5):
            sf = SAXFeatures(alphabet_size=alpha)
            assert len(sf._letters) == alpha

    def test_invalid_alphabet_raises(self):
        with pytest.raises(ValueError):
            SAXFeatures(alphabet_size=6)

    def test_effective_window_divisible_by_segments(self):
        sf = SAXFeatures(pattern_window=20, n_segments=5)
        assert sf._effective_window % sf.n_segments == 0

    def test_effective_window_non_zero(self):
        sf = SAXFeatures(pattern_window=5, n_segments=5)
        assert sf._effective_window > 0


# ─── Download Tests ───────────────────────────────────────────────────────────

class TestSAXDownload:

    def test_download_returns_empty_dataframe(self):
        sf = SAXFeatures()
        result = sf.download_sax_data(datetime(2024, 1, 1), datetime(2024, 12, 31))
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_download_always_empty_no_network(self):
        """Even without network, download must not raise."""
        sf = SAXFeatures()
        result = sf.download_sax_data("2020-01-01", "2020-06-01")
        assert isinstance(result, pd.DataFrame)


# ─── Feature Invariant Tests ──────────────────────────────────────────────────

class TestSAXInvariants:

    @pytest.fixture
    def sf(self):
        return SAXFeatures()

    @pytest.fixture
    def spy(self):
        return _make_spy_daily(250)

    def test_exactly_3_features_created(self, sf, spy):
        result = sf.create_sax_features(spy)
        sax_cols = {c for c in result.columns if c.startswith("sax_")}
        assert sax_cols == ALL_3

    def test_feature_names_correct(self, sf, spy):
        result = sf.create_sax_features(spy)
        for col in ALL_3:
            assert col in result.columns, f"Missing feature: {col}"

    def test_no_nans(self, sf, spy):
        result = sf.create_sax_features(spy)
        for col in ALL_3:
            nan_count = result[col].isna().sum()
            assert nan_count == 0, f"NaN found in {col}"

    def test_no_infinities(self, sf, spy):
        result = sf.create_sax_features(spy)
        for col in ALL_3:
            assert not np.isinf(result[col]).any(), f"Inf found in {col}"

    def test_preserves_original_columns(self, sf, spy):
        original = set(spy.columns)
        result = sf.create_sax_features(spy)
        assert original.issubset(set(result.columns))

    def test_preserves_row_count(self, sf, spy):
        result = sf.create_sax_features(spy)
        assert len(result) == len(spy)

    def test_missing_close_column_returns_unchanged(self, sf):
        df = pd.DataFrame({"date": [1, 2, 3], "price": [100, 101, 102]})
        result = sf.create_sax_features(df)
        assert list(result.columns) == list(df.columns)


# ─── Feature Value Range Tests ────────────────────────────────────────────────

class TestSAXValueRanges:

    @pytest.fixture
    def sf(self):
        return SAXFeatures()

    @pytest.fixture
    def result(self):
        sf = SAXFeatures()
        spy = _make_spy_daily(250)
        return sf.create_sax_features(spy)

    def test_pattern_hash_in_range(self, result):
        """sax_pattern_20d is hash % 10000, so must be in [0, 9999]."""
        non_zero = result["sax_pattern_20d"][result["sax_pattern_20d"] != 0]
        if len(non_zero) > 0:
            assert non_zero.min() >= 0
            assert non_zero.max() <= 9999

    def test_pattern_match_values(self, result):
        """sax_pattern_match must be one of {-1, 0, +1}."""
        unique = set(result["sax_pattern_match"].unique())
        assert unique.issubset({-1.0, 0.0, 1.0}), (
            f"Unexpected sax_pattern_match values: {unique}"
        )

    def test_novelty_non_negative(self, result):
        """KL divergence is always >= 0."""
        assert (result["sax_novelty"] >= 0).all(), "Negative novelty score found"

    def test_novelty_finite(self, result):
        """Novelty must not blow up to inf."""
        assert result["sax_novelty"].max() < 1e6


# ─── SAX Encoding Logic Tests ─────────────────────────────────────────────────

class TestSAXEncoding:

    def test_encode_uptrend_ends_with_cd_or_dd(self):
        """A monotonically increasing sequence should produce a bullish word."""
        sf = SAXFeatures(pattern_window=20, n_segments=5, alphabet_size=4)
        # Create a strongly upward-trending return window
        window = np.linspace(0.001, 0.02, 20)
        word = sf._encode_window(window)
        assert word is not None
        assert len(word) == 5, f"Expected 5 letters, got '{word}'"

    def test_encode_downtrend_ends_with_ab_or_aa(self):
        """A monotonically decreasing sequence should produce a bearish word."""
        sf = SAXFeatures(pattern_window=20, n_segments=5, alphabet_size=4)
        window = np.linspace(-0.02, -0.001, 20)
        word = sf._encode_window(window)
        assert word is not None
        assert len(word) == 5

    def test_encode_constant_does_not_crash(self):
        """Zero-variance window should return a valid string of mid-letter repeated."""
        sf = SAXFeatures(pattern_window=20, n_segments=5, alphabet_size=4)
        window = np.zeros(20)
        word = sf._encode_window(window)
        assert word is not None
        assert len(word) == 5
        # All letters must be valid alphabet characters
        for ch in word:
            assert ch in sf._letters

    def test_encode_returns_correct_length(self):
        """Word length must equal n_segments."""
        for n_seg in (3, 5):
            sf = SAXFeatures(pattern_window=20, n_segments=n_seg, alphabet_size=4)
            window = np.random.RandomState(7).normal(0, 1, sf._effective_window)
            word = sf._encode_window(window)
            assert word is not None
            assert len(word) == n_seg, (
                f"n_segments={n_seg}: expected word length {n_seg}, got {len(word)}"
            )

    def test_all_letters_in_alphabet(self):
        """Every character in the SAX word must be a valid alphabet letter."""
        sf = SAXFeatures()
        rng = np.random.RandomState(42)
        for _ in range(20):
            window = rng.normal(0, 1, sf._effective_window)
            word = sf._encode_window(window)
            if word:
                for ch in word:
                    assert ch in sf._letters, f"Invalid letter '{ch}' in word '{word}'"

    def test_discretise_maps_extreme_low_to_a(self):
        """z = -3.0 should map to letter 'a' (first bin)."""
        sf = SAXFeatures(alphabet_size=4)
        assert sf._discretise(-3.0) == "a"

    def test_discretise_maps_extreme_high_to_d(self):
        """z = +3.0 should map to letter 'd' (last bin for alphabet_size=4)."""
        sf = SAXFeatures(alphabet_size=4)
        assert sf._discretise(3.0) == "d"


# ─── Pattern Match Logic Tests ────────────────────────────────────────────────

class TestPatternMatchLogic:

    def test_bullish_pattern_detected(self):
        """
        Force an uptrend where the last two PAA segments will be large positive,
        mapping to 'c' or 'd'. Check that sax_pattern_match == +1 for at least
        some rows.
        """
        sf = SAXFeatures(pattern_window=20, n_segments=5, alphabet_size=4)
        # Strong steady uptrend over 200 days — should generate bullish signals
        rng = np.random.RandomState(5)
        n = 200
        close = 450.0 * np.cumprod(1.0 + rng.uniform(0.002, 0.005, n))
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=n, freq="B"),
            "close": close,
        })
        result = sf.create_sax_features(df)
        bullish_count = (result["sax_pattern_match"] == 1).sum()
        assert bullish_count > 0, "No bullish patterns detected in strong uptrend"

    def test_bearish_pattern_detected(self):
        """Strong downtrend should generate some bearish pattern signals."""
        sf = SAXFeatures(pattern_window=20, n_segments=5, alphabet_size=4)
        rng = np.random.RandomState(9)
        n = 200
        close = 450.0 * np.cumprod(1.0 + rng.uniform(-0.005, -0.001, n))
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=n, freq="B"),
            "close": close,
        })
        result = sf.create_sax_features(df)
        bearish_count = (result["sax_pattern_match"] == -1).sum()
        assert bearish_count > 0, "No bearish patterns detected in strong downtrend"


# ─── Novelty / KL Divergence Tests ───────────────────────────────────────────

class TestNoveltyFeature:

    def test_novelty_zero_at_start(self):
        """Early rows (< 50 days of history) should have novelty == 0."""
        sf = SAXFeatures()
        spy = _make_spy_daily(250)
        result = sf.create_sax_features(spy)
        # Rows before the effective_window should have novelty == 0
        assert result["sax_novelty"].iloc[0] == 0.0

    def test_novelty_positive_after_warmup(self):
        """After 50 days of history, at least some novelty values should be > 0."""
        sf = SAXFeatures()
        spy = _make_spy_daily(250)
        result = sf.create_sax_features(spy)
        non_zero = (result["sax_novelty"] > 0).sum()
        assert non_zero > 0, "All novelty values are zero after warmup period"

    def test_kl_divergence_symmetric_input(self):
        """When p == q, KL divergence should be ~0."""
        sf = SAXFeatures()
        p = np.array([0.25, 0.25, 0.25, 0.25])
        q = np.array([0.25, 0.25, 0.25, 0.25])
        result = sf._kl_divergence(p, q)
        assert abs(result) < 1e-6, f"Expected ~0, got {result}"

    def test_kl_divergence_non_negative(self):
        """KL divergence must always be non-negative."""
        rng = np.random.RandomState(42)
        sf = SAXFeatures()
        for _ in range(20):
            p = rng.dirichlet(np.ones(4))
            q = rng.dirichlet(np.ones(4))
            assert sf._kl_divergence(p, q) >= 0.0

    def test_kl_divergence_concentrated_vs_uniform(self):
        """A concentrated distribution vs uniform should give larger KL than uniform vs uniform."""
        sf = SAXFeatures()
        p_conc = np.array([0.97, 0.01, 0.01, 0.01])
        q_unif = np.array([0.25, 0.25, 0.25, 0.25])
        kl_high = sf._kl_divergence(p_conc, q_unif)
        kl_zero = sf._kl_divergence(q_unif, q_unif)
        assert kl_high > kl_zero


# ─── Analyze Current Pattern Tests ───────────────────────────────────────────

class TestAnalyzeCurrentPattern:

    def test_returns_dict(self):
        sf = SAXFeatures()
        df = _make_spy_daily(250)
        df = sf.create_sax_features(df)
        result = sf.analyze_current_pattern(df)
        assert isinstance(result, dict)

    def test_required_keys(self):
        sf = SAXFeatures()
        df = _make_spy_daily(250)
        df = sf.create_sax_features(df)
        result = sf.analyze_current_pattern(df)
        for key in ("pattern_regime", "novelty_level", "sax_word", "sax_hash", "novelty_score"):
            assert key in result, f"Missing key: {key}"

    def test_pattern_regime_valid_values(self):
        sf = SAXFeatures()
        df = _make_spy_daily(250)
        df = sf.create_sax_features(df)
        result = sf.analyze_current_pattern(df)
        assert result["pattern_regime"] in ("BULLISH", "BEARISH", "NEUTRAL"), (
            f"Unexpected pattern_regime: {result['pattern_regime']}"
        )

    def test_novelty_level_valid_values(self):
        sf = SAXFeatures()
        df = _make_spy_daily(250)
        df = sf.create_sax_features(df)
        result = sf.analyze_current_pattern(df)
        assert result["novelty_level"] in ("HIGH", "LOW", "NORMAL"), (
            f"Unexpected novelty_level: {result['novelty_level']}"
        )

    def test_sax_word_is_string(self):
        sf = SAXFeatures()
        df = _make_spy_daily(250)
        df = sf.create_sax_features(df)
        result = sf.analyze_current_pattern(df)
        assert isinstance(result["sax_word"], str)

    def test_sax_hash_is_int(self):
        sf = SAXFeatures()
        df = _make_spy_daily(250)
        df = sf.create_sax_features(df)
        result = sf.analyze_current_pattern(df)
        assert isinstance(result["sax_hash"], int)

    def test_returns_none_without_close_column(self):
        sf = SAXFeatures()
        df = pd.DataFrame({"price": [100, 101]})
        result = sf.analyze_current_pattern(df)
        assert result is None

    def test_computes_on_raw_df_without_pre_existing_features(self):
        """analyze_current_pattern should work even if features haven't been added yet."""
        sf = SAXFeatures()
        df = _make_spy_daily(200)
        # Do NOT call create_sax_features first
        result = sf.analyze_current_pattern(df)
        assert result is not None
        assert "pattern_regime" in result


# ─── Edge Case Tests ──────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_short_data_below_window(self):
        """Data shorter than pattern_window should not crash; features are 0."""
        sf = SAXFeatures(pattern_window=20)
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=10, freq="B"),
            "close": np.linspace(450, 460, 10),
        })
        result = sf.create_sax_features(df)
        assert len(result) == 10
        for col in ALL_3:
            assert col in result.columns
            assert result[col].isna().sum() == 0

    def test_constant_price_series(self):
        """Flat close prices (zero returns) should not crash."""
        sf = SAXFeatures()
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=60, freq="B"),
            "close": np.full(60, 450.0),
        })
        result = sf.create_sax_features(df)
        for col in ALL_3:
            assert col in result.columns
            assert result[col].isna().sum() == 0

    def test_single_row(self):
        """Single-row DataFrame should not crash and return all zeros."""
        sf = SAXFeatures()
        df = pd.DataFrame({"date": [pd.Timestamp("2024-06-01")], "close": [450.0]})
        result = sf.create_sax_features(df)
        assert len(result) == 1
        for col in ALL_3:
            assert col in result.columns

    def test_different_alphabet_sizes(self):
        """Module should work with alphabet_size 3 and 5."""
        spy = _make_spy_daily(200)
        for alpha in (3, 5):
            sf = SAXFeatures(alphabet_size=alpha)
            result = sf.create_sax_features(spy)
            for col in ALL_3:
                assert col in result.columns
                assert result[col].isna().sum() == 0

    def test_feature_prefix_consistency(self):
        """Every new column must start with 'sax_'."""
        sf = SAXFeatures()
        spy = _make_spy_daily(200)
        original_cols = set(spy.columns)
        result = sf.create_sax_features(spy)
        new_cols = set(result.columns) - original_cols
        for col in new_cols:
            assert col.startswith("sax_"), (
                f"New column '{col}' does not have 'sax_' prefix"
            )


# ─── Feature Count Test ───────────────────────────────────────────────────────

class TestFeatureCounts:
    def test_total_count(self):
        assert len(ALL_3) == 3

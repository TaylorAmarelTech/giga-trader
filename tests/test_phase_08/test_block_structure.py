"""
Tests for BlockStructureFeatures — multi-day block structure features (54 total).
"""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.block_structure_features import (
    BlockStructureFeatures,
)


# ─── Helpers ────────────────────────────────────────────────────────────

def _make_spy_daily(n_days: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create realistic synthetic SPY daily data for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")

    returns = rng.normal(0.0003, 0.012, n_days)
    # Inject specific events
    if n_days > 50:
        returns[50] = -0.025   # crash
    if n_days > 51:
        returns[51] = 0.015    # rebound
    if n_days > 100:
        returns[100] = -0.011
    if n_days > 150:
        returns[150] = 0.022
    if n_days > 203:
        returns[200] = -0.032
        returns[201] = -0.015
        returns[202] = -0.008
        returns[203] = 0.018
    # Inject trending period (days 220-230: all up)
    if n_days > 230:
        for i in range(220, 230):
            returns[i] = abs(returns[i]) + 0.005

    close = 450.0 * np.cumprod(1 + returns)
    open_price = close * (1 + rng.normal(0, 0.003, n_days))
    high = np.maximum(open_price, close) * (1 + rng.uniform(0, 0.008, n_days))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0, 0.008, n_days))
    volume = rng.randint(50_000_000, 150_000_000, n_days).astype(float)
    if n_days > 80:
        volume[80] = volume[80] * 3

    return pd.DataFrame({
        "date": dates,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def _make_trending_up(n_days: int = 60) -> pd.DataFrame:
    """Create a monotonically rising market."""
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    close = np.linspace(400, 500, n_days)
    return pd.DataFrame({
        "date": dates,
        "open": close * 0.999,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": [1e8] * n_days,
    })


def _make_trending_down(n_days: int = 60) -> pd.DataFrame:
    """Create a monotonically falling market."""
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    close = np.linspace(500, 400, n_days)
    return pd.DataFrame({
        "date": dates,
        "open": close * 1.001,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": [1e8] * n_days,
    })


def _make_choppy(n_days: int = 60, seed: int = 99) -> pd.DataFrame:
    """Create a choppy, mean-reverting market."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    # Alternating up/down
    returns = np.zeros(n_days)
    for i in range(1, n_days):
        returns[i] = -returns[i - 1] + rng.normal(0, 0.002)
        returns[i] = max(-0.02, min(0.02, returns[i]))
    close = 450.0 * np.cumprod(1 + returns)
    return pd.DataFrame({
        "date": dates,
        "open": close * (1 + rng.normal(0, 0.002, n_days)),
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
        "volume": rng.randint(80_000_000, 120_000_000, n_days).astype(float),
    })


# ─── Feature Set Definitions ─────────────────────────────────────────

BLOCK_RETURN_12 = {
    "blk_3d_return", "blk_5d_return",
    "blk_3d_consistency", "blk_5d_consistency",
    "blk_3d_pattern_score", "blk_5d_pattern_score",
    "blk_3d_acceleration", "blk_5d_acceleration",
    "blk_3d_reversal_boundary", "blk_5d_reversal_boundary",
    "blk_3d_vs_5d_agreement", "blk_sequential_strength",
}

CASCADE_10 = {
    "blk_cascade_3_5_10", "blk_cascade_5_10_20", "blk_cascade_full",
    "blk_cascade_break_short", "blk_cascade_break_long",
    "blk_cascade_ratio_3_10", "blk_cascade_ratio_5_20",
    "blk_cascade_vol_expand", "blk_cascade_vol_compress",
    "blk_cascade_return_rank",
}

INTRA_BLOCK_10 = {
    "blk_3d_where_extreme", "blk_5d_where_extreme",
    "blk_3d_front_vs_back", "blk_5d_front_vs_back",
    "blk_3d_range_position", "blk_5d_range_position",
    "blk_3d_max_drawdown", "blk_5d_max_drawdown",
    "blk_3d_recovery_ratio", "blk_5d_recovery_ratio",
}

VOL_PROFILE_8 = {
    "blk_3d_vol_trend", "blk_5d_vol_trend",
    "blk_3d_vol_return_agree", "blk_5d_vol_return_agree",
    "blk_3d_vol_concentration", "blk_5d_vol_concentration",
    "blk_3d_vwap_vs_close", "blk_5d_vwap_vs_close",
}

BOUNDARY_8 = {
    "blk_boundary_3d_gap", "blk_boundary_5d_gap",
    "blk_boundary_3d_vol_shift", "blk_boundary_5d_vol_shift",
    "blk_boundary_3d_streak", "blk_boundary_5d_streak",
    "blk_boundary_3d_revert", "blk_boundary_5d_revert",
}

TEXTURE_6 = {
    "blk_3d_autocorr", "blk_5d_autocorr",
    "blk_return_roughness", "blk_3d_kurtosis",
    "blk_5d_dispersion", "blk_hurst_proxy",
}

ALL_54 = (
    BLOCK_RETURN_12 | CASCADE_10 | INTRA_BLOCK_10 |
    VOL_PROFILE_8 | BOUNDARY_8 | TEXTURE_6
)


# ─── Core Invariant Tests ─────────────────────────────────────────────

class TestBlockStructureInvariants:
    @pytest.fixture
    def blk(self):
        return BlockStructureFeatures()

    @pytest.fixture
    def spy_daily(self):
        return _make_spy_daily(300)

    def test_all_54_features_created(self, blk, spy_daily):
        result = blk.create_block_structure_features(spy_daily)
        blk_cols = {c for c in result.columns if c.startswith("blk_")}
        assert blk_cols == ALL_54, f"Missing: {ALL_54 - blk_cols}, Extra: {blk_cols - ALL_54}"

    def test_no_nans(self, blk, spy_daily):
        result = blk.create_block_structure_features(spy_daily)
        blk_cols = [c for c in result.columns if c.startswith("blk_")]
        for col in blk_cols:
            assert result[col].isna().sum() == 0, f"NaN found in {col}"

    def test_no_infinities(self, blk, spy_daily):
        result = blk.create_block_structure_features(spy_daily)
        blk_cols = [c for c in result.columns if c.startswith("blk_")]
        for col in blk_cols:
            assert not np.isinf(result[col]).any(), f"Inf found in {col}"

    def test_preserves_original_columns(self, blk, spy_daily):
        original = set(spy_daily.columns)
        result = blk.create_block_structure_features(spy_daily)
        assert original.issubset(set(result.columns))

    def test_preserves_row_count(self, blk, spy_daily):
        result = blk.create_block_structure_features(spy_daily)
        assert len(result) == len(spy_daily)

    def test_no_close_column(self, blk):
        df = pd.DataFrame({"date": pd.bdate_range("2024-01-02", periods=10), "price": [100] * 10})
        result = blk.create_block_structure_features(df)
        assert len(result.columns) == len(df.columns)

    def test_works_with_minimal_data(self, blk):
        """Should work with as few as 5 rows."""
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=10),
            "close": np.linspace(100, 110, 10),
            "volume": [1e8] * 10,
        })
        result = blk.create_block_structure_features(df)
        blk_cols = {c for c in result.columns if c.startswith("blk_")}
        assert blk_cols == ALL_54

    def test_works_without_volume(self, blk):
        """Volume features should default to 0.0 when no volume column."""
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=50),
            "close": np.cumsum(np.random.randn(50)) + 450,
        })
        result = blk.create_block_structure_features(df)
        assert "blk_3d_vol_trend" in result.columns
        assert (result["blk_3d_vol_trend"] == 0.0).all()
        assert (result["blk_cascade_vol_expand"] == 0.0).all()

    def test_works_without_open(self, blk):
        """Boundary gap should fallback when no open column."""
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=50),
            "close": np.cumsum(np.random.randn(50)) + 450,
            "volume": [1e8] * 50,
        })
        result = blk.create_block_structure_features(df)
        assert "blk_boundary_3d_gap" in result.columns


# ─── Section 1: Block Return Pattern Tests ────────────────────────────

class TestBlockReturnPatterns:
    @pytest.fixture
    def blk(self):
        return BlockStructureFeatures()

    def test_3d_return_matches_pct_change(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        expected = df["close"].pct_change(3)
        # Compare non-NaN values
        valid = ~expected.isna()
        np.testing.assert_allclose(
            result.loc[valid, "blk_3d_return"].values,
            expected[valid].values, rtol=1e-10
        )

    def test_consistency_bounded_0_1(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        assert (result["blk_3d_consistency"] >= 0).all()
        assert (result["blk_3d_consistency"] <= 1.0).all()
        assert (result["blk_5d_consistency"] >= 0).all()
        assert (result["blk_5d_consistency"] <= 1.0).all()

    def test_consistency_high_in_trending_market(self, blk):
        df = _make_trending_up(60)
        result = blk.create_block_structure_features(df)
        # Last half should have high consistency
        assert result["blk_5d_consistency"].iloc[-1] >= 0.8

    def test_pattern_score_bounded(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        assert (result["blk_3d_pattern_score"] >= -1.0).all()
        assert (result["blk_3d_pattern_score"] <= 1.0).all()
        assert (result["blk_5d_pattern_score"] >= -1.0).all()
        assert (result["blk_5d_pattern_score"] <= 1.0).all()

    def test_pattern_score_positive_in_uptrend(self, blk):
        df = _make_trending_up(60)
        result = blk.create_block_structure_features(df)
        assert result["blk_5d_pattern_score"].iloc[-1] > 0.5

    def test_reversal_boundary_binary(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        vals = set(result["blk_3d_reversal_boundary"].unique())
        assert vals.issubset({0.0, 1.0})

    def test_sequential_strength_clipped(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        assert result["blk_sequential_strength"].max() <= 3.0
        assert result["blk_sequential_strength"].min() >= -3.0

    def test_3d_vs_5d_agreement_values(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        vals = set(result["blk_3d_vs_5d_agreement"].unique())
        assert vals.issubset({-1.0, 1.0})


# ─── Section 2: Multi-Scale Cascade Tests ─────────────────────────────

class TestCascadeFeatures:
    @pytest.fixture
    def blk(self):
        return BlockStructureFeatures()

    def test_cascade_alignment_bounded(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        assert (result["blk_cascade_3_5_10"] >= -1.0).all()
        assert (result["blk_cascade_3_5_10"] <= 1.0).all()

    def test_cascade_full_values(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        vals = set(result["blk_cascade_full"].unique())
        assert vals.issubset({-1.0, 0.0, 1.0})

    def test_cascade_full_positive_in_uptrend(self, blk):
        df = _make_trending_up(60)
        result = blk.create_block_structure_features(df)
        # After enough days, all timescales should align positive
        assert result["blk_cascade_full"].iloc[-1] == 1.0

    def test_cascade_full_negative_in_downtrend(self, blk):
        df = _make_trending_down(60)
        result = blk.create_block_structure_features(df)
        assert result["blk_cascade_full"].iloc[-1] == -1.0

    def test_cascade_break_binary(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        assert set(result["blk_cascade_break_short"].unique()).issubset({0.0, 1.0})
        assert set(result["blk_cascade_break_long"].unique()).issubset({0.0, 1.0})

    def test_cascade_ratios_clipped(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        assert result["blk_cascade_ratio_3_10"].max() <= 5.0
        assert result["blk_cascade_ratio_3_10"].min() >= -5.0

    def test_cascade_vol_expand_binary(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        assert set(result["blk_cascade_vol_expand"].unique()).issubset({0.0, 1.0})

    def test_return_rank_bounded_0_1(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        valid = result["blk_cascade_return_rank"].dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 1.0).all()


# ─── Section 3: Intra-Block Structure Tests ───────────────────────────

class TestIntraBlockStructure:
    @pytest.fixture
    def blk(self):
        return BlockStructureFeatures()

    def test_where_extreme_bounded(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        assert (result["blk_3d_where_extreme"] >= 0).all()
        assert (result["blk_3d_where_extreme"] <= 1.0).all()
        assert (result["blk_5d_where_extreme"] >= 0).all()
        assert (result["blk_5d_where_extreme"] <= 1.0).all()

    def test_range_position_bounded_0_1(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        assert (result["blk_3d_range_position"] >= 0).all()
        assert (result["blk_3d_range_position"] <= 1.0 + 1e-6).all()

    def test_max_drawdown_non_negative(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        assert (result["blk_3d_max_drawdown"] >= 0).all()
        assert (result["blk_5d_max_drawdown"] >= 0).all()

    def test_recovery_ratio_bounded_0_1(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        assert (result["blk_3d_recovery_ratio"] >= 0).all()
        assert (result["blk_3d_recovery_ratio"] <= 1.0).all()

    def test_drawdown_small_in_trending_up(self, blk):
        df = _make_trending_up(60)
        result = blk.create_block_structure_features(df)
        # In a pure uptrend, drawdowns should be minimal
        assert result["blk_5d_max_drawdown"].iloc[-1] < 0.01

    def test_range_position_high_at_end_of_uptrend(self, blk):
        df = _make_trending_up(60)
        result = blk.create_block_structure_features(df)
        # Close should be near top of 5d range in uptrend
        assert result["blk_5d_range_position"].iloc[-1] > 0.7


# ─── Section 4: Volume Profile Tests ─────────────────────────────────

class TestVolumeProfile:
    @pytest.fixture
    def blk(self):
        return BlockStructureFeatures()

    def test_all_8_features_created(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        for feat in VOL_PROFILE_8:
            assert feat in result.columns, f"Missing {feat}"

    def test_vol_trend_clipped(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        assert result["blk_3d_vol_trend"].max() <= 3.0
        assert result["blk_3d_vol_trend"].min() >= -3.0

    def test_vol_concentration_bounded(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        # Concentration = max/sum, bounded [1/window, 1.0]
        assert (result["blk_3d_vol_concentration"] > 0).all()
        assert (result["blk_3d_vol_concentration"] <= 1.0).all()

    def test_vol_return_agree_bounded(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        assert (result["blk_3d_vol_return_agree"] >= 0).all()
        assert (result["blk_3d_vol_return_agree"] <= 1.0).all()

    def test_vwap_reasonable(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        # VWAP vs close should be small (within a few percent)
        assert result["blk_3d_vwap_vs_close"].abs().max() < 0.10


# ─── Section 5: Block Boundary Tests ─────────────────────────────────

class TestBlockBoundary:
    @pytest.fixture
    def blk(self):
        return BlockStructureFeatures()

    def test_all_8_features_created(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        for feat in BOUNDARY_8:
            assert feat in result.columns, f"Missing {feat}"

    def test_boundary_streak_at_least_1(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        assert (result["blk_boundary_3d_streak"] >= 1.0).all()
        assert (result["blk_boundary_5d_streak"] >= 1.0).all()

    def test_boundary_streak_high_in_trend(self, blk):
        df = _make_trending_up(60)
        result = blk.create_block_structure_features(df)
        # Long uptrend should have streak > 1
        assert result["blk_boundary_3d_streak"].max() > 1.0

    def test_revert_clipped(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        assert result["blk_boundary_3d_revert"].max() <= 4.0
        assert result["blk_boundary_3d_revert"].min() >= -4.0

    def test_vol_shift_clipped(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        assert result["blk_boundary_3d_vol_shift"].max() <= 10.0
        # First few rows are 0.0 from fillna (NaN warmup); check only post-warmup
        post_warmup = result["blk_boundary_3d_vol_shift"].iloc[5:]
        nonzero = post_warmup[post_warmup != 0.0]
        if len(nonzero) > 0:
            assert nonzero.min() >= 0.1


# ─── Section 6: Texture Tests ────────────────────────────────────────

class TestTexture:
    @pytest.fixture
    def blk(self):
        return BlockStructureFeatures()

    def test_all_6_features_created(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        for feat in TEXTURE_6:
            assert feat in result.columns, f"Missing {feat}"

    def test_roughness_clipped(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        assert result["blk_return_roughness"].max() <= 20.0
        assert (result["blk_return_roughness"] >= 0).all()

    def test_roughness_low_in_trend(self, blk):
        df = _make_trending_up(60)
        result = blk.create_block_structure_features(df)
        # Smooth trend → low roughness (path ~= displacement)
        assert result["blk_return_roughness"].iloc[-1] < 3.0

    def test_roughness_high_in_choppy(self, blk):
        df = _make_choppy(60)
        result = blk.create_block_structure_features(df)
        # Choppy → high roughness (lots of path, little displacement)
        assert result["blk_return_roughness"].iloc[-1] > 3.0

    def test_hurst_proxy_bounded(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        assert (result["blk_hurst_proxy"] >= 0).all()
        assert (result["blk_hurst_proxy"] <= 1.0).all()

    def test_hurst_high_in_trend(self, blk):
        df = _make_trending_up(60)
        result = blk.create_block_structure_features(df)
        # Trending → sign persists → hurst > 0.5
        assert result["blk_hurst_proxy"].iloc[-1] > 0.5

    def test_kurtosis_clipped(self, blk):
        df = _make_spy_daily(100)
        result = blk.create_block_structure_features(df)
        assert result["blk_3d_kurtosis"].max() <= 50.0
        assert result["blk_3d_kurtosis"].min() >= -10.0

    def test_autocorr_bounded(self, blk):
        df = _make_spy_daily(300)
        result = blk.create_block_structure_features(df)
        valid = result["blk_3d_autocorr"].dropna()
        if len(valid) > 0:
            assert (valid >= -1.0 - 1e-6).all()
            assert (valid <= 1.0 + 1e-6).all()


# ─── Analyze Tests ─────────────────────────────────────────────────────

class TestAnalyzeCurrentStructure:
    def test_returns_dict(self):
        blk = BlockStructureFeatures()
        df = _make_spy_daily(100)
        df = blk.create_block_structure_features(df)
        result = blk.analyze_current_structure(df)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        blk = BlockStructureFeatures()
        df = _make_spy_daily(100)
        df = blk.create_block_structure_features(df)
        result = blk.analyze_current_structure(df)
        assert "cascade_regime" in result
        assert "block_trend" in result
        assert "texture_regime" in result

    def test_cascade_regime_values(self):
        blk = BlockStructureFeatures()
        df = _make_spy_daily(300)
        df = blk.create_block_structure_features(df)
        result = blk.analyze_current_structure(df)
        assert result["cascade_regime"] in {"FULL_BULL", "FULL_BEAR", "MIXED"}

    def test_block_trend_values(self):
        blk = BlockStructureFeatures()
        df = _make_spy_daily(300)
        df = blk.create_block_structure_features(df)
        result = blk.analyze_current_structure(df)
        assert result["block_trend"] in {"STRONG_UP", "STRONG_DOWN", "CHOPPY", "NORMAL"}

    def test_texture_regime_values(self):
        blk = BlockStructureFeatures()
        df = _make_spy_daily(300)
        df = blk.create_block_structure_features(df)
        result = blk.analyze_current_structure(df)
        assert result["texture_regime"] in {"MEAN_REVERTING", "TRENDING", "NEUTRAL"}

    def test_returns_none_without_features(self):
        blk = BlockStructureFeatures()
        df = pd.DataFrame({"date": [1, 2, 3], "close": [100, 101, 102]})
        result = blk.analyze_current_structure(df)
        assert result is None

    def test_full_bull_in_uptrend(self):
        blk = BlockStructureFeatures()
        df = _make_trending_up(60)
        df = blk.create_block_structure_features(df)
        result = blk.analyze_current_structure(df)
        assert result["cascade_regime"] == "FULL_BULL"

    def test_full_bear_in_downtrend(self):
        blk = BlockStructureFeatures()
        df = _make_trending_down(60)
        df = blk.create_block_structure_features(df)
        result = blk.analyze_current_structure(df)
        assert result["cascade_regime"] == "FULL_BEAR"


# ─── Feature Group Registration Tests ─────────────────────────────────

class TestFeatureGroupRegistration:
    def test_block_structure_group_exists(self):
        from src.phase_10_feature_processing.group_aware_processor import FEATURE_GROUPS
        assert "block_structure" in FEATURE_GROUPS
        assert "blk_" in FEATURE_GROUPS["block_structure"]

    def test_all_54_features_assigned_to_group(self):
        from src.phase_10_feature_processing.group_aware_processor import assign_feature_groups
        feature_names = list(ALL_54) + ["rsi_14", "close"]
        groups = assign_feature_groups(feature_names)
        assert "block_structure" in groups
        assert len(groups["block_structure"]) == 54


# ─── Feature Count Tests ──────────────────────────────────────────────

class TestFeatureCounts:
    def test_total_count(self):
        assert len(ALL_54) == 54

    def test_no_overlap_between_categories(self):
        all_sets = [
            BLOCK_RETURN_12, CASCADE_10, INTRA_BLOCK_10,
            VOL_PROFILE_8, BOUNDARY_8, TEXTURE_6,
        ]
        union = set()
        for s in all_sets:
            overlap = union & s
            assert not overlap, f"Overlap: {overlap}"
            union |= s

    def test_category_sizes(self):
        assert len(BLOCK_RETURN_12) == 12
        assert len(CASCADE_10) == 10
        assert len(INTRA_BLOCK_10) == 10
        assert len(VOL_PROFILE_8) == 8
        assert len(BOUNDARY_8) == 8
        assert len(TEXTURE_6) == 6

"""
Tests for Bear Market Synthetic Universe Generation.

Validates that bear universe methods produce correctly structured output
with bearish return distributions to counteract bull market bias.
"""

import numpy as np
import pandas as pd
import pytest

from src.phase_03_synthetic_data.synthetic_universe import SyntheticSPYGenerator


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def spy_data():
    """Create realistic mock SPY daily data with bull market bias (~55% up days)."""
    np.random.seed(42)
    n_days = 500
    dates = pd.bdate_range("2020-01-02", periods=n_days)

    # Simulate bull market: mean +0.05% daily, std ~0.95%
    returns = np.random.normal(0.0005, 0.0095, n_days)

    # Inject a drawdown period (days 100-140) for block bootstrap testing
    returns[100:120] = np.random.normal(-0.008, 0.012, 20)  # Sharp decline
    returns[120:140] = np.random.normal(-0.003, 0.010, 20)  # Continued weakness

    # Inject another drawdown (days 300-325)
    returns[300:315] = np.random.normal(-0.010, 0.015, 15)
    returns[315:325] = np.random.normal(-0.004, 0.010, 10)

    df = pd.DataFrame({
        "date": dates,
        "day_return": returns,
        "close": 450.0 * (1 + np.cumsum(returns)),
    })
    return df


@pytest.fixture
def component_returns():
    """Create mock component daily returns for 20 stocks."""
    np.random.seed(42)
    n_days = 500
    dates = pd.bdate_range("2020-01-02", periods=n_days)

    tickers = [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
        "AMD", "AVGO", "ADBE", "CRM",  # cyclicals
        "JNJ", "PG", "KO", "PEP", "MRK", "ABBV", "VZ", "PFE", "MCD",  # defensives
    ]
    data = {}
    for ticker in tickers:
        data[ticker] = np.random.normal(0.0005, 0.015, n_days)

    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def spy_returns(spy_data):
    """Extract SPY returns Series with normalized dates (matches bear method input)."""
    spy_ret = spy_data.groupby("date")["day_return"].first()
    spy_ret.index = pd.to_datetime(spy_ret.index).normalize()
    return spy_ret


@pytest.fixture
def common_dates(spy_returns):
    """Common dates index."""
    return spy_returns.index


@pytest.fixture
def generator():
    """Create SyntheticSPYGenerator with bear universes enabled."""
    return SyntheticSPYGenerator(
        n_universes=20,
        real_weight=0.6,
        use_bear_universes=True,
    )


@pytest.fixture
def generator_no_bear():
    """Create SyntheticSPYGenerator with bear universes disabled."""
    return SyntheticSPYGenerator(
        n_universes=20,
        real_weight=0.6,
        use_bear_universes=False,
    )


# ─── Schema Validation ─────────────────────────────────────────────────────

REQUIRED_COLUMNS = {"date", "day_return", "synthetic_return", "real_return", "universe_type"}


def _validate_universe_schema(universe: pd.DataFrame, expected_label: str):
    """Validate that a bear universe has the correct output schema."""
    assert isinstance(universe, pd.DataFrame)
    assert set(universe.columns) >= REQUIRED_COLUMNS, (
        f"Missing columns: {REQUIRED_COLUMNS - set(universe.columns)}"
    )
    assert len(universe) > 0
    assert universe["universe_type"].iloc[0] == expected_label
    assert universe["day_return"].dtype == np.float64 or universe["day_return"].dtype == float
    assert not universe["day_return"].isna().all()


# ─── Block Bootstrap Tests ──────────────────────────────────────────────────

class TestBearBlockBootstrap:

    def test_produces_output(self, generator, spy_returns, common_dates):
        result = generator._bear_block_bootstrap(spy_returns, common_dates, seed=1001)
        assert result is not None
        _validate_universe_schema(result, "bear_block_bootstrap_1001")

    def test_correct_length(self, generator, spy_returns, common_dates):
        result = generator._bear_block_bootstrap(spy_returns, common_dates, seed=1001)
        assert len(result) == len(common_dates)

    def test_bearish_distribution(self, generator, spy_returns, common_dates):
        result = generator._bear_block_bootstrap(spy_returns, common_dates, seed=1001)
        # Block bootstrap from drawdowns should have fewer up days than real
        real_up_pct = (spy_returns > 0.0025).mean()
        synth_up_pct = (result["day_return"] > 0.0025).mean()
        assert synth_up_pct < real_up_pct, (
            f"Bear bootstrap should have fewer up days: synth={synth_up_pct:.1%} vs real={real_up_pct:.1%}"
        )

    def test_reproducible(self, generator, spy_returns, common_dates):
        r1 = generator._bear_block_bootstrap(spy_returns, common_dates, seed=1001)
        r2 = generator._bear_block_bootstrap(spy_returns, common_dates, seed=1001)
        np.testing.assert_array_equal(r1["day_return"].values, r2["day_return"].values)

    def test_different_seeds_differ(self, generator, spy_returns, common_dates):
        r1 = generator._bear_block_bootstrap(spy_returns, common_dates, seed=1001)
        r2 = generator._bear_block_bootstrap(spy_returns, common_dates, seed=1002)
        assert not np.array_equal(r1["day_return"].values, r2["day_return"].values)

    def test_skips_with_insufficient_bear_dates(self, generator):
        """If data has no drawdowns, method should return None."""
        n_days = 200
        dates = pd.bdate_range("2020-01-02", periods=n_days)
        # All positive returns — no drawdowns
        spy_ret = pd.Series(np.random.uniform(0.001, 0.01, n_days), index=dates)
        result = generator._bear_block_bootstrap(spy_ret, dates, seed=1001)
        assert result is None


# ─── Regime Balanced Tests ──────────────────────────────────────────────────

class TestBearRegimeBalanced:

    def test_balanced_variant(self, generator, spy_returns, common_dates):
        result = generator._bear_regime_balanced(
            spy_returns, common_dates,
            target_up_pct=0.45, target_down_pct=0.45,
            seed=2001, label="bear_regime_balanced",
        )
        assert result is not None
        _validate_universe_schema(result, "bear_regime_balanced")

        # Check proportions are approximately right
        up_pct = (result["day_return"] > 0.0025).mean()
        assert 0.40 <= up_pct <= 0.50, f"Expected ~45% up, got {up_pct:.1%}"

    def test_bear_heavy_variant(self, generator, spy_returns, common_dates):
        result = generator._bear_regime_balanced(
            spy_returns, common_dates,
            target_up_pct=0.35, target_down_pct=0.55,
            seed=2002, label="bear_regime_heavy",
        )
        assert result is not None

        up_pct = (result["day_return"] > 0.0025).mean()
        down_pct = (result["day_return"] < -0.0025).mean()
        assert up_pct < down_pct, f"Bear-heavy should have more down than up: up={up_pct:.1%}, down={down_pct:.1%}"

    def test_correct_length(self, generator, spy_returns, common_dates):
        result = generator._bear_regime_balanced(
            spy_returns, common_dates,
            target_up_pct=0.45, target_down_pct=0.45,
            seed=2001, label="test",
        )
        assert len(result) == len(common_dates)

    def test_uses_real_returns(self, generator, spy_returns, common_dates):
        """All return values should be actual SPY returns (resampled)."""
        result = generator._bear_regime_balanced(
            spy_returns, common_dates,
            target_up_pct=0.45, target_down_pct=0.45,
            seed=2001, label="test",
        )
        real_set = set(spy_returns.values)
        for val in result["day_return"].values:
            assert val in real_set, f"Return {val} not found in real SPY returns"


# ─── Mean Shift Tests ───────────────────────────────────────────────────────

class TestBearMeanShift:

    def test_5bp_shift(self, generator, spy_returns, common_dates):
        result = generator._bear_mean_shift(spy_returns, common_dates, shift_bps=5)
        assert result is not None
        _validate_universe_schema(result, "bear_mean_shift_5bp")

        # Mean should be ~5bp lower than real
        shift_actual = spy_returns.mean() - result["day_return"].mean()
        assert 0.0004 <= shift_actual <= 0.0006, f"Expected ~5bp shift, got {shift_actual:.6f}"

    def test_10bp_shift(self, generator, spy_returns, common_dates):
        result = generator._bear_mean_shift(spy_returns, common_dates, shift_bps=10)
        assert result is not None
        _validate_universe_schema(result, "bear_mean_shift_10bp")

        shift_actual = spy_returns.mean() - result["day_return"].mean()
        assert 0.0009 <= shift_actual <= 0.0011, f"Expected ~10bp shift, got {shift_actual:.6f}"

    def test_preserves_temporal_structure(self, generator, spy_returns, common_dates):
        """Autocorrelation should be preserved since only the mean changes."""
        result = generator._bear_mean_shift(spy_returns, common_dates, shift_bps=5)

        real = spy_returns.values
        synth = result["day_return"].values

        # Differences between consecutive days should be identical
        real_diffs = np.diff(real)
        synth_diffs = np.diff(synth)
        np.testing.assert_allclose(real_diffs, synth_diffs, atol=1e-10)

    def test_clamped_to_bounds(self, generator):
        """Returns should be clipped to [-10%, +10%]."""
        dates = pd.bdate_range("2020-01-02", periods=5)
        spy_ret = pd.Series([0.09, -0.09, 0.05, -0.05, 0.0], index=dates)
        result = generator._bear_mean_shift(spy_ret, dates, shift_bps=500)  # Extreme shift
        assert result["day_return"].min() >= -0.10
        assert result["day_return"].max() <= 0.10

    def test_correct_length(self, generator, spy_returns, common_dates):
        result = generator._bear_mean_shift(spy_returns, common_dates, shift_bps=5)
        assert len(result) == len(common_dates)

    def test_reduces_up_day_pct(self, generator, spy_returns, common_dates):
        real_up_pct = (spy_returns > 0.0025).mean()
        result = generator._bear_mean_shift(spy_returns, common_dates, shift_bps=10)
        synth_up_pct = (result["day_return"] > 0.0025).mean()
        assert synth_up_pct < real_up_pct, "10bp shift should reduce up-day percentage"


# ─── Volatility Amplified Tests ─────────────────────────────────────────────

class TestBearVolAmplified:

    def test_produces_output(self, generator, spy_returns, common_dates):
        result = generator._bear_vol_amplified(spy_returns, common_dates)
        assert result is not None
        _validate_universe_schema(result, "bear_vol_amplified")

    def test_amplifies_negative_dampens_positive(self, generator, spy_returns, common_dates):
        result = generator._bear_vol_amplified(spy_returns, common_dates)

        real = spy_returns.values
        synth = result["day_return"].values

        # For negative days (before clipping), synth should be more negative
        neg_mask = real < -0.001  # Use margin to avoid near-zero
        if neg_mask.sum() > 0:
            assert synth[neg_mask].mean() < real[neg_mask].mean()

        # For positive days, synth should be less positive
        pos_mask = real > 0.001
        if pos_mask.sum() > 0:
            assert synth[pos_mask].mean() < real[pos_mask].mean()

    def test_net_bearish(self, generator, spy_returns, common_dates):
        result = generator._bear_vol_amplified(spy_returns, common_dates)
        assert result["day_return"].mean() < spy_returns.mean()

    def test_preserves_direction(self, generator, spy_returns, common_dates):
        """Days that were positive stay positive (though possibly below threshold)."""
        result = generator._bear_vol_amplified(spy_returns, common_dates)
        real = spy_returns.values
        synth = result["day_return"].values

        # Signs should match (positive stays positive, negative stays negative)
        # Allow small tolerance for zero-crossing at clipping boundaries
        sign_match = np.sign(real) == np.sign(synth)
        zero_mask = np.abs(real) < 1e-10
        assert (sign_match | zero_mask).mean() > 0.99

    def test_correct_length(self, generator, spy_returns, common_dates):
        result = generator._bear_vol_amplified(spy_returns, common_dates)
        assert len(result) == len(common_dates)


# ─── Defensive Rotation Tests ──────────────────────────────────────────────

class TestBearDefensiveRotation:

    def test_produces_output(self, generator, component_returns, spy_returns, common_dates):
        result = generator._bear_defensive_rotation(component_returns, spy_returns, common_dates)
        assert result is not None
        _validate_universe_schema(result, "bear_defensive_rotation")

    def test_net_bearish(self, generator, component_returns, spy_returns, common_dates):
        result = generator._bear_defensive_rotation(component_returns, spy_returns, common_dates)
        assert result["day_return"].mean() < 0, "Defensive rotation should produce net-bearish returns"

    def test_correct_length(self, generator, component_returns, spy_returns, common_dates):
        result = generator._bear_defensive_rotation(component_returns, spy_returns, common_dates)
        assert len(result) == len(common_dates)

    def test_skips_with_insufficient_stocks(self, generator, spy_returns, common_dates):
        """Should skip if not enough defensive or cyclical stocks."""
        dates = pd.bdate_range("2020-01-02", periods=len(common_dates))
        # Only 2 stocks — not enough for either category
        returns = pd.DataFrame({
            "AAPL": np.random.normal(0, 0.01, len(dates)),
            "JNJ": np.random.normal(0, 0.01, len(dates)),
        }, index=dates)
        result = generator._bear_defensive_rotation(returns, spy_returns, common_dates)
        assert result is None

    def test_volatility_reasonable(self, generator, component_returns, spy_returns, common_dates):
        result = generator._bear_defensive_rotation(component_returns, spy_returns, common_dates)
        real_std = spy_returns.std()
        synth_std = result["day_return"].std()
        # Should be within 0.3x to 3.0x of real volatility
        assert 0.3 * real_std <= synth_std <= 3.0 * real_std, (
            f"Synthetic vol {synth_std:.4f} too far from real vol {real_std:.4f}"
        )


# ─── Integration Tests ──────────────────────────────────────────────────────

class TestBearUniverseIntegration:

    def test_generate_bear_universes_returns_list(self, generator, component_returns, spy_data):
        result = generator._generate_bear_universes(component_returns, spy_data)
        assert isinstance(result, list)
        assert len(result) >= 5  # At least 5 methods should produce output

    def test_all_universes_have_correct_schema(self, generator, component_returns, spy_data):
        universes = generator._generate_bear_universes(component_returns, spy_data)
        for u in universes:
            assert set(u.columns) >= REQUIRED_COLUMNS
            assert u["universe_type"].iloc[0].startswith("bear_")

    def test_bear_disabled_produces_none(self, generator_no_bear, component_returns, spy_data):
        """When use_bear_universes=False, generate_universes should not include bear universes."""
        universes = generator_no_bear.generate_universes(component_returns, spy_data)
        bear_count = sum(
            1 for u in universes
            if "universe_type" in u.columns
            and str(u["universe_type"].iloc[0]).startswith("bear_")
        )
        assert bear_count == 0

    def test_config_controls_mean_shift_bps(self):
        gen = SyntheticSPYGenerator(
            n_universes=20, real_weight=0.6,
            use_bear_universes=True,
            bear_mean_shift_bps=[3, 7, 15],
        )
        assert gen.bear_mean_shift_bps == [3, 7, 15]

    def test_config_controls_vol_factors(self):
        gen = SyntheticSPYGenerator(
            n_universes=20, real_weight=0.6,
            bear_vol_amplify_factor=2.0,
            bear_vol_dampen_factor=0.5,
        )
        assert gen.bear_vol_amplify_factor == 2.0
        assert gen.bear_vol_dampen_factor == 0.5


# ─── AntiOverfitConfig Tests ───────────────────────────────────────────────

class TestAntiOverfitConfigBearFields:

    def test_default_bear_fields_exist(self):
        from src.experiment_config import AntiOverfitConfig
        config = AntiOverfitConfig()
        assert config.use_bear_universes is True
        assert config.bear_mean_shift_bps == [5, 10]
        assert config.bear_vol_amplify_factor == 1.5
        assert config.bear_vol_dampen_factor == 0.7

    def test_bear_disabled(self):
        from src.experiment_config import AntiOverfitConfig
        config = AntiOverfitConfig(use_bear_universes=False)
        assert config.use_bear_universes is False

    def test_custom_shift_bps(self):
        from src.experiment_config import AntiOverfitConfig
        config = AntiOverfitConfig(bear_mean_shift_bps=[3, 8, 15])
        assert config.bear_mean_shift_bps == [3, 8, 15]

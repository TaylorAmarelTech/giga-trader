"""
Tests for Multi-Timescale Regime Bootstrap Universe Generation.

Validates that multiscale bootstrap methods produce correctly structured
output with balanced regime representation at weekly, monthly, magnitude,
and volatility timescales.
"""

import numpy as np
import pandas as pd
import pytest

from src.phase_03_synthetic_data.synthetic_universe import SyntheticSPYGenerator


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def spy_data():
    """Create realistic mock SPY daily data with diverse regimes."""
    np.random.seed(42)
    n_days = 600
    dates = pd.bdate_range("2020-01-02", periods=n_days)

    # Base: bull market with 55% up days
    returns = np.random.normal(0.0005, 0.0095, n_days)

    # Inject bear weeks (days 100-115: strong consecutive declines)
    returns[100:105] = np.random.normal(-0.008, 0.005, 5)
    returns[105:110] = np.random.normal(-0.006, 0.005, 5)
    returns[110:115] = np.random.normal(-0.007, 0.005, 5)

    # Inject drawdown month (days 200-221)
    returns[200:221] = np.random.normal(-0.005, 0.010, 21)

    # Inject high-vol period (days 300-350)
    returns[300:350] = np.random.normal(0.0, 0.020, 50)

    # Inject flat period (days 400-430)
    returns[400:430] = np.random.normal(0.0, 0.002, 30)

    # Inject large swing days scattered through
    for i in [50, 150, 250, 350, 450, 500]:
        returns[i] = np.random.choice([-1, 1]) * np.random.uniform(0.015, 0.03)

    df = pd.DataFrame({
        "date": dates,
        "day_return": returns,
        "close": 450.0 * (1 + np.cumsum(returns)),
    })
    return df


@pytest.fixture
def spy_returns(spy_data):
    """Extract SPY returns Series with normalized dates."""
    spy_ret = spy_data.groupby("date")["day_return"].first()
    spy_ret.index = pd.to_datetime(spy_ret.index).normalize()
    return spy_ret


@pytest.fixture
def common_dates(spy_returns):
    return spy_returns.index


@pytest.fixture
def generator():
    return SyntheticSPYGenerator(
        n_universes=20, real_weight=0.6,
        use_bear_universes=False,  # Focus on multiscale only
        use_multiscale_bootstrap=True,
    )


@pytest.fixture
def generator_disabled():
    return SyntheticSPYGenerator(
        n_universes=20, real_weight=0.6,
        use_multiscale_bootstrap=False,
    )


# ─── Schema ─────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = {"date", "day_return", "synthetic_return", "real_return", "universe_type"}


def _validate_schema(universe: pd.DataFrame, expected_label: str):
    assert isinstance(universe, pd.DataFrame)
    assert set(universe.columns) >= REQUIRED_COLUMNS
    assert len(universe) > 0
    assert universe["universe_type"].iloc[0] == expected_label
    assert not universe["day_return"].isna().all()


# ─── Weekly Regime Tests ────────────────────────────────────────────────────

class TestMultiscaleWeeklyRegime:

    def test_balanced_produces_output(self, generator, spy_returns, common_dates):
        result = generator._multiscale_weekly_regime(
            spy_returns, common_dates,
            target_pcts=(0.40, 0.40, 0.20),
            label="multiscale_weekly_balanced",
            seed=3001,
        )
        assert result is not None
        _validate_schema(result, "multiscale_weekly_balanced")

    def test_correct_length(self, generator, spy_returns, common_dates):
        result = generator._multiscale_weekly_regime(
            spy_returns, common_dates,
            target_pcts=(0.40, 0.40, 0.20),
            label="test",
            seed=3001,
        )
        assert len(result) == len(common_dates)

    def test_bear_variant_has_fewer_up_days(self, generator, spy_returns, common_dates):
        balanced = generator._multiscale_weekly_regime(
            spy_returns, common_dates,
            target_pcts=(0.40, 0.40, 0.20),
            label="balanced",
            seed=3001,
        )
        bear = generator._multiscale_weekly_regime(
            spy_returns, common_dates,
            target_pcts=(0.30, 0.50, 0.20),
            label="bear",
            seed=3002,
        )
        # Bear-heavy should generally have fewer up days
        balanced_up = (balanced["day_return"] > 0.0025).mean()
        bear_up = (bear["day_return"] > 0.0025).mean()
        assert bear_up < balanced_up, (
            f"Bear variant should have fewer up days: bear={bear_up:.1%} vs balanced={balanced_up:.1%}"
        )

    def test_reproducible(self, generator, spy_returns, common_dates):
        r1 = generator._multiscale_weekly_regime(
            spy_returns, common_dates, (0.40, 0.40, 0.20), "test", seed=3001,
        )
        r2 = generator._multiscale_weekly_regime(
            spy_returns, common_dates, (0.40, 0.40, 0.20), "test", seed=3001,
        )
        np.testing.assert_array_equal(r1["day_return"].values, r2["day_return"].values)

    def test_uses_real_returns(self, generator, spy_returns, common_dates):
        """All return values should be actual SPY returns (block-resampled)."""
        result = generator._multiscale_weekly_regime(
            spy_returns, common_dates, (0.40, 0.40, 0.20), "test", seed=3001,
        )
        real_set = set(spy_returns.values)
        for val in result["day_return"].values:
            assert val in real_set


# ─── Monthly Drawdown Tests ────────────────────────────────────────────────

class TestMultiscaleMonthlyDrawdown:

    def test_produces_output(self, generator, spy_returns, common_dates):
        result = generator._multiscale_monthly_drawdown(spy_returns, common_dates, seed=3010)
        assert result is not None
        _validate_schema(result, "multiscale_monthly_drawdown")

    def test_correct_length(self, generator, spy_returns, common_dates):
        result = generator._multiscale_monthly_drawdown(spy_returns, common_dates, seed=3010)
        assert len(result) == len(common_dates)

    def test_oversamples_drawdowns(self, generator, spy_returns, common_dates):
        """Should have more negative-return months than real data."""
        result = generator._multiscale_monthly_drawdown(spy_returns, common_dates, seed=3010)
        # Check 21-day block returns
        returns = result["day_return"].values
        n_blocks = len(returns) // 21
        block_returns = [np.sum(returns[i*21:(i+1)*21]) for i in range(n_blocks)]
        drawdown_pct = np.mean([r < -0.03 for r in block_returns])
        # Should have roughly 40% drawdown months (target)
        assert drawdown_pct >= 0.20, f"Expected ~40% drawdown months, got {drawdown_pct:.1%}"

    def test_reproducible(self, generator, spy_returns, common_dates):
        r1 = generator._multiscale_monthly_drawdown(spy_returns, common_dates, seed=3010)
        r2 = generator._multiscale_monthly_drawdown(spy_returns, common_dates, seed=3010)
        np.testing.assert_array_equal(r1["day_return"].values, r2["day_return"].values)


# ─── Swing Magnitude Tests ─────────────────────────────────────────────────

class TestMultiscaleSwingMagnitude:

    def test_large_swing_oversampling(self, generator, spy_returns, common_dates):
        result = generator._multiscale_swing_magnitude(
            spy_returns, common_dates,
            target_large_pct=0.40, label="multiscale_swing_large", seed=3020,
        )
        assert result is not None
        _validate_schema(result, "multiscale_swing_large")

        # Verify large swings are oversampled
        large_pct = (np.abs(result["day_return"]) > 0.01).mean()
        assert large_pct >= 0.30, f"Expected ~40% large swings, got {large_pct:.1%}"

    def test_flat_day_oversampling(self, generator, spy_returns, common_dates):
        result = generator._multiscale_swing_magnitude(
            spy_returns, common_dates,
            target_flat_pct=0.40, label="multiscale_swing_flat", seed=3021,
        )
        assert result is not None
        _validate_schema(result, "multiscale_swing_flat")

        # Verify flat days are oversampled
        flat_pct = (np.abs(result["day_return"]) <= 0.0025).mean()
        assert flat_pct >= 0.30, f"Expected ~40% flat days, got {flat_pct:.1%}"

    def test_correct_length(self, generator, spy_returns, common_dates):
        result = generator._multiscale_swing_magnitude(
            spy_returns, common_dates,
            target_large_pct=0.40, label="test", seed=3020,
        )
        assert len(result) == len(common_dates)

    def test_uses_real_returns(self, generator, spy_returns, common_dates):
        result = generator._multiscale_swing_magnitude(
            spy_returns, common_dates,
            target_large_pct=0.40, label="test", seed=3020,
        )
        real_set = set(spy_returns.values)
        for val in result["day_return"].values:
            assert val in real_set


# ─── Volatility Regime Block Tests ─────────────────────────────────────────

class TestMultiscaleVolRegime:

    def test_produces_output(self, generator, spy_returns, common_dates):
        result = generator._multiscale_vol_regime_block(spy_returns, common_dates, seed=3030)
        assert result is not None
        _validate_schema(result, "multiscale_vol_regime")

    def test_correct_length(self, generator, spy_returns, common_dates):
        result = generator._multiscale_vol_regime_block(spy_returns, common_dates, seed=3030)
        assert len(result) == len(common_dates)

    def test_reproducible(self, generator, spy_returns, common_dates):
        r1 = generator._multiscale_vol_regime_block(spy_returns, common_dates, seed=3030)
        r2 = generator._multiscale_vol_regime_block(spy_returns, common_dates, seed=3030)
        np.testing.assert_array_equal(r1["day_return"].values, r2["day_return"].values)

    def test_uses_real_returns(self, generator, spy_returns, common_dates):
        result = generator._multiscale_vol_regime_block(spy_returns, common_dates, seed=3030)
        real_set = set(spy_returns.values)
        for val in result["day_return"].values:
            assert val in real_set


# ─── Cascade Tests ──────────────────────────────────────────────────────────

class TestMultiscaleCascade:

    def test_produces_output(self, generator, spy_returns, common_dates):
        result = generator._multiscale_cascade(spy_returns, common_dates, seed=3040)
        assert result is not None
        _validate_schema(result, "multiscale_cascade")

    def test_correct_length(self, generator, spy_returns, common_dates):
        result = generator._multiscale_cascade(spy_returns, common_dates, seed=3040)
        assert len(result) == len(common_dates)

    def test_reproducible(self, generator, spy_returns, common_dates):
        r1 = generator._multiscale_cascade(spy_returns, common_dates, seed=3040)
        r2 = generator._multiscale_cascade(spy_returns, common_dates, seed=3040)
        np.testing.assert_array_equal(r1["day_return"].values, r2["day_return"].values)

    def test_uses_real_returns(self, generator, spy_returns, common_dates):
        result = generator._multiscale_cascade(spy_returns, common_dates, seed=3040)
        real_set = set(spy_returns.values)
        for val in result["day_return"].values:
            assert val in real_set

    def test_too_short_data_skips(self, generator):
        """Should skip if data has fewer than 63 days."""
        dates = pd.bdate_range("2020-01-02", periods=40)
        spy_ret = pd.Series(np.random.normal(0, 0.01, 40), index=dates)
        result = generator._multiscale_cascade(spy_ret, dates, seed=3040)
        assert result is None


# ─── Integration Tests ──────────────────────────────────────────────────────

class TestMultiscaleIntegration:

    def test_orchestrator_returns_list(self, generator, spy_data):
        result = generator._generate_multiscale_universes(spy_data)
        assert isinstance(result, list)
        assert len(result) >= 5  # At least 5 methods should produce output

    def test_all_universes_have_correct_schema(self, generator, spy_data):
        universes = generator._generate_multiscale_universes(spy_data)
        for u in universes:
            assert set(u.columns) >= REQUIRED_COLUMNS
            assert u["universe_type"].iloc[0].startswith("multiscale_")

    def test_disabled_produces_none(self, generator_disabled, spy_data):
        """When use_multiscale_bootstrap=False, generate_universes should not include them."""
        # Need component returns for generate_universes
        np.random.seed(42)
        n_days = len(spy_data)
        dates = pd.bdate_range("2020-01-02", periods=n_days)
        components = pd.DataFrame(
            np.random.normal(0, 0.01, (n_days, 5)),
            index=dates,
            columns=["AAPL", "MSFT", "GOOGL", "JNJ", "PG"],
        )
        universes = generator_disabled.generate_universes(components, spy_data)
        multiscale_count = sum(
            1 for u in universes
            if "universe_type" in u.columns
            and str(u["universe_type"].iloc[0]).startswith("multiscale_")
        )
        assert multiscale_count == 0


# ─── Config Tests ───────────────────────────────────────────────────────────

class TestMultiscaleConfig:

    def test_default_enabled(self):
        from src.experiment_config import AntiOverfitConfig
        config = AntiOverfitConfig()
        assert config.use_multiscale_bootstrap is True

    def test_can_disable(self):
        from src.experiment_config import AntiOverfitConfig
        config = AntiOverfitConfig(use_multiscale_bootstrap=False)
        assert config.use_multiscale_bootstrap is False

    def test_constructor_accepts_flag(self):
        gen = SyntheticSPYGenerator(
            n_universes=20, real_weight=0.6,
            use_multiscale_bootstrap=False,
        )
        assert gen.use_multiscale_bootstrap is False

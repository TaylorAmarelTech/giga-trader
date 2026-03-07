"""
Tests for DiffusionSyntheticGenerator.

Validates that the GBM-aware diffusion-based synthetic price path generator
produces realistic, correctly structured OHLCV daily data.
"""

import numpy as np
import pandas as pd
import pytest

from src.phase_03_synthetic_data.diffusion_generator import (
    DiffusionSyntheticGenerator,
)


# ─── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def realistic_spy_data():
    """Create realistic mock SPY daily OHLCV data (~2 years)."""
    np.random.seed(42)
    n_days = 504
    dates = pd.bdate_range("2022-01-03", periods=n_days)

    # Simulate with ~0.04% daily drift, ~1% daily vol
    returns = np.random.normal(0.0004, 0.01, n_days)
    close = 450.0 * np.exp(np.cumsum(returns))

    # Build realistic OHLCV from close
    open_prices = np.empty(n_days)
    open_prices[0] = 450.0
    open_prices[1:] = close[:-1] * np.exp(np.random.normal(0, 0.001, n_days - 1))

    oc_max = np.maximum(open_prices, close)
    oc_min = np.minimum(open_prices, close)
    high = oc_max + np.abs(np.random.normal(0.003, 0.002, n_days)) * close
    low = oc_min - np.abs(np.random.normal(0.003, 0.002, n_days)) * close
    low = np.maximum(low, 1.0)

    volume = np.random.lognormal(mean=np.log(80_000_000), sigma=0.3, size=n_days).astype(int)

    df = pd.DataFrame({
        "date": dates,
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })
    return df


@pytest.fixture
def close_only_data():
    """Minimal DataFrame with only close column (no OHLV)."""
    np.random.seed(99)
    n = 200
    returns = np.random.normal(0.0003, 0.009, n)
    close = 400.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame({"close": close})


@pytest.fixture
def fitted_generator(realistic_spy_data):
    """Pre-fitted generator for convenience."""
    gen = DiffusionSyntheticGenerator(n_steps=30, seed=123)
    gen.fit(realistic_spy_data)
    return gen


# ─── Construction Tests ─────────────────────────────────────────────────────


class TestConstruction:
    """Test default and custom construction."""

    def test_default_construction(self):
        gen = DiffusionSyntheticGenerator()
        assert gen.n_steps == 50
        assert gen.beta_start == 0.0001
        assert gen.beta_end == 0.02
        assert gen.seed == 42
        assert len(gen.betas) == 50
        assert len(gen.alphas) == 50
        assert len(gen.alpha_bar) == 50
        assert not gen._fitted

    def test_custom_construction(self):
        gen = DiffusionSyntheticGenerator(
            n_steps=100, beta_start=0.001, beta_end=0.05, seed=7
        )
        assert gen.n_steps == 100
        assert gen.beta_start == 0.001
        assert gen.beta_end == 0.05
        assert gen.seed == 7
        assert len(gen.betas) == 100

    def test_noise_schedule_monotonic(self):
        gen = DiffusionSyntheticGenerator(n_steps=50)
        # Betas increase linearly
        assert np.all(np.diff(gen.betas) >= 0)
        # Alpha_bar decreases
        assert np.all(np.diff(gen.alpha_bar) <= 0)
        # Alpha_bar is in (0, 1]
        assert np.all(gen.alpha_bar > 0)
        assert np.all(gen.alpha_bar <= 1.0)


# ─── Fit Tests ───────────────────────────────────────────────────────────────


class TestFit:
    """Test calibration from real data."""

    def test_fit_returns_self(self, realistic_spy_data):
        gen = DiffusionSyntheticGenerator()
        result = gen.fit(realistic_spy_data)
        assert result is gen
        assert gen._fitted

    def test_fit_calibrates_gbm_params(self, realistic_spy_data):
        gen = DiffusionSyntheticGenerator()
        gen.fit(realistic_spy_data)
        # Daily mu should be small (roughly 0.0004 for our fixture)
        assert -0.01 < gen.daily_mu < 0.01
        # Daily sigma should be around 0.01
        assert 0.001 < gen.daily_sigma < 0.05
        # Annualised sigma should be reasonable (5%-50%)
        assert 0.05 < gen.sigma < 0.50

    def test_fit_calibrates_ohlc_ratios(self, realistic_spy_data):
        gen = DiffusionSyntheticGenerator()
        gen.fit(realistic_spy_data)
        assert gen.high_ratio_mean > 0
        assert gen.low_ratio_mean > 0
        assert gen.high_ratio_std > 0
        assert gen.low_ratio_std > 0

    def test_fit_calibrates_volume(self, realistic_spy_data):
        gen = DiffusionSyntheticGenerator()
        gen.fit(realistic_spy_data)
        assert gen.vol_log_mean > 0  # log(volume) > 0 for real volumes
        assert gen.vol_log_std > 0

    def test_fit_close_only(self, close_only_data):
        """Fit with only close column uses default OHLC/volume params."""
        gen = DiffusionSyntheticGenerator()
        gen.fit(close_only_data)
        assert gen._fitted
        # Should use defaults
        assert gen.high_ratio_mean == pytest.approx(0.004, abs=0.001)
        assert gen.vol_log_mean == pytest.approx(np.log(50_000_000), rel=0.01)

    def test_fit_empty_raises(self):
        gen = DiffusionSyntheticGenerator()
        with pytest.raises(ValueError, match="empty"):
            gen.fit(pd.DataFrame())

    def test_fit_too_few_rows_raises(self):
        gen = DiffusionSyntheticGenerator()
        df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
        with pytest.raises(ValueError, match="at least 10"):
            gen.fit(df)


# ─── Generate Tests ──────────────────────────────────────────────────────────


class TestGenerate:
    """Test path generation."""

    def test_generate_requires_fit(self):
        gen = DiffusionSyntheticGenerator()
        with pytest.raises(RuntimeError, match="fit"):
            gen.generate()

    def test_generate_correct_count(self, fitted_generator):
        paths = fitted_generator.generate(n_paths=7, n_days=100)
        assert len(paths) == 7

    def test_generate_correct_length(self, fitted_generator):
        paths = fitted_generator.generate(n_paths=2, n_days=150)
        for df in paths:
            assert len(df) == 150

    def test_generate_required_columns(self, fitted_generator):
        paths = fitted_generator.generate(n_paths=1, n_days=50)
        required = {"date", "open", "high", "low", "close", "volume"}
        for df in paths:
            assert required.issubset(set(df.columns))

    def test_generate_has_daily_return(self, fitted_generator):
        paths = fitted_generator.generate(n_paths=1, n_days=50)
        assert "daily_return" in paths[0].columns

    def test_no_nans(self, fitted_generator):
        paths = fitted_generator.generate(n_paths=3, n_days=100)
        for df in paths:
            numeric_cols = ["open", "high", "low", "close", "volume", "daily_return"]
            for col in numeric_cols:
                assert not df[col].isna().any(), f"NaN found in column {col}"

    def test_ohlc_consistency(self, fitted_generator):
        """High >= max(Open, Close) and Low <= min(Open, Close)."""
        paths = fitted_generator.generate(n_paths=5, n_days=200)
        for i, df in enumerate(paths):
            oc_max = df[["open", "close"]].max(axis=1)
            oc_min = df[["open", "close"]].min(axis=1)
            violations_high = (df["high"] < oc_max - 1e-10).sum()
            violations_low = (df["low"] > oc_min + 1e-10).sum()
            assert violations_high == 0, (
                f"Path {i}: {violations_high} High < max(O,C) violations"
            )
            assert violations_low == 0, (
                f"Path {i}: {violations_low} Low > min(O,C) violations"
            )

    def test_volume_positive(self, fitted_generator):
        paths = fitted_generator.generate(n_paths=3, n_days=100)
        for df in paths:
            assert (df["volume"] > 0).all()

    def test_close_positive(self, fitted_generator):
        paths = fitted_generator.generate(n_paths=3, n_days=252)
        for df in paths:
            assert (df["close"] > 0).all()
            assert (df["open"] > 0).all()
            assert (df["low"] > 0).all()

    def test_return_statistics_reasonable(self, fitted_generator):
        """Mean daily return near zero, vol in reasonable range."""
        paths = fitted_generator.generate(n_paths=20, n_days=252)
        all_returns = []
        for df in paths:
            all_returns.extend(df["daily_return"].values)
        all_returns = np.array(all_returns)

        mean_ret = np.mean(all_returns)
        std_ret = np.std(all_returns)

        # Mean daily return should be near zero (within 1%)
        assert abs(mean_ret) < 0.01, f"Mean return {mean_ret:.6f} too far from 0"
        # Std should be in a reasonable range (0.1% to 5%)
        assert 0.001 < std_ret < 0.05, f"Return std {std_ret:.6f} out of range"

    def test_reproducibility(self, realistic_spy_data):
        """Same seed produces identical paths."""
        gen1 = DiffusionSyntheticGenerator(n_steps=20, seed=42)
        gen1.fit(realistic_spy_data)
        paths1 = gen1.generate(n_paths=2, n_days=50)

        gen2 = DiffusionSyntheticGenerator(n_steps=20, seed=42)
        gen2.fit(realistic_spy_data)
        paths2 = gen2.generate(n_paths=2, n_days=50)

        for p1, p2 in zip(paths1, paths2):
            np.testing.assert_array_almost_equal(
                p1["close"].values, p2["close"].values
            )

    def test_different_seeds_differ(self, realistic_spy_data):
        """Different seeds produce different paths."""
        gen1 = DiffusionSyntheticGenerator(n_steps=20, seed=1)
        gen1.fit(realistic_spy_data)
        paths1 = gen1.generate(n_paths=1, n_days=100)

        gen2 = DiffusionSyntheticGenerator(n_steps=20, seed=999)
        gen2.fit(realistic_spy_data)
        paths2 = gen2.generate(n_paths=1, n_days=100)

        # Paths should differ
        assert not np.allclose(
            paths1[0]["close"].values, paths2[0]["close"].values
        )


# ─── Regime Conditioning Tests ───────────────────────────────────────────────


class TestRegimeConditioned:
    """Test regime-conditioned generation."""

    def test_regime_requires_fit(self):
        gen = DiffusionSyntheticGenerator()
        with pytest.raises(RuntimeError, match="fit"):
            gen.generate_regime_conditioned("bear")

    def test_unknown_regime_raises(self, fitted_generator):
        with pytest.raises(ValueError, match="Unknown regime"):
            fitted_generator.generate_regime_conditioned("hyperinflation")

    def test_all_regimes_produce_output(self, fitted_generator):
        for regime in ["bear", "bull", "sideways", "crisis"]:
            paths = fitted_generator.generate_regime_conditioned(
                regime, n_paths=2, n_days=100
            )
            assert len(paths) == 2
            for df in paths:
                assert len(df) == 100
                assert {"open", "high", "low", "close", "volume"}.issubset(
                    set(df.columns)
                )

    def test_bear_has_negative_drift(self, fitted_generator):
        """Bear paths should have negative mean return on average."""
        paths = fitted_generator.generate_regime_conditioned(
            "bear", n_paths=10, n_days=252
        )
        mean_returns = [df["daily_return"].mean() for df in paths]
        avg = np.mean(mean_returns)
        assert avg < 0, f"Bear regime average return {avg:.6f} should be negative"

    def test_bull_vs_bear_drift_direction(self, fitted_generator):
        """Bull paths should have higher mean return than bear paths."""
        bull = fitted_generator.generate_regime_conditioned(
            "bull", n_paths=10, n_days=252
        )
        bear = fitted_generator.generate_regime_conditioned(
            "bear", n_paths=10, n_days=252
        )
        bull_avg = np.mean([df["daily_return"].mean() for df in bull])
        bear_avg = np.mean([df["daily_return"].mean() for df in bear])
        assert bull_avg > bear_avg, (
            f"Bull avg {bull_avg:.6f} should exceed bear avg {bear_avg:.6f}"
        )

    def test_crisis_higher_volatility(self, fitted_generator):
        """Crisis paths should have higher vol than sideways."""
        crisis = fitted_generator.generate_regime_conditioned(
            "crisis", n_paths=10, n_days=252
        )
        sideways = fitted_generator.generate_regime_conditioned(
            "sideways", n_paths=10, n_days=252
        )
        crisis_vol = np.mean([df["daily_return"].std() for df in crisis])
        sideways_vol = np.mean([df["daily_return"].std() for df in sideways])
        assert crisis_vol > sideways_vol, (
            f"Crisis vol {crisis_vol:.6f} should exceed sideways vol {sideways_vol:.6f}"
        )

    def test_regime_ohlc_consistency(self, fitted_generator):
        """OHLC consistency must hold for all regimes."""
        for regime in ["bear", "bull", "sideways", "crisis"]:
            paths = fitted_generator.generate_regime_conditioned(
                regime, n_paths=3, n_days=100
            )
            for df in paths:
                oc_max = df[["open", "close"]].max(axis=1)
                oc_min = df[["open", "close"]].min(axis=1)
                assert (df["high"] >= oc_max - 1e-10).all(), (
                    f"High < max(O,C) in {regime}"
                )
                assert (df["low"] <= oc_min + 1e-10).all(), (
                    f"Low > min(O,C) in {regime}"
                )

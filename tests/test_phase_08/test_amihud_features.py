"""Tests for AmihudFeatures — Amihud illiquidity ratio features (4 total)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.amihud_features import AmihudFeatures


# ─── Helpers ────────────────────────────────────────────────────────────

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


ALL_4 = {"liq_amihud_raw", "liq_amihud_20d", "liq_amihud_z", "liq_amihud_regime"}


# ─── Invariant Tests ────────────────────────────────────────────────────

class TestAmihudInvariants:
    @pytest.fixture
    def feat(self):
        return AmihudFeatures()

    @pytest.fixture
    def spy(self):
        return _make_spy_daily(200)

    def test_all_4_features_created(self, feat, spy):
        result = feat.create_amihud_features(spy)
        liq_cols = {c for c in result.columns if c.startswith("liq_")}
        assert liq_cols == ALL_4

    def test_no_nans(self, feat, spy):
        result = feat.create_amihud_features(spy)
        for col in ALL_4:
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_no_infinities(self, feat, spy):
        result = feat.create_amihud_features(spy)
        for col in ALL_4:
            assert not np.isinf(result[col]).any(), f"Inf in {col}"

    def test_preserves_original_columns(self, feat, spy):
        original = set(spy.columns)
        result = feat.create_amihud_features(spy)
        assert original.issubset(set(result.columns))

    def test_preserves_row_count(self, feat, spy):
        result = feat.create_amihud_features(spy)
        assert len(result) == len(spy)

    def test_no_close_column(self, feat):
        df = pd.DataFrame({"date": [1, 2], "price": [100, 101]})
        result = feat.create_amihud_features(df)
        assert len(result.columns) == len(df.columns)

    def test_works_without_volume(self, feat):
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=50),
            "close": np.linspace(400, 500, 50),
        })
        result = feat.create_amihud_features(df)
        assert "liq_amihud_raw" in result.columns
        assert (result["liq_amihud_raw"] == 0.0).all()


# ─── Feature Logic Tests ───────────────────────────────────────────────

class TestAmihudLogic:
    @pytest.fixture
    def feat(self):
        return AmihudFeatures()

    def test_amihud_raw_positive(self, feat):
        df = _make_spy_daily(100)
        result = feat.create_amihud_features(df)
        # Amihud ratio should be non-negative (|return| / dollar_volume)
        assert (result["liq_amihud_raw"] >= 0).all()

    def test_amihud_z_bounded(self, feat):
        df = _make_spy_daily(200)
        result = feat.create_amihud_features(df)
        assert result["liq_amihud_z"].max() <= 4.0
        assert result["liq_amihud_z"].min() >= -4.0

    def test_regime_values(self, feat):
        df = _make_spy_daily(200)
        result = feat.create_amihud_features(df)
        vals = set(result["liq_amihud_regime"].unique())
        assert vals.issubset({-1.0, 0.0, 1.0})

    def test_high_volume_means_lower_amihud(self, feat):
        """Doubling volume should halve the Amihud ratio."""
        df = _make_spy_daily(50)
        r1 = feat.create_amihud_features(df)

        df2 = df.copy()
        df2["volume"] = df2["volume"] * 2
        r2 = feat.create_amihud_features(df2)

        # Amihud with doubled volume should be roughly half
        valid = (r1["liq_amihud_raw"] > 0) & (r2["liq_amihud_raw"] > 0)
        if valid.sum() > 5:
            ratio = r2.loc[valid, "liq_amihud_raw"].mean() / r1.loc[valid, "liq_amihud_raw"].mean()
            assert ratio < 0.7  # Should be ~0.5

    def test_rolling_20d_smoothing(self, feat):
        df = _make_spy_daily(100)
        result = feat.create_amihud_features(df)
        # Rolling mean should be smoother than raw
        raw_std = result["liq_amihud_raw"].iloc[30:].std()
        rolling_std = result["liq_amihud_20d"].iloc[30:].std()
        assert rolling_std < raw_std


# ─── Analyze Tests ──────────────────────────────────────────────────────

class TestAnalyzeCurrentLiquidity:
    def test_returns_dict(self):
        feat = AmihudFeatures()
        df = _make_spy_daily(100)
        df = feat.create_amihud_features(df)
        result = feat.analyze_current_liquidity(df)
        assert isinstance(result, dict)
        assert "liquidity_regime" in result

    def test_regime_values(self):
        feat = AmihudFeatures()
        df = _make_spy_daily(200)
        df = feat.create_amihud_features(df)
        result = feat.analyze_current_liquidity(df)
        assert result["liquidity_regime"] in {"ILLIQUID", "LIQUID", "NORMAL"}

    def test_returns_none_without_features(self):
        feat = AmihudFeatures()
        df = pd.DataFrame({"close": [100]})
        assert feat.analyze_current_liquidity(df) is None


# ─── Feature Count Test ─────────────────────────────────────────────────

class TestFeatureCounts:
    def test_total_count(self):
        assert len(ALL_4) == 4

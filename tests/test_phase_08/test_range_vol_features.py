"""Tests for RangeVolFeatures — range-based volatility estimators (8 total)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.range_vol_features import RangeVolFeatures


# ─── Helpers ────────────────────────────────────────────────────────────

def _make_spy_daily(n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    returns = rng.normal(0.0003, 0.012, n_days)
    close = 450.0 * np.cumprod(1 + returns)
    open_p = close * (1 + rng.normal(0, 0.003, n_days))
    high = np.maximum(open_p, close) * (1 + rng.uniform(0, 0.008, n_days))
    low = np.minimum(open_p, close) * (1 - rng.uniform(0, 0.008, n_days))
    volume = rng.randint(50_000_000, 150_000_000, n_days).astype(float)
    return pd.DataFrame({
        "date": dates, "open": open_p, "high": high,
        "low": low, "close": close, "volume": volume,
    })


def _make_high_vol(n_days: int = 100) -> pd.DataFrame:
    rng = np.random.RandomState(99)
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    returns = rng.normal(0, 0.03, n_days)  # 3% daily vol
    close = 450.0 * np.cumprod(1 + returns)
    open_p = close * (1 + rng.normal(0, 0.01, n_days))
    high = np.maximum(open_p, close) * (1 + rng.uniform(0.005, 0.02, n_days))
    low = np.minimum(open_p, close) * (1 - rng.uniform(0.005, 0.02, n_days))
    return pd.DataFrame({
        "date": dates, "open": open_p, "high": high,
        "low": low, "close": close, "volume": [1e8] * n_days,
    })


ALL_8 = {
    "rvol_gk_5d", "rvol_gk_20d", "rvol_yz_5d", "rvol_yz_20d",
    "rvol_rs_5d", "rvol_rs_20d", "rvol_ratio_gk_cc", "rvol_vol_surprise",
}


# ─── Invariant Tests ────────────────────────────────────────────────────

class TestRangeVolInvariants:
    @pytest.fixture
    def feat(self):
        return RangeVolFeatures()

    @pytest.fixture
    def spy(self):
        return _make_spy_daily(200)

    def test_all_8_features_created(self, feat, spy):
        result = feat.create_range_vol_features(spy)
        rvol_cols = {c for c in result.columns if c.startswith("rvol_")}
        assert rvol_cols == ALL_8

    def test_no_nans(self, feat, spy):
        result = feat.create_range_vol_features(spy)
        for col in ALL_8:
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_no_infinities(self, feat, spy):
        result = feat.create_range_vol_features(spy)
        for col in ALL_8:
            assert not np.isinf(result[col]).any(), f"Inf in {col}"

    def test_preserves_original_columns(self, feat, spy):
        original = set(spy.columns)
        result = feat.create_range_vol_features(spy)
        assert original.issubset(set(result.columns))

    def test_preserves_row_count(self, feat, spy):
        result = feat.create_range_vol_features(spy)
        assert len(result) == len(spy)

    def test_no_close_column(self, feat):
        df = pd.DataFrame({"date": [1, 2], "price": [100, 101]})
        result = feat.create_range_vol_features(df)
        assert len(result.columns) == len(df.columns)

    def test_works_without_ohlc(self, feat):
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=50),
            "close": np.linspace(400, 500, 50),
        })
        result = feat.create_range_vol_features(df)
        assert "rvol_gk_5d" in result.columns
        assert (result["rvol_gk_5d"] == 0.0).all()


# ─── Feature Logic Tests ───────────────────────────────────────────────

class TestRangeVolLogic:
    @pytest.fixture
    def feat(self):
        return RangeVolFeatures()

    def test_vol_estimates_positive(self, feat):
        df = _make_spy_daily(100)
        result = feat.create_range_vol_features(df)
        for col in ["rvol_gk_5d", "rvol_gk_20d", "rvol_yz_5d", "rvol_yz_20d",
                     "rvol_rs_5d", "rvol_rs_20d"]:
            assert (result[col] >= 0).all(), f"{col} has negatives"

    def test_high_vol_market_higher_estimates(self, feat):
        normal = _make_spy_daily(100)
        high = _make_high_vol(100)
        r_normal = feat.create_range_vol_features(normal)
        r_high = feat.create_range_vol_features(high)
        # High vol market should have higher vol estimates
        assert r_high["rvol_yz_20d"].iloc[-1] > r_normal["rvol_yz_20d"].iloc[-1]

    def test_gk_ratio_positive(self, feat):
        df = _make_spy_daily(100)
        result = feat.create_range_vol_features(df)
        assert (result["rvol_ratio_gk_cc"] >= 0).all()
        assert (result["rvol_ratio_gk_cc"] <= 5.0).all()

    def test_vol_surprise_bounded(self, feat):
        df = _make_spy_daily(100)
        result = feat.create_range_vol_features(df)
        assert (result["rvol_vol_surprise"] >= 0).all()
        assert (result["rvol_vol_surprise"] <= 5.0).all()

    def test_5d_more_volatile_than_20d(self, feat):
        """5-day rolling vol has higher variance than 20-day."""
        df = _make_spy_daily(200)
        result = feat.create_range_vol_features(df)
        std_5 = result["rvol_gk_5d"].iloc[25:].std()
        std_20 = result["rvol_gk_20d"].iloc[25:].std()
        assert std_5 > std_20  # shorter window = noisier

    def test_all_estimators_reasonable_range(self, feat):
        """All vol estimates should be in roughly the same range."""
        df = _make_spy_daily(200)
        result = feat.create_range_vol_features(df)
        for col in ["rvol_gk_20d", "rvol_yz_20d", "rvol_rs_20d"]:
            valid = result[col].iloc[25:]
            mean_vol = valid.mean()
            # Vol should be between 0 and 100% annualized for SPY-like data
            assert 0 < mean_vol < 1.0, f"{col} mean={mean_vol}"


# ─── Analyze Tests ──────────────────────────────────────────────────────

class TestAnalyzeCurrentVolatility:
    def test_returns_dict(self):
        feat = RangeVolFeatures()
        df = _make_spy_daily(100)
        df = feat.create_range_vol_features(df)
        result = feat.analyze_current_volatility(df)
        assert isinstance(result, dict)
        assert "vol_regime" in result

    def test_regime_values(self):
        feat = RangeVolFeatures()
        df = _make_spy_daily(200)
        df = feat.create_range_vol_features(df)
        result = feat.analyze_current_volatility(df)
        assert result["vol_regime"] in {"HIGH_VOL", "LOW_VOL", "NORMAL_VOL"}

    def test_returns_none_without_features(self):
        feat = RangeVolFeatures()
        df = pd.DataFrame({"close": [100]})
        assert feat.analyze_current_volatility(df) is None


# ─── Feature Count Test ─────────────────────────────────────────────────

class TestFeatureCounts:
    def test_total_count(self):
        assert len(ALL_8) == 8

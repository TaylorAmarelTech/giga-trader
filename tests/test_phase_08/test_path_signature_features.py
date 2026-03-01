"""Tests for PathSignatureFeatures — iterated integral features (17 total)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.path_signature_features import PathSignatureFeatures


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


ALL_17 = {
    "psig_sig1_5d", "psig_sig1_10d", "psig_sig1_20d",
    "psig_sig2_5d", "psig_sig2_10d", "psig_sig2_20d",
    "psig_sig2_anti_5d", "psig_sig2_anti_10d", "psig_sig2_anti_20d",
    "psig_logsig1_5d", "psig_logsig1_10d", "psig_logsig1_20d",
    "psig_path_length_5d", "psig_path_length_10d", "psig_path_length_20d",
    "psig_sig_ratio", "psig_momentum_asym",
}


# ─── Invariant Tests ────────────────────────────────────────────────────

class TestPathSignatureInvariants:
    @pytest.fixture
    def feat(self):
        return PathSignatureFeatures()

    @pytest.fixture
    def spy(self):
        return _make_spy_daily(200)

    def test_all_17_features_created(self, feat, spy):
        result = feat.create_path_signature_features(spy)
        psig_cols = {c for c in result.columns if c.startswith("psig_")}
        assert psig_cols == ALL_17

    def test_no_nans(self, feat, spy):
        result = feat.create_path_signature_features(spy)
        for col in ALL_17:
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_no_infinities(self, feat, spy):
        result = feat.create_path_signature_features(spy)
        for col in ALL_17:
            assert not np.isinf(result[col]).any(), f"Inf in {col}"

    def test_preserves_original_columns(self, feat, spy):
        original = set(spy.columns)
        result = feat.create_path_signature_features(spy)
        assert original.issubset(set(result.columns))

    def test_preserves_row_count(self, feat, spy):
        result = feat.create_path_signature_features(spy)
        assert len(result) == len(spy)

    def test_no_close_column(self, feat):
        df = pd.DataFrame({"date": [1, 2], "price": [100, 101]})
        result = feat.create_path_signature_features(df)
        assert len(result.columns) == len(df.columns)

    def test_feature_count(self):
        assert len(ALL_17) == 17


# ─── NaN / Edge-Case Tests ──────────────────────────────────────────────

class TestPathSignatureEdgeCases:
    @pytest.fixture
    def feat(self):
        return PathSignatureFeatures()

    def test_constant_price_all_zeros(self, feat):
        """Flat close prices should give zero for all signature features."""
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=100),
            "close": np.full(100, 450.0),
            "volume": np.full(100, 100_000_000.0),
        })
        result = feat.create_path_signature_features(df)
        for col in ALL_17:
            assert col in result.columns
            assert result[col].isna().sum() == 0, f"NaN in {col} with flat data"
            assert (result[col] == 0.0).all(), f"{col} should be 0 for flat prices"

    def test_short_dataframe(self, feat):
        """Very short DataFrame (< smallest window) should not crash."""
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=3),
            "close": [450.0, 451.0, 449.0],
        })
        result = feat.create_path_signature_features(df)
        for col in ALL_17:
            assert col in result.columns
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_works_without_volume(self, feat):
        """When volume column is missing, features should still compute (1D path)."""
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=100),
            "close": 450.0 * np.cumprod(1 + np.random.RandomState(7).normal(0, 0.01, 100)),
        })
        result = feat.create_path_signature_features(df)
        psig_cols = {c for c in result.columns if c.startswith("psig_")}
        assert psig_cols == ALL_17


# ─── Feature Logic Tests ────────────────────────────────────────────────

class TestPathSignatureLogic:
    @pytest.fixture
    def feat(self):
        return PathSignatureFeatures()

    def test_sig1_equals_net_return(self, feat):
        """sig1 over window N should equal sum of returns in that window."""
        rng = np.random.RandomState(99)
        n = 50
        close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=n),
            "close": close,
        })
        result = feat.create_path_signature_features(df)

        # Manually compute expected sig1 for window=5 at last row
        returns = np.empty(n)
        returns[0] = 0.0
        returns[1:] = close[1:] / close[:-1] - 1.0
        expected = np.sum(returns[-5:])
        actual = result["psig_sig1_5d"].iloc[-1]
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_path_length_non_negative(self, feat):
        """Path length (total variation) must always be >= 0."""
        df = _make_spy_daily(200)
        result = feat.create_path_signature_features(df)
        for w in [5, 10, 20]:
            col = f"psig_path_length_{w}d"
            assert (result[col] >= 0).all(), f"{col} has negative values"

    def test_path_length_increases_with_volatility(self, feat):
        """Higher volatility should produce larger path lengths."""
        rng = np.random.RandomState(11)
        dates = pd.bdate_range("2024-01-02", periods=100, freq="B")

        # Low vol
        low_close = 450.0 * np.cumprod(1 + rng.normal(0, 0.001, 100))
        df_low = pd.DataFrame({"date": dates, "close": low_close})
        r_low = feat.create_path_signature_features(df_low)

        # High vol
        high_close = 450.0 * np.cumprod(1 + np.random.RandomState(22).normal(0, 0.05, 100))
        df_high = pd.DataFrame({"date": dates, "close": high_close})
        r_high = feat.create_path_signature_features(df_high)

        # Mean path length should be higher for high-vol data
        low_mean = r_low["psig_path_length_20d"].iloc[25:].mean()
        high_mean = r_high["psig_path_length_20d"].iloc[25:].mean()
        assert high_mean > low_mean

    def test_momentum_asym_bounded(self, feat):
        """psig_momentum_asym should be clipped to [-4, 4]."""
        df = _make_spy_daily(200)
        result = feat.create_path_signature_features(df)
        assert result["psig_momentum_asym"].max() <= 4.0
        assert result["psig_momentum_asym"].min() >= -4.0

    def test_anti_symmetric_reverses_sign(self, feat):
        """Anti-symmetric component = sig2(x,y) - sig2(y,x); for identical
        x and y the anti-symmetric part should be zero."""
        # Use constant volume so vol_ratio increments are zero
        # => sig2 and sig2_rev are both zero => anti = 0
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=100),
            "close": 450.0 * np.cumprod(1 + np.random.RandomState(5).normal(0, 0.01, 100)),
            "volume": np.full(100, 100_000_000.0),
        })
        result = feat.create_path_signature_features(df)
        # With constant volume, vol_ratio = volume/ma20 = 1 always after warmup,
        # so vol_ratio increments = 0 => sig2 = 0 and sig2_anti = 0
        # Check from row 25 onward (after volume MA20 warmup)
        for w in [5, 10, 20]:
            vals = result[f"psig_sig2_anti_{w}d"].iloc[25:].values
            np.testing.assert_allclose(vals, 0.0, atol=1e-10,
                                       err_msg=f"sig2_anti_{w}d not ~0 with constant volume")


# ─── No Future Leakage Test ─────────────────────────────────────────────

class TestPathSignatureNoLeakage:
    def test_no_future_leakage(self):
        """Value at index i must depend only on data at indices <= i."""
        feat = PathSignatureFeatures()
        rng = np.random.RandomState(42)

        n = 100
        close = 450.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
        volume = rng.randint(50_000_000, 150_000_000, n).astype(float)

        df_full = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=n),
            "close": close,
            "volume": volume,
        })
        result_full = feat.create_path_signature_features(df_full)

        # Truncate to first 60 rows and recompute
        df_trunc = df_full.iloc[:60].copy()
        result_trunc = feat.create_path_signature_features(df_trunc)

        # Features at index 59 should be identical in both
        for col in ALL_17:
            val_full = result_full[col].iloc[59]
            val_trunc = result_trunc[col].iloc[59]
            np.testing.assert_allclose(
                val_full, val_trunc, atol=1e-12,
                err_msg=f"Future leakage in {col}: full={val_full}, trunc={val_trunc}",
            )

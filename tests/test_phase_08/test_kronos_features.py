"""Tests for KronosFeatures -- random-projection OHLCV embeddings (12 features)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.kronos_features import KronosFeatures


# --- Helpers ----------------------------------------------------------------

def _make_ohlcv(n_days: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data suitable for Kronos features."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    returns = rng.normal(0.0003, 0.012, n_days)
    close = 450.0 * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0.001, 0.01, n_days))
    low = close * (1 - rng.uniform(0.001, 0.01, n_days))
    volume = rng.randint(50_000_000, 150_000_000, n_days).astype(float)
    return pd.DataFrame({
        "date": dates,
        "open": close * (1 + rng.normal(0, 0.003, n_days)),
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


ALL_12 = {
    "kron_embed_0", "kron_embed_1", "kron_embed_2", "kron_embed_3",
    "kron_embed_4", "kron_embed_5", "kron_embed_6", "kron_embed_7",
    "kron_recon_error", "kron_volatility_mode", "kron_trend_strength",
    "kron_regime_proxy",
}


# --- Construction Tests -----------------------------------------------------

class TestKronosConstruction:
    def test_default_construction(self):
        feat = KronosFeatures()
        assert feat.window == 20
        assert feat.n_embed == 8
        assert feat.seed == 42

    def test_custom_params(self):
        feat = KronosFeatures(window=10, n_embed=4, seed=99)
        assert feat.window == 10
        assert feat.n_embed == 4
        assert feat.seed == 99

    def test_feature_names_defined(self):
        assert len(KronosFeatures.FEATURE_NAMES) == 12
        for name in KronosFeatures.FEATURE_NAMES:
            assert name.startswith("kron_"), f"Feature {name} lacks kron_ prefix"

    def test_projection_matrix_shape(self):
        feat = KronosFeatures(window=20, n_embed=8)
        expected_rows = 20 * 4  # window * len(_INPUT_COLS)
        assert feat._proj.shape == (expected_rows, 8)

    def test_projection_matrix_deterministic(self):
        """Same seed -> identical projection matrix."""
        a = KronosFeatures(window=15, n_embed=6, seed=123)
        b = KronosFeatures(window=15, n_embed=6, seed=123)
        np.testing.assert_array_equal(a._proj, b._proj)

    def test_different_seeds_differ(self):
        a = KronosFeatures(seed=1)
        b = KronosFeatures(seed=2)
        assert not np.allclose(a._proj, b._proj)


# --- Feature Creation Tests -------------------------------------------------

class TestKronosFeatureCreation:
    @pytest.fixture
    def feat(self):
        return KronosFeatures(window=20, n_embed=8)

    @pytest.fixture
    def ohlcv(self):
        return _make_ohlcv(100)

    def test_creates_all_12_features(self, feat, ohlcv):
        result = feat.create_kronos_features(ohlcv)
        kron_cols = {c for c in result.columns if c.startswith("kron_")}
        assert kron_cols == ALL_12

    def test_no_nans_in_output(self, feat, ohlcv):
        result = feat.create_kronos_features(ohlcv)
        for col in ALL_12:
            assert not result[col].isna().any(), f"NaN in {col}"

    def test_no_infs_in_output(self, feat, ohlcv):
        result = feat.create_kronos_features(ohlcv)
        for col in ALL_12:
            assert not np.isinf(result[col]).any(), f"Inf in {col}"

    def test_preserves_original_columns(self, feat, ohlcv):
        original = set(ohlcv.columns)
        result = feat.create_kronos_features(ohlcv)
        assert original.issubset(set(result.columns))

    def test_preserves_row_count(self, feat, ohlcv):
        result = feat.create_kronos_features(ohlcv)
        assert len(result) == len(ohlcv)

    def test_recon_error_nonnegative(self, feat, ohlcv):
        result = feat.create_kronos_features(ohlcv)
        assert (result["kron_recon_error"] >= 0.0).all()

    def test_volatility_mode_nonnegative(self, feat, ohlcv):
        result = feat.create_kronos_features(ohlcv)
        assert (result["kron_volatility_mode"] >= 0.0).all()

    def test_trend_strength_bounded(self, feat, ohlcv):
        """Cosine similarity should be in [-1, 1]."""
        result = feat.create_kronos_features(ohlcv)
        assert result["kron_trend_strength"].min() >= -1.0 - 1e-9
        assert result["kron_trend_strength"].max() <= 1.0 + 1e-9

    def test_regime_proxy_values(self, feat, ohlcv):
        """Regime proxy should only be -1, 0, or 1."""
        result = feat.create_kronos_features(ohlcv)
        vals = set(result["kron_regime_proxy"].unique())
        assert vals.issubset({-1.0, 0.0, 1.0})


# --- Edge Cases --------------------------------------------------------------

class TestKronosEdgeCases:
    def test_missing_columns_zero_fills(self):
        """If required columns are missing, all features should be zero."""
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=50),
            "price": np.linspace(400, 410, 50),
        })
        feat = KronosFeatures()
        result = feat.create_kronos_features(df)
        for col in ALL_12:
            assert col in result.columns
            assert (result[col] == 0.0).all(), f"Expected all zeros for {col}"

    def test_short_data_zero_fills(self):
        """Data shorter than window -> zero-fill."""
        df = _make_ohlcv(n_days=5)
        feat = KronosFeatures(window=20)
        result = feat.create_kronos_features(df)
        for col in ALL_12:
            assert col in result.columns
            assert (result[col] == 0.0).all(), f"Expected zeros for short data: {col}"

    def test_exact_window_size_data(self):
        """Data with exactly window rows should produce at least one non-zero row."""
        df = _make_ohlcv(n_days=20)
        feat = KronosFeatures(window=20)
        result = feat.create_kronos_features(df)
        # The last row should have been computed
        assert any(result[col].iloc[-1] != 0.0 for col in ALL_12 if col != "kron_regime_proxy")

    def test_constant_data(self):
        """Constant input -> embeddings should be near zero, no crashes."""
        n = 50
        df = pd.DataFrame({
            "close": np.full(n, 100.0),
            "high": np.full(n, 100.0),
            "low": np.full(n, 100.0),
            "volume": np.full(n, 1_000_000.0),
        })
        feat = KronosFeatures(window=10)
        result = feat.create_kronos_features(df)
        for col in ALL_12:
            assert not result[col].isna().any()
            assert not np.isinf(result[col]).any()

    def test_data_with_nans_in_input(self):
        """NaN values in input columns should be handled gracefully."""
        df = _make_ohlcv(n_days=60)
        # Inject NaNs
        df.loc[10:15, "close"] = np.nan
        df.loc[20:25, "volume"] = np.nan
        feat = KronosFeatures(window=10)
        result = feat.create_kronos_features(df)
        for col in ALL_12:
            assert not result[col].isna().any(), f"NaN leaked into {col}"
            assert not np.isinf(result[col]).any(), f"Inf leaked into {col}"

    def test_small_n_embed(self):
        """n_embed < 8 -> remaining kron_embed_* columns should be zero."""
        df = _make_ohlcv(n_days=50)
        feat = KronosFeatures(window=10, n_embed=3)
        result = feat.create_kronos_features(df)
        # Columns 3-7 should all be zero
        for j in range(3, 8):
            col = f"kron_embed_{j}"
            assert (result[col] == 0.0).all(), f"{col} should be zero with n_embed=3"
        # Columns 0-2 should have some non-zero values
        assert result["kron_embed_0"].abs().sum() > 0

    def test_minimum_window_clamp(self):
        """Window=1 should be clamped to 2."""
        feat = KronosFeatures(window=1)
        assert feat.window == 2


# --- Determinism Tests -------------------------------------------------------

class TestKronosDeterminism:
    def test_same_input_same_output(self):
        """Identical inputs must produce identical outputs."""
        df = _make_ohlcv(60)
        feat = KronosFeatures(window=10, seed=42)
        r1 = feat.create_kronos_features(df)
        r2 = feat.create_kronos_features(df)
        for col in ALL_12:
            np.testing.assert_array_equal(r1[col].values, r2[col].values)


# --- Feature Count Test ------------------------------------------------------

class TestFeatureCounts:
    def test_total_feature_count(self):
        assert len(ALL_12) == 12

    def test_all_feature_names_classmethod(self):
        names = KronosFeatures._all_feature_names()
        assert len(names) == 12
        assert set(names) == ALL_12

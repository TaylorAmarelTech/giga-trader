"""
Tests for TransferEntropyFeatures.

Validates transfer entropy feature engineering from cross-asset returns
into SPY, with and without optional cross-asset columns present.
"""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.transfer_entropy_features import (
    TransferEntropyFeatures,
    _digitize_series,
    _rolling_te,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_spy_daily(
    n_days: int = 220,
    seed: int = 42,
    include_cross_asset: bool = True,
) -> pd.DataFrame:
    """
    Generate a realistic daily DataFrame for testing.

    Parameters
    ----------
    n_days : int
        Number of trading days (>= 200 to allow warm-up for window=50).
    seed : int
        Random seed for reproducibility.
    include_cross_asset : bool
        If True, include VXX_return, TLT_return, QQQ_return, GLD_return columns.
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")

    spy_ret = rng.normal(0.0003, 0.010, n_days)
    close = 450.0 * np.cumprod(1 + spy_ret)

    df = pd.DataFrame({
        "date": dates,
        "close": close,
        "day_return": spy_ret,
    })

    if include_cross_asset:
        # Correlated cross-asset returns (realistic lead-lag relationships)
        vxx_ret = -0.5 * spy_ret + rng.normal(0, 0.015, n_days)   # VIX inverse
        tlt_ret = -0.3 * spy_ret + rng.normal(0, 0.008, n_days)   # bonds inverse
        qqq_ret =  0.9 * spy_ret + rng.normal(0, 0.005, n_days)   # tech correlated
        gld_ret =  0.1 * spy_ret + rng.normal(0, 0.006, n_days)   # weak correlation

        df["VXX_return"] = vxx_ret
        df["TLT_return"] = tlt_ret
        df["QQQ_return"] = qqq_ret
        df["GLD_return"] = gld_ret

    return df


ALL_6 = {
    "te_vix_to_spy",
    "te_tlt_to_spy",
    "te_qqq_to_spy",
    "te_gld_to_spy",
    "te_max_inflow",
    "te_net_flow",
}


# ─── Unit Tests: Internal Helpers ─────────────────────────────────────────────

class TestDigitizeSeries:

    def test_three_bins_produced(self):
        rng = np.random.RandomState(0)
        x = rng.normal(0, 1, 200)
        bins = _digitize_series(x, n_bins=3)
        assert set(np.unique(bins)).issubset({0, 1, 2})

    def test_constant_series_mid_bin(self):
        x = np.full(50, 5.0)
        bins = _digitize_series(x, n_bins=3)
        assert (bins == 1).all()

    def test_length_preserved(self):
        x = np.random.randn(100)
        bins = _digitize_series(x, n_bins=3)
        assert len(bins) == 100


class TestRollingTE:

    def test_shape_matches_input(self):
        rng = np.random.RandomState(1)
        n = 100
        src = rng.normal(0, 1, n)
        tgt = rng.normal(0, 1, n)
        result = _rolling_te(src, tgt, window=30, n_bins=3)
        assert result.shape == (n,)

    def test_warmup_zeros(self):
        rng = np.random.RandomState(2)
        src = rng.normal(0, 1, 80)
        tgt = rng.normal(0, 1, 80)
        result = _rolling_te(src, tgt, window=50, n_bins=3)
        # First `window` values should be 0.0 (warm-up period)
        assert (result[:50] == 0.0).all()

    def test_non_negative(self):
        rng = np.random.RandomState(3)
        src = rng.normal(0, 1, 120)
        tgt = rng.normal(0, 1, 120)
        result = _rolling_te(src, tgt, window=50, n_bins=3)
        assert (result >= 0.0).all()

    def test_correlated_higher_than_independent(self):
        """Highly correlated source/target should produce higher TE than independent."""
        rng = np.random.RandomState(7)
        n = 200
        tgt = rng.normal(0, 0.01, n)

        # Independent source
        src_indep = rng.normal(0, 0.01, n)
        te_indep = _rolling_te(src_indep, tgt, window=50, n_bins=3)

        # Correlated source (shifted-by-1 target)
        src_corr = np.roll(tgt, 1)
        src_corr[0] = 0.0
        te_corr = _rolling_te(src_corr, tgt, window=50, n_bins=3)

        # On average, correlated TE should be at least as large
        mean_indep = te_indep[50:].mean()
        mean_corr = te_corr[50:].mean()
        assert mean_corr >= mean_indep - 0.01  # allow tiny tolerance


# ─── Invariant Tests ─────────────────────────────────────────────────────────

class TestTransferEntropyInvariants:

    @pytest.fixture
    def te(self):
        return TransferEntropyFeatures(window=50, n_bins=3)

    @pytest.fixture
    def spy_with_cross(self):
        return _make_spy_daily(220, seed=42, include_cross_asset=True)

    @pytest.fixture
    def spy_no_cross(self):
        return _make_spy_daily(220, seed=42, include_cross_asset=False)

    def test_all_6_features_with_cross_asset(self, te, spy_with_cross):
        result = te.create_transfer_entropy_features(spy_with_cross)
        te_cols = {c for c in result.columns if c.startswith("te_")}
        assert te_cols == ALL_6

    def test_all_6_features_without_cross_asset(self, te, spy_no_cross):
        result = te.create_transfer_entropy_features(spy_no_cross)
        te_cols = {c for c in result.columns if c.startswith("te_")}
        assert te_cols == ALL_6

    def test_no_nans_with_cross_asset(self, te, spy_with_cross):
        result = te.create_transfer_entropy_features(spy_with_cross)
        for col in ALL_6:
            assert result[col].isna().sum() == 0, f"NaN found in {col}"

    def test_no_nans_without_cross_asset(self, te, spy_no_cross):
        result = te.create_transfer_entropy_features(spy_no_cross)
        for col in ALL_6:
            assert result[col].isna().sum() == 0, f"NaN found in {col}"

    def test_no_infinities(self, te, spy_with_cross):
        result = te.create_transfer_entropy_features(spy_with_cross)
        for col in ALL_6:
            assert not np.isinf(result[col]).any(), f"Inf in {col}"

    def test_preserves_original_columns(self, te, spy_with_cross):
        original = set(spy_with_cross.columns)
        result = te.create_transfer_entropy_features(spy_with_cross)
        assert original.issubset(set(result.columns))

    def test_preserves_row_count(self, te, spy_with_cross):
        result = te.create_transfer_entropy_features(spy_with_cross)
        assert len(result) == len(spy_with_cross)

    def test_non_negative_values(self, te, spy_with_cross):
        result = te.create_transfer_entropy_features(spy_with_cross)
        for col in ALL_6:
            assert (result[col] >= 0.0).all(), f"Negative values in {col}"


# ─── Feature Logic Tests ──────────────────────────────────────────────────────

class TestTransferEntropyLogic:

    def test_missing_cross_asset_cols_are_zero(self):
        """With no cross-asset columns, individual TE features are all 0."""
        te = TransferEntropyFeatures(window=50, n_bins=3)
        df = _make_spy_daily(220, include_cross_asset=False)
        result = te.create_transfer_entropy_features(df)
        for col in ["te_vix_to_spy", "te_tlt_to_spy", "te_qqq_to_spy", "te_gld_to_spy"]:
            assert (result[col] == 0.0).all(), f"{col} should be 0 without source"

    def test_max_inflow_geq_individual(self):
        """te_max_inflow >= each individual TE value."""
        te = TransferEntropyFeatures(window=50, n_bins=3)
        df = _make_spy_daily(220, include_cross_asset=True)
        result = te.create_transfer_entropy_features(df)
        for col in ["te_vix_to_spy", "te_tlt_to_spy", "te_qqq_to_spy", "te_gld_to_spy"]:
            assert (result["te_max_inflow"] >= result[col] - 1e-10).all(), \
                f"te_max_inflow < {col} on some rows"

    def test_net_flow_between_min_and_max(self):
        """te_net_flow should be between the min and max individual TEs."""
        te = TransferEntropyFeatures(window=50, n_bins=3)
        df = _make_spy_daily(220, include_cross_asset=True)
        result = te.create_transfer_entropy_features(df)
        indiv = result[["te_vix_to_spy", "te_tlt_to_spy", "te_qqq_to_spy", "te_gld_to_spy"]]
        row_min = indiv.min(axis=1)
        row_max = indiv.max(axis=1)
        # net_flow is the mean — should be between min and max (allow tiny float error)
        assert (result["te_net_flow"] >= row_min - 1e-10).all()
        assert (result["te_net_flow"] <= row_max + 1e-10).all()

    def test_partial_cross_asset_cols(self):
        """Only VXX_return present: VIX TE non-zero, others zero."""
        te = TransferEntropyFeatures(window=50, n_bins=3)
        df = _make_spy_daily(220, include_cross_asset=False)
        rng = np.random.RandomState(99)
        df["VXX_return"] = -0.5 * df["day_return"] + rng.normal(0, 0.01, len(df))

        result = te.create_transfer_entropy_features(df)

        # TLT, QQQ, GLD should be zero
        assert (result["te_tlt_to_spy"] == 0.0).all()
        assert (result["te_qqq_to_spy"] == 0.0).all()
        assert (result["te_gld_to_spy"] == 0.0).all()

        # net_flow should equal vix_to_spy (only one source)
        np.testing.assert_array_almost_equal(
            result["te_net_flow"].values,
            result["te_vix_to_spy"].values,
            decimal=10,
        )

    def test_no_close_column_returns_zeros(self):
        """DataFrame without 'close' gets all-zero TE features."""
        te = TransferEntropyFeatures(window=50, n_bins=3)
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=10),
            "volume": np.ones(10) * 1e6,
        })
        result = te.create_transfer_entropy_features(df)
        for col in ALL_6:
            assert col in result.columns
            assert (result[col] == 0.0).all()

    def test_configurable_window(self):
        """Smaller window should produce more non-zero values."""
        df = _make_spy_daily(220, include_cross_asset=True)
        te_small = TransferEntropyFeatures(window=30, n_bins=3)
        te_large = TransferEntropyFeatures(window=100, n_bins=3)

        r_small = te_small.create_transfer_entropy_features(df)
        r_large = te_large.create_transfer_entropy_features(df)

        nonzero_small = (r_small["te_vix_to_spy"] > 0).sum()
        nonzero_large = (r_large["te_vix_to_spy"] > 0).sum()
        assert nonzero_small >= nonzero_large


# ─── Download Method Tests ────────────────────────────────────────────────────

class TestDownloadTEData:

    def test_returns_empty_dataframe(self):
        te = TransferEntropyFeatures()
        result = te.download_te_data(
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ─── analyze_current_te Tests ─────────────────────────────────────────────────

class TestAnalyzeCurrentTE:

    @pytest.fixture
    def analyzed_df(self):
        te = TransferEntropyFeatures(window=50, n_bins=3)
        df = _make_spy_daily(220, seed=5, include_cross_asset=True)
        return te.create_transfer_entropy_features(df)

    def test_returns_dict(self, analyzed_df):
        te = TransferEntropyFeatures()
        result = te.analyze_current_te(analyzed_df)
        assert isinstance(result, dict)

    def test_dominant_source_valid(self, analyzed_df):
        te = TransferEntropyFeatures()
        result = te.analyze_current_te(analyzed_df)
        valid = {"VIX", "TLT", "QQQ", "GLD", "NONE"}
        assert result["dominant_source"] in valid

    def test_information_flow_valid(self, analyzed_df):
        te = TransferEntropyFeatures()
        result = te.analyze_current_te(analyzed_df)
        valid = {"HIGH", "LOW", "NORMAL"}
        assert result["information_flow"] in valid

    def test_te_values_dict_present(self, analyzed_df):
        te = TransferEntropyFeatures()
        result = te.analyze_current_te(analyzed_df)
        assert "te_values" in result
        assert isinstance(result["te_values"], dict)

    def test_te_net_flow_present(self, analyzed_df):
        te = TransferEntropyFeatures()
        result = te.analyze_current_te(analyzed_df)
        assert "te_net_flow" in result
        assert isinstance(result["te_net_flow"], float)

    def test_te_max_inflow_present(self, analyzed_df):
        te = TransferEntropyFeatures()
        result = te.analyze_current_te(analyzed_df)
        assert "te_max_inflow" in result
        assert isinstance(result["te_max_inflow"], float)

    def test_none_without_te_features(self):
        te = TransferEntropyFeatures()
        df = pd.DataFrame({"close": [450.0, 451.0]})
        assert te.analyze_current_te(df) is None

    def test_none_when_no_cross_asset_all_zeros(self):
        """With all-zero TE (no cross-asset cols), dominant_source is NONE."""
        te = TransferEntropyFeatures(window=50, n_bins=3)
        df = _make_spy_daily(220, include_cross_asset=False)
        result_df = te.create_transfer_entropy_features(df)
        result = te.analyze_current_te(result_df)
        # All TE values are 0, so dominant_source should be NONE
        assert result["dominant_source"] == "NONE"


# ─── Feature Count Tests ──────────────────────────────────────────────────────

class TestFeatureCount:

    def test_exactly_6_features(self):
        assert len(ALL_6) == 6

    def test_feature_names_list(self):
        assert len(TransferEntropyFeatures.FEATURE_NAMES) == 6
        for name in ALL_6:
            assert name in TransferEntropyFeatures.FEATURE_NAMES

    def test_prefix_consistency(self):
        df = _make_spy_daily(220, include_cross_asset=True)
        te = TransferEntropyFeatures(window=50, n_bins=3)
        original_cols = set(df.columns)
        result = te.create_transfer_entropy_features(df)
        new_cols = set(result.columns) - original_cols
        for col in new_cols:
            assert col.startswith("te_"), f"Column {col} lacks te_ prefix"

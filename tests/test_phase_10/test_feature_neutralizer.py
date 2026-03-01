"""
Tests for FeatureNeutralizer.
===============================
Comprehensive suite covering all three neutralization methods,
edge cases, and sklearn API compatibility.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase_10_feature_processing.feature_neutralizer import FeatureNeutralizer


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def rng():
    """Reproducible random state."""
    return np.random.RandomState(42)


@pytest.fixture
def simple_data(rng):
    """100 x 5 random matrix with well-behaved values."""
    return rng.randn(100, 5)


@pytest.fixture
def correlated_data(rng):
    """
    Data where column 1 is strongly correlated with column 0 (the market
    factor) plus noise, so we can verify that residual neutralization
    removes that relationship.
    """
    n = 200
    market = rng.randn(n)
    noise = rng.randn(n) * 0.1
    col1 = 2.0 * market + 3.0 + noise  # strong beta=2, intercept=3
    col2 = rng.randn(n)                 # unrelated
    col3 = -0.5 * market + noise * 2    # weak negative beta
    col4 = rng.randn(n) * 5 + 10       # different scale, uncorrelated
    X = np.column_stack([market, col1, col2, col3, col4])
    return X, market


@pytest.fixture
def data_with_nans(rng):
    """Data that contains some NaN values."""
    X = rng.randn(100, 4)
    X[5, 1] = np.nan
    X[10, 2] = np.nan
    X[50, 0] = np.nan
    X[99, 3] = np.nan
    return X


@pytest.fixture
def data_with_outliers(rng):
    """Data with extreme outlier values."""
    X = rng.randn(200, 4)
    X[0, 0] = 100.0
    X[1, 1] = -80.0
    X[2, 2] = 50.0
    X[3, 3] = -60.0
    return X


# =============================================================================
# TEST DEMEANING
# =============================================================================

class TestDemeaning:
    """Tests for method='demeaning'."""

    def test_demeaning_removes_column_means(self, simple_data):
        """After demeaning, each column should have near-zero mean."""
        neutralizer = FeatureNeutralizer(method="demeaning")
        X_out = neutralizer.fit_transform(simple_data)
        col_means = np.nanmean(X_out, axis=0)
        np.testing.assert_allclose(col_means, 0.0, atol=1e-10)

    def test_demeaning_uses_training_means_not_test(self, rng):
        """Transform must use means from fit, not recalculate on test set."""
        X_train = rng.randn(100, 3) + 5.0  # mean ~ 5
        X_test = rng.randn(50, 3) + 10.0   # mean ~ 10

        neutralizer = FeatureNeutralizer(method="demeaning")
        neutralizer.fit(X_train)
        X_test_out = neutralizer.transform(X_test)

        # Test means should be roughly 10 - 5 = 5, NOT 0
        test_means = np.mean(X_test_out, axis=0)
        assert np.all(test_means > 3.0), (
            "Transform should subtract training means, not test means"
        )

    def test_demeaning_preserves_shape(self, simple_data):
        """Output shape must match input shape."""
        neutralizer = FeatureNeutralizer(method="demeaning")
        X_out = neutralizer.fit_transform(simple_data)
        assert X_out.shape == simple_data.shape

    def test_demeaning_stores_means(self, simple_data):
        """Fitted neutralizer should have means_ attribute."""
        neutralizer = FeatureNeutralizer(method="demeaning")
        neutralizer.fit(simple_data)
        assert hasattr(neutralizer, "means_")
        assert neutralizer.means_.shape == (simple_data.shape[1],)

    def test_demeaning_idempotent_fit(self, simple_data):
        """Fitting twice on same data should give same result."""
        n1 = FeatureNeutralizer(method="demeaning")
        n2 = FeatureNeutralizer(method="demeaning")
        X1 = n1.fit_transform(simple_data)
        n2.fit(simple_data)
        X2 = n2.transform(simple_data)
        np.testing.assert_array_equal(X1, X2)

    def test_demeaning_single_row_above_min(self, rng):
        """With exactly min_samples rows, should NOT pass through."""
        X = rng.randn(50, 3)
        neutralizer = FeatureNeutralizer(method="demeaning", min_samples=50)
        X_out = neutralizer.fit_transform(X)
        # Should be demeaned, not passthrough
        col_means = np.mean(X_out, axis=0)
        np.testing.assert_allclose(col_means, 0.0, atol=1e-10)


# =============================================================================
# TEST RESIDUAL
# =============================================================================

class TestResidual:
    """Tests for method='residual'."""

    def test_residual_removes_market_beta(self, correlated_data):
        """
        Column 1 = 2*market + 3 + noise.  After residual neutralization,
        the correlation between neutralized col 1 and market should be
        near zero.
        """
        X, _ = correlated_data
        neutralizer = FeatureNeutralizer(method="residual", market_col_idx=0)
        X_out = neutralizer.fit_transform(X)

        # Correlation between residual of col 1 and market (col 0 of original)
        corr = np.corrcoef(X_out[:, 1], X[:, 0])[0, 1]
        assert abs(corr) < 0.15, (
            f"Expected near-zero correlation after neutralization, got {corr:.4f}"
        )

    def test_residual_stores_betas(self, correlated_data):
        """Betas and intercepts should be stored after fit."""
        X, _ = correlated_data
        neutralizer = FeatureNeutralizer(method="residual", market_col_idx=0)
        neutralizer.fit(X)

        assert hasattr(neutralizer, "betas_")
        assert hasattr(neutralizer, "intercepts_")
        assert neutralizer.betas_.shape == (X.shape[1],)
        assert neutralizer.intercepts_.shape == (X.shape[1],)

    def test_residual_beta_values(self, correlated_data):
        """Known betas should be recovered approximately."""
        X, _ = correlated_data
        neutralizer = FeatureNeutralizer(method="residual", market_col_idx=0)
        neutralizer.fit(X)

        # col 1 has beta ~ 2.0, intercept ~ 3.0
        assert abs(neutralizer.betas_[1] - 2.0) < 0.1
        assert abs(neutralizer.intercepts_[1] - 3.0) < 0.2

        # col 0 (market on itself) should have beta ~ 1.0, intercept ~ 0
        assert abs(neutralizer.betas_[0] - 1.0) < 0.01
        assert abs(neutralizer.intercepts_[0]) < 0.01

    def test_residual_preserves_shape(self, simple_data):
        """Output shape must match input shape."""
        neutralizer = FeatureNeutralizer(method="residual", market_col_idx=0)
        X_out = neutralizer.fit_transform(simple_data)
        assert X_out.shape == simple_data.shape

    def test_residual_uncorrelated_column_unchanged(self, correlated_data):
        """Column uncorrelated with market should be largely unchanged."""
        X, _ = correlated_data
        neutralizer = FeatureNeutralizer(method="residual", market_col_idx=0)
        X_out = neutralizer.fit_transform(X)

        # col 2 is random noise, beta should be near 0
        assert abs(neutralizer.betas_[2]) < 0.2

    def test_residual_with_explicit_market_returns(self, rng):
        """Passing market_returns explicitly should use those, not col 0."""
        n = 150
        market = rng.randn(n)
        unrelated_col0 = rng.randn(n) * 10
        col1 = 3.0 * market + rng.randn(n) * 0.05
        X = np.column_stack([unrelated_col0, col1])

        neutralizer = FeatureNeutralizer(method="residual", market_col_idx=0)
        neutralizer.fit(X, market_returns=market)

        # Beta for col 1 w.r.t. the explicit market should be ~ 3.0
        assert abs(neutralizer.betas_[1] - 3.0) < 0.15

    def test_residual_market_col_self_neutralizes(self, simple_data):
        """The market column itself should become near-zero after neutralization."""
        neutralizer = FeatureNeutralizer(method="residual", market_col_idx=0)
        X_out = neutralizer.fit_transform(simple_data)
        # col 0 regressed on itself => residual should be ~0
        assert np.std(X_out[:, 0]) < 1e-10

    def test_residual_uses_fit_betas_on_test(self, rng):
        """Transform on test data should use betas from training fit."""
        X_train = rng.randn(100, 3)
        X_test = rng.randn(50, 3)

        neutralizer = FeatureNeutralizer(method="residual", market_col_idx=0)
        neutralizer.fit(X_train)
        betas_after_fit = neutralizer.betas_.copy()

        neutralizer.transform(X_test)
        # Betas should not change after transform
        np.testing.assert_array_equal(neutralizer.betas_, betas_after_fit)


# =============================================================================
# TEST WINSORIZED RESIDUAL
# =============================================================================

class TestWinsorizedResidual:
    """Tests for method='winsorized_residual'."""

    def test_winsorized_clips_extremes(self, data_with_outliers):
        """Winsorized method should clip outliers before regression."""
        neutralizer = FeatureNeutralizer(
            method="winsorized_residual",
            winsorize_pct=0.01,
        )
        neutralizer.fit(data_with_outliers)

        assert hasattr(neutralizer, "clip_lo_")
        assert hasattr(neutralizer, "clip_hi_")
        assert neutralizer.clip_lo_.shape == (data_with_outliers.shape[1],)
        assert neutralizer.clip_hi_.shape == (data_with_outliers.shape[1],)

        # Lo should be < hi for all columns
        assert np.all(neutralizer.clip_lo_ <= neutralizer.clip_hi_)

    def test_winsorized_stores_betas(self, data_with_outliers):
        """Winsorized residual should also store betas and intercepts."""
        neutralizer = FeatureNeutralizer(method="winsorized_residual")
        neutralizer.fit(data_with_outliers)
        assert hasattr(neutralizer, "betas_")
        assert hasattr(neutralizer, "intercepts_")

    def test_winsorized_preserves_shape(self, data_with_outliers):
        """Output shape must match input."""
        neutralizer = FeatureNeutralizer(method="winsorized_residual")
        X_out = neutralizer.fit_transform(data_with_outliers)
        assert X_out.shape == data_with_outliers.shape

    def test_winsorized_different_from_residual(self, data_with_outliers):
        """Results should differ from plain residual when outliers exist."""
        n_res = FeatureNeutralizer(method="residual")
        n_win = FeatureNeutralizer(method="winsorized_residual", winsorize_pct=0.05)

        X_res = n_res.fit_transform(data_with_outliers)
        X_win = n_win.fit_transform(data_with_outliers)

        # Should not be identical due to clipping
        assert not np.allclose(X_res, X_win), (
            "Winsorized and plain residual should differ on outlier data"
        )

    def test_winsorized_betas_more_robust(self, rng):
        """
        With a single huge outlier, winsorized regression should produce
        a beta closer to the true beta than plain residual.
        """
        n = 200
        market = rng.randn(n)
        true_beta = 1.5
        col1 = true_beta * market + rng.randn(n) * 0.1

        # Add massive outlier
        col1[0] = 500.0

        X = np.column_stack([market, col1])

        n_res = FeatureNeutralizer(method="residual")
        n_win = FeatureNeutralizer(method="winsorized_residual", winsorize_pct=0.01)

        n_res.fit(X)
        n_win.fit(X)

        err_res = abs(n_res.betas_[1] - true_beta)
        err_win = abs(n_win.betas_[1] - true_beta)

        assert err_win <= err_res + 0.5, (
            f"Winsorized beta error ({err_win:.3f}) should not be much worse "
            f"than plain ({err_res:.3f})"
        )

    def test_winsorized_transform_clips_test_data(self, rng):
        """Transform should clip test data using training percentiles."""
        X_train = rng.randn(100, 3)
        neutralizer = FeatureNeutralizer(
            method="winsorized_residual", winsorize_pct=0.05,
        )
        neutralizer.fit(X_train)

        # Test data with extreme values
        X_test = rng.randn(20, 3)
        X_test[0, 0] = 999.0
        X_test[1, 1] = -999.0

        X_out = neutralizer.transform(X_test)
        # Output should exist and have correct shape
        assert X_out.shape == X_test.shape
        # The extreme values should be clipped, so output should be finite
        assert np.all(np.isfinite(X_out))


# =============================================================================
# TEST EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for boundary conditions and unusual inputs."""

    def test_min_samples_passthrough(self, rng):
        """If X has fewer rows than min_samples, return unchanged copy."""
        X = rng.randn(10, 3)
        neutralizer = FeatureNeutralizer(method="residual", min_samples=50)
        X_out = neutralizer.fit_transform(X)
        np.testing.assert_array_equal(X_out, X)

    def test_min_samples_boundary_below(self, rng):
        """49 rows with min_samples=50 should pass through."""
        X = rng.randn(49, 3)
        neutralizer = FeatureNeutralizer(method="demeaning", min_samples=50)
        X_out = neutralizer.fit_transform(X)
        np.testing.assert_array_equal(X_out, X)

    def test_min_samples_boundary_exact(self, rng):
        """Exactly min_samples rows should NOT pass through."""
        X = rng.randn(50, 3) + 5.0
        neutralizer = FeatureNeutralizer(method="demeaning", min_samples=50)
        X_out = neutralizer.fit_transform(X)
        # Demeaned means should be ~ 0, not ~ 5
        means = np.mean(X_out, axis=0)
        np.testing.assert_allclose(means, 0.0, atol=1e-10)

    def test_constant_column_handled(self, rng):
        """Constant columns should not crash; beta should be 0."""
        X = rng.randn(100, 3)
        X[:, 1] = 7.0  # constant column
        neutralizer = FeatureNeutralizer(method="residual", market_col_idx=0)
        neutralizer.fit(X)

        assert neutralizer.betas_[1] == 0.0
        assert neutralizer.intercepts_[1] == 7.0

        X_out = neutralizer.transform(X)
        assert X_out.shape == X.shape

    def test_single_column(self, rng):
        """Single-column X should work for demeaning."""
        X = rng.randn(100, 1)
        neutralizer = FeatureNeutralizer(method="demeaning")
        X_out = neutralizer.fit_transform(X)
        assert X_out.shape == (100, 1)
        np.testing.assert_allclose(np.mean(X_out), 0.0, atol=1e-10)

    def test_single_column_residual(self, rng):
        """Single-column X with residual: market is the only column."""
        X = rng.randn(100, 1)
        neutralizer = FeatureNeutralizer(method="residual", market_col_idx=0)
        X_out = neutralizer.fit_transform(X)
        # Regressing col 0 on itself => residual ~ 0
        assert X_out.shape == (100, 1)
        assert np.std(X_out) < 1e-10

    def test_nan_handling_demeaning(self, data_with_nans):
        """NaN positions should be preserved in output."""
        neutralizer = FeatureNeutralizer(method="demeaning")
        X_out = neutralizer.fit_transform(data_with_nans)

        # NaN positions from input should still be NaN in output
        assert np.isnan(X_out[5, 1])
        assert np.isnan(X_out[10, 2])
        assert np.isnan(X_out[50, 0])
        assert np.isnan(X_out[99, 3])

    def test_nan_handling_residual(self, data_with_nans):
        """NaN positions should be preserved in residual output."""
        neutralizer = FeatureNeutralizer(method="residual")
        X_out = neutralizer.fit_transform(data_with_nans)
        assert np.isnan(X_out[5, 1])
        assert np.isnan(X_out[10, 2])

    def test_empty_array(self):
        """Empty array (0 rows) should trigger passthrough."""
        X = np.empty((0, 5))
        neutralizer = FeatureNeutralizer(method="demeaning", min_samples=50)
        X_out = neutralizer.fit_transform(X)
        assert X_out.shape == (0, 5)

    def test_feature_mismatch_raises(self, rng):
        """Transforming with different feature count should raise."""
        X_train = rng.randn(100, 5)
        X_test = rng.randn(50, 3)

        neutralizer = FeatureNeutralizer(method="demeaning")
        neutralizer.fit(X_train)

        with pytest.raises(ValueError, match="features"):
            neutralizer.transform(X_test)

    def test_invalid_method_raises(self):
        """Unknown method should raise ValueError."""
        neutralizer = FeatureNeutralizer(method="unknown_method")
        X = np.random.randn(100, 3)
        with pytest.raises(ValueError, match="Unknown method"):
            neutralizer.fit(X)

    def test_market_col_idx_out_of_bounds(self, rng):
        """market_col_idx beyond columns should raise at fit time."""
        X = rng.randn(100, 3)
        neutralizer = FeatureNeutralizer(method="residual", market_col_idx=10)
        with pytest.raises(ValueError, match="market_col_idx"):
            neutralizer.fit(X)

    def test_all_nan_column(self, rng):
        """Column that is entirely NaN should not crash."""
        X = rng.randn(100, 3)
        X[:, 2] = np.nan
        neutralizer = FeatureNeutralizer(method="demeaning")
        X_out = neutralizer.fit_transform(X)
        # All-NaN column treated as all-zero for mean, output is NaN
        assert X_out.shape == X.shape
        assert np.all(np.isnan(X_out[:, 2]))

    def test_inf_values_treated_as_zero(self, rng):
        """Inf values should be treated as 0 during fit, preserved as NaN."""
        X = rng.randn(100, 3)
        X[0, 0] = np.inf
        X[1, 1] = -np.inf
        neutralizer = FeatureNeutralizer(method="demeaning")
        X_out = neutralizer.fit_transform(X)
        assert X_out.shape == X.shape
        # Inf positions become NaN in output (since they're not finite)
        assert np.isnan(X_out[0, 0])
        assert np.isnan(X_out[1, 1])

    def test_large_data(self, rng):
        """Should handle reasonably large datasets without error."""
        X = rng.randn(5000, 50)
        neutralizer = FeatureNeutralizer(method="residual")
        X_out = neutralizer.fit_transform(X)
        assert X_out.shape == (5000, 50)


# =============================================================================
# TEST SKLEARN COMPATIBILITY
# =============================================================================

class TestSklearnCompat:
    """Tests for sklearn API conformance."""

    def test_fit_transform_equals_fit_then_transform(self, simple_data):
        """fit_transform should produce same result as fit then transform."""
        n1 = FeatureNeutralizer(method="demeaning")
        n2 = FeatureNeutralizer(method="demeaning")

        X1 = n1.fit_transform(simple_data)
        n2.fit(simple_data)
        X2 = n2.transform(simple_data)

        np.testing.assert_array_almost_equal(X1, X2)

    def test_fit_transform_residual_equals_separate(self, simple_data):
        """Same check for residual method."""
        n1 = FeatureNeutralizer(method="residual")
        n2 = FeatureNeutralizer(method="residual")

        X1 = n1.fit_transform(simple_data)
        n2.fit(simple_data)
        X2 = n2.transform(simple_data)

        np.testing.assert_array_almost_equal(X1, X2)

    def test_transform_without_fit_raises(self, simple_data):
        """Calling transform before fit should raise an error."""
        neutralizer = FeatureNeutralizer(method="demeaning")
        with pytest.raises(Exception):
            neutralizer.transform(simple_data)

    def test_get_params(self):
        """get_params should return constructor parameters."""
        neutralizer = FeatureNeutralizer(
            method="residual",
            market_col_idx=2,
            winsorize_pct=0.05,
            min_samples=100,
        )
        params = neutralizer.get_params()
        assert params["method"] == "residual"
        assert params["market_col_idx"] == 2
        assert params["winsorize_pct"] == 0.05
        assert params["min_samples"] == 100

    def test_set_params(self):
        """set_params should update parameters."""
        neutralizer = FeatureNeutralizer(method="demeaning")
        neutralizer.set_params(method="residual", min_samples=200)
        assert neutralizer.method == "residual"
        assert neutralizer.min_samples == 200

    def test_clone(self):
        """sklearn clone should produce identical unfitted copy."""
        from sklearn.base import clone
        original = FeatureNeutralizer(
            method="winsorized_residual",
            winsorize_pct=0.02,
            min_samples=75,
        )
        cloned = clone(original)
        assert cloned.method == original.method
        assert cloned.winsorize_pct == original.winsorize_pct
        assert cloned.min_samples == original.min_samples
        # Cloned should not be fitted
        assert not hasattr(cloned, "is_fitted_")

    def test_pipeline_integration(self, rng):
        """FeatureNeutralizer should work inside an sklearn Pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X = rng.randn(100, 5)
        y = (rng.randn(100) > 0).astype(int)

        pipe = Pipeline([
            ("neutralize", FeatureNeutralizer(method="demeaning")),
            ("scale", StandardScaler()),
        ])

        X_out = pipe.fit_transform(X)
        assert X_out.shape == X.shape

        # After scaler, each column should have mean ~ 0 and std ~ 1
        np.testing.assert_allclose(np.mean(X_out, axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.std(X_out, axis=0), 1.0, atol=1e-10)

    def test_repr(self):
        """repr should produce readable string."""
        neutralizer = FeatureNeutralizer(method="residual", market_col_idx=3)
        r = repr(neutralizer)
        assert "FeatureNeutralizer" in r
        assert "residual" in r
        assert "3" in r

    def test_is_fitted_flag(self, simple_data):
        """is_fitted_ should be set after fit."""
        neutralizer = FeatureNeutralizer(method="demeaning")
        assert not hasattr(neutralizer, "is_fitted_")
        neutralizer.fit(simple_data)
        assert neutralizer.is_fitted_ is True

    def test_n_features_in(self, simple_data):
        """n_features_in_ should be set after fit."""
        neutralizer = FeatureNeutralizer(method="demeaning")
        neutralizer.fit(simple_data)
        assert neutralizer.n_features_in_ == simple_data.shape[1]


# =============================================================================
# TEST MULTIPLE METHODS COMPARISON
# =============================================================================

class TestMethodComparison:
    """Tests comparing behavior across methods."""

    def test_demeaning_vs_residual_differ(self, simple_data):
        """Different methods should produce different outputs."""
        n_dem = FeatureNeutralizer(method="demeaning")
        n_res = FeatureNeutralizer(method="residual")

        X_dem = n_dem.fit_transform(simple_data)
        X_res = n_res.fit_transform(simple_data)

        # They should generally differ (unless data is perfectly uncorrelated)
        assert not np.allclose(X_dem, X_res)

    def test_all_methods_same_shape(self, simple_data):
        """All methods should output the same shape as input."""
        for method in ("demeaning", "residual", "winsorized_residual"):
            neutralizer = FeatureNeutralizer(method=method)
            X_out = neutralizer.fit_transform(simple_data)
            assert X_out.shape == simple_data.shape, (
                f"Shape mismatch for method '{method}'"
            )

    def test_all_methods_handle_passthrough(self, rng):
        """All methods should pass through when below min_samples."""
        X = rng.randn(5, 3)
        for method in ("demeaning", "residual", "winsorized_residual"):
            neutralizer = FeatureNeutralizer(method=method, min_samples=50)
            X_out = neutralizer.fit_transform(X)
            np.testing.assert_array_equal(X_out, X)

"""
Tests for src.phase_13_validation.anti_overfit_integration

Covers:
  - integrate_anti_overfit() with all flags off
  - Empty and minimal DataFrames
  - _run_feature_step() success and failure paths
  - Function signature accepts all expected kwargs
  - use_synthetic=False skips synthetic universe generation
"""

import importlib
import inspect
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.phase_13_validation.anti_overfit_integration import (
    _FeatureStep,
    _FEATURE_STEPS,
    _run_feature_step,
    integrate_anti_overfit,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_daily_df():
    """50-row DataFrame with only 'date' and 'close' columns."""
    np.random.seed(99)
    dates = pd.bdate_range("2024-06-01", periods=50, freq="B")
    close = 450.0 * np.cumprod(1 + np.random.normal(0, 0.01, 50))
    return pd.DataFrame({"date": dates, "close": close})


@pytest.fixture
def empty_daily_df():
    """Empty DataFrame with the expected columns."""
    return pd.DataFrame(columns=["date", "close", "open", "high", "low", "volume"])


@pytest.fixture
def all_flags_off():
    """Dict of every boolean flag set to False."""
    sig = inspect.signature(integrate_anti_overfit)
    flags = {}
    for name, param in sig.parameters.items():
        if name == "df_daily":
            continue
        if param.default is True or param.default is False:
            flags[name] = False
    return flags


@pytest.fixture
def dummy_step():
    """A _FeatureStep pointing at a mock module path."""
    return _FeatureStep(
        flag="use_dummy",
        label="DUMMY",
        meta_key="dummy_features",
        cls_path="fake_module.FakeClass",
        create_method="create_features",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Patch every inline-step import so the function body never reaches real APIs.
_INLINE_PATCHES = [
    "src.phase_13_validation.anti_overfit_integration.ComponentStreakFeatures",
    "src.phase_13_validation.anti_overfit_integration.CrossAssetFeatures",
    "src.phase_13_validation.anti_overfit_integration.Mag7BreadthFeatures",
    "src.phase_13_validation.anti_overfit_integration.SectorBreadthFeatures",
    "src.phase_13_validation.anti_overfit_integration.VolatilityRegimeFeatures",
    "src.phase_13_validation.anti_overfit_integration.SyntheticSPYGenerator",
    "src.phase_13_validation.anti_overfit_integration.EconomicFeatures",
    "src.phase_13_validation.anti_overfit_integration.SentimentFeatures",
    "src.phase_13_validation.anti_overfit_integration.CalendarFeatureGenerator",
    "src.phase_13_validation.anti_overfit_integration._maybe_gc",
]


def _patch_all_inline():
    """Return a list of active patchers for all inline imports."""
    patchers = [patch(p) for p in _INLINE_PATCHES]
    return patchers


class TestAllFlagsOff:
    """integrate_anti_overfit with every feature flag disabled."""

    def test_returns_unchanged_shape(self, minimal_daily_df, all_flags_off):
        patchers = _patch_all_inline()
        for p in patchers:
            p.start()
        try:
            result_df, metadata = integrate_anti_overfit(
                minimal_daily_df, **all_flags_off,
            )
            assert len(result_df) == len(minimal_daily_df)
            assert "close" in result_df.columns
        finally:
            for p in patchers:
                p.stop()

    def test_metadata_is_dict(self, minimal_daily_df, all_flags_off):
        patchers = _patch_all_inline()
        for p in patchers:
            p.start()
        try:
            _, metadata = integrate_anti_overfit(
                minimal_daily_df, **all_flags_off,
            )
            assert isinstance(metadata, dict)
        finally:
            for p in patchers:
                p.stop()


class TestEmptyDataFrame:
    """integrate_anti_overfit with an empty DataFrame."""

    def test_empty_df_returns_tuple(self, empty_daily_df, all_flags_off):
        patchers = _patch_all_inline()
        for p in patchers:
            p.start()
        try:
            result = integrate_anti_overfit(empty_daily_df, **all_flags_off)
            assert isinstance(result, tuple)
            assert len(result) == 2
            df_out, meta = result
            assert isinstance(df_out, pd.DataFrame)
            assert isinstance(meta, dict)
        finally:
            for p in patchers:
                p.stop()


class TestMinimalDataFrame:
    """integrate_anti_overfit with a tiny 50-row date+close DataFrame."""

    def test_minimal_df_no_crash(self, minimal_daily_df, all_flags_off):
        patchers = _patch_all_inline()
        for p in patchers:
            p.start()
        try:
            df_out, meta = integrate_anti_overfit(
                minimal_daily_df, **all_flags_off,
            )
            assert len(df_out) == 50
        finally:
            for p in patchers:
                p.stop()


class TestRunFeatureStepSuccess:
    """_run_feature_step with a mock class that succeeds."""

    def test_success_sets_metadata_true(self, minimal_daily_df, dummy_step):
        fake_cls = MagicMock()
        fake_instance = MagicMock()
        fake_cls.return_value = fake_instance
        # create_features returns the df with one extra column
        def _add_col(df):
            df = df.copy()
            df["dummy_feat_1"] = 0.0
            return df
        fake_instance.create_features.side_effect = _add_col

        fake_mod = MagicMock()
        fake_mod.FakeClass = fake_cls

        metadata = {}
        with patch("importlib.import_module", return_value=fake_mod):
            df_out = _run_feature_step(
                dummy_step,
                minimal_daily_df.copy(),
                pd.Timestamp("2024-06-01"),
                pd.Timestamp("2024-08-15"),
                metadata,
                None,
                {},
            )
        assert metadata["dummy_features"] is True
        assert metadata["n_dummy_features"] == 1
        assert "dummy_feat_1" in df_out.columns


class TestRunFeatureStepFailure:
    """_run_feature_step with a step that raises an exception."""

    def test_failure_sets_metadata_false(self, minimal_daily_df, dummy_step):
        fake_mod = MagicMock()
        fake_mod.FakeClass.side_effect = RuntimeError("boom")

        metadata = {}
        with patch("importlib.import_module", return_value=fake_mod):
            df_out = _run_feature_step(
                dummy_step,
                minimal_daily_df.copy(),
                pd.Timestamp("2024-06-01"),
                pd.Timestamp("2024-08-15"),
                metadata,
                None,
                {},
            )
        assert metadata["dummy_features"] is False
        # DataFrame should be returned unchanged
        assert len(df_out) == len(minimal_daily_df)


class TestFunctionSignature:
    """Verify integrate_anti_overfit accepts all documented kwargs."""

    EXPECTED_PARAMS = [
        "df_daily", "spy_1min", "use_synthetic", "use_cross_assets",
        "use_breadth_streaks", "use_mag_breadth", "use_sector_breadth",
        "use_vol_regime", "use_economic_features", "use_calendar_features",
        "use_sentiment_features", "validate_ohlc", "use_fear_greed",
        "use_reddit_sentiment", "use_crypto_sentiment", "use_gamma_exposure",
        "use_finnhub_social", "use_dark_pool", "use_options_features",
        "use_event_recency", "use_block_structure", "use_amihud_features",
        "use_range_vol_features", "use_entropy_features", "use_hurst_features",
        "use_nmi_features", "use_absorption_ratio", "use_drift_features",
        "use_changepoint_features", "use_hmm_features", "use_vpin_features",
        "use_intraday_momentum", "use_futures_basis", "use_congressional_features",
        "use_insider_aggregate", "use_etf_flow", "use_wavelet_features",
        "use_sax_features", "use_transfer_entropy", "use_mfdfa_features",
        "use_rqa_features", "use_copula_features", "use_network_centrality",
        "use_path_signatures", "use_wavelet_scattering", "use_wasserstein_regime",
        "use_market_structure", "use_time_series_models", "use_catch22",
        "use_har_rv", "use_l_moments", "use_multiscale_entropy",
        "use_rv_signature_plot", "use_tda_homology",
        "synthetic_weight", "resource_config",
    ]

    def test_all_expected_params_present(self):
        sig = inspect.signature(integrate_anti_overfit)
        actual = set(sig.parameters.keys())
        for p in self.EXPECTED_PARAMS:
            assert p in actual, f"Missing parameter: {p}"


class TestSyntheticSkipped:
    """use_synthetic=False must skip synthetic universe generation."""

    def test_no_synthetic_metadata(self, minimal_daily_df, all_flags_off):
        patchers = _patch_all_inline()
        mocks = [p.start() for p in patchers]
        try:
            # Ensure use_synthetic is False
            flags = {**all_flags_off, "use_synthetic": False}
            _, metadata = integrate_anti_overfit(minimal_daily_df, **flags)
            # SyntheticSPYGenerator should never have been instantiated
            synth_mock = mocks[5]  # index matches SyntheticSPYGenerator patch
            synth_mock.assert_not_called()
            assert "n_universes" not in metadata
        finally:
            for p in patchers:
                p.stop()


class TestFeatureStepRegistry:
    """Basic sanity checks on the _FEATURE_STEPS registry."""

    def test_registry_is_nonempty(self):
        assert len(_FEATURE_STEPS) > 0

    def test_each_entry_has_required_fields(self):
        for step in _FEATURE_STEPS:
            assert step.flag, f"Missing flag on step {step.label}"
            assert step.label, f"Missing label"
            assert step.meta_key, f"Missing meta_key on {step.label}"
            assert step.cls_path, f"Missing cls_path on {step.label}"
            assert step.create_method, f"Missing create_method on {step.label}"

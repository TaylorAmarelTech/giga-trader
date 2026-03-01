"""
Tests for ETFFlowFeatures class.

Validates ETF fund-flow proxy feature engineering from close/volume data
without requiring any live API calls or external packages.
"""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.etf_flow_features import ETFFlowFeatures


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_spy_daily(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic SPY-like daily DataFrame.

    Parameters
    ----------
    n : int
        Number of rows (trading days).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: date, close, volume (and auxiliary open/high/low/return).
    """
    np.random.seed(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    log_returns = np.random.normal(0.0003, 0.01, n)
    close = 450.0 * np.exp(np.cumsum(log_returns))
    # Realistic SPY daily volume: 60–150 M shares with occasional spikes
    base_vol = np.random.randint(60_000_000, 150_000_000, n).astype(float)
    # Inject a handful of creation/redemption-style spikes (high vol, low move)
    spike_idx = np.random.choice(n, size=20, replace=False)
    base_vol[spike_idx] *= 2.0
    log_returns[spike_idx] = log_returns[spike_idx] * 0.1  # small price change

    return pd.DataFrame({
        "date": pd.to_datetime(dates.date),
        "close": close,
        "volume": base_vol,
        "day_return": log_returns,
    })


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def spy_daily():
    return _make_spy_daily(n=250)


@pytest.fixture
def spy_daily_large():
    """Larger dataset — ensures rolling windows are populated."""
    return _make_spy_daily(n=400, seed=7)


@pytest.fixture
def engine():
    return ETFFlowFeatures()


@pytest.fixture
def engine_custom():
    return ETFFlowFeatures(flow_window=10, z_window=30)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Constructor / Init Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestETFFlowFeaturesInit:

    def test_default_constructor(self, engine):
        assert isinstance(engine, ETFFlowFeatures)

    def test_default_flow_window(self, engine):
        assert engine.flow_window == 20

    def test_default_z_window(self, engine):
        assert engine.z_window == 60

    def test_custom_windows(self, engine_custom):
        assert engine_custom.flow_window == 10
        assert engine_custom.z_window == 30


# ═══════════════════════════════════════════════════════════════════════════════
# 2. download_etf_flow_data — Always returns empty DataFrame
# ═══════════════════════════════════════════════════════════════════════════════

class TestDownloadEtfFlowData:

    def test_returns_dataframe(self, engine):
        from datetime import datetime
        result = engine.download_etf_flow_data(
            datetime(2023, 1, 1), datetime(2023, 12, 31)
        )
        assert isinstance(result, pd.DataFrame)

    def test_returns_empty(self, engine):
        from datetime import datetime
        result = engine.download_etf_flow_data(
            datetime(2023, 1, 1), datetime(2023, 12, 31)
        )
        assert result.empty


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Feature Engineering — Core Behaviour
# ═══════════════════════════════════════════════════════════════════════════════

class TestCreateEtfFlowFeatures:

    def test_returns_dataframe(self, engine, spy_daily):
        result = engine.create_etf_flow_features(spy_daily)
        assert isinstance(result, pd.DataFrame)

    def test_exact_four_features_added(self, engine, spy_daily):
        result = engine.create_etf_flow_features(spy_daily)
        etf_cols = [c for c in result.columns if c.startswith("etf_flow_")]
        assert len(etf_cols) == 4

    def test_expected_column_names(self, engine, spy_daily):
        result = engine.create_etf_flow_features(spy_daily)
        expected = {
            "etf_flow_spy_20d",
            "etf_flow_spy_z",
            "etf_flow_creation_redemption",
            "etf_flow_short_interest_ratio",
        }
        for col in expected:
            assert col in result.columns, f"Missing: {col}"

    def test_all_prefixed_etf_flow(self, engine, spy_daily):
        result = engine.create_etf_flow_features(spy_daily)
        new_cols = set(result.columns) - set(spy_daily.columns)
        for col in new_cols:
            assert col.startswith("etf_flow_"), f"Unexpected prefix: {col}"

    def test_preserves_original_columns(self, engine, spy_daily):
        original_cols = set(spy_daily.columns)
        result = engine.create_etf_flow_features(spy_daily)
        assert original_cols.issubset(set(result.columns))

    def test_same_row_count(self, engine, spy_daily):
        result = engine.create_etf_flow_features(spy_daily)
        assert len(result) == len(spy_daily)

    def test_no_nans_in_etf_cols(self, engine, spy_daily):
        result = engine.create_etf_flow_features(spy_daily)
        etf_cols = [c for c in result.columns if c.startswith("etf_flow_")]
        nan_count = result[etf_cols].isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values in ETF flow columns"

    def test_does_not_mutate_input(self, engine, spy_daily):
        original_cols = list(spy_daily.columns)
        original_len = len(spy_daily)
        _ = engine.create_etf_flow_features(spy_daily)
        assert list(spy_daily.columns) == original_cols
        assert len(spy_daily) == original_len


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Feature Value Validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestEtfFlowFeatureValues:

    def test_creation_redemption_in_zero_one(self, engine, spy_daily_large):
        result = engine.create_etf_flow_features(spy_daily_large)
        col = result["etf_flow_creation_redemption"]
        assert col.min() >= 0.0
        assert col.max() <= 1.0

    def test_short_interest_ratio_in_zero_one(self, engine, spy_daily_large):
        result = engine.create_etf_flow_features(spy_daily_large)
        col = result["etf_flow_short_interest_ratio"]
        assert col.min() >= 0.0
        assert col.max() <= 1.0

    def test_spy_20d_is_finite(self, engine, spy_daily_large):
        result = engine.create_etf_flow_features(spy_daily_large)
        assert np.isfinite(result["etf_flow_spy_20d"]).all()

    def test_spy_z_is_finite(self, engine, spy_daily_large):
        result = engine.create_etf_flow_features(spy_daily_large)
        assert np.isfinite(result["etf_flow_spy_z"]).all()

    def test_custom_window_produces_features(self, engine_custom, spy_daily):
        result = engine_custom.create_etf_flow_features(spy_daily)
        etf_cols = [c for c in result.columns if c.startswith("etf_flow_")]
        assert len(etf_cols) == 4

    def test_all_values_numeric(self, engine, spy_daily):
        result = engine.create_etf_flow_features(spy_daily)
        etf_cols = [c for c in result.columns if c.startswith("etf_flow_")]
        for col in etf_cols:
            assert result[col].dtype in (np.float64, np.float32, float), (
                f"{col} is not float: {result[col].dtype}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Edge / Guard Cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEtfFlowEdgeCases:

    def test_missing_close_returns_unchanged(self, engine):
        df = pd.DataFrame({"volume": [1e6, 2e6, 3e6]})
        result = engine.create_etf_flow_features(df)
        assert list(result.columns) == ["volume"]

    def test_missing_volume_returns_unchanged(self, engine):
        df = pd.DataFrame({"close": [450.0, 451.0, 452.0]})
        result = engine.create_etf_flow_features(df)
        assert list(result.columns) == ["close"]

    def test_empty_dataframe_returns_unchanged(self, engine):
        df = pd.DataFrame({"close": pd.Series(dtype=float), "volume": pd.Series(dtype=float)})
        result = engine.create_etf_flow_features(df)
        assert result.empty

    def test_short_dataframe_still_works(self, engine):
        """Even with fewer rows than flow_window, NaN fill should keep 0 values."""
        df = pd.DataFrame({
            "close": [450.0, 451.0, 449.5, 452.0, 450.5],
            "volume": [80e6, 200e6, 90e6, 95e6, 85e6],
        })
        result = engine.create_etf_flow_features(df)
        etf_cols = [c for c in result.columns if c.startswith("etf_flow_")]
        assert len(etf_cols) == 4
        assert result[etf_cols].isna().sum().sum() == 0

    def test_constant_price_no_crash(self, engine):
        """Flat price (zero returns) should not raise division errors."""
        df = pd.DataFrame({
            "close": [450.0] * 100,
            "volume": [80e6] * 100,
        })
        result = engine.create_etf_flow_features(df)
        etf_cols = [c for c in result.columns if c.startswith("etf_flow_")]
        assert result[etf_cols].isna().sum().sum() == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. analyze_current_flows Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnalyzeCurrentFlows:

    def test_returns_dict(self, engine, spy_daily):
        result = engine.create_etf_flow_features(spy_daily)
        analysis = engine.analyze_current_flows(result)
        assert isinstance(analysis, dict)

    def test_expected_keys(self, engine, spy_daily):
        result = engine.create_etf_flow_features(spy_daily)
        analysis = engine.analyze_current_flows(result)
        expected_keys = {"flow_proxy", "flow_z", "cr_fraction", "short_fraction", "flow_regime"}
        assert expected_keys.issubset(set(analysis.keys()))

    def test_flow_regime_valid_values(self, engine, spy_daily):
        result = engine.create_etf_flow_features(spy_daily)
        analysis = engine.analyze_current_flows(result)
        assert analysis["flow_regime"] in ("INFLOW", "NEUTRAL", "OUTFLOW")

    def test_returns_none_when_no_etf_cols(self, engine, spy_daily):
        """If called on a raw df with no etf_flow_ cols, should return None."""
        analysis = engine.analyze_current_flows(spy_daily)
        assert analysis is None

    def test_returns_none_on_empty_df(self, engine):
        df = pd.DataFrame({"etf_flow_spy_20d": pd.Series(dtype=float)})
        analysis = engine.analyze_current_flows(df)
        assert analysis is None

    def test_fractions_in_range(self, engine, spy_daily):
        result = engine.create_etf_flow_features(spy_daily)
        analysis = engine.analyze_current_flows(result)
        assert 0.0 <= analysis["cr_fraction"] <= 1.0
        assert 0.0 <= analysis["short_fraction"] <= 1.0

"""
Tests for FuturesBasisFeatures class.

Validates futures-spot basis feature engineering without requiring
live yfinance downloads.  All tests use pre-built synthetic data that
exercise the proxy path (no network calls).
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.phase_08_features_breadth.futures_basis_features import FuturesBasisFeatures


# ─── Helpers ─────────────────────────────────────────────────────────────────

N_ROWS = 200


def make_spy_daily(n: int = N_ROWS, seed: int = 42) -> pd.DataFrame:
    """Return a realistic SPY daily DataFrame with OHLCV and a date column."""
    np.random.seed(seed)
    dates = pd.bdate_range("2023-01-03", periods=n)
    close = 440.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, n)))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = close * (1 + np.random.normal(0, 0.003, n))
    volume = np.random.randint(60_000_000, 200_000_000, n)

    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates.date),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


EXPECTED_FEATURES = [
    "basis_spread",
    "basis_spread_z",
    "basis_change_5d",
    "basis_regime",
]


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def spy_daily() -> pd.DataFrame:
    return make_spy_daily()


@pytest.fixture
def engine() -> FuturesBasisFeatures:
    """Return a FuturesBasisFeatures with no downloaded futures data (proxy path)."""
    return FuturesBasisFeatures()


@pytest.fixture
def engine_with_result(engine, spy_daily) -> pd.DataFrame:
    """DataFrame that has already had basis features added via proxy."""
    return engine.create_futures_basis_features(spy_daily)


# ─── TestFuturesBasisInvariants ───────────────────────────────────────────────

class TestFuturesBasisInvariants:
    """Core structural guarantees for create_futures_basis_features."""

    def test_all_4_features_created(self, engine_with_result):
        basis_cols = [c for c in engine_with_result.columns if c.startswith("basis_")]
        assert len(basis_cols) == 4, (
            f"Expected 4 basis_ features, got {len(basis_cols)}: {basis_cols}"
        )

    def test_feature_names_exact(self, engine_with_result):
        for col in EXPECTED_FEATURES:
            assert col in engine_with_result.columns, f"Missing feature: {col}"

    def test_no_nans(self, engine_with_result):
        basis_cols = [c for c in engine_with_result.columns if c.startswith("basis_")]
        nan_total = engine_with_result[basis_cols].isna().sum().sum()
        assert nan_total == 0, f"Found {nan_total} NaN values in basis columns"

    def test_no_infinities(self, engine_with_result):
        basis_cols = [c for c in engine_with_result.columns if c.startswith("basis_")]
        inf_total = np.isinf(engine_with_result[basis_cols].values).sum()
        assert inf_total == 0, f"Found {inf_total} infinite values in basis columns"

    def test_preserves_original_columns(self, engine, spy_daily):
        original_cols = set(spy_daily.columns)
        result = engine.create_futures_basis_features(spy_daily)
        assert original_cols.issubset(
            set(result.columns)
        ), "Original columns were removed"

    def test_preserves_row_count(self, engine, spy_daily):
        result = engine.create_futures_basis_features(spy_daily)
        assert len(result) == len(spy_daily), (
            f"Row count changed: {len(spy_daily)} -> {len(result)}"
        )

    def test_no_close_column_returns_unchanged(self, engine):
        """DataFrame without 'close' column must be returned unchanged."""
        df = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "open": [440.0]})
        result = engine.create_futures_basis_features(df)
        assert list(result.columns) == list(df.columns)
        assert len(result) == len(df)


# ─── TestFuturesBasisLogic ────────────────────────────────────────────────────

class TestFuturesBasisLogic:
    """Validate the numeric behaviour of the computed features."""

    def test_z_bounded_reasonable(self, engine_with_result):
        """Z-scores should lie in a reasonable range (not wildly exploding)."""
        z = engine_with_result["basis_spread_z"]
        # After fillna(0) the first few NaN-warm-up rows become 0; ignore those
        non_zero = z[z != 0.0]
        if len(non_zero) > 0:
            assert non_zero.abs().max() < 20.0, (
                f"basis_spread_z has extreme value: {non_zero.abs().max()}"
            )

    def test_regime_values(self, engine_with_result):
        """basis_regime must only contain -1, 0, or 1."""
        valid = {-1.0, 0.0, 1.0}
        unique = set(engine_with_result["basis_regime"].unique())
        assert unique.issubset(valid), f"Unexpected regime values: {unique - valid}"

    def test_change_5d_exists(self, engine_with_result):
        """basis_change_5d should be present and numeric."""
        col = engine_with_result["basis_change_5d"]
        assert col.dtype in (np.float64, np.float32, float)

    def test_contango_when_z_above_threshold(self):
        """When z-score > 1.0, regime should be CONTANGO (+1).

        Strategy: use a long-enough series (200 rows, z_window=20) with a
        clearly varying return-acceleration signal so the rolling std > 0 and
        some entries push the z-score above +1.
        """
        np.random.seed(0)
        n = 200
        # Base: mild random walk
        base_ret = np.random.normal(0.0003, 0.005, n)
        # Inject several large positive return-acceleration spikes
        spike_idx = [50, 80, 120, 160]
        for i in spike_idx:
            base_ret[i] += 0.04   # large single-day jump → big positive acceleration
        close = 440.0 * np.cumprod(1 + base_ret)

        eng = FuturesBasisFeatures(z_window=20)
        df = pd.DataFrame({"close": close})
        result = eng.create_futures_basis_features(df)
        assert (result["basis_regime"] == 1.0).any(), (
            "Expected at least one CONTANGO (+1) reading from z-score > 1.0"
        )

    def test_backwardation_when_z_below_threshold(self):
        """When z-score < -1.0, regime should be BACKWARDATION (-1).

        Strategy: inject several large negative return-acceleration spikes so
        the z-score drops well below -1.0 at those points.
        """
        np.random.seed(2)
        n = 200
        base_ret = np.random.normal(0.0003, 0.005, n)
        # Inject large negative acceleration events
        spike_idx = [50, 90, 130, 170]
        for i in spike_idx:
            base_ret[i] -= 0.04   # large single-day crash → big negative acceleration
        close = 440.0 * np.cumprod(1 + base_ret)

        eng = FuturesBasisFeatures(z_window=20)
        df = pd.DataFrame({"close": close})
        result = eng.create_futures_basis_features(df)
        assert (result["basis_regime"] == -1.0).any(), (
            "Expected at least one BACKWARDATION (-1) reading from z-score < -1.0"
        )


# ─── TestAnalyze ──────────────────────────────────────────────────────────────

class TestAnalyze:
    """Tests for analyze_current_basis."""

    def test_returns_dict(self, engine_with_result, engine):
        result = engine.analyze_current_basis(engine_with_result)
        assert isinstance(result, dict)

    def test_regime_values(self, engine_with_result, engine):
        result = engine.analyze_current_basis(engine_with_result)
        assert result["basis_regime"] in ("CONTANGO", "NORMAL", "BACKWARDATION")

    def test_returns_none_without_features(self, engine, spy_daily):
        """DataFrame that hasn't had basis features added must return None."""
        result = engine.analyze_current_basis(spy_daily)
        assert result is None

    def test_all_keys_present(self, engine_with_result, engine):
        result = engine.analyze_current_basis(engine_with_result)
        for key in ("basis_spread", "basis_spread_z", "basis_change_5d",
                    "basis_regime", "is_extreme"):
            assert key in result, f"Key '{key}' missing from analyze result"

    def test_is_extreme_boolean(self, engine_with_result, engine):
        result = engine.analyze_current_basis(engine_with_result)
        assert isinstance(result["is_extreme"], bool)

    def test_returns_none_empty_df(self, engine):
        empty = pd.DataFrame()
        assert engine.analyze_current_basis(empty) is None


# ─── TestFeatureCounts ───────────────────────────────────────────────────────

class TestFeatureCounts:
    """Sanity-check on total feature production."""

    def test_total_count(self, engine_with_result):
        basis_cols = [c for c in engine_with_result.columns if c.startswith("basis_")]
        assert len(basis_cols) == 4

    def test_no_duplicate_columns(self, engine_with_result):
        assert len(engine_with_result.columns) == len(set(engine_with_result.columns))


# ─── TestProxyFallback ────────────────────────────────────────────────────────

class TestProxyFallback:
    """Ensure all features are created even when yfinance download fails."""

    def test_works_without_futures_data(self, spy_daily):
        """Proxy path: no downloaded futures data → features still produced."""
        eng = FuturesBasisFeatures()
        # _futures_data is empty by default (no download called)
        assert eng._futures_data.empty

        result = eng.create_futures_basis_features(spy_daily)

        basis_cols = [c for c in result.columns if c.startswith("basis_")]
        assert len(basis_cols) == 4, (
            f"Proxy path should still produce 4 features, got {len(basis_cols)}"
        )

    def test_proxy_no_nans(self, spy_daily):
        """Proxy path must not leave NaN values."""
        eng = FuturesBasisFeatures()
        result = eng.create_futures_basis_features(spy_daily)
        basis_cols = [c for c in result.columns if c.startswith("basis_")]
        assert result[basis_cols].isna().sum().sum() == 0

    def test_proxy_no_infinities(self, spy_daily):
        """Proxy path must not produce infinite values."""
        eng = FuturesBasisFeatures()
        result = eng.create_futures_basis_features(spy_daily)
        basis_cols = [c for c in result.columns if c.startswith("basis_")]
        assert np.isinf(result[basis_cols].values).sum() == 0

    def test_mocked_yfinance_failure_uses_proxy(self, spy_daily):
        """Simulate yfinance raising an exception; proxy should kick in."""
        eng = FuturesBasisFeatures()

        with patch("yfinance.download", side_effect=Exception("network failure")):
            eng.download_futures_data(
                start_date=pd.Timestamp("2023-01-03").to_pydatetime(),
                end_date=pd.Timestamp("2023-12-29").to_pydatetime(),
            )

        # _futures_data should still be empty (download failed gracefully)
        assert eng._futures_data.empty

        result = eng.create_futures_basis_features(spy_daily)
        basis_cols = [c for c in result.columns if c.startswith("basis_")]
        assert len(basis_cols) == 4

    def test_mocked_yfinance_empty_response_uses_proxy(self, spy_daily):
        """Simulate yfinance returning empty DataFrame; proxy should kick in."""
        eng = FuturesBasisFeatures()

        with patch("yfinance.download", return_value=pd.DataFrame()):
            eng.download_futures_data(
                start_date=pd.Timestamp("2023-01-03").to_pydatetime(),
                end_date=pd.Timestamp("2023-12-29").to_pydatetime(),
            )

        assert eng._futures_data.empty
        result = eng.create_futures_basis_features(spy_daily)
        basis_cols = [c for c in result.columns if c.startswith("basis_")]
        assert len(basis_cols) == 4

    def test_works_without_date_column(self):
        """Proxy works even when no date column is present."""
        np.random.seed(99)
        n = 200
        close = 440 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, n)))
        df = pd.DataFrame({"close": close})

        eng = FuturesBasisFeatures()
        result = eng.create_futures_basis_features(df)
        basis_cols = [c for c in result.columns if c.startswith("basis_")]
        assert len(basis_cols) == 4
        assert result[basis_cols].isna().sum().sum() == 0


# ─── TestFuturesDataMerge ─────────────────────────────────────────────────────

class TestFuturesDataMerge:
    """
    Test the real-basis path by injecting synthetic futures data directly
    into _futures_data (no network calls).
    """

    def _make_futures_data(self, spy_daily: pd.DataFrame, spread: float = 0.002) -> pd.DataFrame:
        """Return a synthetic futures DataFrame aligned with spy_daily."""
        return pd.DataFrame(
            {
                "date": spy_daily["date"],
                "es_close": spy_daily["close"] * (1 + spread),
            }
        ).reset_index(drop=True)

    def test_real_basis_path_produces_features(self, spy_daily):
        eng = FuturesBasisFeatures()
        eng._futures_data = self._make_futures_data(spy_daily, spread=0.002)
        result = eng.create_futures_basis_features(spy_daily)
        basis_cols = [c for c in result.columns if c.startswith("basis_")]
        assert len(basis_cols) == 4

    def test_real_basis_spread_approximately_correct(self, spy_daily):
        """With a fixed 0.2% spread, basis_spread should be near 0.002."""
        eng = FuturesBasisFeatures()
        eng._futures_data = self._make_futures_data(spy_daily, spread=0.002)
        result = eng.create_futures_basis_features(spy_daily)
        non_zero = result["basis_spread"][result["basis_spread"] != 0.0]
        if len(non_zero) > 0:
            assert abs(non_zero.mean()) < 0.05, (
                f"basis_spread mean unexpectedly large: {non_zero.mean()}"
            )

    def test_insufficient_overlap_falls_back_to_proxy(self):
        """If <30 dates match, fall back silently to the proxy."""
        np.random.seed(7)
        n = 200
        spy = make_spy_daily(n=n, seed=7)

        # Futures data with only 5 matching dates
        futures_dates = spy["date"].iloc[:5]
        futures_data = pd.DataFrame(
            {"date": futures_dates, "es_close": spy["close"].iloc[:5] * 1.002}
        ).reset_index(drop=True)

        eng = FuturesBasisFeatures()
        eng._futures_data = futures_data
        result = eng.create_futures_basis_features(spy)

        basis_cols = [c for c in result.columns if c.startswith("basis_")]
        assert len(basis_cols) == 4
        assert result[basis_cols].isna().sum().sum() == 0


# ─── TestConstructor ─────────────────────────────────────────────────────────

class TestConstructor:

    def test_default_z_window(self):
        eng = FuturesBasisFeatures()
        assert eng.z_window == 60

    def test_custom_z_window(self):
        eng = FuturesBasisFeatures(z_window=30)
        assert eng.z_window == 30

    def test_futures_data_initially_empty(self):
        eng = FuturesBasisFeatures()
        assert eng._futures_data.empty

    def test_futures_ticker_constant(self):
        assert FuturesBasisFeatures.FUTURES_TICKER == "ES=F"

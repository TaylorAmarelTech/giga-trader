"""
Tests for CongressionalFeatures class.

Validates proxy smart-money feature engineering based on volume-filtered
up/down days.  No live API calls are made — all tests use synthetic data.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from src.phase_08_features_breadth.congressional_features import CongressionalFeatures


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_spy_daily(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic SPY daily DataFrame with close and volume columns."""
    np.random.seed(seed)
    dates = pd.bdate_range("2023-01-03", periods=n)
    close = 450.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, n)))
    # Randomise volume around 80M shares/day
    volume = np.abs(np.random.normal(80_000_000, 20_000_000, n))
    return pd.DataFrame({
        "date": pd.to_datetime(dates.date),
        "close": close,
        "volume": volume,
    })


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def spy_daily():
    return make_spy_daily()


@pytest.fixture
def cf():
    return CongressionalFeatures()


@pytest.fixture
def cf_with_features(spy_daily):
    """CongressionalFeatures instance; features already computed."""
    inst = CongressionalFeatures()
    return inst, inst.create_congressional_features(spy_daily)


# ─── Class TestInvariants ────────────────────────────────────────────────────

class TestInvariants:
    """Structural / API guarantees that must always hold."""

    def test_default_constructor(self):
        cf = CongressionalFeatures()
        assert cf.window == 30
        assert cf.volume_threshold == 1.2
        assert isinstance(cf.data, pd.DataFrame)
        assert cf.data.empty

    def test_custom_constructor(self):
        cf = CongressionalFeatures(window=20, volume_threshold=1.5)
        assert cf.window == 20
        assert cf.volume_threshold == 1.5

    def test_download_returns_empty_df(self):
        cf = CongressionalFeatures()
        result = cf.download_congressional_data(
            datetime(2024, 1, 1), datetime(2024, 12, 31)
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_feature_cols_constant(self):
        expected = [
            "congress_net_buys_30d",
            "congress_volume_z",
            "congress_buy_ratio",
            "congress_sentiment",
        ]
        assert CongressionalFeatures.FEATURE_COLS == expected

    def test_returns_dataframe(self, cf, spy_daily):
        result = cf.create_congressional_features(spy_daily)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_original_columns(self, cf, spy_daily):
        original_cols = set(spy_daily.columns)
        result = cf.create_congressional_features(spy_daily)
        assert original_cols.issubset(set(result.columns))

    def test_same_row_count(self, cf, spy_daily):
        result = cf.create_congressional_features(spy_daily)
        assert len(result) == len(spy_daily)

    def test_does_not_mutate_input(self, cf, spy_daily):
        original_cols = list(spy_daily.columns)
        _ = cf.create_congressional_features(spy_daily)
        assert list(spy_daily.columns) == original_cols

    def test_missing_columns_returns_unchanged(self, cf):
        """If required columns are absent, return df as-is."""
        df = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=5)})
        result = cf.create_congressional_features(df)
        assert list(result.columns) == list(df.columns)

    def test_all_congress_cols_present(self, cf, spy_daily):
        result = cf.create_congressional_features(spy_daily)
        for col in CongressionalFeatures.FEATURE_COLS:
            assert col in result.columns, f"Missing: {col}"

    def test_no_nans_in_congress_cols(self, cf, spy_daily):
        result = cf.create_congressional_features(spy_daily)
        nan_count = result[CongressionalFeatures.FEATURE_COLS].isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values"

    def test_prefix_consistency(self, cf, spy_daily):
        original_cols = set(spy_daily.columns)
        result = cf.create_congressional_features(spy_daily)
        new_cols = set(result.columns) - original_cols
        for col in new_cols:
            assert col.startswith("congress_"), (
                f"Feature '{col}' does not have the 'congress_' prefix"
            )


# ─── Class TestLogic ─────────────────────────────────────────────────────────

class TestLogic:
    """Verify that each feature has the expected mathematical properties."""

    def test_net_buys_range(self, cf, spy_daily):
        """congress_net_buys_30d must lie in [-1, 1]."""
        result = cf.create_congressional_features(spy_daily)
        col = result["congress_net_buys_30d"]
        assert col.min() >= -1.0 - 1e-9
        assert col.max() <= 1.0 + 1e-9

    def test_buy_ratio_range(self, cf, spy_daily):
        """congress_buy_ratio must lie in [0, 1]."""
        result = cf.create_congressional_features(spy_daily)
        col = result["congress_buy_ratio"]
        assert col.min() >= -1e-9
        assert col.max() <= 1.0 + 1e-9

    def test_sentiment_range(self, cf, spy_daily):
        """congress_sentiment must be clipped to [-1, 1]."""
        result = cf.create_congressional_features(spy_daily)
        col = result["congress_sentiment"]
        assert col.min() >= -1.0 - 1e-9
        assert col.max() <= 1.0 + 1e-9

    def test_net_buys_all_positive_days(self):
        """If every high-volume day is an up-day, net_buys > 0.

        Use volume_threshold=0.9 so that constant high volume (all values
        identical) still clears the threshold bar (volume > 0.9 * ma is True
        for all rows once the MA stabilises).
        """
        np.random.seed(0)
        n = 100
        # Strictly increasing close so every day's return is positive
        close = 450.0 + np.arange(n) * 0.5
        # Add tiny jitter so the 20d MA is slightly below the actual value
        volume = 200_000_000.0 + np.arange(n, dtype=float) * 1.0
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=n),
            "close": close,
            "volume": volume,
        })
        # threshold=0.9 so volume (which is always > 0.9 * its rolling mean)
        # reliably qualifies as a high-volume day
        cf = CongressionalFeatures(window=10, volume_threshold=0.9)
        result = cf.create_congressional_features(df)
        # After warm-up, net_buys should be positive
        assert result["congress_net_buys_30d"].iloc[-1] > 0

    def test_net_buys_all_negative_days(self):
        """If every high-volume day is a down-day, net_buys < 0."""
        np.random.seed(0)
        n = 100
        close = 450.0 - np.arange(n) * 0.5          # strictly decreasing
        volume = 200_000_000.0 + np.arange(n, dtype=float) * 1.0
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=n),
            "close": close,
            "volume": volume,
        })
        cf = CongressionalFeatures(window=10, volume_threshold=0.9)
        result = cf.create_congressional_features(df)
        assert result["congress_net_buys_30d"].iloc[-1] < 0

    def test_buy_ratio_all_hv_up(self):
        """If all high-volume days are up, buy_ratio should be 1.0."""
        n = 100
        close = 450.0 + np.arange(n) * 0.5
        volume = 200_000_000.0 + np.arange(n, dtype=float) * 1.0
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=n),
            "close": close,
            "volume": volume,
        })
        cf = CongressionalFeatures(window=10, volume_threshold=0.9)
        result = cf.create_congressional_features(df)
        # After warm-up, all hv days are up → ratio = 1.0
        last_ratio = result["congress_buy_ratio"].iloc[-1]
        assert abs(last_ratio - 1.0) < 1e-6

    def test_volume_z_is_numeric(self, cf, spy_daily):
        result = cf.create_congressional_features(spy_daily)
        col = result["congress_volume_z"]
        assert col.dtype in (np.float64, np.float32, float)

    def test_short_dataframe_no_crash(self, cf):
        """5-row DataFrame should not raise."""
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=5),
            "close": [100.0, 101.0, 99.5, 102.0, 100.5],
            "volume": [1e8, 1.5e8, 2e8, 0.5e8, 1e8],
        })
        result = cf.create_congressional_features(df)
        assert len(result) == 5

    def test_custom_window_respected(self, spy_daily):
        """Verify that a custom window of 10 days produces the same columns."""
        cf = CongressionalFeatures(window=10)
        result = cf.create_congressional_features(spy_daily)
        for col in CongressionalFeatures.FEATURE_COLS:
            assert col in result.columns

    def test_custom_volume_threshold_respected(self, spy_daily):
        """Low threshold (1.0) means more days qualify as high-volume."""
        cf_low = CongressionalFeatures(window=30, volume_threshold=1.0)
        cf_high = CongressionalFeatures(window=30, volume_threshold=5.0)
        result_low = cf_low.create_congressional_features(spy_daily)
        result_high = cf_high.create_congressional_features(spy_daily)
        # With threshold=5.0 almost no days qualify → near-zero net_buys
        # With threshold=1.0 many days qualify → non-zero net_buys
        abs_low = result_low["congress_net_buys_30d"].abs().mean()
        abs_high = result_high["congress_net_buys_30d"].abs().mean()
        assert abs_low >= abs_high


# ─── Class TestAnalyze ───────────────────────────────────────────────────────

class TestAnalyze:
    """Validate the analyze_current_congressional() helper."""

    def test_returns_dict(self, cf, spy_daily):
        result = cf.analyze_current_congressional(spy_daily)
        assert isinstance(result, dict)

    def test_regime_present(self, cf, spy_daily):
        result = cf.analyze_current_congressional(spy_daily)
        assert "congressional_regime" in result
        assert result["congressional_regime"] in {"BUYING", "NEUTRAL", "SELLING"}

    def test_numeric_fields_present(self, cf, spy_daily):
        result = cf.analyze_current_congressional(spy_daily)
        for field in ("net_buys", "buy_ratio", "sentiment"):
            assert field in result
            assert isinstance(result[field], float)

    def test_date_included_when_date_col_exists(self, cf, spy_daily):
        result = cf.analyze_current_congressional(spy_daily)
        assert "date" in result

    def test_buying_regime(self):
        """sentiment > 0.2 → BUYING.

        Use volume_threshold=0.9 so that increasing volume series reliably
        clears the high-volume bar (avoids volume == MA edge case).
        """
        n = 200
        close = 450.0 + np.arange(n) * 0.5       # always rising
        volume = 200_000_000.0 + np.arange(n, dtype=float) * 1.0
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=n),
            "close": close,
            "volume": volume,
        })
        cf = CongressionalFeatures(window=30, volume_threshold=0.9)
        result = cf.analyze_current_congressional(df)
        assert result["congressional_regime"] == "BUYING"

    def test_selling_regime(self):
        """sentiment < -0.2 → SELLING."""
        n = 200
        close = 450.0 - np.arange(n) * 0.5       # always falling
        volume = 200_000_000.0 + np.arange(n, dtype=float) * 1.0
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=n),
            "close": close,
            "volume": volume,
        })
        cf = CongressionalFeatures(window=30, volume_threshold=0.9)
        result = cf.analyze_current_congressional(df)
        assert result["congressional_regime"] == "SELLING"

    def test_returns_none_for_empty_df(self, cf):
        result = cf.analyze_current_congressional(pd.DataFrame())
        assert result is None

    def test_returns_none_for_missing_cols(self, cf):
        df = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=5)})
        result = cf.analyze_current_congressional(df)
        assert result is None

    def test_works_when_features_already_computed(self, cf, spy_daily):
        """Should not recompute if congress_ cols already present."""
        precomputed = cf.create_congressional_features(spy_daily)
        result = cf.analyze_current_congressional(precomputed)
        assert isinstance(result, dict)
        assert "congressional_regime" in result


# ─── Class TestFeatureCounts ─────────────────────────────────────────────────

class TestFeatureCounts:
    """Ensure the exact number of features is produced."""

    def test_exactly_4_congress_features(self, cf, spy_daily):
        result = cf.create_congressional_features(spy_daily)
        congress_cols = [c for c in result.columns if c.startswith("congress_")]
        assert len(congress_cols) == 4, (
            f"Expected 4 congress_ features, got {len(congress_cols)}: {congress_cols}"
        )

    def test_config_has_flag(self):
        from src.experiment_config import AntiOverfitConfig
        config = AntiOverfitConfig()
        assert hasattr(config, "use_congressional_features"), (
            "AntiOverfitConfig must have use_congressional_features flag"
        )
        assert config.use_congressional_features is True

"""Tests for SeasonalityFeatures -- calendar-based seasonal signals (8 total)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.seasonality_features import SeasonalityFeatures


# ---- Helpers ----------------------------------------------------------------

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


ALL_8 = {
    "seas_turn_of_month",
    "seas_january_effect",
    "seas_pre_fomc_drift",
    "seas_holiday_drift",
    "seas_santa_claus",
    "seas_quad_witching",
    "seas_sell_in_may",
    "seas_day_of_week",
}


# ---- Invariant Tests --------------------------------------------------------

class TestSeasonalityInvariants:
    @pytest.fixture
    def feat(self):
        return SeasonalityFeatures()

    @pytest.fixture
    def spy(self):
        return _make_spy_daily(200)

    def test_all_8_features_created(self, feat, spy):
        result = feat.create_seasonality_features(spy)
        seas_cols = {c for c in result.columns if c.startswith("seas_")}
        assert seas_cols == ALL_8

    def test_no_nans(self, feat, spy):
        result = feat.create_seasonality_features(spy)
        for col in ALL_8:
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_no_infinities(self, feat, spy):
        result = feat.create_seasonality_features(spy)
        for col in ALL_8:
            assert not np.isinf(result[col]).any(), f"Inf in {col}"

    def test_preserves_original_columns(self, feat, spy):
        original = set(spy.columns)
        result = feat.create_seasonality_features(spy)
        assert original.issubset(set(result.columns))

    def test_preserves_row_count(self, feat, spy):
        result = feat.create_seasonality_features(spy)
        assert len(result) == len(spy)

    def test_no_required_column(self, feat):
        """No close column -> returns df unchanged."""
        df = pd.DataFrame({"date": [1, 2], "price": [100, 101]})
        result = feat.create_seasonality_features(df)
        assert len(result.columns) == len(df.columns)

    def test_works_with_flat_data(self, feat):
        """Flat close prices should not crash."""
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=50),
            "close": np.full(50, 450.0),
        })
        result = feat.create_seasonality_features(df)
        for col in ALL_8:
            assert col in result.columns
            assert result[col].isna().sum() == 0, f"NaN in {col} with flat data"


# ---- Feature Logic Tests ----------------------------------------------------

class TestSeasonalityLogic:
    @pytest.fixture
    def feat(self):
        return SeasonalityFeatures()

    def test_january_effect_correct_months(self, feat):
        """January effect should be 1.0 only for January dates."""
        # Full year of data
        df = _make_spy_daily(260, seed=10)
        result = feat.create_seasonality_features(df)
        months = pd.to_datetime(result["date"]).dt.month
        for i, row in result.iterrows():
            if months.iloc[i] == 1:
                assert row["seas_january_effect"] == 1.0
            else:
                assert row["seas_january_effect"] == 0.0

    def test_sell_in_may_correct_months(self, feat):
        """Sell in May should be 1.0 for May-October only."""
        df = _make_spy_daily(260, seed=10)
        result = feat.create_seasonality_features(df)
        months = pd.to_datetime(result["date"]).dt.month
        for i, row in result.iterrows():
            if 5 <= months.iloc[i] <= 10:
                assert row["seas_sell_in_may"] == 1.0
            else:
                assert row["seas_sell_in_may"] == 0.0

    def test_day_of_week_range(self, feat):
        """Day of week should be 0-4 (Mon-Fri)."""
        df = _make_spy_daily(200)
        result = feat.create_seasonality_features(df)
        assert result["seas_day_of_week"].min() >= 0
        assert result["seas_day_of_week"].max() <= 4

    def test_binary_features_are_binary(self, feat):
        """Binary features should only contain 0.0 or 1.0."""
        df = _make_spy_daily(260, seed=10)
        result = feat.create_seasonality_features(df)
        binary_cols = [
            "seas_turn_of_month", "seas_january_effect", "seas_pre_fomc_drift",
            "seas_holiday_drift", "seas_santa_claus", "seas_quad_witching",
            "seas_sell_in_may",
        ]
        for col in binary_cols:
            vals = set(result[col].unique())
            assert vals.issubset({0.0, 1.0}), f"{col} has values {vals}"


# ---- Analyze Tests -----------------------------------------------------------

class TestAnalyzeCurrentSeasonality:
    def test_returns_dict(self):
        feat = SeasonalityFeatures()
        df = _make_spy_daily(100)
        df = feat.create_seasonality_features(df)
        result = feat.analyze_current_seasonality(df)
        assert isinstance(result, dict)
        assert "active_effects" in result
        assert "day_of_week" in result

    def test_regime_values(self):
        feat = SeasonalityFeatures()
        df = _make_spy_daily(200)
        df = feat.create_seasonality_features(df)
        result = feat.analyze_current_seasonality(df)
        assert isinstance(result["active_effects"], list)
        assert result["day_of_week"] in {0, 1, 2, 3, 4}

    def test_returns_none_without_features(self):
        feat = SeasonalityFeatures()
        df = pd.DataFrame({"close": [100]})
        assert feat.analyze_current_seasonality(df) is None


# ---- Feature Count Test ------------------------------------------------------

class TestFeatureCounts:
    def test_total_count(self):
        assert len(ALL_8) == 8

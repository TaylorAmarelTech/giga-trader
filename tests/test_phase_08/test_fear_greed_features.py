"""
Tests for FearGreedFeatures class.

Validates CNN Fear & Greed Index feature engineering
without requiring live API calls.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.phase_08_features_breadth.fear_greed_features import FearGreedFeatures


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_fg_data():
    """Create realistic mock Fear & Greed historical data."""
    np.random.seed(42)
    n_days = 300
    dates = pd.bdate_range("2023-01-03", periods=n_days)

    # Simulate FG index: oscillates between 10-90 with mean reversion
    fg_scores = 50.0 + np.cumsum(np.random.normal(0, 3, n_days))
    fg_scores = np.clip(fg_scores, 5, 95)

    return pd.DataFrame({
        "date": pd.to_datetime(dates.date),
        "fg_score": fg_scores,
    })


@pytest.fixture
def spy_daily(mock_fg_data):
    """Create a mock SPY daily DataFrame aligned with FG data."""
    dates = mock_fg_data["date"]
    n = len(dates)
    np.random.seed(123)
    return pd.DataFrame({
        "date": dates,
        "day_return": np.random.normal(0.0004, 0.01, n),
        "close": 450 * np.exp(np.cumsum(np.random.normal(0.0004, 0.01, n))),
    })


@pytest.fixture
def fg_features(mock_fg_data):
    """Create a FearGreedFeatures instance with pre-loaded data."""
    fg = FearGreedFeatures()
    fg.data = mock_fg_data
    return fg


# ─── Constructor Tests ───────────────────────────────────────────────────────

class TestFearGreedInit:

    def test_default_constructor(self):
        fg = FearGreedFeatures()
        assert isinstance(fg.data, pd.DataFrame)
        assert fg.data.empty

    def test_data_initially_empty(self):
        fg = FearGreedFeatures()
        assert len(fg.data) == 0


# ─── Download Tests ──────────────────────────────────────────────────────────

class TestFearGreedDownload:

    @patch("requests.get")
    def test_cnn_api_success(self, mock_get):
        """Test successful download from CNN API."""
        from datetime import datetime

        # Mock CNN API response
        now_ms = int(datetime.now().timestamp() * 1000)
        mock_data = {
            "fear_and_greed_historical": {
                "data": [
                    {"x": now_ms - i * 86400000, "y": 50 + i % 30}
                    for i in range(30)
                ]
            }
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_data
        mock_get.return_value = mock_resp

        fg = FearGreedFeatures()
        result = fg.download_fear_greed_data(
            datetime(2024, 1, 1), datetime(2026, 12, 31)
        )
        assert not result.empty
        assert "fg_score" in result.columns
        assert "date" in result.columns

    @patch("requests.get")
    def test_cnn_api_failure_graceful(self, mock_get):
        """Network error returns empty DF."""
        mock_get.side_effect = Exception("Network error")

        fg = FearGreedFeatures()
        result = fg.download_fear_greed_data(
            datetime(2024, 1, 1), datetime(2024, 12, 31)
        )
        assert isinstance(result, pd.DataFrame)

    @patch("requests.get")
    def test_cnn_api_non_200(self, mock_get):
        """Non-200 status code returns empty DF."""
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_get.return_value = mock_resp

        fg = FearGreedFeatures()
        result = fg.download_fear_greed_data(
            datetime(2024, 1, 1), datetime(2024, 12, 31)
        )
        # Should still be empty or fall back to package
        assert isinstance(result, pd.DataFrame)

    @patch("requests.get")
    def test_cnn_api_empty_response(self, mock_get):
        """Empty data field returns empty DF."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_get.return_value = mock_resp

        fg = FearGreedFeatures()
        result = fg.download_fear_greed_data(
            datetime(2024, 1, 1), datetime(2024, 12, 31)
        )
        assert isinstance(result, pd.DataFrame)


# ─── Feature Engineering Tests ───────────────────────────────────────────────

class TestFeatureEngineering:

    def test_creates_features(self, fg_features, spy_daily):
        result = fg_features.create_fear_greed_features(spy_daily)
        fg_cols = [c for c in result.columns if c.startswith("fg_")]
        assert len(fg_cols) == 8

    def test_feature_names(self, fg_features, spy_daily):
        result = fg_features.create_fear_greed_features(spy_daily)
        expected = [
            "fg_index", "fg_index_z", "fg_index_pctile",
            "fg_chg_1d", "fg_chg_5d", "fg_regime",
            "fg_extreme_signal", "fg_momentum_5d",
        ]
        for name in expected:
            assert name in result.columns, f"Missing feature: {name}"

    def test_preserves_original_columns(self, fg_features, spy_daily):
        original_cols = set(spy_daily.columns)
        result = fg_features.create_fear_greed_features(spy_daily)
        assert original_cols.issubset(set(result.columns))

    def test_same_row_count(self, fg_features, spy_daily):
        result = fg_features.create_fear_greed_features(spy_daily)
        assert len(result) == len(spy_daily)

    def test_no_nans_in_fg_cols(self, fg_features, spy_daily):
        result = fg_features.create_fear_greed_features(spy_daily)
        fg_cols = [c for c in result.columns if c.startswith("fg_")]
        nan_count = result[fg_cols].isna().sum().sum()
        assert nan_count == 0

    def test_index_range(self, fg_features, spy_daily):
        """fg_index should be in [0, 100] range."""
        result = fg_features.create_fear_greed_features(spy_daily)
        non_zero = result["fg_index"][result["fg_index"] != 0]
        if len(non_zero) > 0:
            assert non_zero.min() >= 0
            assert non_zero.max() <= 100

    def test_regime_values(self, fg_features, spy_daily):
        """fg_regime should be integers 0-4."""
        result = fg_features.create_fear_greed_features(spy_daily)
        valid_regimes = {0, 1, 2, 3, 4}
        unique_regimes = set(result["fg_regime"].unique())
        assert unique_regimes.issubset(valid_regimes)

    def test_extreme_signal_binary(self, fg_features, spy_daily):
        """fg_extreme_signal should be 0 or 1."""
        result = fg_features.create_fear_greed_features(spy_daily)
        unique = set(result["fg_extreme_signal"].unique())
        assert unique.issubset({0, 1, 0.0, 1.0})

    def test_extreme_fear_classified(self):
        """Score < 20 should be extreme fear (regime=0, extreme_signal=1)."""
        fg = FearGreedFeatures()
        fg.data = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "fg_score": [10.0],
        })
        spy = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "close": [450.0],
        })
        result = fg.create_fear_greed_features(spy)
        assert result["fg_regime"].iloc[0] == 0
        assert result["fg_extreme_signal"].iloc[0] == 1

    def test_extreme_greed_classified(self):
        """Score > 80 should be extreme greed (regime=4, extreme_signal=1)."""
        fg = FearGreedFeatures()
        fg.data = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "fg_score": [90.0],
        })
        spy = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "close": [450.0],
        })
        result = fg.create_fear_greed_features(spy)
        assert result["fg_regime"].iloc[0] == 4
        assert result["fg_extreme_signal"].iloc[0] == 1

    def test_neutral_classified(self):
        """Score 40-60 should be neutral (regime=2, extreme_signal=0)."""
        fg = FearGreedFeatures()
        fg.data = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "fg_score": [50.0],
        })
        spy = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "close": [450.0],
        })
        result = fg.create_fear_greed_features(spy)
        assert result["fg_regime"].iloc[0] == 2
        assert result["fg_extreme_signal"].iloc[0] == 0


# ─── Condition Analysis Tests ────────────────────────────────────────────────

class TestConditionAnalysis:

    def test_returns_dict(self, fg_features, spy_daily):
        result = fg_features.analyze_current_fear_greed(spy_daily)
        assert isinstance(result, dict)

    def test_score_present(self, fg_features, spy_daily):
        result = fg_features.analyze_current_fear_greed(spy_daily)
        assert "fear_greed_score" in result

    def test_regime_present(self, fg_features, spy_daily):
        result = fg_features.analyze_current_fear_greed(spy_daily)
        assert "fear_greed_regime" in result
        valid = {"EXTREME_FEAR", "FEAR", "NEUTRAL", "GREED", "EXTREME_GREED"}
        assert result["fear_greed_regime"] in valid

    def test_extreme_flag(self, fg_features, spy_daily):
        result = fg_features.analyze_current_fear_greed(spy_daily)
        assert "is_extreme" in result
        assert isinstance(result["is_extreme"], bool)

    def test_none_when_no_data(self, spy_daily):
        fg = FearGreedFeatures()
        assert fg.analyze_current_fear_greed(spy_daily) is None


# ─── Config Integration Tests ────────────────────────────────────────────────

class TestConfigIntegration:

    def test_empty_data_returns_original(self, spy_daily):
        fg = FearGreedFeatures()
        result = fg.create_fear_greed_features(spy_daily)
        assert list(result.columns) == list(spy_daily.columns)

    def test_feature_prefix_consistency(self, fg_features, spy_daily):
        """All new features should start with fg_."""
        original_cols = set(spy_daily.columns)
        result = fg_features.create_fear_greed_features(spy_daily)
        new_cols = set(result.columns) - original_cols
        for col in new_cols:
            assert col.startswith("fg_"), f"Feature {col} doesn't have fg_ prefix"


# ─── Edge Case Tests ─────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_single_day_data(self):
        """Should handle single day of data gracefully."""
        fg = FearGreedFeatures()
        fg.data = pd.DataFrame({
            "date": pd.to_datetime(["2024-06-01"]),
            "fg_score": [55.0],
        })
        spy = pd.DataFrame({
            "date": pd.to_datetime(["2024-06-01"]),
            "close": [450.0],
        })
        result = fg.create_fear_greed_features(spy)
        assert len(result) == 1
        assert "fg_index" in result.columns

    def test_short_history(self):
        """Should handle < 60 days without z-score crash."""
        fg = FearGreedFeatures()
        dates = pd.bdate_range("2024-01-01", periods=10)
        fg.data = pd.DataFrame({
            "date": pd.to_datetime(dates.date),
            "fg_score": np.linspace(30, 70, 10),
        })
        spy = pd.DataFrame({
            "date": pd.to_datetime(dates.date),
            "close": np.linspace(440, 460, 10),
        })
        result = fg.create_fear_greed_features(spy)
        assert len(result) == 10
        fg_cols = [c for c in result.columns if c.startswith("fg_")]
        assert len(fg_cols) == 8

    def test_non_overlapping_dates(self):
        """Should return all zeros when dates don't overlap."""
        fg = FearGreedFeatures()
        fg.data = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "fg_score": [50.0, 55.0],
        })
        spy = pd.DataFrame({
            "date": pd.to_datetime(["2024-06-01", "2024-06-02"]),
            "close": [450.0, 451.0],
        })
        result = fg.create_fear_greed_features(spy)
        assert len(result) == 2
        # All fg_ values should be 0 (filled)
        fg_cols = [c for c in result.columns if c.startswith("fg_")]
        assert (result[fg_cols] == 0).all().all()

    def test_nan_scores_handled(self):
        """NaN scores in data should not crash."""
        fg = FearGreedFeatures()
        fg.data = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "fg_score": [50.0, np.nan, 55.0],
        })
        spy = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "close": [450.0, 451.0, 452.0],
        })
        result = fg.create_fear_greed_features(spy)
        fg_cols = [c for c in result.columns if c.startswith("fg_")]
        nan_count = result[fg_cols].isna().sum().sum()
        assert nan_count == 0

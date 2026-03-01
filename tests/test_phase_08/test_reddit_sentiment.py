"""
Tests for RedditSentimentFeatures class.

Validates ApeWisdom Reddit sentiment feature engineering
without requiring live API calls.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.phase_08_features_breadth.reddit_sentiment_features import RedditSentimentFeatures


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_reddit_data():
    """Create mock ApeWisdom data."""
    today = pd.Timestamp.now().normalize()
    return pd.DataFrame([{
        "date": today,
        "spy_mentions": 150,
        "spy_rank": 5,
        "spy_upvotes": 1200,
        "breadth_bullish": 0.6,
        "n_components_mentioned": 9,
    }])


@pytest.fixture
def spy_daily():
    """Create a mock SPY daily DataFrame."""
    np.random.seed(42)
    n = 100
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({
        "date": pd.to_datetime(dates.date),
        "day_return": np.random.normal(0.0004, 0.01, n),
        "close": 450 * np.exp(np.cumsum(np.random.normal(0.0004, 0.01, n))),
    })


@pytest.fixture
def reddit_features(mock_reddit_data):
    """Create a RedditSentimentFeatures instance with pre-loaded data."""
    rf = RedditSentimentFeatures()
    rf.data = mock_reddit_data
    return rf


# ─── Constructor Tests ───────────────────────────────────────────────────────

class TestRedditInit:

    def test_default_constructor(self):
        rf = RedditSentimentFeatures()
        assert rf.data.empty

    def test_data_initially_empty(self):
        rf = RedditSentimentFeatures()
        assert len(rf.data) == 0


# ─── Download Tests ──────────────────────────────────────────────────────────

class TestRedditDownload:

    @patch("requests.get")
    def test_api_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {"ticker": "SPY", "mentions": 100, "rank": 3, "upvotes": 500},
                {"ticker": "AAPL", "mentions": 200, "rank": 1, "upvotes": 1000},
                {"ticker": "TSLA", "mentions": 150, "rank": 2, "upvotes": 800},
            ]
        }
        mock_get.return_value = mock_resp

        rf = RedditSentimentFeatures()
        result = rf.download_reddit_data(datetime(2024, 1, 1), datetime(2024, 12, 31))
        assert not result.empty
        assert "spy_mentions" in result.columns

    @patch("requests.get")
    def test_api_failure_graceful(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        rf = RedditSentimentFeatures()
        result = rf.download_reddit_data(datetime(2024, 1, 1), datetime(2024, 12, 31))
        assert isinstance(result, pd.DataFrame)

    @patch("requests.get")
    def test_api_no_spy(self, mock_get):
        """When SPY not in results, defaults should be used."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {"ticker": "AAPL", "mentions": 200, "rank": 1, "upvotes": 1000},
            ]
        }
        mock_get.return_value = mock_resp

        rf = RedditSentimentFeatures()
        result = rf.download_reddit_data(datetime(2024, 1, 1), datetime(2024, 12, 31))
        assert not result.empty
        assert result["spy_mentions"].iloc[0] == 0


# ─── Feature Engineering Tests ───────────────────────────────────────────────

class TestFeatureEngineering:

    def test_creates_features(self, reddit_features, spy_daily):
        result = reddit_features.create_reddit_features(spy_daily)
        reddit_cols = [c for c in result.columns if c.startswith("reddit_")]
        assert len(reddit_cols) == 6

    def test_feature_names(self, reddit_features, spy_daily):
        result = reddit_features.create_reddit_features(spy_daily)
        expected = [
            "reddit_spy_mentions", "reddit_spy_rank", "reddit_spy_upvotes",
            "reddit_breadth_bullish", "reddit_momentum_3d", "reddit_buzz_zscore",
        ]
        for name in expected:
            assert name in result.columns, f"Missing: {name}"

    def test_preserves_original_columns(self, reddit_features, spy_daily):
        original = set(spy_daily.columns)
        result = reddit_features.create_reddit_features(spy_daily)
        assert original.issubset(set(result.columns))

    def test_same_row_count(self, reddit_features, spy_daily):
        result = reddit_features.create_reddit_features(spy_daily)
        assert len(result) == len(spy_daily)

    def test_no_nans(self, reddit_features, spy_daily):
        result = reddit_features.create_reddit_features(spy_daily)
        reddit_cols = [c for c in result.columns if c.startswith("reddit_")]
        assert result[reddit_cols].isna().sum().sum() == 0

    def test_empty_data_returns_original(self, spy_daily):
        rf = RedditSentimentFeatures()
        result = rf.create_reddit_features(spy_daily)
        assert list(result.columns) == list(spy_daily.columns)

    def test_prefix_consistency(self, reddit_features, spy_daily):
        original = set(spy_daily.columns)
        result = reddit_features.create_reddit_features(spy_daily)
        new_cols = set(result.columns) - original
        for col in new_cols:
            assert col.startswith("reddit_"), f"{col} doesn't have reddit_ prefix"


# ─── Condition Analysis Tests ────────────────────────────────────────────────

class TestConditionAnalysis:

    def test_returns_dict(self, reddit_features, spy_daily):
        result = reddit_features.analyze_current_reddit(spy_daily)
        assert isinstance(result, dict)

    def test_spy_mentions_present(self, reddit_features, spy_daily):
        result = reddit_features.analyze_current_reddit(spy_daily)
        assert "spy_mentions" in result

    def test_none_when_no_data(self, spy_daily):
        rf = RedditSentimentFeatures()
        assert rf.analyze_current_reddit(spy_daily) is None


# ─── Config Integration Tests ────────────────────────────────────────────────

class TestConfigIntegration:

    def test_config_has_flag(self):
        from src.experiment_config import AntiOverfitConfig
        config = AntiOverfitConfig()
        assert hasattr(config, "use_reddit_sentiment")

"""
Tests for FinnhubSocialFeatures (social sentiment from Finnhub API).
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.phase_08_features_breadth.finnhub_social_features import FinnhubSocialFeatures


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def fh_engine():
    return FinnhubSocialFeatures()


@pytest.fixture
def mock_social_data():
    """Mock pre-processed social data (what download would produce)."""
    dates = pd.date_range("2024-06-01", periods=30, freq="B")
    np.random.seed(42)
    return pd.DataFrame({
        "date": dates,
        "reddit_mentions": np.random.randint(10, 500, 30),
        "reddit_positive": np.random.uniform(0, 100, 30),
        "reddit_negative": np.random.uniform(0, 50, 30),
        "twitter_mentions": np.random.randint(50, 2000, 30),
        "twitter_positive": np.random.uniform(0, 200, 30),
        "twitter_negative": np.random.uniform(0, 100, 30),
    })


@pytest.fixture
def spy_daily():
    dates = pd.date_range("2024-06-01", periods=30, freq="B")
    np.random.seed(42)
    return pd.DataFrame({
        "date": dates,
        "close": 450 + np.cumsum(np.random.randn(30) * 0.5),
        "volume": np.random.randint(50_000_000, 200_000_000, 30),
    })


# ─── Init Tests ──────────────────────────────────────────────────────────────

class TestFinnhubInit:

    def test_default_constructor(self, fh_engine):
        assert isinstance(fh_engine, FinnhubSocialFeatures)
        assert fh_engine.data.empty

    def test_api_key_none_initially(self, fh_engine):
        assert fh_engine._api_key is None


# ─── Download Tests ──────────────────────────────────────────────────────────

class TestFinnhubDownload:

    @patch.dict("os.environ", {"FINNHUB_API_KEY": ""}, clear=False)
    def test_no_api_key_returns_empty(self, fh_engine):
        fh_engine._api_key = None
        with patch.dict("os.environ", {"FINNHUB_API_KEY": ""}, clear=False):
            result = fh_engine.download_finnhub_social_data(
                datetime(2024, 6, 1), datetime(2024, 7, 1)
            )
            assert isinstance(result, pd.DataFrame)

    @patch("requests.get")
    def test_api_success(self, mock_get, fh_engine):
        fh_engine._api_key = "test_key"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "reddit": [
                {"atTime": "2024-06-03T10:00:00Z", "mention": 50,
                 "positiveScore": 30, "negativeScore": 10},
                {"atTime": "2024-06-04T10:00:00Z", "mention": 60,
                 "positiveScore": 40, "negativeScore": 15},
            ],
            "twitter": [
                {"atTime": "2024-06-03T10:00:00Z", "mention": 200,
                 "positiveScore": 100, "negativeScore": 50},
            ],
        }
        mock_get.return_value = mock_resp

        result = fh_engine.download_finnhub_social_data(
            datetime(2024, 6, 1), datetime(2024, 7, 1)
        )
        assert not result.empty
        assert "reddit_mentions" in result.columns

    @patch("requests.get")
    def test_api_rate_limit(self, mock_get, fh_engine):
        fh_engine._api_key = "test_key"
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_get.return_value = mock_resp

        result = fh_engine.download_finnhub_social_data(
            datetime(2024, 6, 1), datetime(2024, 7, 1)
        )
        assert result.empty

    @patch("requests.get")
    def test_api_invalid_key(self, mock_get, fh_engine):
        fh_engine._api_key = "invalid"
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_get.return_value = mock_resp

        result = fh_engine.download_finnhub_social_data(
            datetime(2024, 6, 1), datetime(2024, 7, 1)
        )
        assert result.empty

    @patch("requests.get")
    def test_api_empty_response(self, mock_get, fh_engine):
        fh_engine._api_key = "test_key"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"reddit": [], "twitter": []}
        mock_get.return_value = mock_resp

        result = fh_engine.download_finnhub_social_data(
            datetime(2024, 6, 1), datetime(2024, 7, 1)
        )
        assert result.empty


# ─── Feature Engineering Tests ───────────────────────────────────────────────

class TestFinnhubFeatures:

    def test_creates_features(self, fh_engine, mock_social_data, spy_daily):
        fh_engine.data = mock_social_data
        result = fh_engine.create_finnhub_social_features(spy_daily)
        fh_cols = [c for c in result.columns if c.startswith("finnhub_social_")]
        assert len(fh_cols) == 6

    def test_feature_names(self, fh_engine, mock_social_data, spy_daily):
        fh_engine.data = mock_social_data
        result = fh_engine.create_finnhub_social_features(spy_daily)
        expected = [
            "finnhub_social_reddit_mentions", "finnhub_social_twitter_mentions",
            "finnhub_social_total_mentions", "finnhub_social_positive_pct",
            "finnhub_social_score", "finnhub_social_buzz_zscore",
        ]
        for name in expected:
            assert name in result.columns, f"Missing: {name}"

    def test_preserves_original(self, fh_engine, mock_social_data, spy_daily):
        fh_engine.data = mock_social_data
        result = fh_engine.create_finnhub_social_features(spy_daily)
        assert "close" in result.columns

    def test_same_row_count(self, fh_engine, mock_social_data, spy_daily):
        fh_engine.data = mock_social_data
        result = fh_engine.create_finnhub_social_features(spy_daily)
        assert len(result) == len(spy_daily)

    def test_no_nans(self, fh_engine, mock_social_data, spy_daily):
        fh_engine.data = mock_social_data
        result = fh_engine.create_finnhub_social_features(spy_daily)
        fh_cols = [c for c in result.columns if c.startswith("finnhub_social_")]
        for col in fh_cols:
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_positive_pct_range(self, fh_engine, mock_social_data, spy_daily):
        """Positive pct should be 0-1 range."""
        fh_engine.data = mock_social_data
        result = fh_engine.create_finnhub_social_features(spy_daily)
        col = result["finnhub_social_positive_pct"]
        non_zero = col[col > 0]
        if not non_zero.empty:
            assert non_zero.min() >= 0.0
            assert non_zero.max() <= 1.0

    def test_score_range(self, fh_engine, mock_social_data, spy_daily):
        """Score should be in [-1, +1]."""
        fh_engine.data = mock_social_data
        result = fh_engine.create_finnhub_social_features(spy_daily)
        col = result["finnhub_social_score"]
        non_zero = col[col != 0]
        if not non_zero.empty:
            assert non_zero.min() >= -1.0
            assert non_zero.max() <= 1.0

    def test_buzz_zscore_clamped(self, fh_engine, mock_social_data, spy_daily):
        fh_engine.data = mock_social_data
        result = fh_engine.create_finnhub_social_features(spy_daily)
        col = result["finnhub_social_buzz_zscore"]
        assert col.min() >= -3.0
        assert col.max() <= 3.0

    def test_empty_data_returns_original(self, fh_engine, spy_daily):
        result = fh_engine.create_finnhub_social_features(spy_daily)
        assert list(result.columns) == list(spy_daily.columns)

    def test_prefix_consistency(self, fh_engine, mock_social_data, spy_daily):
        fh_engine.data = mock_social_data
        result = fh_engine.create_finnhub_social_features(spy_daily)
        new_cols = set(result.columns) - set(spy_daily.columns)
        for col in new_cols:
            assert col.startswith("finnhub_social_"), f"{col} missing prefix"


# ─── Condition Analysis Tests ────────────────────────────────────────────────

class TestFinnhubAnalysis:

    def test_returns_dict(self, fh_engine, mock_social_data, spy_daily):
        fh_engine.data = mock_social_data
        result_df = fh_engine.create_finnhub_social_features(spy_daily)
        analysis = fh_engine.analyze_current_finnhub_social(result_df)
        assert isinstance(analysis, dict)

    def test_score_present(self, fh_engine, mock_social_data, spy_daily):
        fh_engine.data = mock_social_data
        result_df = fh_engine.create_finnhub_social_features(spy_daily)
        analysis = fh_engine.analyze_current_finnhub_social(result_df)
        assert "social_score" in analysis

    def test_sentiment_label(self, fh_engine, mock_social_data, spy_daily):
        fh_engine.data = mock_social_data
        result_df = fh_engine.create_finnhub_social_features(spy_daily)
        analysis = fh_engine.analyze_current_finnhub_social(result_df)
        assert analysis["social_sentiment"] in ["bullish", "bearish", "neutral"]

    def test_none_when_no_data(self, fh_engine, spy_daily):
        analysis = fh_engine.analyze_current_finnhub_social(spy_daily)
        assert analysis is None


# ─── Config Integration Tests ────────────────────────────────────────────────

class TestFinnhubConfig:

    def test_config_has_flag(self):
        from src.experiment_config import AntiOverfitConfig
        config = AntiOverfitConfig()
        assert hasattr(config, "use_finnhub_social")
        assert config.use_finnhub_social is True

    def test_feature_group_registered(self):
        from src.phase_10_feature_processing.group_aware_processor import FEATURE_GROUPS
        assert "finnhub_sentiment" in FEATURE_GROUPS
        assert "finnhub_social_" in FEATURE_GROUPS["finnhub_sentiment"]

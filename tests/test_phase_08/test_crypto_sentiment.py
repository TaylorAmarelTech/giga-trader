"""
Tests for CryptoSentimentFeatures class.

Validates Alternative.me Crypto Fear & Greed feature engineering
without requiring live API calls.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.phase_08_features_breadth.crypto_sentiment_features import CryptoSentimentFeatures


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_crypto_data():
    """Create realistic mock Crypto Fear & Greed data."""
    np.random.seed(42)
    n_days = 200
    dates = pd.bdate_range("2023-06-01", periods=n_days)
    scores = 50 + np.cumsum(np.random.normal(0, 3, n_days))
    scores = np.clip(scores, 5, 95)
    return pd.DataFrame({
        "date": pd.to_datetime(dates.date),
        "crypto_score": scores,
    })


@pytest.fixture
def spy_daily(mock_crypto_data):
    dates = mock_crypto_data["date"]
    n = len(dates)
    np.random.seed(123)
    return pd.DataFrame({
        "date": dates,
        "day_return": np.random.normal(0.0004, 0.01, n),
        "close": 450 * np.exp(np.cumsum(np.random.normal(0.0004, 0.01, n))),
    })


@pytest.fixture
def crypto_features(mock_crypto_data):
    cf = CryptoSentimentFeatures()
    cf.data = mock_crypto_data
    return cf


# ─── Constructor Tests ───────────────────────────────────────────────────────

class TestCryptoInit:

    def test_default_constructor(self):
        cf = CryptoSentimentFeatures()
        assert cf.data.empty


# ─── Download Tests ──────────────────────────────────────────────────────────

class TestCryptoDownload:

    @patch("requests.get")
    def test_api_success(self, mock_get):
        import time
        now = int(time.time())
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"timestamp": str(now - i * 86400), "value": str(50 + i % 20)}
                for i in range(30)
            ]
        }
        mock_get.return_value = mock_resp

        cf = CryptoSentimentFeatures()
        result = cf.download_crypto_data(datetime(2024, 1, 1), datetime(2026, 12, 31))
        assert not result.empty
        assert "crypto_score" in result.columns

    @patch("requests.get")
    def test_api_failure(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        cf = CryptoSentimentFeatures()
        result = cf.download_crypto_data(datetime(2024, 1, 1), datetime(2024, 12, 31))
        assert isinstance(result, pd.DataFrame)

    @patch("requests.get")
    def test_api_empty_data(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        cf = CryptoSentimentFeatures()
        result = cf.download_crypto_data(datetime(2024, 1, 1), datetime(2024, 12, 31))
        assert isinstance(result, pd.DataFrame)


# ─── Feature Engineering Tests ───────────────────────────────────────────────

class TestFeatureEngineering:

    def test_creates_features(self, crypto_features, spy_daily):
        result = crypto_features.create_crypto_features(spy_daily)
        crypto_cols = [c for c in result.columns if c.startswith("crypto_")]
        assert len(crypto_cols) == 5

    def test_feature_names(self, crypto_features, spy_daily):
        result = crypto_features.create_crypto_features(spy_daily)
        expected = [
            "crypto_fg_index", "crypto_fg_zscore", "crypto_fg_chg_5d",
            "crypto_fg_regime", "crypto_risk_proxy",
        ]
        for name in expected:
            assert name in result.columns, f"Missing: {name}"

    def test_preserves_original_columns(self, crypto_features, spy_daily):
        original = set(spy_daily.columns)
        result = crypto_features.create_crypto_features(spy_daily)
        assert original.issubset(set(result.columns))

    def test_same_row_count(self, crypto_features, spy_daily):
        result = crypto_features.create_crypto_features(spy_daily)
        assert len(result) == len(spy_daily)

    def test_no_nans(self, crypto_features, spy_daily):
        result = crypto_features.create_crypto_features(spy_daily)
        crypto_cols = [c for c in result.columns if c.startswith("crypto_")]
        assert result[crypto_cols].isna().sum().sum() == 0

    def test_regime_values(self, crypto_features, spy_daily):
        result = crypto_features.create_crypto_features(spy_daily)
        valid = {0, 1, 2, 3, 4}
        assert set(result["crypto_fg_regime"].unique()).issubset(valid)

    def test_risk_proxy_binary(self, crypto_features, spy_daily):
        result = crypto_features.create_crypto_features(spy_daily)
        assert set(result["crypto_risk_proxy"].unique()).issubset({0, 1, 0.0, 1.0})

    def test_empty_data_returns_original(self, spy_daily):
        cf = CryptoSentimentFeatures()
        result = cf.create_crypto_features(spy_daily)
        assert list(result.columns) == list(spy_daily.columns)

    def test_prefix_consistency(self, crypto_features, spy_daily):
        original = set(spy_daily.columns)
        result = crypto_features.create_crypto_features(spy_daily)
        new_cols = set(result.columns) - original
        for col in new_cols:
            assert col.startswith("crypto_"), f"{col} doesn't have crypto_ prefix"


# ─── Condition Analysis Tests ────────────────────────────────────────────────

class TestConditionAnalysis:

    def test_returns_dict(self, crypto_features, spy_daily):
        result = crypto_features.analyze_current_crypto(spy_daily)
        assert isinstance(result, dict)

    def test_score_present(self, crypto_features, spy_daily):
        result = crypto_features.analyze_current_crypto(spy_daily)
        assert "crypto_fg_score" in result

    def test_regime_present(self, crypto_features, spy_daily):
        result = crypto_features.analyze_current_crypto(spy_daily)
        assert "crypto_regime" in result
        valid = {"EXTREME_FEAR", "FEAR", "NEUTRAL", "GREED", "EXTREME_GREED"}
        assert result["crypto_regime"] in valid

    def test_none_when_no_data(self, spy_daily):
        cf = CryptoSentimentFeatures()
        assert cf.analyze_current_crypto(spy_daily) is None


# ─── Config Integration Tests ────────────────────────────────────────────────

class TestConfigIntegration:

    def test_config_has_flag(self):
        from src.experiment_config import AntiOverfitConfig
        config = AntiOverfitConfig()
        assert hasattr(config, "use_crypto_sentiment")

"""Tests for OnlineUpdater."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB

from src.phase_21_continuous.online_updater import OnlineUpdater


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n: int = 50, n_features: int = 5, seed: int = 42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    y = (X[:, 0] > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOnlineUpdater:

    def test_partial_fit_model_updated(self):
        """SGDClassifier should be updated via partial_fit."""
        X, y = _make_data(50)
        model = SGDClassifier(loss="log_loss", random_state=0)
        model.partial_fit(X[:30], y[:30], classes=[0, 1])

        updater = OnlineUpdater()
        result = updater.update(model, X[30:35], y[30:35])

        assert result["method"] == "partial_fit"
        assert result["model_updated"] is True
        assert result["n_samples"] == 5
        assert result["needs_retrain"] is False

    def test_non_partial_fit_buffers(self):
        """LogisticRegression should buffer data (no partial_fit)."""
        X, y = _make_data(50)
        model = LogisticRegression(max_iter=200, random_state=0).fit(X[:30], y[:30])

        updater = OnlineUpdater(retrain_threshold=20)
        result = updater.update(model, X[30:35], y[30:35])

        assert result["method"] == "buffered"
        assert result["model_updated"] is False
        assert result["buffer_size"] == 5

    def test_buffer_fills_correctly(self):
        """Buffer should accumulate samples across multiple updates."""
        X, y = _make_data(80)
        model = LogisticRegression(max_iter=200, random_state=0).fit(X[:30], y[:30])

        updater = OnlineUpdater(retrain_threshold=20)
        updater.update(model, X[30:40], y[30:40])  # 10 samples
        updater.update(model, X[40:47], y[40:47])  # 7 more

        buf_X, buf_y = updater.get_buffer()
        assert buf_X.shape[0] == 17
        assert buf_y.shape[0] == 17

    def test_needs_retrain_triggered(self):
        """needs_retrain should be True when buffer reaches threshold."""
        X, y = _make_data(100)
        model = LogisticRegression(max_iter=200, random_state=0).fit(X[:30], y[:30])

        updater = OnlineUpdater(retrain_threshold=10)
        # First batch: 8 samples
        r1 = updater.update(model, X[30:38], y[30:38])
        assert r1["needs_retrain"] is False

        # Second batch: 5 more -> total 13 >= 10 but deque maxlen=10 so capped
        r2 = updater.update(model, X[38:43], y[38:43])
        assert r2["needs_retrain"] is True

    def test_clear_buffer(self):
        """clear_buffer should empty the buffer."""
        X, y = _make_data(50)
        model = LogisticRegression(max_iter=200, random_state=0).fit(X[:30], y[:30])

        updater = OnlineUpdater(retrain_threshold=20)
        updater.update(model, X[30:40], y[30:40])
        updater.clear_buffer()

        buf_X, buf_y = updater.get_buffer()
        assert buf_X.shape[0] == 0
        assert buf_y.shape[0] == 0

    def test_get_buffer_returns_arrays(self):
        """get_buffer should return numpy arrays."""
        X, y = _make_data(50)
        model = LogisticRegression(max_iter=200, random_state=0).fit(X[:30], y[:30])

        updater = OnlineUpdater()
        updater.update(model, X[30:35], y[30:35])

        buf_X, buf_y = updater.get_buffer()
        assert isinstance(buf_X, np.ndarray)
        assert isinstance(buf_y, np.ndarray)
        assert buf_X.shape == (5, 5)
        assert buf_y.shape == (5,)

    def test_supports_partial_fit(self):
        """Correctly detect partial_fit support."""
        updater = OnlineUpdater()

        assert updater.supports_partial_fit(SGDClassifier()) is True
        assert updater.supports_partial_fit(GaussianNB()) is True
        assert updater.supports_partial_fit(LogisticRegression()) is False

    def test_n_updates_increments(self):
        """n_updates should increment with each update call."""
        X, y = _make_data(50)
        model = SGDClassifier(loss="log_loss", random_state=0)
        model.partial_fit(X[:30], y[:30], classes=[0, 1])

        updater = OnlineUpdater()
        assert updater.n_updates == 0

        updater.update(model, X[30:35], y[30:35])
        assert updater.n_updates == 1

        updater.update(model, X[35:40], y[35:40])
        assert updater.n_updates == 2

    def test_update_history_tracks(self):
        """update_history should contain one entry per update."""
        X, y = _make_data(60)
        model = SGDClassifier(loss="log_loss", random_state=0)
        model.partial_fit(X[:30], y[:30], classes=[0, 1])

        updater = OnlineUpdater()
        updater.update(model, X[30:35], y[30:35])
        updater.update(model, X[35:40], y[35:40])

        history = updater.update_history
        assert len(history) == 2
        assert history[0]["method"] == "partial_fit"
        assert history[1]["method"] == "partial_fit"
        # Should contain timestamps
        assert "timestamp" in history[0]

    def test_single_sample_update(self):
        """Should handle a single-sample update (1D input)."""
        X, y = _make_data(50)
        model = SGDClassifier(loss="log_loss", random_state=0)
        model.partial_fit(X[:30], y[:30], classes=[0, 1])

        updater = OnlineUpdater()
        result = updater.update(model, X[30:31], y[30:31])

        assert result["n_samples"] == 1
        assert result["model_updated"] is True

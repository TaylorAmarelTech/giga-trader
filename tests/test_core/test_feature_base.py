"""Tests for FeatureModuleBase."""

import numpy as np
import pandas as pd
import pytest

from src.core.feature_base import FeatureModuleBase


class DummyFeatures(FeatureModuleBase):
    """Concrete subclass for testing."""
    REQUIRED_COLS = {"close", "volume"}
    FEATURE_NAMES = ["feat_a", "feat_b", "feat_c"]


class TestFeatureModuleBase:
    def _make_df(self, n=50):
        return pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=n, freq="B"),
            "close": np.random.uniform(400, 500, n),
            "volume": np.random.randint(1_000_000, 10_000_000, n),
        })

    def test_validate_input_ok(self):
        feat = DummyFeatures()
        df = self._make_df()
        assert feat._validate_input(df) is True

    def test_validate_input_missing_col(self):
        feat = DummyFeatures()
        df = self._make_df().drop(columns=["volume"])
        assert feat._validate_input(df) is False

    def test_validate_input_too_few_rows(self):
        feat = DummyFeatures()
        df = self._make_df(1)
        assert feat._validate_input(df) is False

    def test_cleanup_features(self):
        feat = DummyFeatures()
        df = self._make_df()
        df["feat_a"] = np.nan
        df["feat_b"] = np.inf
        df["feat_c"] = -np.inf
        df = feat._cleanup_features(df)
        assert df["feat_a"].isna().sum() == 0
        assert np.isinf(df["feat_b"]).sum() == 0
        assert np.isinf(df["feat_c"]).sum() == 0

    def test_zero_fill_all(self):
        feat = DummyFeatures()
        df = self._make_df()
        df = feat._zero_fill_all(df)
        for col in DummyFeatures.FEATURE_NAMES:
            assert (df[col] == 0.0).all()

    def test_all_feature_names_classmethod(self):
        names = DummyFeatures._all_feature_names()
        assert names == ["feat_a", "feat_b", "feat_c"]

    def test_all_feature_names_instance(self):
        feat = DummyFeatures()
        assert feat._all_feature_names() == ["feat_a", "feat_b", "feat_c"]

    def test_cleanup_missing_cols_no_error(self):
        """Cleanup should not fail if FEATURE_NAMES columns don't exist yet."""
        feat = DummyFeatures()
        df = self._make_df()
        # No feat_a/b/c columns exist
        df = feat._cleanup_features(df)
        assert "feat_a" not in df.columns  # should not create them

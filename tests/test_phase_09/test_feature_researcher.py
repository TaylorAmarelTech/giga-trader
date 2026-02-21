"""Tests for Wave 32 Feature Research Agent."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from src.phase_09_features_calendar.feature_researcher import (
    FeatureCandidate,
    FeatureResearchAgent,
    TEMPLATE_REGISTRY,
    _compute_ratio,
    _compute_interaction,
    _compute_lag_diff,
    _compute_zscore,
    _compute_kernel_rbf,
    _compute_rank_ratio,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_df():
    """Create a sample DataFrame with common feature columns."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, freq="B"),
        "pm_return": np.random.normal(0, 0.01, n),
        "pm_range": np.abs(np.random.normal(0.005, 0.003, n)),
        "pm_direction": np.random.choice([-1, 0, 1], n),
        "pm_vwap_dev": np.random.normal(0, 0.002, n),
        "gap_open_pct": np.random.normal(0, 0.005, n),
        "atr_pct_at_0930": np.abs(np.random.normal(0.01, 0.003, n)),
        "rsi_14_at_0930": np.random.uniform(20, 80, n),
        "macd_at_0930": np.random.normal(0, 0.5, n),
        "bb_position_at_0930": np.random.uniform(0, 1, n),
        "volume_ratio_at_0930": np.random.uniform(0.5, 2.0, n),
        "return_at_1000": np.random.normal(0, 0.005, n),
        "return_at_1030": np.random.normal(0, 0.005, n),
        "return_at_1100": np.random.normal(0, 0.005, n),
        "day_return_lag1": np.random.normal(0, 0.01, n),
        "day_range_lag1": np.abs(np.random.normal(0.01, 0.005, n)),
        "up_streak": np.random.randint(0, 5, n),
        "down_streak": np.random.randint(0, 5, n),
        "vol_ratio_5d": np.random.uniform(0.5, 2.0, n),
        "mom_5_at_0930": np.random.normal(0, 0.02, n),
        "mom_15_at_0930": np.random.normal(0, 0.03, n),
        "target_up": np.random.randint(0, 2, n),
    })
    return df


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def agent(temp_data_dir):
    """Create a FeatureResearchAgent with temp directory."""
    return FeatureResearchAgent(data_dir=temp_data_dir)


# =============================================================================
# TEMPLATE FUNCTION TESTS
# =============================================================================

class TestTemplates:
    """Test individual template compute functions."""

    def test_ratio(self, sample_df):
        result = _compute_ratio(sample_df, "pm_return", "atr_pct_at_0930")
        assert len(result) == len(sample_df)
        assert not result.isna().all()

    def test_interaction(self, sample_df):
        result = _compute_interaction(sample_df, "pm_return", "gap_open_pct")
        assert len(result) == len(sample_df)
        # Product of two centered values should be small
        assert abs(result.mean()) < 0.01

    def test_lag_diff(self, sample_df):
        result = _compute_lag_diff(sample_df, "rsi_14_at_0930", lag=5)
        assert len(result) == len(sample_df)
        # First 5 values should be NaN
        assert result.iloc[:5].isna().all()

    def test_zscore(self, sample_df):
        result = _compute_zscore(sample_df, "pm_return", window=20)
        assert len(result) == len(sample_df)
        # Z-scores should be roughly centered
        valid = result.dropna()
        if len(valid) > 20:
            assert abs(valid.mean()) < 1.0

    def test_kernel_rbf(self, sample_df):
        result = _compute_kernel_rbf(sample_df, "pm_return", "gap_open_pct", gamma=1.0)
        assert len(result) == len(sample_df)
        # RBF output is in (0, 1]
        assert result.min() >= 0
        assert result.max() <= 1.0

    def test_rank_ratio(self, sample_df):
        result = _compute_rank_ratio(sample_df, "pm_return", "gap_open_pct")
        assert len(result) == len(sample_df)
        # Rank ratios should be positive
        assert (result > 0).all()

    def test_all_templates_registered(self):
        expected = {"ratio", "interaction", "lag_diff", "zscore", "kernel_rbf", "rank_ratio"}
        assert set(TEMPLATE_REGISTRY.keys()) == expected


# =============================================================================
# FEATURE CANDIDATE TESTS
# =============================================================================

class TestFeatureCandidate:
    """Test FeatureCandidate dataclass."""

    def test_creation(self):
        c = FeatureCandidate(
            name="rc_ratio_pm_ret_atr",
            template_type="ratio",
            source_features=["pm_return", "atr_pct_at_0930"],
        )
        assert c.name == "rc_ratio_pm_ret_atr"
        assert c.n_experiments == 0
        assert not c.graduated

    def test_round_trip(self):
        c = FeatureCandidate(
            name="test_feature",
            template_type="interaction",
            source_features=["a", "b"],
            params={"scale": 2.0},
            n_experiments=5,
            n_tier1_pass=3,
        )
        d = c.to_dict()
        c2 = FeatureCandidate.from_dict(d)
        assert c2.name == c.name
        assert c2.n_experiments == 5
        assert c2.params == {"scale": 2.0}

    def test_from_dict_ignores_unknown(self):
        d = {"name": "x", "template_type": "ratio", "source_features": ["a", "b"],
             "unknown_field": 123}
        c = FeatureCandidate.from_dict(d)
        assert c.name == "x"


# =============================================================================
# FEATURE RESEARCH AGENT TESTS
# =============================================================================

class TestFeatureResearchAgent:
    """Test FeatureResearchAgent main class."""

    def test_init_empty(self, agent):
        assert len(agent._candidates) == 0
        assert len(agent._graduated) == 0

    def test_generate_candidates(self, agent):
        candidates = agent.generate_candidates(n_candidates=3)
        assert len(candidates) <= 3
        assert len(candidates) > 0
        for c in candidates:
            assert isinstance(c, FeatureCandidate)
            assert c.template_type in TEMPLATE_REGISTRY
            assert len(c.source_features) > 0

    def test_generate_persists(self, agent, temp_data_dir):
        agent.generate_candidates(n_candidates=2)
        assert agent.candidates_path.is_file()
        # Reload
        agent2 = FeatureResearchAgent(data_dir=temp_data_dir)
        assert len(agent2._candidates) == len(agent._candidates)

    def test_inject_candidates(self, agent, sample_df):
        candidates = agent.generate_candidates(n_candidates=3)
        config_mock = MagicMock()
        config_mock.metadata = {"candidates": [c.to_dict() for c in candidates]}
        config_mock.experiment_type = "feature_research"

        df = sample_df.copy()
        n_cols_before = len(df.columns)
        added = agent.inject_candidates(df, config_mock)

        # Should have added some columns (some may fail if source features missing)
        assert isinstance(added, list)
        # At least verify no crash and df is modified in-place
        assert len(df.columns) >= n_cols_before

    def test_inject_graduated_features(self, agent, sample_df):
        # Manually add a graduated feature
        grad = FeatureCandidate(
            name="rc_ratio_pm_return_atr_grad",
            template_type="ratio",
            source_features=["pm_return", "atr_pct_at_0930"],
            graduated=True,
        )
        agent._graduated["rc_ratio_pm_return_atr_grad"] = grad

        config_mock = MagicMock()
        config_mock.metadata = {"candidates": []}
        df = sample_df.copy()
        added = agent.inject_candidates(df, config_mock)
        assert "rc_ratio_pm_return_atr_grad" in added
        assert "rc_ratio_pm_return_atr_grad" in df.columns

    def test_update_candidate_stats(self, agent):
        candidates = agent.generate_candidates(n_candidates=2)
        names = [c.name for c in candidates]

        agent.update_candidate_stats(names, tier1_passed=True, wmes_score=0.55, walk_forward_passed=True)
        for name in names:
            assert agent._candidates[name].n_experiments == 1
            assert agent._candidates[name].n_tier1_pass == 1
            assert agent._candidates[name].avg_wmes == 0.55

        # Second update
        agent.update_candidate_stats(names, tier1_passed=False, wmes_score=0.40, walk_forward_passed=False)
        for name in names:
            assert agent._candidates[name].n_experiments == 2
            assert agent._candidates[name].avg_wmes == pytest.approx(0.475, abs=0.01)

    def test_graduation(self, agent):
        # Create a candidate manually
        c = FeatureCandidate(
            name="test_grad",
            template_type="ratio",
            source_features=["pm_return", "atr_pct_at_0930"],
            n_experiments=5,
            n_tier1_pass=4,  # 80% pass rate
            avg_wmes=0.55,
        )
        agent._candidates["test_grad"] = c

        graduated = agent.check_graduations(baseline_tier1_rate=0.30)
        assert "test_grad" in graduated
        assert agent._graduated["test_grad"].graduated

    def test_no_premature_graduation(self, agent):
        # Too few experiments
        c = FeatureCandidate(
            name="too_few",
            template_type="ratio",
            source_features=["pm_return", "atr_pct_at_0930"],
            n_experiments=2,  # Less than MIN_EXPERIMENTS_TO_GRADUATE
            n_tier1_pass=2,
        )
        agent._candidates["too_few"] = c

        graduated = agent.check_graduations(baseline_tier1_rate=0.30)
        assert len(graduated) == 0

    def test_max_active_candidates(self, agent):
        # Fill up candidates
        for i in range(agent.MAX_ACTIVE_CANDIDATES + 5):
            agent._candidates[f"fake_{i}"] = FeatureCandidate(
                name=f"fake_{i}", template_type="ratio",
                source_features=["pm_return", "atr_pct_at_0930"],
            )
        # Should not generate more, but return existing untested ones
        result = agent.generate_candidates(n_candidates=3)
        assert len(result) <= 3
        # Total should not exceed MAX + 5 (no new ones added)
        total_non_graduated = sum(1 for c in agent._candidates.values() if not c.graduated)
        assert total_non_graduated <= agent.MAX_ACTIVE_CANDIDATES + 5

    def test_get_candidates_for_config(self, agent):
        specs = agent.get_candidates_for_config(n=3)
        assert isinstance(specs, list)
        for spec in specs:
            assert isinstance(spec, dict)
            assert "name" in spec
            assert "template_type" in spec

    def test_summary(self, agent):
        agent.generate_candidates(n_candidates=2)
        s = agent.summary()
        assert "Active candidates" in s
        assert "Graduated" in s

    def test_compute_candidate_column_missing_source(self, agent, sample_df):
        """Candidate with missing source feature should return None."""
        c = FeatureCandidate(
            name="test_missing",
            template_type="ratio",
            source_features=["nonexistent_col", "pm_return"],
        )
        result = agent._compute_candidate_column(sample_df, c)
        assert result is None

    def test_compute_candidate_column_success(self, agent, sample_df):
        """Valid candidate should produce a Series."""
        c = FeatureCandidate(
            name="test_valid",
            template_type="ratio",
            source_features=["pm_return", "atr_pct_at_0930"],
        )
        result = agent._compute_candidate_column(sample_df, c)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_df)
        # Should not have inf or NaN (replaced by _compute_candidate_column)
        assert not np.isinf(result).any()


# =============================================================================
# CONFIG METADATA TESTS
# =============================================================================

class TestConfigMetadata:
    """Test that ExperimentConfig metadata round-trips correctly."""

    def test_metadata_in_config(self):
        from src.experiment_config import ExperimentConfig
        config = ExperimentConfig(metadata={"candidates": [{"name": "test"}]})
        d = config.to_dict()
        assert "metadata" in d
        assert d["metadata"]["candidates"][0]["name"] == "test"

    def test_metadata_from_dict(self):
        from src.experiment_config import ExperimentConfig, create_default_config
        original = create_default_config("test")
        original.metadata = {"candidates": [{"name": "feat1", "template_type": "ratio"}]}
        d = original.to_dict()
        restored = ExperimentConfig.from_dict(d)
        assert restored.metadata["candidates"][0]["name"] == "feat1"

    def test_metadata_default_empty(self):
        from src.experiment_config import ExperimentConfig
        config = ExperimentConfig()
        assert config.metadata == {}

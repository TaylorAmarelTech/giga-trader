"""
Tests for GroupAwareFeatureProcessor - feature protection and
hierarchical dimensionality reduction.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase_10_feature_processing.group_aware_processor import (
    GroupAwareFeatureProcessor,
    FEATURE_GROUPS,
    assign_feature_groups,
)
from src.phase_18_persistence.registry_enums import FeatureGroupMode
from src.phase_18_persistence.registry_configs import (
    FeatureGroupConfig,
    ModelEntry,
)


# =============================================================================
# HELPERS
# =============================================================================

def _make_feature_names(n_per_group=5):
    """Create realistic feature names spanning multiple groups."""
    names = []
    # Premarket
    for i in range(n_per_group):
        names.append(f"pm_feat_{i}")
    for i in range(n_per_group):
        names.append(f"ah_return_lag{i+1}")
    names.append("overnight_return")
    # Breadth streak
    for n in [2, 3, 4, 5, 6]:
        names.append(f"pct_green_{n}d")
        names.append(f"pct_red_{n}d")
    names.append("breadth_divergence")
    # MAG breadth
    for prefix in ["mag3_", "mag5_", "mag7_"]:
        names.append(f"{prefix}pct_advancing")
        names.append(f"{prefix}avg_return")
        names.append(f"{prefix}momentum_5d")
    # Sector
    for feat in ["sector_pct_advancing", "sector_avg_return", "xlk_momentum_5d"]:
        names.append(feat)
    # Cross-asset
    for asset in ["TLT_", "QQQ_", "GLD_"]:
        names.append(f"{asset}return")
        names.append(f"{asset}rsi")
    # Volatility
    names.extend(["vxx_level", "vxx_return", "vol_contango"])
    # Calendar
    names.extend(["cal_day_of_week", "fomc_is_meeting_day", "opex_is_monthly"])
    # Intraday
    names.extend(["return_at_0945", "return_at_1015", "high_to_0945", "low_to_0945"])
    # Daily
    names.extend(["day_return", "day_range", "return_ma3", "up_streak", "vol_ratio_5_20"])
    # Technical
    names.extend(["rsi_14", "macd_hist", "bb_pos", "stoch_k", "atr_14"])
    # Unmatched
    names.extend(["custom_feature_1", "custom_feature_2"])
    return names


def _make_data(n_samples=200, feature_names=None):
    """Create random data matching feature names."""
    if feature_names is None:
        feature_names = _make_feature_names()
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, len(feature_names))
    y = (X[:, 0] > 0).astype(float)
    return X, y, feature_names


# =============================================================================
# TESTS: FEATURE_GROUPS constant
# =============================================================================

class TestFeatureGroups:
    """Tests for FEATURE_GROUPS constant."""

    def test_has_minimum_groups(self):
        assert len(FEATURE_GROUPS) >= 24  # 11 orig + 13 new (A-I waves)

    def test_all_groups_have_prefixes(self):
        for name, prefixes in FEATURE_GROUPS.items():
            assert len(prefixes) > 0, f"Group {name} has no prefixes"
            for p in prefixes:
                assert isinstance(p, str), f"Prefix {p} in {name} is not a string"

    def test_known_groups_present(self):
        expected = [
            "premarket", "breadth_streak", "mag_breadth", "sector",
            "cross_asset", "volatility", "calendar", "intraday",
            "daily", "technical",
        ]
        for g in expected:
            assert g in FEATURE_GROUPS, f"Missing group: {g}"


# =============================================================================
# TESTS: assign_feature_groups
# =============================================================================

class TestAssignFeatureGroups:
    """Tests for the assign_feature_groups function."""

    def test_basic_assignment(self):
        names = ["pm_return", "vxx_level", "cal_day_of_week", "custom"]
        groups = assign_feature_groups(names)
        assert 0 in groups["premarket"]
        assert 1 in groups["volatility"]
        assert 2 in groups["calendar"]
        assert 3 in groups["other"]

    def test_all_features_assigned(self):
        names = _make_feature_names()
        groups = assign_feature_groups(names)
        all_indices = set()
        for indices in groups.values():
            all_indices.update(indices)
        assert all_indices == set(range(len(names)))

    def test_no_double_assignment(self):
        names = _make_feature_names()
        groups = assign_feature_groups(names)
        seen = set()
        for indices in groups.values():
            for idx in indices:
                assert idx not in seen, f"Index {idx} assigned to multiple groups"
                seen.add(idx)

    def test_unmatched_go_to_other(self):
        names = ["unknown_feature_1", "unknown_feature_2"]
        groups = assign_feature_groups(names)
        assert "other" in groups
        assert len(groups["other"]) == 2

    def test_empty_groups_excluded(self):
        names = ["pm_return", "pm_range"]
        groups = assign_feature_groups(names)
        assert "premarket" in groups
        assert "sector" not in groups


# =============================================================================
# TESTS: FeatureGroupConfig
# =============================================================================

class TestFeatureGroupConfig:
    """Tests for the FeatureGroupConfig dataclass."""

    def test_default_values(self):
        cfg = FeatureGroupConfig()
        assert cfg.enabled is False
        assert cfg.group_mode == "flat"
        assert cfg.protected_groups == []
        assert cfg.budget_mode == "proportional"
        assert cfg.total_components == 40

    def test_model_entry_has_feature_group_config(self):
        entry = ModelEntry()
        assert hasattr(entry, "feature_group_config")
        assert isinstance(entry.feature_group_config, FeatureGroupConfig)
        assert entry.feature_group_config.enabled is False


# =============================================================================
# TESTS: GroupAwareFeatureProcessor - FLAT MODE
# =============================================================================

class TestFlatMode:
    """Tests for flat mode (current behavior fallback)."""

    def test_flat_mode_produces_output(self):
        X, y, names = _make_data()
        proc = GroupAwareFeatureProcessor(
            feature_names=names,
            group_mode="flat",
            n_features=15,
            n_components=8,
        )
        X_out = proc.fit_transform(X, y)
        assert X_out.shape[0] == X.shape[0]
        assert X_out.shape[1] <= X.shape[1]
        assert X_out.shape[1] > 0

    def test_flat_mode_transform_matches_fit_transform(self):
        X, y, names = _make_data()
        proc = GroupAwareFeatureProcessor(
            feature_names=names,
            group_mode="flat",
            n_features=15,
            n_components=8,
        )
        proc.fit(X, y)
        X_t = proc.transform(X)
        X_ft = proc.fit_transform(X, y)
        # Should be same shape
        assert X_t.shape == X_ft.shape


# =============================================================================
# TESTS: GroupAwareFeatureProcessor - PROTECTED MODE
# =============================================================================

class TestProtectedMode:
    """Tests for protected mode."""

    def test_protected_mode_output_shape(self):
        X, y, names = _make_data()
        proc = GroupAwareFeatureProcessor(
            feature_names=names,
            group_mode="protected",
            protected_groups=["premarket"],
            total_components=10,
        )
        X_out = proc.fit_transform(X, y)
        assert X_out.shape[0] == X.shape[0]
        # Protected premarket features + reduced others
        assert X_out.shape[1] > 0

    def test_protected_features_more_than_reduced(self):
        """Protected + reduced should have more features than just reduced."""
        X, y, names = _make_data()
        proc_protected = GroupAwareFeatureProcessor(
            feature_names=names,
            group_mode="protected",
            protected_groups=["premarket", "calendar"],
            total_components=10,
        )
        proc_flat = GroupAwareFeatureProcessor(
            feature_names=names,
            group_mode="flat",
            n_features=10,
            n_components=5,
        )
        X_prot = proc_protected.fit_transform(X, y)
        X_flat = proc_flat.fit_transform(X, y)
        # Protected mode should produce more features since protected pass through
        assert X_prot.shape[1] > X_flat.shape[1]


# =============================================================================
# TESTS: GroupAwareFeatureProcessor - GROUPED MODE
# =============================================================================

class TestGroupedMode:
    """Tests for grouped (per-group reduction) mode."""

    def test_grouped_mode_output(self):
        X, y, names = _make_data()
        proc = GroupAwareFeatureProcessor(
            feature_names=names,
            group_mode="grouped",
            total_components=20,
            min_components_per_group=2,
        )
        X_out = proc.fit_transform(X, y)
        assert X_out.shape[0] == X.shape[0]
        assert X_out.shape[1] > 0

    def test_grouped_mode_transform_consistent(self):
        X, y, names = _make_data()
        proc = GroupAwareFeatureProcessor(
            feature_names=names,
            group_mode="grouped",
            total_components=20,
        )
        proc.fit(X, y)
        X_t = proc.transform(X)
        assert X_t.shape[0] == X.shape[0]
        assert X_t.shape[1] > 0


# =============================================================================
# TESTS: GroupAwareFeatureProcessor - GROUPED_PROTECTED MODE
# =============================================================================

class TestGroupedProtectedMode:
    """Tests for grouped + protected mode."""

    def test_grouped_protected_output(self):
        X, y, names = _make_data()
        proc = GroupAwareFeatureProcessor(
            feature_names=names,
            group_mode="grouped_protected",
            protected_groups=["premarket", "calendar"],
            total_components=15,
            min_components_per_group=2,
        )
        X_out = proc.fit_transform(X, y)
        assert X_out.shape[0] == X.shape[0]
        # Should have protected premarket + calendar features + reduced others
        assert X_out.shape[1] > 15  # More than just reduced components

    def test_grouped_protected_more_than_grouped(self):
        """GP mode should have more features than pure grouped (due to protection)."""
        X, y, names = _make_data()
        proc_gp = GroupAwareFeatureProcessor(
            feature_names=names,
            group_mode="grouped_protected",
            protected_groups=["premarket", "calendar"],
            total_components=15,
        )
        proc_g = GroupAwareFeatureProcessor(
            feature_names=names,
            group_mode="grouped",
            total_components=15,
        )
        X_gp = proc_gp.fit_transform(X, y)
        X_g = proc_g.fit_transform(X, y)
        assert X_gp.shape[1] >= X_g.shape[1]


# =============================================================================
# TESTS: Budget Allocation
# =============================================================================

class TestBudgetAllocation:
    """Tests for component budget computation."""

    def test_proportional_budget(self):
        proc = GroupAwareFeatureProcessor(
            feature_names=["a"] * 10,
            total_components=20,
            min_components_per_group=2,
        )
        groups = {"big": list(range(80)), "small": list(range(20))}
        budgets = proc._compute_budgets(groups)
        assert budgets["big"] > budgets["small"]

    def test_equal_budget(self):
        proc = GroupAwareFeatureProcessor(
            feature_names=["a"] * 10,
            budget_mode="equal",
            total_components=20,
            min_components_per_group=2,
        )
        groups = {"a": list(range(80)), "b": list(range(20))}
        budgets = proc._compute_budgets(groups)
        assert budgets["a"] == budgets["b"]

    def test_min_components_enforced(self):
        proc = GroupAwareFeatureProcessor(
            feature_names=["a"],
            total_components=5,
            min_components_per_group=3,
        )
        groups = {"tiny": [0, 1], "large": list(range(2, 100))}
        budgets = proc._compute_budgets(groups)
        assert budgets["tiny"] >= 3


# =============================================================================
# TESTS: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_feature_group(self):
        X = np.random.randn(50, 3)
        y = (X[:, 0] > 0).astype(float)
        proc = GroupAwareFeatureProcessor(
            feature_names=["pm_a", "pm_b", "pm_c"],
            group_mode="protected",
            protected_groups=["premarket"],
        )
        X_out = proc.fit_transform(X, y)
        # All features are in one protected group
        assert X_out.shape[1] <= 3

    def test_all_unmatched_features(self):
        X = np.random.randn(50, 5)
        y = (X[:, 0] > 0).astype(float)
        proc = GroupAwareFeatureProcessor(
            feature_names=["x1", "x2", "x3", "x4", "x5"],
            group_mode="grouped",
            total_components=3,
        )
        X_out = proc.fit_transform(X, y)
        assert X_out.shape[0] == 50
        assert X_out.shape[1] > 0

    def test_transform_without_fit_raises(self):
        proc = GroupAwareFeatureProcessor(
            feature_names=["pm_a"],
            group_mode="flat",
        )
        with pytest.raises((ValueError, AttributeError)):
            proc.transform(np.array([[1.0]]))


# =============================================================================
# TESTS: FeatureGroupMode enum
# =============================================================================

class TestFeatureGroupModeEnum:
    """Tests for the FeatureGroupMode enum."""

    def test_all_modes_exist(self):
        assert FeatureGroupMode.FLAT.value == "flat"
        assert FeatureGroupMode.PROTECTED.value == "protected"
        assert FeatureGroupMode.GROUPED.value == "grouped"
        assert FeatureGroupMode.GROUPED_PROTECTED.value == "grouped_protected"

    def test_enum_has_4_values(self):
        assert len(FeatureGroupMode) == 4


# =============================================================================
# TESTS: Grid Search Integration
# =============================================================================

class TestGridSearchIntegration:
    """Tests for grid search integration."""

    def test_minimal_grid_includes_group_mode(self):
        from src.phase_18_persistence.grid_search_generator import (
            GridSearchConfigGenerator,
        )
        gen = GridSearchConfigGenerator.create_minimal_grid()
        assert "feature_group_config.group_mode" in gen.dimensions

    def test_minimal_grid_count(self):
        from src.phase_18_persistence.grid_search_generator import (
            GridSearchConfigGenerator,
        )
        gen = GridSearchConfigGenerator.create_minimal_grid()
        total = gen.count_combinations()
        # 2 res x 5 dim x 2 group_mode x 2 model x 4 cascade = 160
        assert total == 160

    def test_standard_grid_includes_group_mode(self):
        from src.phase_18_persistence.grid_search_generator import (
            GridSearchConfigGenerator,
        )
        gen = GridSearchConfigGenerator.create_standard_grid()
        assert "feature_group_config.group_mode" in gen.dimensions


# =============================================================================
# TESTS: ThickWeaveSearch Integration
# =============================================================================

class TestThickWeaveSearchIntegration:
    """Tests for ThickWeaveSearch integration."""

    def test_categorical_dims_includes_group_mode(self):
        from src.phase_23_analytics.thick_weave_search import CATEGORICAL_DIMS
        assert "feature_group_config.group_mode" in CATEGORICAL_DIMS
        modes = CATEGORICAL_DIMS["feature_group_config.group_mode"]
        assert "flat" in modes
        assert "grouped_protected" in modes
        assert len(modes) == 4

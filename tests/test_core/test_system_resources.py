"""
Tests for src.core.system_resources — system resource detection and adaptive scaling.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.system_resources import (
    ResourceTier,
    SystemResources,
    ResourceConfig,
    create_resource_config,
    get_system_resources,
    check_memory_pressure,
    maybe_gc,
    _TIER_DEFAULTS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sys_resources(total_gb: float, cpu_physical: int = 8) -> SystemResources:
    """Create a SystemResources instance with mocked hardware values."""
    SystemResources._reset()
    mock_vm = MagicMock()
    mock_vm.total = int(total_gb * 1024 ** 3)
    mock_vm.available = int(total_gb * 0.5 * 1024 ** 3)

    with patch("psutil.virtual_memory", return_value=mock_vm), \
         patch("psutil.cpu_count", return_value=cpu_physical), \
         patch("os.cpu_count", return_value=cpu_physical * 2):
        sr = SystemResources()
    return sr


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton between tests."""
    SystemResources._reset()
    yield
    SystemResources._reset()


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------

class TestTierClassification:
    def test_low_tier(self):
        sr = _make_sys_resources(8.0)
        assert sr.tier == ResourceTier.LOW

    def test_medium_tier(self):
        sr = _make_sys_resources(24.0)
        assert sr.tier == ResourceTier.MEDIUM

    def test_high_tier(self):
        sr = _make_sys_resources(64.0)
        assert sr.tier == ResourceTier.HIGH

    def test_ultra_tier(self):
        sr = _make_sys_resources(512.0)
        assert sr.tier == ResourceTier.ULTRA

    def test_boundary_16gb_is_medium(self):
        sr = _make_sys_resources(16.0)
        assert sr.tier == ResourceTier.MEDIUM

    def test_boundary_32gb_is_high(self):
        sr = _make_sys_resources(32.0)
        assert sr.tier == ResourceTier.HIGH

    def test_boundary_128gb_is_ultra(self):
        sr = _make_sys_resources(128.0)
        assert sr.tier == ResourceTier.ULTRA


# ---------------------------------------------------------------------------
# Singleton behaviour
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_same_instance(self):
        sr1 = SystemResources()
        sr2 = SystemResources()
        assert sr1 is sr2

    def test_reset_creates_new(self):
        sr1 = SystemResources()
        SystemResources._reset()
        sr2 = SystemResources()
        assert sr1 is not sr2

    def test_get_system_resources_returns_singleton(self):
        sr1 = get_system_resources()
        sr2 = get_system_resources()
        assert sr1 is sr2


# ---------------------------------------------------------------------------
# n_jobs computation
# ---------------------------------------------------------------------------

class TestNJobs:
    def test_low_tier_single_job(self):
        sr = _make_sys_resources(8.0, cpu_physical=10)
        assert sr.recommended_n_jobs() == 1

    def test_medium_tier_half_cores(self):
        sr = _make_sys_resources(24.0, cpu_physical=10)
        assert sr.recommended_n_jobs() == 5

    def test_high_tier_cores_minus_one(self):
        sr = _make_sys_resources(64.0, cpu_physical=10)
        assert sr.recommended_n_jobs() == 9

    def test_ultra_tier_all_cores(self):
        sr = _make_sys_resources(512.0, cpu_physical=10)
        assert sr.recommended_n_jobs() == 10

    def test_n_jobs_minimum_one(self):
        sr = _make_sys_resources(24.0, cpu_physical=1)
        assert sr.recommended_n_jobs() >= 1


# ---------------------------------------------------------------------------
# ResourceConfig defaults
# ---------------------------------------------------------------------------

class TestResourceConfigDefaults:
    def test_low_tier_config(self):
        sr = _make_sys_resources(8.0)
        rc = create_resource_config()
        assert rc.tier == "low"
        assert rc.n_synthetic_universes == 3
        assert rc.stability_skip_expensive is True
        assert rc.gc_between_feature_steps is True
        assert rc.feature_cache_max_size == 1

    def test_medium_tier_config(self):
        sr = _make_sys_resources(24.0)
        rc = create_resource_config()
        assert rc.tier == "medium"
        assert rc.n_synthetic_universes == 10
        assert rc.stability_skip_expensive is False
        assert rc.gc_between_feature_steps is True

    def test_high_tier_config(self):
        sr = _make_sys_resources(64.0)
        rc = create_resource_config()
        assert rc.tier == "high"
        assert rc.n_synthetic_universes == 20
        assert rc.gc_between_feature_steps is False
        assert rc.kernel_pca_nystroem_threshold == 10000

    def test_ultra_tier_config(self):
        sr = _make_sys_resources(512.0)
        rc = create_resource_config()
        assert rc.tier == "ultra"
        assert rc.n_synthetic_universes == 30
        assert rc.feature_cache_max_size == 10
        assert rc.optuna_n_trials == 100

    def test_all_tiers_have_defaults(self):
        for tier in ResourceTier:
            assert tier in _TIER_DEFAULTS


# ---------------------------------------------------------------------------
# User overrides
# ---------------------------------------------------------------------------

class TestOverrides:
    def test_override_n_jobs(self):
        _make_sys_resources(24.0)
        rc = create_resource_config(overrides={"n_jobs": 2})
        assert rc.n_jobs == 2

    def test_override_synthetic_universes(self):
        _make_sys_resources(8.0)
        rc = create_resource_config(overrides={"n_synthetic_universes": 50})
        assert rc.n_synthetic_universes == 50
        assert rc.tier == "low"  # tier unchanged

    def test_unknown_override_ignored(self):
        _make_sys_resources(24.0)
        # Should not raise
        rc = create_resource_config(overrides={"nonexistent_field": 999})
        assert not hasattr(rc, "nonexistent_field") or rc.tier == "medium"


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

class TestGPUDetection:
    def test_no_gpu_graceful(self):
        sr = _make_sys_resources(24.0)
        # By default in test (no real GPU), has_cuda depends on system
        # The key invariant: it must not crash
        assert isinstance(sr.has_cuda, bool)
        assert isinstance(sr.cuda_device_name, str)

    def test_xgb_tree_method_with_gpu(self):
        sr = _make_sys_resources(64.0)
        if sr.has_cuda:
            assert sr.recommended_xgb_tree_method() == "gpu_hist"
        else:
            assert sr.recommended_xgb_tree_method() == "hist"


# ---------------------------------------------------------------------------
# summary() and to_dict()
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_summary_is_string(self):
        sr = _make_sys_resources(24.0)
        s = sr.summary()
        assert isinstance(s, str)
        assert "SYSTEM RESOURCES" in s
        assert "Tier:" in s
        assert "RAM:" in s
        assert "CPU:" in s

    def test_to_dict_serializable(self):
        import json
        sr = _make_sys_resources(24.0)
        d = sr.to_dict()
        # Must be JSON-serializable
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
        assert d["tier"] == "medium"
        assert d["total_ram_gb"] == 24.0


# ---------------------------------------------------------------------------
# maybe_gc helper
# ---------------------------------------------------------------------------

class TestMaybeGC:
    def test_no_config_no_crash(self):
        maybe_gc(None, "test")

    def test_gc_disabled_does_nothing(self):
        rc = ResourceConfig(gc_between_feature_steps=False)
        # Should return without doing anything
        maybe_gc(rc, "test")

    def test_check_memory_returns_float(self):
        mem = check_memory_pressure()
        assert isinstance(mem, float)
        assert mem >= 0.0


# ---------------------------------------------------------------------------
# ResourceConfig in ExperimentConfig
# ---------------------------------------------------------------------------

class TestExperimentConfigIntegration:
    def test_default_config_has_resources(self):
        from src.experiment_config import create_default_config
        config = create_default_config()
        assert hasattr(config, "resources")
        assert isinstance(config.resources, ResourceConfig)
        assert config.resources.tier in {"low", "medium", "high", "ultra"}

    def test_round_trip_dict(self):
        from src.experiment_config import ExperimentConfig
        config = ExperimentConfig()
        d = config.to_dict()
        assert "resources" in d
        # Reconstruct
        config2 = ExperimentConfig.from_dict(d)
        assert config2.resources.tier == config.resources.tier
        assert config2.resources.n_jobs == config.resources.n_jobs

    def test_from_dict_missing_resources_auto_detects(self):
        from src.experiment_config import ExperimentConfig
        # Old-style config dict without "resources" key
        d = {"experiment_name": "legacy_test"}
        config = ExperimentConfig.from_dict(d)
        assert hasattr(config, "resources")
        assert config.resources.tier in {"low", "medium", "high", "ultra"}

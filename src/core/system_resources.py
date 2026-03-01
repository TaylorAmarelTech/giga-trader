"""
GIGA TRADER - System Resource Detection & Adaptive Scaling
===========================================================
Detects hardware (RAM, CPU, GPU) once at import time and provides
tier-aware configuration defaults that scale the pipeline up or down.

Tiers:
  LOW    (<16 GB RAM)  — aggressive memory saving, skip expensive methods
  MEDIUM (16-32 GB)    — balanced defaults, GC between feature steps
  HIGH   (32-128 GB)   — full pipeline, no GC pressure
  ULTRA  (128 GB+)     — maximum parallelism and data expansion

Usage:
    from src.core.system_resources import get_system_resources, create_resource_config

    # Print hardware summary
    print(get_system_resources().summary())

    # Get tier-aware config (with optional overrides)
    rc = create_resource_config(overrides={"n_jobs": 4})
"""

import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# RESOURCE TIER
# ═══════════════════════════════════════════════════════════════════════════════

class ResourceTier(str, Enum):
    LOW = "low"        # <16 GB RAM
    MEDIUM = "medium"  # 16-32 GB
    HIGH = "high"      # 32-128 GB
    ULTRA = "ultra"    # 128 GB+


# ═══════════════════════════════════════════════════════════════════════════════
# TIER DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════

_TIER_DEFAULTS: Dict[ResourceTier, Dict[str, Any]] = {
    ResourceTier.LOW: {
        "n_synthetic_universes": 3,
        "n_breadth_components": 20,
        "feature_cache_max_size": 1,
        "optuna_n_trials": 10,
        "walk_forward_max_windows": 3,
        "stability_n_cpcv_groups": 3,
        "stability_n_stability_sel": 5,
        "stability_n_rashomon": 3,
        "stability_skip_expensive": True,
        "memory_pressure_threshold_mb": 1500,
        "memory_critical_threshold_mb": 2500,
        "gc_between_feature_steps": True,
        "kernel_pca_nystroem_threshold": 2000,
    },
    ResourceTier.MEDIUM: {
        "n_synthetic_universes": 10,
        "n_breadth_components": 50,
        "feature_cache_max_size": 3,
        "optuna_n_trials": 30,
        "walk_forward_max_windows": 5,
        "stability_n_cpcv_groups": 5,
        "stability_n_stability_sel": 10,
        "stability_n_rashomon": 5,
        "stability_skip_expensive": False,
        "memory_pressure_threshold_mb": 3500,
        "memory_critical_threshold_mb": 5000,
        "gc_between_feature_steps": True,
        "kernel_pca_nystroem_threshold": 5000,
    },
    ResourceTier.HIGH: {
        "n_synthetic_universes": 20,
        "n_breadth_components": 100,
        "feature_cache_max_size": 5,
        "optuna_n_trials": 50,
        "walk_forward_max_windows": 7,
        "stability_n_cpcv_groups": 5,
        "stability_n_stability_sel": 10,
        "stability_n_rashomon": 5,
        "stability_skip_expensive": False,
        "memory_pressure_threshold_mb": 8000,
        "memory_critical_threshold_mb": 16000,
        "gc_between_feature_steps": False,
        "kernel_pca_nystroem_threshold": 10000,
    },
    ResourceTier.ULTRA: {
        "n_synthetic_universes": 30,
        "n_breadth_components": 200,
        "feature_cache_max_size": 10,
        "optuna_n_trials": 100,
        "walk_forward_max_windows": 10,
        "stability_n_cpcv_groups": 8,
        "stability_n_stability_sel": 20,
        "stability_n_rashomon": 10,
        "stability_skip_expensive": False,
        "memory_pressure_threshold_mb": 32000,
        "memory_critical_threshold_mb": 64000,
        "gc_between_feature_steps": False,
        "kernel_pca_nystroem_threshold": 20000,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM RESOURCES (singleton)
# ═══════════════════════════════════════════════════════════════════════════════

class SystemResources:
    """Detect and cache system hardware info. Singleton — created once."""

    _instance: Optional["SystemResources"] = None

    def __new__(cls) -> "SystemResources":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._detect()
        return cls._instance

    def _detect(self) -> None:
        """Probe hardware. All detection wrapped in try/except for robustness."""
        self._detected_at = datetime.now().isoformat()

        # ── RAM ──
        try:
            import psutil
            vm = psutil.virtual_memory()
            self.total_ram_gb: float = round(vm.total / (1024 ** 3), 1)
            self.available_ram_gb: float = round(vm.available / (1024 ** 3), 1)
        except Exception:
            self.total_ram_gb = 16.0  # conservative fallback
            self.available_ram_gb = 8.0

        # ── CPU ──
        self.cpu_count: int = os.cpu_count() or 4
        try:
            import psutil
            self.cpu_physical: int = psutil.cpu_count(logical=False) or self.cpu_count
        except Exception:
            self.cpu_physical = max(1, self.cpu_count // 2)

        # ── GPU (CUDA) ──
        self.has_cuda: bool = False
        self.cuda_device_name: str = ""
        self.cuda_memory_gb: float = 0.0
        self._detect_gpu()

        # ── Tier ──
        self._tier = self._classify_tier()

    def _detect_gpu(self) -> None:
        """Try torch CUDA first, then XGBoost GPU."""
        # Try PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                self.has_cuda = True
                self.cuda_device_name = torch.cuda.get_device_name(0)
                self.cuda_memory_gb = round(
                    torch.cuda.get_device_properties(0).total_mem / (1024 ** 3), 1
                )
                return
        except Exception:
            pass

        # Try XGBoost GPU (lightweight check)
        try:
            import xgboost as xgb
            # XGBoost 2.0+ uses device="cuda"
            bst = xgb.XGBClassifier(
                device="cuda", n_estimators=1, max_depth=1,
                verbosity=0, use_label_encoder=False,
            )
            # Don't actually fit — just check if construction succeeds
            self.has_cuda = True
            self.cuda_device_name = "XGBoost GPU (device detected)"
        except Exception:
            pass

    def _classify_tier(self) -> ResourceTier:
        if self.total_ram_gb < 16:
            return ResourceTier.LOW
        elif self.total_ram_gb < 32:
            return ResourceTier.MEDIUM
        elif self.total_ram_gb < 128:
            return ResourceTier.HIGH
        else:
            return ResourceTier.ULTRA

    @property
    def tier(self) -> ResourceTier:
        return self._tier

    def recommended_n_jobs(self) -> int:
        """Compute n_jobs based on tier and CPU count."""
        if self._tier == ResourceTier.LOW:
            return 1
        elif self._tier == ResourceTier.MEDIUM:
            return max(1, self.cpu_physical // 2)
        elif self._tier == ResourceTier.HIGH:
            return max(1, self.cpu_physical - 1)
        else:  # ULTRA
            return self.cpu_physical

    def recommended_xgb_tree_method(self) -> str:
        """Recommend XGBoost tree_method based on GPU availability."""
        if self.has_cuda:
            return "gpu_hist"
        return "hist"

    def summary(self) -> str:
        """Human-readable summary for startup logging."""
        gpu_line = f"  GPU:  {self.cuda_device_name} ({self.cuda_memory_gb} GB)" if self.has_cuda else "  GPU:  None detected"
        return (
            f"SYSTEM RESOURCES (detected {self._detected_at})\n"
            f"  Tier: {self._tier.value.upper()}\n"
            f"  RAM:  {self.total_ram_gb} GB total, {self.available_ram_gb} GB available\n"
            f"  CPU:  {self.cpu_physical} physical cores, {self.cpu_count} logical\n"
            f"{gpu_line}\n"
            f"  n_jobs recommendation: {self.recommended_n_jobs()}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializable summary for logging/registry metadata."""
        return {
            "tier": self._tier.value,
            "total_ram_gb": self.total_ram_gb,
            "available_ram_gb": self.available_ram_gb,
            "cpu_count": self.cpu_count,
            "cpu_physical": self.cpu_physical,
            "has_cuda": self.has_cuda,
            "cuda_device_name": self.cuda_device_name,
            "cuda_memory_gb": self.cuda_memory_gb,
            "detected_at": self._detected_at,
        }

    @classmethod
    def _reset(cls) -> None:
        """Reset singleton (for testing only)."""
        cls._instance = None


# ═══════════════════════════════════════════════════════════════════════════════
# RESOURCE CONFIG (dataclass)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ResourceConfig:
    """Resource-aware pipeline configuration. Auto-populated from SystemResources."""

    # ── Informational (auto-detected) ──
    tier: str = "medium"
    total_ram_gb: float = 16.0
    cpu_count: int = 4
    has_cuda: bool = False

    # ── Parallelism ──
    n_jobs: int = -1  # -1 = auto-detect from tier

    # ── Synthetic data scaling ──
    n_synthetic_universes: int = 10
    n_breadth_components: int = 50

    # ── Caching ──
    feature_cache_max_size: int = 3

    # ── Optimization ──
    optuna_n_trials: int = 30
    walk_forward_max_windows: int = 5

    # ── Stability suite ──
    stability_n_cpcv_groups: int = 5
    stability_n_stability_sel: int = 10
    stability_n_rashomon: int = 5
    stability_skip_expensive: bool = False

    # ── Memory management ──
    memory_pressure_threshold_mb: int = 3500
    memory_critical_threshold_mb: int = 5000
    gc_between_feature_steps: bool = True

    # ── Feature processing ──
    kernel_pca_nystroem_threshold: int = 5000

    # ── GPU ──
    xgb_tree_method: str = "hist"


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_system_resources() -> SystemResources:
    """Return the cached SystemResources singleton."""
    return SystemResources()


def create_resource_config(overrides: Optional[Dict[str, Any]] = None) -> ResourceConfig:
    """Create a ResourceConfig with tier-aware defaults, then apply overrides.

    Args:
        overrides: Optional dict of field names -> values. User wins over auto-detection.

    Returns:
        ResourceConfig populated from detected hardware tier.
    """
    sys_res = get_system_resources()
    tier = sys_res.tier
    defaults = _TIER_DEFAULTS[tier]

    # Compute n_jobs from tier
    n_jobs = sys_res.recommended_n_jobs()

    config = ResourceConfig(
        tier=tier.value,
        total_ram_gb=sys_res.total_ram_gb,
        cpu_count=sys_res.cpu_count,
        has_cuda=sys_res.has_cuda,
        n_jobs=n_jobs,
        n_synthetic_universes=defaults["n_synthetic_universes"],
        n_breadth_components=defaults["n_breadth_components"],
        feature_cache_max_size=defaults["feature_cache_max_size"],
        optuna_n_trials=defaults["optuna_n_trials"],
        walk_forward_max_windows=defaults["walk_forward_max_windows"],
        stability_n_cpcv_groups=defaults["stability_n_cpcv_groups"],
        stability_n_stability_sel=defaults["stability_n_stability_sel"],
        stability_n_rashomon=defaults["stability_n_rashomon"],
        stability_skip_expensive=defaults["stability_skip_expensive"],
        memory_pressure_threshold_mb=defaults["memory_pressure_threshold_mb"],
        memory_critical_threshold_mb=defaults["memory_critical_threshold_mb"],
        gc_between_feature_steps=defaults["gc_between_feature_steps"],
        kernel_pca_nystroem_threshold=defaults["kernel_pca_nystroem_threshold"],
        xgb_tree_method=sys_res.recommended_xgb_tree_method(),
    )

    # Apply user overrides (user wins)
    if overrides:
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"ResourceConfig: unknown override key '{key}', ignoring")

    return config


def check_memory_pressure(resource_config: Optional["ResourceConfig"] = None) -> float:
    """Check current process memory in MB. For use between pipeline steps.

    Returns:
        Current RSS in MB, or 0.0 if psutil unavailable.
    """
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def maybe_gc(
    resource_config: Optional["ResourceConfig"],
    step_name: str = "",
) -> None:
    """Conditionally run gc.collect() if memory pressure is high.

    Only acts when gc_between_feature_steps=True and current RSS exceeds
    the pressure threshold.
    """
    if resource_config is None:
        return
    if not resource_config.gc_between_feature_steps:
        return

    mem_mb = check_memory_pressure()
    if mem_mb > resource_config.memory_pressure_threshold_mb:
        import gc
        gc.collect()
        mem_after = check_memory_pressure()
        logger.info(
            f"[RESOURCE] GC after {step_name}: {mem_mb:.0f}MB -> {mem_after:.0f}MB"
        )

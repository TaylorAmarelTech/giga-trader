"""
Mega Ensemble System - Multi-layer ensemble for robust predictions.

Architecture:
    Layer 1: Extended Grid Search (500+ base models)
    Layer 2: Diversity-Selected Registry Ensemble (voting + stacking)
    Layer 3: Interpolated Config Fabric (hyperparameter mesh)
    Layer 4: Temporal Cascade (time-slice ensemble)
    Layer 5: Final Meta-Learner (combines all layers)
"""

from .extended_grid_search import ExtendedGridConfig, ExtendedGridSearchGenerator
from .diversity_selector import DiversityConfig, DiversitySelector
from .registry_ensemble import RegistryEnsembleConfig, RegistryEnsemble
from .config_interpolator import InterpolationConfig, ConfigInterpolator, FabricOfPoints
from .mega_ensemble import MegaEnsembleConfig, CascadeLineage, MegaEnsemble
from .pipeline import MegaEnsemblePipeline

__all__ = [
    # Extended Grid Search
    "ExtendedGridConfig",
    "ExtendedGridSearchGenerator",
    # Diversity Selection
    "DiversityConfig",
    "DiversitySelector",
    # Registry Ensemble
    "RegistryEnsembleConfig",
    "RegistryEnsemble",
    # Config Interpolation
    "InterpolationConfig",
    "ConfigInterpolator",
    "FabricOfPoints",
    # Mega Ensemble
    "MegaEnsembleConfig",
    "CascadeLineage",
    "MegaEnsemble",
    # Pipeline
    "MegaEnsemblePipeline",
]

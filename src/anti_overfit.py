"""
GIGA TRADER - Anti-Overfitting Module (Shim)
==============================================
This module re-exports all classes and functions from the decomposed files
so that `from src.anti_overfit import X` continues to work.

The actual implementations live in:
  - src/phase_01_data_acquisition/alpaca_data_helper.py
  - src/phase_03_synthetic_data/synthetic_universe.py
  - src/phase_08_features_breadth/streak_features.py
  - src/phase_08_features_breadth/cross_asset_features.py
  - src/phase_08_features_breadth/mag7_breadth.py
  - src/phase_08_features_breadth/sector_breadth.py
  - src/phase_08_features_breadth/volatility_regime.py
  - src/phase_13_validation/weighted_evaluation.py
  - src/phase_13_validation/anti_overfit_integration.py
  - src/phase_14_robustness/stability_analyzer.py
  - src/phase_14_robustness/robustness_ensemble.py
"""

# Phase 01: Data Acquisition
from src.phase_01_data_acquisition.alpaca_data_helper import (
    AlpacaDataHelper,
    get_alpaca_helper,
)

# Phase 03: Synthetic Data
from src.phase_03_synthetic_data.synthetic_universe import (
    SyntheticSPYGenerator,
)

# Phase 08: Breadth Features
from src.phase_08_features_breadth.streak_features import (
    ComponentStreakFeatures,
)
from src.phase_08_features_breadth.cross_asset_features import (
    CrossAssetFeatures,
)
from src.phase_08_features_breadth.mag7_breadth import (
    Mag7BreadthFeatures,
)
from src.phase_08_features_breadth.sector_breadth import (
    SectorBreadthFeatures,
)
from src.phase_08_features_breadth.volatility_regime import (
    VolatilityRegimeFeatures,
)

# Phase 13: Validation
from src.phase_13_validation.weighted_evaluation import (
    WeightedModelEvaluator,
    compute_weighted_evaluation,
)
from src.phase_13_validation.anti_overfit_integration import (
    integrate_anti_overfit,
)

# Phase 14: Robustness
from src.phase_14_robustness.stability_analyzer import (
    StabilityAnalyzer,
)
from src.phase_14_robustness.robustness_ensemble import (
    RobustnessEnsemble,
    create_robustness_ensemble,
)

__all__ = [
    # Phase 01
    "AlpacaDataHelper",
    "get_alpaca_helper",
    # Phase 03
    "SyntheticSPYGenerator",
    # Phase 08
    "ComponentStreakFeatures",
    "CrossAssetFeatures",
    "Mag7BreadthFeatures",
    "SectorBreadthFeatures",
    "VolatilityRegimeFeatures",
    # Phase 13
    "WeightedModelEvaluator",
    "compute_weighted_evaluation",
    "integrate_anti_overfit",
    # Phase 14
    "StabilityAnalyzer",
    "RobustnessEnsemble",
    "create_robustness_ensemble",
]


if __name__ == "__main__":
    # Test the module
    print("Anti-Overfit Module - Test Run")
    print("=" * 50)

    # Test WeightedModelEvaluator
    evaluator = WeightedModelEvaluator()
    print(f"Default weights: {evaluator.weights}")

    # Test SyntheticSPYGenerator
    synth = SyntheticSPYGenerator()
    print(f"Synthetic universes: {synth.n_universes}")
    print(f"Real weight: {synth.real_weight}")

    # Test CrossAssetFeatures
    cross = CrossAssetFeatures()
    print(f"Cross assets: {cross.assets}")

    print("\nModule loaded successfully!")

"""
Mega Ensemble Pipeline - Orchestrates full training.

Steps:
1. Run extended grid search → registry
2. Build diversity-selected ensemble (Layer 2)
3. Generate and train interpolated fabric (Layer 3)
4. Optionally train temporal cascade (Layer 4)
5. Train final meta-learner (Layer 5)
"""

import sys
import logging
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model_registry_v2 import ModelRegistryV2, ModelEntry
from src.training_pipeline_v2 import TrainingPipelineV2, DataLoader, FeatureEngineer, TargetCreator
from src.mega_ensemble.extended_grid_search import ExtendedGridConfig, ExtendedGridSearchGenerator, create_quick_extended_grid
from src.mega_ensemble.comprehensive_grid_search import ComprehensiveGridSearchGenerator, GRID_LEVELS
from src.mega_ensemble.config_interpolator import FabricOfPoints, InterpolationConfig
from src.mega_ensemble.mega_ensemble import MegaEnsemble, MegaEnsembleConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger("MEGA_PIPELINE")


@dataclass
class MegaPipelineConfig:
    """Configuration for the mega ensemble pipeline."""

    # Grid search settings - COMPREHENSIVE (ALL 13 PIPELINE STEPS)
    grid_search_level: str = "comprehensive"  # minimal, standard, comprehensive, full
    grid_search_max_configs: int = 100
    grid_include_boosting: bool = True
    grid_include_trees: bool = True
    grid_include_neural: bool = False  # Slow, off by default
    use_comprehensive_grid: bool = True  # Use new 13-step comprehensive grid search

    # Fabric settings (Layer 3 - interpolated configs)
    fabric_enabled: bool = True
    fabric_max_models: int = 30
    fabric_n_anchor_models: int = 10

    # Meta-ensemble settings (Layer 2 - diversity selection)
    diversity_n_models: int = 10
    final_meta_model: str = "logistic"

    # Data settings
    data_period: str = "3Y"
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    # Temporal cascade (Layer 4)
    use_temporal_cascade: bool = False


class MegaEnsemblePipeline:
    """
    Training pipeline for the full mega ensemble.

    Orchestrates:
    1. Extended grid search (populates registry with diverse base models)
    2. Mega ensemble training (selects diverse models, builds fabric, trains meta)
    """

    def __init__(
        self,
        registry: ModelRegistryV2 = None,
        config: MegaPipelineConfig = None,
    ):
        self.registry = registry or ModelRegistryV2()
        self.config = config or MegaPipelineConfig()

        self.base_pipeline = TrainingPipelineV2(self.registry)

        # Results
        self.grid_search_results: List[ModelEntry] = []
        self.fabric_results: List[ModelEntry] = []
        self.mega_ensemble: Optional[MegaEnsemble] = None

    def run_extended_grid_search(
        self,
        target_type: str = "swing",
    ) -> List[ModelEntry]:
        """
        Run extended grid search to populate registry with diverse base models.

        If use_comprehensive_grid is True, this uses the FULL 13-step grid search
        that varies all pipeline parameters (data, synthetic, preprocess, features,
        target, feature_selection, dim_reduction, model, cascade, sample_weights,
        training, evaluation).

        Returns:
            List of trained ModelEntry objects
        """
        logger.info("")
        logger.info("=" * 70)
        if self.config.use_comprehensive_grid:
            logger.info("MEGA PIPELINE - Step 1: COMPREHENSIVE Grid Search (ALL 13 STEPS)")
        else:
            logger.info("MEGA PIPELINE - Step 1: Extended Grid Search")
        logger.info("=" * 70)

        if self.config.use_comprehensive_grid:
            # Use COMPREHENSIVE grid search - ALL 13 pipeline steps
            gen = ComprehensiveGridSearchGenerator(
                level=self.config.grid_search_level,
                max_configs=self.config.grid_search_max_configs,
                random_sample=True,
            )
            logger.info(gen.get_summary())

            # Generate configs
            configs = gen.generate_configs(target_type=target_type)
        else:
            # Use original extended grid search (model types + dim reduction only)
            grid_config = ExtendedGridConfig(
                include_logistic=True,
                include_trees=self.config.grid_include_trees,
                include_boosting=self.config.grid_include_boosting,
                include_neural=self.config.grid_include_neural,
                include_svm=False,
                max_configs=self.config.grid_search_max_configs,
                random_sample=True,
            )

            gen = ExtendedGridSearchGenerator(grid_config)
            logger.info(gen.get_summary())

            # Generate configs
            configs = gen.generate_configs(target_type=target_type)

        logger.info(f"Generated {len(configs)} configurations")

        # Show sample configuration details for first config
        if configs:
            c = configs[0]
            logger.info("")
            logger.info("Sample configuration (first of generated):")
            logger.info(f"  Model: {c.model_config.model_type}")
            logger.info(f"  DimReduction: {c.dim_reduction_config.method} ({c.dim_reduction_config.n_components} components)")
            logger.info(f"  Cascade: {c.cascade_config.cascade_type}")
            logger.info(f"  Preprocessing: outlier={c.preprocess_config.outlier_method}, scaling={c.preprocess_config.scaling_method}")
            logger.info(f"  FeatureSelection: {c.feature_selection_config.method}")
            logger.info(f"  Target: threshold={c.target_config.swing_threshold}, soft={c.target_config.use_soft_targets}")
            logger.info(f"  Training: cv={c.training_config.cv_method}, folds={c.training_config.cv_folds}")

        # Train all
        logger.info("")
        results = self.base_pipeline.train_batch(configs)

        # Filter successful
        self.grid_search_results = [r for r in results if r.status == "trained"]
        n_success = len(self.grid_search_results)
        n_total = len(results)

        logger.info("")
        logger.info(f"Grid search complete: {n_success}/{n_total} models trained successfully")

        # Find best
        if self.grid_search_results:
            best = max(self.grid_search_results, key=lambda m: m.metrics.cv_auc)
            logger.info(f"Best model: {best.model_config.model_type} "
                       f"AUC={best.metrics.cv_auc:.4f} Gap={best.metrics.train_test_gap:.4f}")

        return self.grid_search_results

    def run_fabric_training(
        self,
        anchor_models: List[ModelEntry] = None,
    ) -> List[ModelEntry]:
        """
        Generate and train interpolated fabric models.

        Args:
            anchor_models: Best models to interpolate between (uses top from registry if None)

        Returns:
            List of trained fabric ModelEntry objects
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info("MEGA PIPELINE - Step 2: Fabric Generation & Training")
        logger.info("=" * 70)

        if not self.config.fabric_enabled:
            logger.info("Fabric training disabled")
            return []

        # Get anchor models
        if anchor_models is None:
            # Query registry for best models
            all_models = list(self.registry.models.values())
            trained_models = [m for m in all_models if m.status == "trained"]
            trained_models.sort(key=lambda m: m.metrics.cv_auc if m.metrics else 0, reverse=True)
            anchor_models = trained_models[:self.config.fabric_n_anchor_models]

        if len(anchor_models) < 2:
            logger.warning("Need at least 2 anchor models for fabric generation")
            return []

        logger.info(f"Using {len(anchor_models)} anchor models for fabric generation")

        # Generate fabric configs
        interp_config = InterpolationConfig(
            n_interpolation_points=2,
            max_fabric_models=self.config.fabric_max_models,
            interpolate_continuous=True,
            interpolate_discrete=True,
            create_hybrid_dim_reduction=True,
        )

        fabric = FabricOfPoints(anchor_models, interp_config)
        fabric_configs, fabric_metrics = fabric.generate_fabric()

        if not fabric_configs:
            logger.warning("No fabric configs generated")
            return []

        logger.info(f"Generated {len(fabric_configs)} fabric configurations")

        # Train fabric models
        fabric_results = self.base_pipeline.train_batch(fabric_configs)
        self.fabric_results = [r for r in fabric_results if r.status == "trained"]

        n_success = len(self.fabric_results)
        n_total = len(fabric_results)
        logger.info(f"Fabric training complete: {n_success}/{n_total} models trained")

        return self.fabric_results

    def train_mega_ensemble(
        self,
        df_1min: pd.DataFrame = None,
    ) -> Tuple[MegaEnsemble, Dict[str, Any]]:
        """
        Train the mega ensemble using registry models.

        Args:
            df_1min: 1-minute data for temporal cascade (optional)

        Returns:
            Tuple of (trained MegaEnsemble, metrics dict)
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info("MEGA PIPELINE - Step 3: Mega Ensemble Training")
        logger.info("=" * 70)

        # Load data for training/validation split
        logger.info("Loading data for ensemble training...")

        # Create a dummy ModelEntry to load data
        dummy_entry = ModelEntry(target_type="swing")
        dummy_entry.data_config.period = self.config.data_period

        df_1min_data = df_1min
        if df_1min_data is None:
            df_1min_data = DataLoader.load(dummy_entry)

        # Engineer features
        logger.info("Engineering features...")
        df_daily = FeatureEngineer.engineer(df_1min_data, dummy_entry)

        # Create targets
        df_daily, swing_target, timing_target = TargetCreator.create_targets(df_daily, dummy_entry)

        # Prepare features (exclude target columns)
        exclude_patterns = ["target", "soft_target", "smoothed", "label", "is_up", "is_down",
                          "low_before_high", "sample_weight", "forward_return"]
        feature_cols = [c for c in df_daily.columns
                       if not any(p in c.lower() for p in exclude_patterns)
                       and c not in ['date', 'timestamp', 'open', 'high', 'low', 'close', 'volume',
                                    'day_return', 'day_range', 'high_minutes', 'low_minutes']]

        X = df_daily[feature_cols].fillna(0).values
        y = (df_daily[swing_target] > 0.5).astype(int).values if swing_target in df_daily.columns else df_daily['is_up_day'].values

        # Split data
        n = len(X)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]

        logger.info(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X) - val_end}")

        # Create and fit mega ensemble
        mega_config = MegaEnsembleConfig(
            diversity_n_models=self.config.diversity_n_models,
            fabric_max_models=self.config.fabric_max_models,
            final_meta_model=self.config.final_meta_model,
            use_temporal_cascade=self.config.use_temporal_cascade,
        )

        self.mega_ensemble = MegaEnsemble(mega_config)

        metrics = self.mega_ensemble.fit(
            registry=self.registry,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            trained_fabric_models=self.fabric_results,
            df_1min=df_1min_data if self.config.use_temporal_cascade else None,
        )

        return self.mega_ensemble, metrics

    def train_full_mega_ensemble(
        self,
        target_type: str = "swing",
        df_1min: pd.DataFrame = None,
    ) -> Tuple[MegaEnsemble, Dict[str, Any]]:
        """
        Execute full mega ensemble training pipeline.

        Steps:
        1. Extended grid search (Layer 1 → registry)
        2. Fabric generation and training (Layer 3)
        3. Mega ensemble training (Layers 2, 4, 5)

        Args:
            target_type: Target variable type
            df_1min: 1-minute data (optional, will be downloaded if not provided)

        Returns:
            Tuple of (trained MegaEnsemble, full metrics dict)
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info("MEGA ENSEMBLE PIPELINE - Full Training")
        logger.info("=" * 70)
        logger.info(f"Started: {datetime.now().isoformat()}")
        logger.info("")

        full_metrics = {}

        # Step 1: Extended Grid Search
        grid_results = self.run_extended_grid_search(target_type=target_type)
        full_metrics["grid_search"] = {
            "n_configs": self.config.grid_search_max_configs,
            "n_trained": len(grid_results),
            "best_auc": max([m.metrics.cv_auc for m in grid_results]) if grid_results else 0,
        }

        # Step 2: Fabric Training
        if self.config.fabric_enabled:
            fabric_results = self.run_fabric_training()
            full_metrics["fabric"] = {
                "n_generated": len(fabric_results) + len([f for f in self.fabric_results if f.status != "trained"]),
                "n_trained": len(self.fabric_results),
            }

        # Step 3: Mega Ensemble Training
        ensemble, ensemble_metrics = self.train_mega_ensemble(df_1min=df_1min)
        full_metrics["ensemble"] = ensemble_metrics

        # Final summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("MEGA PIPELINE - Complete")
        logger.info("=" * 70)
        logger.info(f"Total registry models: {len(self.registry.models)}")
        logger.info(f"Grid search models: {full_metrics['grid_search']['n_trained']}")
        if self.config.fabric_enabled:
            logger.info(f"Fabric models: {full_metrics['fabric']['n_trained']}")
        logger.info(f"Final ensemble validation AUC: {ensemble_metrics.get('layer5_meta', {}).get('val_auc', 0):.4f}")
        logger.info(f"Completed: {datetime.now().isoformat()}")
        logger.info("=" * 70)

        return ensemble, full_metrics


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train Mega Ensemble")
    parser.add_argument("--mode", choices=["quick", "standard", "full"], default="quick",
                       help="Training mode: quick (50 configs), standard (100), full (500)")
    parser.add_argument("--grid-level", choices=["minimal", "standard", "comprehensive", "full"], default="comprehensive",
                       help="Grid search level: minimal (model+dimred), standard (+preprocess+cascade), "
                            "comprehensive (+features+target+selection), full (ALL 13 STEPS)")
    parser.add_argument("--max-configs", type=int, default=None,
                       help="Override max configs for grid search")
    parser.add_argument("--fabric", action="store_true", default=True,
                       help="Enable fabric training")
    parser.add_argument("--no-fabric", action="store_true",
                       help="Disable fabric training")
    parser.add_argument("--temporal", action="store_true",
                       help="Enable temporal cascade (requires more data)")
    parser.add_argument("--use-extended-only", action="store_true",
                       help="Use original extended grid search (not comprehensive 13-step)")

    args = parser.parse_args()

    # Set config based on mode
    if args.mode == "quick":
        max_configs = args.max_configs or 50
        fabric_max = 20
    elif args.mode == "standard":
        max_configs = args.max_configs or 100
        fabric_max = 30
    else:
        max_configs = args.max_configs or 500
        fabric_max = 50

    config = MegaPipelineConfig(
        grid_search_level=args.grid_level,
        grid_search_max_configs=max_configs,
        grid_include_boosting=True,
        grid_include_trees=True,
        grid_include_neural=(args.mode == "full"),
        fabric_enabled=not args.no_fabric,
        fabric_max_models=fabric_max,
        use_temporal_cascade=args.temporal,
        use_comprehensive_grid=not args.use_extended_only,
    )

    # Run pipeline
    registry = ModelRegistryV2()
    pipeline = MegaEnsemblePipeline(registry, config)

    ensemble, metrics = pipeline.train_full_mega_ensemble(target_type="swing")

    # Print registry summary
    print("\n" + registry.summary())

    return ensemble, metrics


if __name__ == "__main__":
    main()

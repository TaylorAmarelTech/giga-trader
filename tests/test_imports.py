"""
Test that ALL shim re-exports work correctly.

Each original monolithic module was decomposed into focused modules in phase
directories, with the original file becoming a backward-compatible shim.  These
tests verify that every expected public name is still importable from the shim.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# SHIM RE-EXPORT TESTS
# ============================================================================

def test_model_registry_v2_shim():
    """Verify model_registry_v2 shim re-exports all expected names."""
    from src.model_registry_v2 import (
        ModelRegistryV2,
        ModelEntry,
        GridSearchConfigGenerator,
        list_all_options,
        DataSource,
        ModelType,
    )
    assert ModelRegistryV2 is not None
    assert ModelEntry is not None
    assert GridSearchConfigGenerator is not None
    assert callable(list_all_options)
    assert hasattr(DataSource, "ALPACA")
    assert hasattr(ModelType, "LOGISTIC_L1")


def test_anti_overfit_shim():
    """Verify anti_overfit shim re-exports all expected names."""
    from src.anti_overfit import (
        AlpacaDataHelper,
        SyntheticSPYGenerator,
        ComponentStreakFeatures,
        CrossAssetFeatures,
        Mag7BreadthFeatures,
        SectorBreadthFeatures,
        VolatilityRegimeFeatures,
        WeightedModelEvaluator,
        compute_weighted_evaluation,
        StabilityAnalyzer,
        RobustnessEnsemble,
        create_robustness_ensemble,
        integrate_anti_overfit,
    )
    for name, obj in [
        ("AlpacaDataHelper", AlpacaDataHelper),
        ("SyntheticSPYGenerator", SyntheticSPYGenerator),
        ("ComponentStreakFeatures", ComponentStreakFeatures),
        ("CrossAssetFeatures", CrossAssetFeatures),
        ("Mag7BreadthFeatures", Mag7BreadthFeatures),
        ("SectorBreadthFeatures", SectorBreadthFeatures),
        ("VolatilityRegimeFeatures", VolatilityRegimeFeatures),
        ("WeightedModelEvaluator", WeightedModelEvaluator),
        ("StabilityAnalyzer", StabilityAnalyzer),
        ("RobustnessEnsemble", RobustnessEnsemble),
    ]:
        assert obj is not None, f"{name} should not be None"
    assert callable(compute_weighted_evaluation)
    assert callable(create_robustness_ensemble)
    assert callable(integrate_anti_overfit)


def test_paper_trading_shim():
    """Verify paper_trading shim re-exports all expected names."""
    from src.paper_trading import (
        AlpacaPaperClient,
        SignalGenerator,
        RiskManager,
        OrderManager,
        TradingBot,
    )
    for cls in [AlpacaPaperClient, SignalGenerator, RiskManager, OrderManager, TradingBot]:
        assert cls is not None


def test_enhanced_signal_generator_shim():
    """Verify enhanced_signal_generator shim re-exports."""
    from src.enhanced_signal_generator import (
        RobustEnsembleSignalGenerator,
        get_enhanced_signal_generator,
    )
    assert RobustEnsembleSignalGenerator is not None
    assert callable(get_enhanced_signal_generator)


def test_backtest_engine_shim():
    """Verify backtest_engine shim re-exports."""
    from src.backtest_engine import (
        Trade,
        Portfolio,
        BacktestEngine,
        run_full_backtest,
        WalkForwardBacktest,
        MonteCarloSimulator,
    )
    for cls in [Trade, Portfolio, BacktestEngine, WalkForwardBacktest, MonteCarloSimulator]:
        assert cls is not None
    assert callable(run_full_backtest)


def test_entry_exit_model_shim():
    """Verify entry_exit_model shim re-exports."""
    from src.entry_exit_model import (
        TargetLabeler,
        TimingFeatureEngineer,
        EntryExitTimingModel,
    )
    for cls in [TargetLabeler, TimingFeatureEngineer, EntryExitTimingModel]:
        assert cls is not None


def test_leak_proof_cv_shim():
    """Verify leak_proof_cv shim re-exports."""
    from src.leak_proof_cv import (
        LeakProofFeatureSelector,
        LeakProofDimReducer,
        LeakProofCV,
        EnsembleReducer,
        LeakProofPipeline,
        train_with_leak_proof_cv,
    )
    for cls in [LeakProofFeatureSelector, LeakProofDimReducer, LeakProofCV,
                EnsembleReducer, LeakProofPipeline]:
        assert cls is not None
    assert callable(train_with_leak_proof_cv)


def test_experiment_engine_shim():
    """Verify experiment_engine shim re-exports."""
    from src.experiment_engine import (
        WalkForwardCV,
        ExperimentEngine,
        UnifiedExperimentRunner,
    )
    for cls in [WalkForwardCV, ExperimentEngine, UnifiedExperimentRunner]:
        assert cls is not None


def test_pipeline_grid_shim():
    """Verify pipeline_grid shim re-exports."""
    from src.pipeline_grid import (
        PipelineGridSearch,
        MultiObjectiveOptimizer,
        IntegratedGridSearch,
    )
    for cls in [PipelineGridSearch, MultiObjectiveOptimizer, IntegratedGridSearch]:
        assert cls is not None


def test_dynamic_model_selector_shim():
    """Verify dynamic_model_selector shim re-exports."""
    from src.dynamic_model_selector import (
        DynamicModelSelector,
        EnsemblePrediction,
        ModelCandidate,
    )
    for cls in [DynamicModelSelector, EnsemblePrediction, ModelCandidate]:
        assert cls is not None


def test_data_manager_shim():
    """Verify data_manager shim re-exports."""
    from src.data_manager import (
        DataManager,
        get_data_manager,
        get_spy_data,
    )
    assert DataManager is not None
    assert callable(get_data_manager)
    assert callable(get_spy_data)


def test_dashboard_shim():
    """Verify dashboard shim re-exports."""
    from src.dashboard import (
        ModelAnalyzer,
        BacktestAnalyzer,
        generate_html_dashboard,
    )
    for cls in [ModelAnalyzer, BacktestAnalyzer]:
        assert cls is not None
    assert callable(generate_html_dashboard)


# ============================================================================
# PHASE DIRECTORY IMPORT TESTS
# ============================================================================

def test_phase_01_imports():
    """Verify phase_01_data_acquisition package exports."""
    from src.phase_01_data_acquisition import (
        HistoricalConstituentProvider,
        AlpacaDataHelper,
        DataManager,
    )
    for cls in [HistoricalConstituentProvider, AlpacaDataHelper, DataManager]:
        assert cls is not None


def test_phase_09_imports():
    """Verify phase_09_features_calendar package exports."""
    from src.phase_09_features_calendar import (
        FOMCFeatures,
        CalendarFeatureGenerator,
    )
    for cls in [FOMCFeatures, CalendarFeatureGenerator]:
        assert cls is not None


def test_phase_18_imports():
    """Verify phase_18_persistence package exports."""
    from src.phase_18_persistence import (
        ModelRegistryV2,
        ModelEntry,
        GridSearchConfigGenerator,
    )
    for cls in [ModelRegistryV2, ModelEntry, GridSearchConfigGenerator]:
        assert cls is not None


def test_phase_20_imports():
    """Verify phase_20_monitoring package exports."""
    from src.phase_20_monitoring import (
        HealthChecker,
        AlertManager,
    )
    for cls in [HealthChecker, AlertManager]:
        assert cls is not None


def test_core_imports():
    """Verify src.core package exports."""
    from src.core import (
        PhaseRunner,
        StateManager,
    )
    for cls in [PhaseRunner, StateManager]:
        assert cls is not None


def test_phase_03_imports():
    """Verify phase_03_synthetic_data exports."""
    from src.phase_03_synthetic_data import SyntheticSPYGenerator
    assert SyntheticSPYGenerator is not None


def test_phase_08_imports():
    """Verify phase_08_features_breadth exports."""
    from src.phase_08_features_breadth import (
        ComponentStreakFeatures,
        CrossAssetFeatures,
    )
    for cls in [ComponentStreakFeatures, CrossAssetFeatures]:
        assert cls is not None


def test_phase_14_imports():
    """Verify phase_14_robustness exports."""
    from src.phase_14_robustness import (
        StabilityAnalyzer,
        RobustnessEnsemble,
        create_robustness_ensemble,
    )
    assert StabilityAnalyzer is not None
    assert RobustnessEnsemble is not None
    assert callable(create_robustness_ensemble)


def test_phase_15_imports():
    """Verify phase_15_strategy exports."""
    from src.phase_15_strategy import (
        SignalDirection,
        StrategySignal,
        MomentumStrategy,
        ContrarianStrategy,
        RobustEnsembleSignalGenerator,
    )
    for cls in [SignalDirection, StrategySignal, MomentumStrategy,
                ContrarianStrategy, RobustEnsembleSignalGenerator]:
        assert cls is not None


def test_phase_16_imports():
    """Verify phase_16_backtesting exports."""
    from src.phase_16_backtesting import (
        Trade,
        Portfolio,
        BacktestEngine,
        run_full_backtest,
        WalkForwardBacktest,
        MonteCarloSimulator,
    )
    for cls in [Trade, Portfolio, BacktestEngine, WalkForwardBacktest, MonteCarloSimulator]:
        assert cls is not None
    assert callable(run_full_backtest)


def test_phase_19_imports():
    """Verify phase_19_paper_trading exports."""
    from src.phase_19_paper_trading import (
        AlpacaPaperClient,
        SignalGenerator,
        RiskManager,
        OrderManager,
        TradingBot,
    )
    for cls in [AlpacaPaperClient, SignalGenerator, RiskManager, OrderManager, TradingBot]:
        assert cls is not None

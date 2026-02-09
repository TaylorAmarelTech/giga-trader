"""
Integration tests: verify that key imports across all phases can be combined
and that objects from multiple phases can be instantiated together.

These tests do NOT train models or call APIs -- they verify that the
decomposed module structure is coherent and that cross-phase object creation
works.
"""

import sys
from pathlib import Path
from datetime import datetime, date

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Test: Key imports from across all phases can coexist
# ---------------------------------------------------------------------------

def test_cross_phase_imports():
    """Importing from many phases simultaneously should not cause conflicts."""
    # Phase 01 - Data
    from src.phase_01_data_acquisition import HistoricalConstituentProvider

    # Phase 09 - Calendar Features
    from src.phase_09_features_calendar import FOMCFeatures, CalendarFeatureGenerator

    # Phase 14 - Robustness
    from src.phase_14_robustness import RobustnessEnsemble, StabilityAnalyzer

    # Phase 15 - Strategy
    from src.phase_15_strategy import (
        SignalDirection,
        StrategySignal,
        MomentumStrategy,
    )

    # Phase 16 - Backtesting
    from src.phase_16_backtesting import Trade, Portfolio, BacktestEngine

    # Phase 18 - Persistence
    from src.phase_18_persistence import (
        ModelRegistryV2,
        ModelEntry,
        GridSearchConfigGenerator,
    )
    from src.phase_18_persistence.registry_enums import (
        DataSource,
        ModelType,
        TargetType,
    )

    # Phase 20 - Monitoring
    from src.phase_20_monitoring import HealthChecker, AlertManager

    # Core
    from src.core import PhaseConfig, PhaseResult, PipelineOrchestrator, StateManager

    # All imports should be non-None
    assert all(cls is not None for cls in [
        HistoricalConstituentProvider, FOMCFeatures, CalendarFeatureGenerator,
        RobustnessEnsemble, StabilityAnalyzer,
        SignalDirection, StrategySignal, MomentumStrategy,
        Trade, Portfolio, BacktestEngine,
        ModelRegistryV2, ModelEntry, GridSearchConfigGenerator,
        DataSource, ModelType, TargetType,
        HealthChecker, AlertManager,
        PhaseConfig, PhaseResult, PipelineOrchestrator, StateManager,
    ])


# ---------------------------------------------------------------------------
# Test: Instantiate objects from multiple phases in sequence
# ---------------------------------------------------------------------------

def test_create_multi_phase_objects(tmp_path):
    """Objects from different phases should coexist without conflict."""
    from src.core.base import PhaseConfig, PipelineOrchestrator
    from src.core.state_manager import StateManager
    from src.phase_18_persistence.model_registry import ModelRegistryV2
    from src.phase_18_persistence.registry_configs import ModelEntry
    from src.phase_18_persistence.registry_enums import TargetType, ModelType
    from src.phase_14_robustness.robustness_ensemble import RobustnessEnsemble
    from src.phase_15_strategy.trading_strategies import MomentumStrategy
    from src.phase_16_backtesting.portfolio import Portfolio, Trade
    from src.phase_20_monitoring.health_checker import HealthChecker, AlertManager

    # Core
    orchestrator = PipelineOrchestrator()
    assert isinstance(orchestrator, PipelineOrchestrator)

    state_manager = StateManager(state_dir=str(tmp_path / "state"))
    state_manager.update(mode="TESTING")
    assert state_manager.state.mode == "TESTING"

    # Phase 18 - Registry
    registry = ModelRegistryV2(
        registry_dir=tmp_path / "registry",
        models_dir=tmp_path / "artifacts",
    )
    entry = ModelEntry(target_type=TargetType.SWING.value)
    entry.model_config.model_type = ModelType.GRADIENT_BOOSTING.value
    entry.metrics.cv_auc = 0.77
    registry.register(entry)
    assert len(registry.models) == 1

    # Phase 14 - Robustness
    ensemble = RobustnessEnsemble(n_dimension_variants=2)
    variants = ensemble.create_dimension_variants(optimal_dims=30)
    assert 30 in variants

    # Phase 15 - Strategy
    strategy = MomentumStrategy(lookback=5, threshold=0.3)
    assert strategy.name == "momentum"

    # Phase 16 - Portfolio
    portfolio = Portfolio(initial_capital=100000)
    assert portfolio.cash == 100000

    # Phase 20 - Monitoring
    checker = HealthChecker(alert_manager=AlertManager())
    assert isinstance(checker, HealthChecker)


# ---------------------------------------------------------------------------
# Test: Shim imports and phase imports return the same classes
# ---------------------------------------------------------------------------

def test_shim_and_phase_imports_are_identical():
    """Shim re-exports should point to the same class objects as phase imports."""
    # Backtest engine
    from src.backtest_engine import Trade as ShimTrade, Portfolio as ShimPortfolio
    from src.phase_16_backtesting.portfolio import Trade as PhaseTrade, Portfolio as PhasePortfolio
    assert ShimTrade is PhaseTrade
    assert ShimPortfolio is PhasePortfolio

    # Model registry
    from src.model_registry_v2 import ModelRegistryV2 as ShimRegistry
    from src.phase_18_persistence.model_registry import ModelRegistryV2 as PhaseRegistry
    assert ShimRegistry is PhaseRegistry

    # Enhanced signal generator
    from src.enhanced_signal_generator import RobustEnsembleSignalGenerator as ShimGen
    from src.phase_15_strategy.ensemble_signal_generator import RobustEnsembleSignalGenerator as PhaseGen
    assert ShimGen is PhaseGen

    # Anti-overfit
    from src.anti_overfit import RobustnessEnsemble as ShimRE
    from src.phase_14_robustness.robustness_ensemble import RobustnessEnsemble as PhaseRE
    assert ShimRE is PhaseRE


# ---------------------------------------------------------------------------
# Test: Minimal end-to-end object flow (no training)
# ---------------------------------------------------------------------------

def test_minimal_pipeline_flow(tmp_path, sample_daily_df):
    """
    Simulate a minimal pipeline flow:
    1. Create model registry
    2. Register a model entry
    3. Create a portfolio
    4. Execute a trade
    5. Record daily metrics
    6. Save state

    This verifies integration without training.
    """
    from src.phase_18_persistence.model_registry import ModelRegistryV2
    from src.phase_18_persistence.registry_configs import ModelEntry
    from src.phase_18_persistence.registry_enums import TargetType, ModelStatus
    from src.phase_16_backtesting.portfolio import Portfolio
    from src.core.state_manager import StateManager

    # Step 1: Registry
    registry = ModelRegistryV2(
        registry_dir=tmp_path / "reg",
        models_dir=tmp_path / "art",
    )

    # Step 2: Register model
    entry = ModelEntry(target_type=TargetType.SWING.value)
    entry.metrics.cv_auc = 0.77
    entry.status = ModelStatus.TRAINED.value
    model_id = registry.register(entry)
    assert registry.get(model_id) is not None

    # Step 3: Portfolio
    portfolio = Portfolio(initial_capital=100000, slippage_pct=0.0)

    # Step 4: Trade
    trade = portfolio.open_trade(
        date=datetime(2025, 1, 6, 10, 0),
        direction="LONG",
        price=450.0,
        position_value=10000.0,
    )
    assert trade is not None
    portfolio.close_trade(
        trade=trade,
        date=datetime(2025, 1, 6, 15, 0),
        price=452.0,
        reason="signal",
    )
    assert len(portfolio.closed_trades) == 1
    assert portfolio.closed_trades[0].realized_pnl > 0

    # Step 5: Record daily
    portfolio.record_daily(datetime(2025, 1, 6), 451.0)
    assert len(portfolio.equity_curve) == 1

    # Step 6: Save state
    sm = StateManager(state_dir=str(tmp_path / "state"))
    sm.update(
        mode="BACKTESTING",
        experiments_completed=1,
        models_trained=1,
        best_auc=0.77,
    )
    sm.save()
    assert sm.state_file.exists()


# ---------------------------------------------------------------------------
# Test: Calendar features applied to synthetic data
# ---------------------------------------------------------------------------

def test_calendar_features_on_synthetic_data(sample_daily_df):
    """Calendar features should work on the shared fixture data."""
    from src.phase_09_features_calendar import CalendarFeatureGenerator

    gen = CalendarFeatureGenerator()
    result = gen.create_all_features(sample_daily_df)

    # Should have more columns than original
    assert len(result.columns) > len(sample_daily_df.columns)

    # Original data should be unchanged
    pd.testing.assert_series_equal(
        result["close"],
        sample_daily_df["close"],
        check_names=True,
    )


# ---------------------------------------------------------------------------
# Test: Historical constituents with survivorship report
# ---------------------------------------------------------------------------

def test_historical_constituents_integration():
    """Historical constituent provider should work end-to-end."""
    from src.phase_01_data_acquisition.historical_constituents import (
        HistoricalConstituentProvider,
    )

    provider = HistoricalConstituentProvider(changes_csv="__nonexistent__.csv")
    # Use the fallback
    provider._current_constituents = provider._get_top50_fallback()

    # Get constituents at a past date
    constituents = provider.get_constituents_at_date(date(2022, 6, 15))
    assert isinstance(constituents, list)
    assert len(constituents) > 0

    # Get survivorship report
    report = provider.get_survivorship_bias_report(date(2020, 1, 1), date(2024, 12, 31))
    assert report["bias_risk"] in ("LOW", "MEDIUM", "HIGH")

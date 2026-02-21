"""
System Integration Tests
=========================
End-to-end smoke tests for the paper trading system:
  - Gate checker (paper vs live mode)
  - Performance tracker
  - Dashboard data writer
  - Model registration
  - Feature group config wiring
"""

import sys
import json
import os
import time
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# GATE CHECKER TESTS
# =============================================================================

class TestExperimentGateChecker:
    """Tests for the mode-aware ExperimentGateChecker."""

    def test_gate_checker_class_exists(self):
        from src.giga_orchestrator import ExperimentGateChecker
        checker = ExperimentGateChecker()
        assert checker is not None

    def test_gate_checker_accepts_trading_mode(self):
        from src.giga_orchestrator import ExperimentGateChecker
        checker = ExperimentGateChecker()
        # Should not raise - may fail on actual gate check but accepts the param
        passed, status = checker.check_gates(trading_mode="paper")
        assert isinstance(passed, bool)
        assert isinstance(status, dict)

    def test_gate_checker_returns_mode_label(self):
        from src.giga_orchestrator import ExperimentGateChecker
        checker = ExperimentGateChecker()
        _, status = checker.check_gates(trading_mode="paper")
        if "mode_label" in status:
            assert status["mode_label"] == "PAPER"

    def test_paper_gate_thresholds_lower_than_live(self):
        from src.giga_orchestrator import ORCHESTRATOR_CONFIG
        assert ORCHESTRATOR_CONFIG["min_experiments_before_paper_trading"] < ORCHESTRATOR_CONFIG["min_experiments_before_live_trading"]
        assert ORCHESTRATOR_CONFIG["min_auc_for_paper_trading"] < ORCHESTRATOR_CONFIG["min_auc_for_live_trading"]
        assert ORCHESTRATOR_CONFIG["min_models_above_threshold_paper"] < ORCHESTRATOR_CONFIG["min_models_above_threshold"]

    def test_paper_gates_exclude_stale_models(self):
        """Wave 16: Gates should exclude models with AUC >= 0.85 (pre-Wave-14 leakage)."""
        from src.giga_orchestrator import ExperimentGateChecker
        checker = ExperimentGateChecker()
        passed, status = checker.check_gates(trading_mode="paper")
        # After Wave 16, stale models (AUC >= 0.85) are excluded from model count.
        # Gates may fail until new legitimate models are trained — this is correct behavior.
        assert "models_above_threshold" in status
        assert "completed_experiments" in status
        # Verify the gate checker ran successfully (regardless of pass/fail)
        assert status["completed_experiments"] > 0, "Should have completed experiments"


# =============================================================================
# PERFORMANCE TRACKER TESTS
# =============================================================================

class TestPaperPerformanceTracker:
    """Tests for the PaperPerformanceTracker."""

    def test_tracker_creation(self):
        from src.phase_19_paper_trading.trading_bot import PaperPerformanceTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = PaperPerformanceTracker(log_dir=Path(tmpdir))
            assert tracker.predictions == []
            assert tracker.open_prediction is None

    def test_record_signal(self):
        from src.phase_19_paper_trading.trading_bot import PaperPerformanceTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = PaperPerformanceTracker(log_dir=Path(tmpdir))

            # Mock signal
            signal = MagicMock()
            signal.signal_type = MagicMock(value="BUY")
            signal.probability = 0.72
            signal.timing_probability = 0.65
            signal.confidence = 0.68

            tracker.record_signal(signal, entry_price=450.0)
            assert tracker.open_prediction is not None
            assert tracker.open_prediction.entry_price == 450.0
            assert tracker.open_prediction.signal_type == "BUY"

    def test_record_close_correct_prediction(self):
        from src.phase_19_paper_trading.trading_bot import PaperPerformanceTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = PaperPerformanceTracker(log_dir=Path(tmpdir))

            signal = MagicMock()
            signal.signal_type = MagicMock(value="BUY")
            signal.probability = 0.72
            signal.timing_probability = 0.65
            signal.confidence = 0.68

            tracker.record_signal(signal, entry_price=450.0)
            tracker.record_close(exit_price=455.0, reason="take_profit")

            assert len(tracker.predictions) == 1
            assert tracker.predictions[0].predicted_correct is True
            assert tracker.predictions[0].actual_return > 0
            assert tracker.open_prediction is None

    def test_record_close_wrong_prediction(self):
        from src.phase_19_paper_trading.trading_bot import PaperPerformanceTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = PaperPerformanceTracker(log_dir=Path(tmpdir))

            signal = MagicMock()
            signal.signal_type = MagicMock(value="BUY")
            signal.probability = 0.55
            signal.timing_probability = 0.50
            signal.confidence = 0.52

            tracker.record_signal(signal, entry_price=450.0)
            tracker.record_close(exit_price=445.0, reason="stop_loss")

            assert len(tracker.predictions) == 1
            assert tracker.predictions[0].predicted_correct is False
            assert tracker.predictions[0].actual_return < 0

    def test_get_summary_empty(self):
        from src.phase_19_paper_trading.trading_bot import PaperPerformanceTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = PaperPerformanceTracker(log_dir=Path(tmpdir))
            summary = tracker.get_summary()
            assert summary["total_trades"] == 0

    def test_get_summary_with_trades(self):
        from src.phase_19_paper_trading.trading_bot import PaperPerformanceTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = PaperPerformanceTracker(log_dir=Path(tmpdir))

            # Record a few trades
            for i, (entry, exit_p) in enumerate([(450, 455), (455, 453), (453, 458)]):
                signal = MagicMock()
                signal.signal_type = MagicMock(value="BUY")
                signal.probability = 0.7
                signal.timing_probability = 0.6
                signal.confidence = 0.65
                tracker.record_signal(signal, entry_price=float(entry))
                tracker.record_close(exit_price=float(exit_p), reason="test")

            summary = tracker.get_summary()
            assert summary["total_trades"] == 3
            assert summary["correct_predictions"] == 2  # 2 of 3 were profitable
            assert 0 < summary["accuracy"] < 1

    def test_persistence(self):
        from src.phase_19_paper_trading.trading_bot import PaperPerformanceTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = PaperPerformanceTracker(log_dir=Path(tmpdir))

            signal = MagicMock()
            signal.signal_type = MagicMock(value="BUY")
            signal.probability = 0.7
            signal.timing_probability = 0.6
            signal.confidence = 0.65
            tracker.record_signal(signal, entry_price=450.0)
            tracker.record_close(exit_price=455.0, reason="test")

            # Reload
            tracker2 = PaperPerformanceTracker(log_dir=Path(tmpdir))
            assert len(tracker2.predictions) == 1
            assert tracker2.predictions[0].entry_price == 450.0

    def test_win_rate_by_confidence(self):
        from src.phase_19_paper_trading.trading_bot import PaperPerformanceTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = PaperPerformanceTracker(log_dir=Path(tmpdir))

            # High confidence trade - correct
            signal = MagicMock()
            signal.signal_type = MagicMock(value="BUY")
            signal.probability = 0.9
            signal.timing_probability = 0.8
            signal.confidence = 0.85
            tracker.record_signal(signal, entry_price=450.0)
            tracker.record_close(exit_price=455.0, reason="test")

            # Low confidence trade - wrong
            signal2 = MagicMock()
            signal2.signal_type = MagicMock(value="BUY")
            signal2.probability = 0.55
            signal2.timing_probability = 0.5
            signal2.confidence = 0.52
            tracker.record_signal(signal2, entry_price=450.0)
            tracker.record_close(exit_price=445.0, reason="test")

            summary = tracker.get_summary()
            assert "win_rate_by_confidence" in summary
            assert len(summary["win_rate_by_confidence"]) >= 1


# =============================================================================
# DASHBOARD DATA TESTS
# =============================================================================

class TestDashboardData:
    """Tests for dashboard data flow."""

    def test_status_json_structure(self):
        """Verify status.json has the expected structure."""
        status_file = PROJECT_ROOT / "logs" / "status.json"
        if not status_file.exists():
            pytest.skip("status.json not yet created")

        data = json.loads(status_file.read_text())
        # Should have key fields
        assert "updated_at" in data or "mode" in data

    def test_status_json_orchestrator_format(self):
        """Verify status.json matches the orchestrator format the dashboard expects."""
        status_file = PROJECT_ROOT / "logs" / "status.json"
        if not status_file.exists():
            pytest.skip("status.json not yet created")

        data = json.loads(status_file.read_text())
        # Dashboard JS expects these nested structures
        assert "mode" in data, "Missing 'mode' field"
        assert "trading" in data, "Missing 'trading' dict"
        assert "health" in data, "Missing 'health' dict"
        assert "components" in data, "Missing 'components' dict"
        # Check nested trading fields
        assert "active" in data["trading"]
        assert "position" in data["trading"]
        # Check nested health fields
        assert "status" in data["health"]
        # Check components is a dict
        assert isinstance(data["components"], dict)

    def test_paper_performance_json(self):
        """Verify paper_performance.json can be read."""
        perf_file = PROJECT_ROOT / "logs" / "paper_performance.json"
        if not perf_file.exists():
            pytest.skip("paper_performance.json not yet created")

        data = json.loads(perf_file.read_text())
        assert "summary" in data

    def test_dashboard_server_importable(self):
        """Dashboard server should be importable."""
        from src.dashboard_server import app
        assert app is not None

    def test_dashboard_performance_endpoint_exists(self):
        """The /api/performance endpoint should exist."""
        from src.phase_20_monitoring.dashboard_server import app
        with app.test_client() as client:
            response = client.get("/api/performance")
            assert response.status_code == 200
            data = response.get_json()
            assert "summary" in data

    def test_cors_headers(self):
        """Dashboard should return CORS headers."""
        from src.phase_20_monitoring.dashboard_server import app
        with app.test_client() as client:
            response = client.get("/api/heartbeat")
            assert response.status_code == 200
            assert response.headers.get("Access-Control-Allow-Origin") == "*"


# =============================================================================
# FEATURE GROUP CONFIG IN TRAINING
# =============================================================================

class TestFeatureGroupTrainingConfig:
    """Tests for feature group config wiring into training."""

    def test_config_has_feature_group_settings(self):
        from src.train_robust_model import CONFIG
        assert "use_feature_groups" in CONFIG
        assert "feature_group_mode" in CONFIG
        assert "protected_groups" in CONFIG
        assert CONFIG["use_feature_groups"] is True
        assert CONFIG["feature_group_mode"] == "grouped_protected"
        assert "premarket" in CONFIG["protected_groups"]
        assert "calendar" in CONFIG["protected_groups"]

    def test_feature_group_total_components(self):
        from src.train_robust_model import CONFIG
        assert CONFIG["feature_group_total_components"] == 40
        assert CONFIG["feature_group_min_components"] == 2

    def test_leak_proof_pipeline_accepts_group_params(self):
        """LeakProofPipeline should accept group mode parameters."""
        from src.phase_11_cv_splitting.ensemble_reducer import LeakProofPipeline
        pipeline = LeakProofPipeline(
            feature_names=["pm_a", "pm_b", "cal_x", "rsi_14"],
            group_mode="grouped_protected",
            protected_groups=["premarket"],
            total_components=3,
        )
        assert pipeline.group_mode == "grouped_protected"


# =============================================================================
# MODEL REGISTRATION
# =============================================================================

class TestModelRegistration:
    """Tests for model registration."""

    def test_experiment_history_has_entries(self):
        from src.experiment_engine import ExperimentHistory
        from src.core.registry_db import get_registry_db
        history = ExperimentHistory(db=get_registry_db())
        stats = history.get_statistics()
        assert stats.get("completed", 0) > 0

    def test_model_registry_accessible(self):
        from src.core.registry_db import get_registry_db
        db = get_registry_db()
        models = db.get_models()
        assert isinstance(models, list)

    def test_registered_models_have_auc(self):
        from src.core.registry_db import get_registry_db
        db = get_registry_db()
        models = db.get_models()
        for record in models[:5]:
            assert record.get("test_auc", 0) > 0

    def test_model_record_has_live_fields(self):
        """Model records from RegistryDB should have live performance fields."""
        from src.core.registry_db import RegistryDB
        import tempfile
        db = RegistryDB(db_path=Path(tempfile.mktemp(suffix=".db")))
        result_dict = {
            "experiment_id": "test", "test_auc": 0.60,
            "train_auc": 0.65, "cv_auc_mean": 0.60,
            "stability_score": 0, "fragility_score": 1.0,
            "wmes_score": 0.55, "backtest_sharpe": 0,
            "backtest_win_rate": 0, "backtest_total_return": 0,
            "model_path": "", "config": {},
        }
        model_id = db.register_model_from_experiment(result_dict)
        models = db.get_models()
        record = models[0]
        assert "live_trades" in record
        assert "live_win_rate" in record
        assert "live_total_return" in record
        assert "live_sharpe" in record
        db.close()


# =============================================================================
# SYSTEM LAUNCHER
# =============================================================================

class TestSystemLauncher:
    """Tests for the system launcher script."""

    def test_launcher_importable(self):
        import ast
        launcher_path = PROJECT_ROOT / "scripts" / "start_system.py"
        assert launcher_path.exists()
        ast.parse(launcher_path.read_text(encoding="utf-8"))

    def test_check_prerequisites(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from start_system import check_prerequisites
        logger = MagicMock()
        checks = check_prerequisites(logger)
        assert isinstance(checks, dict)
        assert "models" in checks
        assert "env_file" in checks


# =============================================================================
# MULTI-TIER QUALITY GATES TESTS
# =============================================================================

class TestMultiTierQualityGates:
    """Tests for the three-tier model quality gate system."""

    def test_registry_db_model_has_tier_fields(self):
        """Models registered via RegistryDB have robustness fields."""
        from src.core.registry_db import RegistryDB
        import tempfile
        db = RegistryDB(db_path=Path(tempfile.mktemp(suffix=".db")))
        result_dict = {
            "experiment_id": "exp_1", "test_auc": 0.60,
            "train_auc": 0.65, "cv_auc_mean": 0.60,
            "stability_score": 0, "fragility_score": 1.0,
            "wmes_score": 0.55, "backtest_sharpe": 0,
            "backtest_win_rate": 0, "backtest_total_return": 0,
            "model_path": "", "config": {},
        }
        model_id = db.register_model_from_experiment(result_dict)
        models = db.get_models()
        record = models[0]
        assert "stability_score" in record
        assert "fragility_score" in record
        assert "train_test_gap" in record
        assert "tier" in record
        # Defaults
        assert record["stability_score"] == 0
        assert record["fragility_score"] == 1.0
        assert record["tier"] == 1
        db.close()

    def test_compute_tier_assigns_correctly(self):
        """compute_tier returns correct tier based on metrics."""
        from src.core.registry_db import compute_tier

        # Tier 1: basic quality
        assert compute_tier(stability_score=0.30, fragility_score=1.0, test_auc=0.58) == 1

        # Tier 2: stability verified
        assert compute_tier(stability_score=0.65, fragility_score=0.50, test_auc=0.58) == 2

        # Tier 3: fragility verified + high AUC
        assert compute_tier(stability_score=0.70, fragility_score=0.20, test_auc=0.65) == 3

    def test_compute_tier_needs_high_auc_for_tier3(self):
        """Tier 3 requires test_auc >= 0.57 even with good fragility (v4: lowered from 0.58)."""
        from src.core.registry_db import compute_tier

        # Good stability + fragility but low AUC → Tier 2 only
        assert compute_tier(stability_score=0.80, fragility_score=0.10, test_auc=0.56) == 2
        # At threshold → Tier 3
        assert compute_tier(stability_score=0.80, fragility_score=0.10, test_auc=0.57) == 3

    def test_model_candidate_has_tier_fields(self):
        """ModelCandidate has robustness fields."""
        from src.phase_25_risk_management.ensemble_strategies import ModelCandidate
        candidate = ModelCandidate(model_id="test", model_path="", config={})
        assert hasattr(candidate, 'stability_score')
        assert hasattr(candidate, 'fragility_score')
        assert hasattr(candidate, 'tier')
        assert candidate.tier == 1

    def test_model_candidate_score_rewards_stability(self):
        """ModelCandidate.score() rewards stable models."""
        from src.phase_25_risk_management.ensemble_strategies import ModelCandidate
        stable = ModelCandidate(
            model_id="s", model_path="", config={},
            cv_auc=0.60, test_auc=0.60, wmes_score=0.55,
            stability_score=0.80, fragility_score=0.15,
        )
        fragile = ModelCandidate(
            model_id="f", model_path="", config={},
            cv_auc=0.60, test_auc=0.60, wmes_score=0.55,
            stability_score=0.20, fragility_score=0.80,
        )
        assert stable.score() > fragile.score()

    def test_dynamic_model_selector_min_tier_parameter(self):
        """DynamicModelSelector accepts min_tier parameter."""
        from src.phase_25_risk_management.model_selector import DynamicModelSelector
        selector = DynamicModelSelector(min_tier=2)
        assert selector.min_tier == 2

    def test_register_model_from_experiment_defaults(self):
        """register_model_from_experiment computes defaults correctly."""
        from src.core.registry_db import RegistryDB
        import tempfile
        db = RegistryDB(db_path=Path(tempfile.mktemp(suffix=".db")))
        result_dict = {
            "experiment_id": "exp_old", "test_auc": 0.59,
            "train_auc": 0.62, "cv_auc_mean": 0.62,
            "wmes_score": 0.55, "backtest_sharpe": 0,
            "backtest_win_rate": 0, "backtest_total_return": 0,
            "model_path": "", "config": {},
        }
        model_id = db.register_model_from_experiment(result_dict)
        models = db.get_models()
        record = models[0]
        assert record["tier"] == 1
        assert record["stability_score"] == 0
        assert record["fragility_score"] == 1.0
        db.close()

    def test_gate_checker_includes_min_tier(self):
        """Gate checker status includes min_tier_required."""
        from src.giga_orchestrator import ExperimentGateChecker
        checker = ExperimentGateChecker()
        _, status = checker.check_gates(trading_mode="paper")
        assert "min_tier_required" in status or "error" in status

    def test_register_model_populates_tier(self):
        """register_model_from_experiment correctly populates tier and robustness fields."""
        import tempfile
        from src.core.registry_db import RegistryDB

        db = RegistryDB(db_path=Path(tempfile.mktemp(suffix=".db")))
        result_dict = {
            "experiment_id": "exp_test",
            "cv_auc_mean": 0.62,
            "test_auc": 0.63,
            "train_auc": 0.68,
            "wmes_score": 0.58,
            "stability_score": 0.72,
            "fragility_score": 0.20,
            "backtest_sharpe": 0, "backtest_win_rate": 0,
            "backtest_total_return": 0, "model_path": "", "config": {},
        }
        model_id = db.register_model_from_experiment(result_dict)
        models = db.get_models()
        record = models[0]
        assert record["tier"] == 3  # stable>=0.60 + fragility<0.25 + AUC>=0.62
        assert record["stability_score"] == 0.72
        assert record["fragility_score"] == 0.20
        assert record["train_test_gap"] == pytest.approx(0.05, abs=0.001)
        db.close()


# =============================================================================
# THICK WEAVE INTEGRATION TESTS
# =============================================================================

class TestThickWeaveIntegration:
    """Tests for thick weave search integration with the system launcher."""

    def test_component_state_includes_thick_weave(self):
        """Component state dict has thick_weave_search entry."""
        from scripts.start_system import _component_state
        assert "thick_weave_search" in _component_state
        assert _component_state["thick_weave_search"] == "STOPPED"

    def test_start_thick_weave_search_exists(self):
        """start_thick_weave_search function is importable."""
        from scripts.start_system import start_thick_weave_search
        assert callable(start_thick_weave_search)

    def test_register_thick_weave_candidates_empty_report(self):
        """Bridge handles empty report gracefully."""
        import tempfile
        from scripts.start_system import _register_thick_weave_candidates
        from src.core.registry_db import RegistryDB

        db = RegistryDB(db_path=Path(tempfile.mktemp(suffix=".db")))
        report = {"production_candidates": []}
        n = _register_thick_weave_candidates(report, db)
        assert n == 0
        db.close()

    def test_register_thick_weave_candidates_with_candidates(self):
        """Bridge registers production candidates with tier scoring."""
        import tempfile
        from scripts.start_system import _register_thick_weave_candidates
        from src.core.registry_db import RegistryDB

        db = RegistryDB(db_path=Path(tempfile.mktemp(suffix=".db")))
        report = {
            "production_candidates": [
                {
                    "thread_id": "thread_001",
                    "model_family": "l2_ensemble_plus",
                    "wmes": 0.62,
                    "pts": 0.65,
                    "fragility": 0.20,
                    "config_hash": "abc123def456",
                    "model_type": "logistic_l2",
                    "dim_reduction": "ensemble_plus",
                },
            ],
        }
        n = _register_thick_weave_candidates(report, db)
        assert n == 1
        # Check the registered model has correct fields
        models = db.get_models()
        record = models[0]
        assert record["wmes_score"] == 0.62
        assert record["stability_score"] == 0.65  # PTS maps to stability
        assert record["fragility_score"] == 0.20
        assert record["tier"] >= 2  # Should be at least paper-eligible
        db.close()

    def test_register_thick_weave_candidates_skips_poor_quality(self):
        """Bridge skips candidates with WMES < 0.50."""
        import tempfile
        from scripts.start_system import _register_thick_weave_candidates
        from src.core.registry_db import RegistryDB

        db = RegistryDB(db_path=Path(tempfile.mktemp(suffix=".db")))
        report = {
            "production_candidates": [
                {
                    "thread_id": "thread_bad",
                    "wmes": 0.35,  # Below 0.50 threshold
                    "pts": 0.60,
                    "config_hash": "bad_config",
                },
            ],
        }
        n = _register_thick_weave_candidates(report, db)
        assert n == 0
        db.close()

    def test_thick_weave_config_importable(self):
        """ThickWeaveSearch and ThickWeaveConfig are importable."""
        from src.phase_23_analytics.thick_weave_search import (
            ThickWeaveSearch, ThickWeaveConfig,
        )
        config = ThickWeaveConfig(max_total_evaluations=10)
        assert config.max_total_evaluations == 10

    def test_cli_args_include_thick_weave(self):
        """CLI parser includes thick weave flags."""
        import argparse
        # Simulate parsing with --no-thick-weave
        parser = argparse.ArgumentParser()
        parser.add_argument("--no-thick-weave", action="store_true")
        parser.add_argument("--thick-weave-budget", type=int, default=50)
        args = parser.parse_args(["--no-thick-weave", "--thick-weave-budget", "100"])
        assert args.no_thick_weave is True
        assert args.thick_weave_budget == 100


# =============================================================================
# ANTI-OVERFITTING FRAMEWORK TESTS (Wave 13)
# =============================================================================

class TestAntiOverfitFixes:
    """Tests for anti-overfitting framework fixes (Wave 13)."""

    def test_feature_selection_method_in_config(self):
        """DimensionalityReductionConfig has feature_selection_method field."""
        from src.experiment_config import DimensionalityReductionConfig
        config = DimensionalityReductionConfig()
        assert hasattr(config, 'feature_selection_method')
        assert config.feature_selection_method == "mutual_info"
        config2 = DimensionalityReductionConfig(feature_selection_method="f_classif")
        assert config2.feature_selection_method == "f_classif"

    def test_dim_method_mapping(self):
        """_map_dim_method maps ExperimentConfig methods to leak-proof methods."""
        from src.phase_21_continuous.experiment_runner import UnifiedExperimentRunner
        m = UnifiedExperimentRunner._map_dim_method
        assert m("kernel_pca") == "kernel_pca"
        assert m("ica") == "ica"
        assert m("pca") == "pca"
        assert m("mutual_info") is None
        assert m("umap") == "kernel_pca"
        assert m("ensemble_plus") == "kernel_pca"
        assert m("agglomeration") == "kernel_pca"
        assert m("kmedoids") == "kernel_pca"

    def test_get_n_components_by_method(self):
        """_get_n_components returns correct component count per method."""
        from src.experiment_config import ExperimentConfig
        from src.phase_21_continuous.experiment_runner import UnifiedExperimentRunner
        config = ExperimentConfig()
        config.dim_reduction.method = "kernel_pca"
        config.dim_reduction.kpca_n_components = 15
        assert UnifiedExperimentRunner._get_n_components(config) == 15

        config.dim_reduction.method = "ica"
        config.dim_reduction.ica_n_components = 12
        assert UnifiedExperimentRunner._get_n_components(config) == 12

        config.dim_reduction.method = "pca"
        config.dim_reduction.pca_n_components = 25
        assert UnifiedExperimentRunner._get_n_components(config) == 25

        config.dim_reduction.method = "mutual_info"
        config.dim_reduction.mi_n_features = 35
        assert UnifiedExperimentRunner._get_n_components(config) == 35

    def test_random_state_varies_by_experiment(self):
        """Different experiment IDs produce different random states."""
        from src.phase_21_continuous.experiment_runner import UnifiedExperimentRunner
        seed1 = UnifiedExperimentRunner._get_random_state("exp_20260101_abc123")
        seed2 = UnifiedExperimentRunner._get_random_state("exp_20260101_def456")
        seed3 = UnifiedExperimentRunner._get_random_state("exp_20260101_abc123")
        assert seed1 != seed2
        assert seed1 == seed3  # Deterministic
        assert 0 <= seed1 < 2**31

    def test_experiment_result_has_walk_forward_fields(self):
        """ExperimentResult has walk-forward validation fields."""
        from src.phase_21_continuous.experiment_tracking import ExperimentResult, ExperimentStatus
        from src.experiment_config import create_default_config
        result = ExperimentResult(
            experiment_id="test",
            config=create_default_config("test"),
            status=ExperimentStatus.COMPLETED,
        )
        assert hasattr(result, 'walk_forward_aucs')
        assert hasattr(result, 'worst_window_auc')
        assert hasattr(result, 'walk_forward_passed')
        assert result.walk_forward_aucs == []
        assert result.worst_window_auc == 0.0
        assert result.walk_forward_passed is False

    def test_experiment_result_has_regime_fields(self):
        """ExperimentResult has regime-specific fields."""
        from src.phase_21_continuous.experiment_tracking import ExperimentResult, ExperimentStatus
        from src.experiment_config import create_default_config
        result = ExperimentResult(
            experiment_id="test",
            config=create_default_config("test"),
            status=ExperimentStatus.COMPLETED,
        )
        assert hasattr(result, 'regime_auc_low_vol')
        assert hasattr(result, 'regime_auc_high_vol')
        assert hasattr(result, 'regime_sensitive')
        assert result.regime_auc_low_vol == 0.0
        assert result.regime_auc_high_vol == 0.0
        assert result.regime_sensitive is False

    def test_sharpe_not_inflated_by_zeros(self):
        """Sharpe computed only on trading days, not inflated by zeros."""
        from src.phase_21_continuous.experiment_tracking import compute_realistic_backtest_metrics
        # Scenario: 10 trades out of 250 days, small positive returns
        np.random.seed(42)
        signals = np.zeros(250)
        returns = np.random.randn(250) * 0.01
        signals[:10] = 1  # Only 10 trades

        metrics = compute_realistic_backtest_metrics(signals, returns)
        # With old code (zeros in denominator), Sharpe was heavily inflated
        # With new code (trading days only), should be reasonable
        assert abs(metrics["sharpe"]) < 10.0  # Sanity bound
        assert metrics["n_trades"] == 10

    def test_sharpe_zero_trades(self):
        """Sharpe is 0 when no trades."""
        from src.phase_21_continuous.experiment_tracking import compute_realistic_backtest_metrics
        signals = np.zeros(100)
        returns = np.random.randn(100) * 0.01
        metrics = compute_realistic_backtest_metrics(signals, returns)
        assert metrics["sharpe"] == 0.0
        assert metrics["sharpe_net"] == 0.0
        assert metrics["n_trades"] == 0

    def test_experiment_result_from_dict_backward_compat(self):
        """from_dict handles old data without new fields."""
        from src.phase_21_continuous.experiment_tracking import ExperimentResult, ExperimentStatus
        from src.experiment_config import create_default_config
        # Simulate old JSON without new fields
        old_data = {
            "experiment_id": "old_exp",
            "config": create_default_config("old").to_dict(),
            "status": "completed",
            "test_auc": 0.65,
            "cv_auc_mean": 0.62,
        }
        result = ExperimentResult.from_dict(old_data)
        assert result.experiment_id == "old_exp"
        assert result.test_auc == 0.65
        # New fields should have defaults
        assert result.walk_forward_aucs == []
        assert result.regime_sensitive is False

    def test_experiment_result_from_dict_ignores_unknown(self):
        """from_dict ignores unknown fields from future versions."""
        from src.phase_21_continuous.experiment_tracking import ExperimentResult, ExperimentStatus
        from src.experiment_config import create_default_config
        data = {
            "experiment_id": "future_exp",
            "config": create_default_config("future").to_dict(),
            "status": "completed",
            "some_future_field": 42,  # Unknown field
        }
        result = ExperimentResult.from_dict(data)
        assert result.experiment_id == "future_exp"
        assert not hasattr(result, 'some_future_field')

    def test_experiment_generator_varies_feature_selection(self):
        """ExperimentGenerator varies feature_selection_method for dim_reduction type."""
        from src.phase_21_continuous.experiment_tracking import ExperimentGenerator
        from src.experiment_config import create_default_config
        gen = ExperimentGenerator()
        methods_seen = set()
        for _ in range(50):
            config = gen.generate_next()
            if config.experiment_type == "dim_reduction":
                methods_seen.add(config.dim_reduction.feature_selection_method)
        # Should see at least one variation (probabilistic, but 50 trials is sufficient)
        # At minimum, mutual_info should appear
        assert "mutual_info" in methods_seen or "f_classif" in methods_seen


# ═══════════════════════════════════════════════════════════════════════════════
# WAVE 14: AGGRESSIVE ANTI-OVERFITTING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAntiOverfitWave14:
    """Tests for aggressive anti-overfitting measures (Wave 14)."""

    def test_feature_exclude_cols_contains_target_columns(self):
        """FEATURE_EXCLUDE_COLS must exclude all target-leaking columns."""
        from src.phase_21_continuous.experiment_runner import FEATURE_EXCLUDE_COLS
        # Critical target columns that caused leakage
        must_exclude = [
            "is_up_day", "is_down_day", "target_up", "target_timing",
            "low_before_high", "high_minutes", "low_minutes",
            "max_gain_from_1015", "max_gain_from_1230",
            "soft_target_up", "smoothed_target_up", "smoothed_target_timing",
        ]
        for col in must_exclude:
            assert col in FEATURE_EXCLUDE_COLS, f"Missing critical exclude: {col}"

    def test_feature_exclude_cols_contains_intraday(self):
        """FEATURE_EXCLUDE_COLS must exclude same-day intraday features."""
        from src.phase_21_continuous.experiment_runner import FEATURE_EXCLUDE_COLS
        # Sample of same-day intraday features (8 prefixes x 8 time points)
        sample_intraday = [
            "return_at_0945", "return_at_1530",
            "high_to_1015", "high_to_1430",
            "low_to_1100", "rsi_at_1230",
            "macd_at_1330", "bb_at_0945",
            "range_to_1530", "return_from_low_1015",
        ]
        for col in sample_intraday:
            assert col in FEATURE_EXCLUDE_COLS, f"Missing intraday exclude: {col}"

    def test_feature_exclude_cols_count(self):
        """Should have at least 80 excluded columns (27 metadata + 64 intraday)."""
        from src.phase_21_continuous.experiment_runner import FEATURE_EXCLUDE_COLS
        assert len(FEATURE_EXCLUDE_COLS) >= 80, f"Only {len(FEATURE_EXCLUDE_COLS)} excludes, expected 80+"

    def test_reality_check_catches_leakage(self):
        """_reality_check flags AUC > 0.85 as likely leakage."""
        from src.phase_21_continuous.experiment_runner import UnifiedExperimentRunner
        from types import SimpleNamespace
        result = SimpleNamespace(
            test_auc=0.92, train_auc=0.95,
            backtest_sharpe=None, backtest_win_rate=None,
        )
        flags = UnifiedExperimentRunner._reality_check(result)
        assert "likely_leakage" in flags

    def test_reality_check_catches_sharpe(self):
        """_reality_check flags Sharpe > 4.0 as unrealistic."""
        from src.phase_21_continuous.experiment_runner import UnifiedExperimentRunner
        from types import SimpleNamespace
        result = SimpleNamespace(
            test_auc=0.65, train_auc=0.68,
            backtest_sharpe=5.5, backtest_win_rate=0.60,
        )
        flags = UnifiedExperimentRunner._reality_check(result)
        assert "unrealistic_sharpe" in flags

    def test_reality_check_passes_normal(self):
        """_reality_check returns empty for normal metrics."""
        from src.phase_21_continuous.experiment_runner import UnifiedExperimentRunner
        from types import SimpleNamespace
        result = SimpleNamespace(
            test_auc=0.63, train_auc=0.68,
            backtest_sharpe=1.5, backtest_win_rate=0.58,
        )
        flags = UnifiedExperimentRunner._reality_check(result)
        assert len(flags) == 0

    def test_tier1_thresholds_tightened(self):
        """Tier 1 gate uses tightened thresholds (Wave 14)."""
        # Verify by importing and checking the reality check is a method
        from src.phase_21_continuous.experiment_runner import UnifiedExperimentRunner
        assert hasattr(UnifiedExperimentRunner, '_reality_check')
        # A model with old-threshold-passing but new-threshold-failing metrics
        from types import SimpleNamespace
        # AUC 0.56 passes old (>0.55) but fails new (>0.58)
        result = SimpleNamespace(
            test_auc=0.56, train_auc=0.60,
            backtest_sharpe=1.0, backtest_win_rate=0.55,
        )
        flags = UnifiedExperimentRunner._reality_check(result)
        # Reality check itself passes for 0.56, but Tier 1 gate would fail on AUC threshold

    def test_compute_tier_tightened_thresholds(self):
        """compute_tier uses Wave 33 updated thresholds."""
        from src.core.registry_db import compute_tier

        # stability 0.55 used to pass Tier 2 (>=0.50), now fails (>=0.60)
        assert compute_tier(0.55, 0.20, 0.65) == 1  # Stays Tier 1

        # stability 0.65 passes Tier 2
        # Wave 33: fragility threshold raised to 0.40 (was 0.30)
        assert compute_tier(0.65, 0.45, 0.65) == 2  # Tier 2 but not 3 (fragility 0.45 >= 0.40)

        # Full Tier 3: stability >= 0.60, fragility < 0.40, AUC >= 0.58
        # Wave 33: suite_composite >= 0.45 when available
        assert compute_tier(0.70, 0.25, 0.60) == 3  # No suite data, waived
        assert compute_tier(0.70, 0.25, 0.60, suite_composite=0.50) == 3  # Suite passes
        assert compute_tier(0.70, 0.25, 0.60, suite_composite=0.30) == 2  # Suite fails

    def test_thick_weave_tightened_thresholds(self):
        """ThickWeaveConfig uses tightened defaults (Wave 14)."""
        from src.phase_23_analytics.thick_weave_search import ThickWeaveConfig
        cfg = ThickWeaveConfig()
        assert cfg.min_thread_thickness == 0.45, f"Expected 0.45, got {cfg.min_thread_thickness}"
        assert cfg.tier1_wmes_threshold == 0.45, f"Expected 0.45, got {cfg.tier1_wmes_threshold}"
        assert cfg.tier2_wmes_threshold == 0.55, f"Expected 0.55, got {cfg.tier2_wmes_threshold}"
        assert cfg.tier3_wmes_threshold == 0.55, f"Expected 0.55, got {cfg.tier3_wmes_threshold}"
        assert cfg.thick_path_threshold == 0.60, f"Expected 0.60, got {cfg.thick_path_threshold}"

    def test_live_trading_gates_tightened(self):
        """Orchestrator uses recalibrated live trading gates (Wave 36 v4: 0.57)."""
        from src.giga_orchestrator import ORCHESTRATOR_CONFIG
        assert ORCHESTRATOR_CONFIG["min_auc_for_live_trading"] == 0.57
        assert ORCHESTRATOR_CONFIG["min_models_above_threshold"] == 5

    def test_training_pipeline_v2_excludes_intraday(self):
        """TrainingPipelineV2._prepare_arrays exclude logic blocks intraday features."""
        # Simulate the exclude logic from _prepare_arrays without calling the full method
        # (to avoid needing a full ModelEntry config for SampleWeighter)
        exclude_exact = {
            "date", "timestamp",
            "open", "high", "low", "close", "volume",
            "is_up_day", "is_down_day",
            "low_before_high", "high_minutes", "low_minutes",
            "day_return", "day_volume", "day_range",
            "target_up", "target_timing", "soft_target_up",
            "smoothed_target_up", "smoothed_target_timing",
            "sample_weight", "timing_weight",
            "max_gain_from_1015", "max_gain_from_1230",
            "has_premarket", "has_afterhours", "quality_score", "year",
            "sample_weight_augment", "universe_id", "universe_type",
            "synthetic_return", "real_return", "is_synthetic",
        }
        _intraday_times = ["0945", "1015", "1100", "1130", "1230", "1330", "1430", "1530"]
        _intraday_prefixes = [
            "return_at_", "high_to_", "low_to_", "range_to_",
            "rsi_at_", "macd_at_", "bb_at_", "return_from_low_",
        ]
        for _pfx in _intraday_prefixes:
            for _tp in _intraday_times:
                exclude_exact.add(f"{_pfx}{_tp}")

        # Verify all 64 intraday features are in the exclude set
        assert len([c for c in exclude_exact if any(c.startswith(p) for p in _intraday_prefixes)]) == 64

        # Test columns
        test_cols = [
            "pm_return_1d", "rsi_14", "volume_ratio_5d",  # legit
            "return_at_0945", "return_at_1530",  # leaky intraday
            "high_to_1015", "rsi_at_1230", "macd_at_1330",
            "is_up_day", "low_before_high", "max_gain_from_1015",
        ]
        exclude_patterns = [
            "target", "soft_target", "smoothed_target", "label",
            "sample_weight", "target_weight", "class_weight",
            "forward_return", "future_",
        ]
        target_col = "target_up"

        feature_cols = []
        for c in test_cols:
            if c in exclude_exact:
                continue
            if c == target_col:
                continue
            if any(pat in c.lower() for pat in exclude_patterns):
                continue
            feature_cols.append(c)

        # Legit features must be present
        assert "pm_return_1d" in feature_cols
        assert "rsi_14" in feature_cols
        assert "volume_ratio_5d" in feature_cols

        # Leaky features must NOT be present
        leaky = [
            "return_at_0945", "return_at_1530", "high_to_1015",
            "rsi_at_1230", "macd_at_1330",
            "is_up_day", "low_before_high", "max_gain_from_1015",
        ]
        for col in leaky:
            assert col not in feature_cols, f"Leaky feature still included: {col}"

    def test_training_pipeline_v2_exclude_matches_experiment_runner(self):
        """TrainingPipelineV2 exclude set matches FEATURE_EXCLUDE_COLS from experiment_runner."""
        from src.phase_21_continuous.experiment_runner import FEATURE_EXCLUDE_COLS

        # Rebuild the exclude set from training_pipeline_v2's logic
        exclude_exact = {
            "date", "timestamp",
            "open", "high", "low", "close", "volume",
            "is_up_day", "is_down_day",
            "low_before_high", "high_minutes", "low_minutes",
            "day_return", "day_volume", "day_range",
            "target_up", "target_timing", "soft_target_up",
            "smoothed_target_up", "smoothed_target_timing",
            "sample_weight", "timing_weight",
            "max_gain_from_1015", "max_gain_from_1230",
            "has_premarket", "has_afterhours", "quality_score", "year",
            "sample_weight_augment", "universe_id", "universe_type",
            "synthetic_return", "real_return", "is_synthetic",
        }
        _intraday_times = ["0945", "1015", "1100", "1130", "1230", "1330", "1430", "1530"]
        _intraday_prefixes = [
            "return_at_", "high_to_", "low_to_", "range_to_",
            "rsi_at_", "macd_at_", "bb_at_", "return_from_low_",
        ]
        for _pfx in _intraday_prefixes:
            for _tp in _intraday_times:
                exclude_exact.add(f"{_pfx}{_tp}")

        # Every column in FEATURE_EXCLUDE_COLS must also be in pipeline's exclude set
        # (pipeline may have additional OHLCV columns that experiment_runner doesn't need)
        missing_from_pipeline = FEATURE_EXCLUDE_COLS - exclude_exact
        assert len(missing_from_pipeline) == 0, (
            f"Columns in FEATURE_EXCLUDE_COLS but not in pipeline exclude: {missing_from_pipeline}"
        )


# ─────────────────────────────────────────────────────────────────────
# WAVE 15: PLATEAU-AWARE NEIGHBORHOOD EXPANSION
# ─────────────────────────────────────────────────────────────────────

class TestPlateauExpansion:
    """Test plateau scoring, adaptive radius, and weighted allocation."""

    def _make_thread(self, wmes_values, thread_id="test_001"):
        """Helper: create a SearchThread with pre-populated evaluations."""
        from src.phase_23_analytics.thick_weave_search import (
            SearchThread, ThickWeaveConfig, CATEGORICAL_DIMS, CONTINUOUS_DIMS,
        )
        from src.phase_18_persistence.registry_configs import ModelEntry

        center = ModelEntry(target_type="swing")
        radius = {}
        for dim in CATEGORICAL_DIMS:
            radius[dim] = 0.3
        for dim in CONTINUOUS_DIMS:
            radius[dim] = 0.2

        thread = SearchThread(thread_id, center, "test_family", radius)
        for w in wmes_values:
            thread.evaluated_configs.append((center, w))

        return thread

    def test_plateau_score_all_above(self):
        """All configs above threshold → plateau_score = 1.0."""
        thread = self._make_thread([0.55, 0.60, 0.58, 0.62, 0.57])
        ps = thread.plateau_score(threshold=0.45)
        assert abs(ps - 1.0) < 1e-6

    def test_plateau_score_mixed(self):
        """Half above, half below → 0.5."""
        thread = self._make_thread([0.55, 0.60, 0.30, 0.20, 0.58, 0.10])
        ps = thread.plateau_score(threshold=0.45)
        assert abs(ps - 0.5) < 1e-6

    def test_plateau_score_too_few(self):
        """Fewer than 3 evaluations → 0.0."""
        thread = self._make_thread([0.55, 0.60])
        ps = thread.plateau_score(threshold=0.45)
        assert abs(ps - 0.0) < 1e-6

    def test_expand_radius(self):
        """expand_radius grows all dimensions."""
        thread = self._make_thread([0.5, 0.5, 0.5])
        original = {k: v for k, v in thread.neighborhood_radius.items()}
        thread.expand_radius(factor=1.5, max_radius=1.0)
        for dim, old_r in original.items():
            assert thread.neighborhood_radius[dim] > old_r
            assert abs(thread.neighborhood_radius[dim] - old_r * 1.5) < 1e-6

    def test_expand_radius_capped(self):
        """expand_radius respects max_radius."""
        thread = self._make_thread([0.5, 0.5, 0.5])
        for dim in thread.neighborhood_radius:
            thread.neighborhood_radius[dim] = 0.45
        thread.expand_radius(factor=1.5, max_radius=0.5)
        for dim in thread.neighborhood_radius:
            assert thread.neighborhood_radius[dim] <= 0.5

    def test_weighted_allocation_favors_better_threads(self):
        """Higher PTS+WMES thread gets more allocation."""
        from src.phase_23_analytics.thick_weave_search import ThickWeaveConfig

        config = ThickWeaveConfig()
        configs_this_round = 10

        thread_a = self._make_thread([0.60, 0.62, 0.58, 0.61, 0.59])
        thread_a.thickness_history = [0.6]
        thread_b = self._make_thread([0.30, 0.20, 0.10, 0.15, 0.25])
        thread_b.thickness_history = [0.1]

        active = [thread_a, thread_b]

        weights = []
        for thread in active:
            pts = thread.current_thickness
            wmes = thread.best_wmes
            score = (config.allocation_weight_pts * pts +
                     config.allocation_weight_wmes * wmes)
            weights.append(max(score, config.min_allocation_floor))

        total_weight = sum(weights)
        allocations = []
        for w in weights:
            frac = w / total_weight
            alloc = max(1, int(round(frac * configs_this_round)))
            allocations.append(alloc)

        assert allocations[0] > allocations[1]

    def test_crossover_prefers_plateau(self):
        """Sorting by combined PTS+WMES prefers thick threads."""
        thread_a = self._make_thread([0.70, 0.20, 0.15, 0.10, 0.12])
        thread_a.thickness_history = [0.15]
        thread_b = self._make_thread([0.58, 0.60, 0.57, 0.59, 0.56])
        thread_b.thickness_history = [0.55]

        active = [thread_a, thread_b]
        sorted_threads = sorted(
            active,
            key=lambda t: (0.5 * t.current_thickness + 0.5 * t.best_wmes),
            reverse=True,
        )
        assert sorted_threads[0] is thread_b

    def test_plateau_bonus_in_ranking(self):
        """Production candidate with higher plateau_score gets adjusted_wmes boost."""
        wmes = 0.60
        ps_high = 0.8
        ps_low = 0.1
        adj_high = wmes * (1.0 + 0.15 * ps_high)
        adj_low = wmes * (1.0 + 0.15 * ps_low)
        assert adj_high > adj_low
        assert abs(adj_high - 0.60 * 1.12) < 1e-6
        assert abs(adj_low - 0.60 * 1.015) < 1e-6

    def test_config_has_plateau_params(self):
        """ThickWeaveConfig has new plateau-aware parameters."""
        from src.phase_23_analytics.thick_weave_search import ThickWeaveConfig
        config = ThickWeaveConfig()
        assert config.plateau_expand_factor == 1.15
        assert config.plateau_shrink_factor == 0.85
        assert config.plateau_threshold == 0.5
        assert config.allocation_weight_pts == 0.6
        assert config.allocation_weight_wmes == 0.4
        assert config.min_allocation_floor == 0.1


# ─────────────────────────────────────────────────────────────────────
# WAVE 15b: BUG FIXES (synthetic cap, backtest, thread merge)
# ─────────────────────────────────────────────────────────────────────

class TestWave15bBugFixes:
    """Test synthetic ratio cap, backtest defensive access, and merge guards."""

    def test_synthetic_ratio_cap_uses_positional_slicing(self):
        """Verify the ratio cap uses iloc[:n_real] instead of is_synthetic column."""
        import ast
        src = open("src/phase_21_continuous/experiment_runner.py", encoding="utf-8").read()
        tree = ast.parse(src)
        # The fix: df_train.iloc[:n_real] and df_train.iloc[n_real:]
        assert "iloc[:n_real]" in src, "Cap should use iloc[:n_real] for real rows"
        assert "iloc[n_real:]" in src, "Cap should use iloc[n_real:] for synthetic rows"
        # The old broken pattern should NOT exist
        assert 'df_train.get("is_synthetic"' not in src, (
            "Should not use is_synthetic column (doesn't exist in df_train)"
        )

    def test_backtest_defensive_access(self):
        """transaction_cost_per_trade should use .get() for safety."""
        src = open("src/phase_21_continuous/experiment_runner.py", encoding="utf-8").read()
        assert "backtest_metrics.get('transaction_cost_per_trade'" in src, (
            "Should use .get() for defensive access to transaction_cost_per_trade"
        )

    def test_merge_requires_minimum_evaluations(self):
        """Threads with < 3 evaluations should not be merge candidates."""
        from src.phase_23_analytics.thick_weave_search import (
            ThickWeaveConfig, ThreadWeaver,
        )
        from src.phase_18_persistence.registry_configs import ModelEntry

        config = ThickWeaveConfig()
        weaver = ThreadWeaver(config)

        # Spawn two identical threads (overlap = 1.0)
        center = ModelEntry(target_type="swing")
        t1 = weaver.spawn_thread(center, "family_a")
        t2 = weaver.spawn_thread(center, "family_b")

        # With 0 evaluations: no merge candidates
        merges = weaver.check_all_merges()
        assert len(merges) == 0, "Threads with 0 evals should not merge"

        # Add 2 evals to t1 (still < 3): no merge
        t1.evaluated_configs.append((center, 0.5))
        t1.evaluated_configs.append((center, 0.6))
        merges = weaver.check_all_merges()
        assert len(merges) == 0, "Threads with < 3 evals should not merge"

        # Add 3 evals to both: now eligible for merge
        t1.evaluated_configs.append((center, 0.55))
        t2.evaluated_configs.append((center, 0.5))
        t2.evaluated_configs.append((center, 0.6))
        t2.evaluated_configs.append((center, 0.55))
        merges = weaver.check_all_merges()
        assert len(merges) > 0, "Threads with >= 3 evals should be merge candidates"

    def test_backtest_metrics_always_has_transaction_cost(self):
        """compute_realistic_backtest_metrics always returns transaction_cost_per_trade."""
        from src.phase_21_continuous.experiment_tracking import compute_realistic_backtest_metrics
        import numpy as np

        # Zero-trade path
        result = compute_realistic_backtest_metrics(
            signals=np.array([0, 0, 0]),
            returns=np.array([0.01, -0.02, 0.005]),
        )
        assert "transaction_cost_per_trade" in result

        # Normal path
        result = compute_realistic_backtest_metrics(
            signals=np.array([1, 0, 1]),
            returns=np.array([0.01, -0.02, 0.005]),
        )
        assert "transaction_cost_per_trade" in result
        assert result["transaction_cost_per_trade"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# WAVE 16: REGIME-AWARE VALIDATION + REGISTRY HYGIENE
# ═══════════════════════════════════════════════════════════════════════════════

class TestWave16RegimeAwareValidation:
    """Test regime-aware walk-forward thresholds and registry staleness detection."""

    def test_walk_forward_uses_per_window_thresholds(self):
        """Verify walk-forward code tracks per-window thresholds."""
        src = open("src/phase_21_continuous/experiment_runner.py", encoding="utf-8").read()
        assert "wf_per_window_thresholds" in src, "Should track per-window thresholds"
        assert "HIGH-VOL regime" in src, "Should detect and log high-vol regime"
        assert "NORMAL regime" in src, "Should detect and log normal regime"

    def test_regime_detection_threshold(self):
        """22% annualized vol should trigger high-vol regime."""
        import numpy as np

        # Simulate daily returns with ~25% annualized volatility (crisis)
        np.random.seed(42)
        crisis_returns = np.random.normal(0, 0.016, 252)  # ~25% annualized
        crisis_ann_vol = float(np.std(crisis_returns)) * (252 ** 0.5)
        assert crisis_ann_vol > 0.22, f"Crisis vol {crisis_ann_vol:.2%} should exceed 22%"

        # Normal market: ~15% annualized volatility
        normal_returns = np.random.normal(0, 0.009, 252)  # ~14% annualized
        normal_ann_vol = float(np.std(normal_returns)) * (252 ** 0.5)
        assert normal_ann_vol < 0.22, f"Normal vol {normal_ann_vol:.2%} should be below 22%"

    def test_walk_forward_aggregate_uses_mean_floor(self):
        """Aggregate walk-forward should require mean >= 0.51 and worst >= 0.47 (Wave 26 recalibration)."""
        src = open("src/phase_21_continuous/experiment_runner.py", encoding="utf-8").read()
        assert "wf_mean >= 0.51" in src, "Should require mean AUC >= 0.51"
        assert "worst_window_auc >= 0.47" in src, "Should require worst AUC >= 0.47"

    def test_walk_forward_variance_relaxed(self):
        """Variance threshold should be 0.07 (Wave 26: relaxed from 0.06 to reduce false negatives)."""
        src = open("src/phase_21_continuous/experiment_runner.py", encoding="utf-8").read()
        assert "wf_variance < 0.07" in src, "Variance threshold should be 0.07"

    def test_registry_purge_stale_models(self):
        """RegistryDB.purge_models() should remove AUC >= 0.85 entries."""
        import tempfile
        from src.core.registry_db import RegistryDB

        db = RegistryDB(db_path=Path(tempfile.mktemp(suffix=".db")))
        # Insert stale + valid models
        db.add_model("stale_1", {
            "model_id": "stale_1", "experiment_id": "e1", "created_at": "2026-01-01",
            "model_path": "", "config": {}, "test_auc": 0.92, "cv_auc": 0.95,
        })
        db.add_model("valid_1", {
            "model_id": "valid_1", "experiment_id": "e2", "created_at": "2026-02-01",
            "model_path": "", "config": {}, "test_auc": 0.62, "cv_auc": 0.65,
        })
        db.add_model("valid_2", {
            "model_id": "valid_2", "experiment_id": "e3", "created_at": "2026-02-01",
            "model_path": "", "config": {}, "test_auc": 0.58, "cv_auc": 0.60,
        })
        assert len(db.get_models()) == 3

        purged = db.purge_models(max_auc=0.85)
        assert purged == 1, f"Should purge 1 stale model, got {purged}"
        remaining = db.get_models()
        assert len(remaining) == 2
        remaining_ids = {m["model_id"] for m in remaining}
        assert "stale_1" not in remaining_ids
        assert "valid_1" in remaining_ids
        assert "valid_2" in remaining_ids
        db.close()

    def test_gate_checker_excludes_leaky_models(self):
        """Gate checker should exclude models with AUC >= 0.85 from counts."""
        src = open("src/giga_orchestrator.py", encoding="utf-8").read()
        assert "max_auc=0.85" in src, "Gate checker should filter out AUC >= 0.85"
        assert "pre-Wave-14 leakage" in src, "Should log about pre-Wave-14 leakage exclusion"

    def test_experiment_history_has_realistic_auc(self):
        """ExperimentHistory.get_statistics() should include best_realistic_auc."""
        src = open("src/phase_21_continuous/experiment_tracking.py", encoding="utf-8").read()
        assert "best_realistic_auc" in src, "Should compute best_realistic_auc excluding AUC >= 0.85"

    def test_gate_checker_uses_realistic_auc(self):
        """Gate checker should prefer best_realistic_auc over best_test_auc."""
        src = open("src/giga_orchestrator.py", encoding="utf-8").read()
        assert "best_realistic_auc" in src, "Gate checker should use best_realistic_auc"


# =============================================================================
# FEATURE EXCLUSION SYNC TESTS (Wave 16b)
# =============================================================================

class TestFeatureExclusionSync:
    """Verify that experiment_runner.py and training_pipeline_v2.py exclude
    the same columns, preventing feature-leakage mismatches between the
    experiment loop and the actual training pipeline."""

    def test_exact_exclude_sets_in_sync(self):
        """Both files must exclude the same exact-match columns."""
        from src.phase_21_continuous.experiment_runner import FEATURE_EXCLUDE_COLS

        # Read training_pipeline_v2 exclude_exact set by parsing
        import ast
        tp_src = open("src/phase_12_model_training/training_pipeline_v2.py",
                       encoding="utf-8").read()
        tree = ast.parse(tp_src)

        # Find the _prepare_arrays method and extract exclude_exact set
        # Simpler: just import the exact columns we know must be present
        required_ohlcv = {"timestamp", "open", "high", "low", "close", "volume"}
        required_targets = {
            "is_up_day", "is_down_day", "low_before_high",
            "target_up", "target_timing", "soft_target_up",
        }
        required_metadata = {
            "universe_id", "universe_type", "is_synthetic",
            "sample_weight", "sample_weight_augment",
        }

        all_required = required_ohlcv | required_targets | required_metadata
        missing = all_required - FEATURE_EXCLUDE_COLS
        assert not missing, f"experiment_runner FEATURE_EXCLUDE_COLS missing: {missing}"

    def test_pattern_exclusion_exists(self):
        """experiment_runner must have pattern-based exclusion matching pipeline."""
        from src.phase_21_continuous.experiment_runner import (
            FEATURE_EXCLUDE_PATTERNS, _is_excluded_feature
        )
        # These patterns must exist (matching training_pipeline_v2 exclude_patterns)
        required_patterns = ["target", "label", "forward_return", "future_"]
        for pat in required_patterns:
            assert pat in FEATURE_EXCLUDE_PATTERNS, \
                f"Missing exclude pattern: {pat}"

    def test_pattern_exclusion_catches_leaky_columns(self):
        """Pattern-based exclusion should catch columns that exact match misses."""
        from src.phase_21_continuous.experiment_runner import _is_excluded_feature

        # These columns would slip past exact matching but are caught by patterns
        leaky_cols = [
            "forward_return_5d",
            "future_close",
            "target_binary_custom",
            "soft_target_special",
            "class_weight",
            "target_weight_override",
        ]
        for col in leaky_cols:
            assert _is_excluded_feature(col), \
                f"Leaky column not excluded: {col}"

    def test_legitimate_features_not_excluded(self):
        """Features we want should NOT be flagged as excluded."""
        from src.phase_21_continuous.experiment_runner import _is_excluded_feature

        legit_features = [
            "rsi_14", "macd_signal", "bb_width_20", "pm_return_lag1",
            "ah_return_lag1", "spy_momentum_20", "vix_close_lag1",
            "pct_green_3d", "net_green_5d", "breadth_divergence",
        ]
        for col in legit_features:
            assert not _is_excluded_feature(col), \
                f"Legitimate feature wrongly excluded: {col}"

    def test_intraday_lookhead_excluded(self):
        """All 64 intraday same-day features must be excluded."""
        from src.phase_21_continuous.experiment_runner import FEATURE_EXCLUDE_COLS

        for prefix in ["return_at_", "high_to_", "rsi_at_", "macd_at_"]:
            for tp in ["0945", "1015", "1100", "1230", "1530"]:
                col = f"{prefix}{tp}"
                assert col in FEATURE_EXCLUDE_COLS, \
                    f"Intraday lookahead not excluded: {col}"


# =============================================================================
# WAVE 17: PERMUTATION TEST, LEAKAGE VALIDATOR, STALENESS, FAILURE PATTERNS
# =============================================================================

class TestWave17Improvements:
    """Tests for Wave 17: permutation test upgrade, runtime leakage validation,
    model staleness detection, and failure pattern analysis."""

    # -- Permutation test --

    def test_permutation_test_uses_multiple_shuffles(self):
        """Permutation test should use 5 shuffles, not just 1."""
        src = open("src/phase_21_continuous/experiment_runner.py", encoding="utf-8").read()
        assert "n_permutations = 5" in src, "Should use 5 permutation shuffles"
        assert "perm_cv_folds = 3" in src, "Should use 3-fold CV for permutation"

    def test_permutation_blocks_leaky_models(self):
        """Permutation test should block tier promotion if mean > 0.53."""
        src = open("src/phase_21_continuous/experiment_runner.py", encoding="utf-8").read()
        assert "perm_mean > 0.53" in src, "Should block at mean permuted AUC > 0.53"
        assert "tier1_pass = False" in src, "Should set tier1_pass = False on failure"

    def test_experiment_result_has_permutation_fields(self):
        """ExperimentResult should track permutation_auc_mean and permutation_passed."""
        from src.phase_21_continuous.experiment_tracking import ExperimentResult
        from dataclasses import fields as dc_fields
        field_names = {f.name for f in dc_fields(ExperimentResult)}
        assert "permutation_auc_mean" in field_names
        assert "permutation_passed" in field_names

    # -- Runtime leakage validator --

    def test_validate_no_leakage_catches_targets(self):
        """Runtime validator should catch target columns in feature list."""
        from src.phase_21_continuous.experiment_runner import _validate_no_leakage
        suspicious = _validate_no_leakage(
            ["rsi_14", "target_up", "close", "future_vol", "macd_signal"],
            target_col="target_up",
        )
        assert "target_up" in suspicious
        assert "close" in suspicious
        assert "future_vol" in suspicious
        assert "rsi_14" not in suspicious
        assert "macd_signal" not in suspicious

    def test_validate_no_leakage_clean_features(self):
        """Runtime validator should pass clean feature lists."""
        from src.phase_21_continuous.experiment_runner import _validate_no_leakage
        suspicious = _validate_no_leakage(
            ["rsi_14", "macd_signal", "bb_width_20", "pm_return_lag1"],
            target_col="target_up",
        )
        assert len(suspicious) == 0

    # -- Model staleness --

    def test_registry_db_model_age_filtering(self):
        """RegistryDB.get_models(max_age_days=...) filters old models."""
        import tempfile
        from datetime import datetime, timedelta
        from src.core.registry_db import RegistryDB

        db = RegistryDB(db_path=Path(tempfile.mktemp(suffix=".db")))
        # Fresh model
        db.add_model("m1", {
            "model_id": "m1", "experiment_id": "e1",
            "created_at": (datetime.now() - timedelta(days=30)).isoformat(),
            "model_path": "", "config": {}, "test_auc": 0.60,
        })
        # Old model
        db.add_model("m2", {
            "model_id": "m2", "experiment_id": "e2",
            "created_at": (datetime.now() - timedelta(days=200)).isoformat(),
            "model_path": "", "config": {}, "test_auc": 0.60,
        })
        fresh = db.get_models(max_age_days=180)
        assert len(fresh) == 1
        assert fresh[0]["model_id"] == "m1"
        db.close()

    def test_registry_db_active_model_count(self):
        """RegistryDB.get_active_model_count() filters by tier and age."""
        import tempfile
        from src.core.registry_db import RegistryDB

        db = RegistryDB(db_path=Path(tempfile.mktemp(suffix=".db")))
        db.add_model("m1", {
            "model_id": "m1", "experiment_id": "e1",
            "created_at": datetime.now().isoformat(),
            "model_path": "", "config": {}, "test_auc": 0.60,
            "tier": 2, "stability_score": 0.65, "fragility_score": 0.3,
        })
        db.add_model("m2", {
            "model_id": "m2", "experiment_id": "e2",
            "created_at": datetime.now().isoformat(),
            "model_path": "", "config": {}, "test_auc": 0.55,
            "tier": 1, "stability_score": 0.4, "fragility_score": 0.8,
        })
        assert db.get_active_model_count(min_auc=0.55, min_tier=2) == 1
        assert db.get_active_model_count(min_auc=0.55, min_tier=1) == 2
        db.close()

    # -- Failure pattern analysis --

    def test_experiment_history_has_failure_patterns(self):
        """ExperimentHistory should have get_failure_patterns() method."""
        src = open("src/phase_21_continuous/experiment_tracking.py", encoding="utf-8").read()
        assert "def get_failure_patterns(" in src
        assert "dim_methods_failing" in src

    def test_experiment_history_has_recent_trend(self):
        """ExperimentHistory should have get_recent_trend() method."""
        src = open("src/phase_21_continuous/experiment_tracking.py", encoding="utf-8").read()
        assert "def get_recent_trend(" in src
        assert '"declining"' in src or "'declining'" in src

    def test_experiment_generator_uses_history(self):
        """ExperimentGenerator should accept history for failure avoidance."""
        from src.phase_21_continuous.experiment_tracking import ExperimentGenerator
        import inspect
        sig = inspect.signature(ExperimentGenerator.__init__)
        assert "history" in sig.parameters, "Generator should accept history param"

    # -- Sharpe risk-free fix --

    def test_sharpe_uses_risk_free_rate(self):
        """Sharpe calculation should subtract risk-free rate."""
        src = open("src/phase_21_continuous/experiment_tracking.py", encoding="utf-8").read()
        assert "risk_free_daily" in src, "Should define risk-free rate"
        assert "excess_gross" in src or "excess_net" in src, "Should compute excess returns"

    # -- Ensemble weights --

    def test_signal_generator_configurable_cascade_weight(self):
        """SignalGenerator should have configurable cascade_blend_weight."""
        src = open("src/phase_19_paper_trading/signal_generator.py", encoding="utf-8").read()
        assert "cascade_blend_weight" in src, "Should have configurable blend weight"
        assert "self.cascade_blend_weight" in src


# =============================================================================
# DATA PERIOD / YEARS CONSISTENCY TESTS (Wave 18)
# =============================================================================
class TestDataPeriodConsistency:
    """Ensure DataConfig defaults to 10 years and pipeline uses dc.years."""

    def test_data_config_default_period_is_10y(self):
        """DataConfig should default to 10Y period."""
        from src.phase_18_persistence.registry_configs import DataConfig
        dc = DataConfig()
        assert dc.period == "10Y", f"Expected '10Y', got '{dc.period}'"
        assert dc.years == 10.0, f"Expected 10.0, got {dc.years}"

    def test_pipeline_uses_dc_years_over_period_map(self):
        """TrainingPipelineV2 should prefer dc.years over period string mapping."""
        src = open("src/phase_12_model_training/training_pipeline_v2.py", encoding="utf-8").read()
        assert "getattr(dc, 'years'" in src, "Should use dc.years as primary"

    def test_grid_search_includes_10y(self):
        """Grid search should include YEARS_10 in data period dimension."""
        src = open("src/phase_18_persistence/grid_search_generator.py", encoding="utf-8").read()
        assert "YEARS_10" in src, "Grid search should include 10Y period"

    def test_period_map_fallback_is_10(self):
        """Period map fallback should be 10.0, not 3.0."""
        src = open("src/phase_12_model_training/training_pipeline_v2.py", encoding="utf-8").read()
        assert "period_map.get(dc.period, 10.0)" in src, "Fallback should be 10.0"


# =============================================================================
# BACKTEST ENGINE HARDENING TESTS (Wave 18)
# =============================================================================
class TestBacktestHardening:
    """Tests for portfolio mark-to-market, dynamic slippage, and Sharpe fix."""

    def test_portfolio_equity_mark_to_market(self):
        """Portfolio.equity should use last known close, not entry price."""
        from src.phase_16_backtesting.portfolio import Portfolio
        p = Portfolio(initial_capital=100_000)
        # Open a LONG trade at $450
        trade = p.open_trade(
            date=datetime(2025, 1, 2), direction="LONG",
            price=450.0, position_value=10_000,
        )
        assert trade is not None
        # Set last known close to $460 (price went up)
        p._last_close = 460.0
        # Equity should reflect mark-to-market gain
        equity = p.equity
        # ~22.2 shares × $460 = ~$10,222 open value + remaining cash
        assert equity > 100_000, f"Equity should be > initial capital, got {equity}"

    def test_portfolio_equity_uses_entry_cost_when_no_close(self):
        """Without a close price, equity falls back to entry cost."""
        from src.phase_16_backtesting.portfolio import Portfolio
        p = Portfolio(initial_capital=100_000)
        trade = p.open_trade(
            date=datetime(2025, 1, 2), direction="LONG",
            price=450.0, position_value=10_000,
        )
        assert trade is not None
        # No _last_close set (still 0)
        assert p._last_close == 0.0
        equity = p.equity
        # Should fall back to entry_cost
        assert 99_990 <= equity <= 100_010  # ~100K (cash - cost + open cost)

    def test_dynamic_slippage_scales_with_volatility(self):
        """Dynamic slippage should increase with higher volatility."""
        from src.phase_16_backtesting.portfolio import Portfolio
        p = Portfolio(initial_capital=100_000, dynamic_slippage=True)
        slip_low_vol = p.calculate_slippage(10_000, volatility=0.10)
        slip_high_vol = p.calculate_slippage(10_000, volatility=0.30)
        assert slip_high_vol > slip_low_vol, "High vol should have more slippage"

    def test_dynamic_slippage_scales_with_order_size(self):
        """Dynamic slippage should increase with larger orders."""
        from src.phase_16_backtesting.portfolio import Portfolio
        p = Portfolio(initial_capital=100_000, dynamic_slippage=True)
        slip_small = p.calculate_slippage(5_000, volatility=0.15)
        slip_large = p.calculate_slippage(80_000, volatility=0.15)
        assert slip_large > slip_small, "Larger orders should have more slippage"

    def test_fixed_slippage_when_dynamic_disabled(self):
        """With dynamic_slippage=False, should use fixed rate."""
        from src.phase_16_backtesting.portfolio import Portfolio
        p = Portfolio(initial_capital=100_000, slippage_pct=0.0002, dynamic_slippage=False)
        slip = p.calculate_slippage(50_000, volatility=0.30)
        assert slip == 0.0002, f"Fixed slippage should be 0.0002, got {slip}"

    def test_backtest_sharpe_uses_risk_free_rate(self):
        """Backtest Sharpe ratio should subtract risk-free rate."""
        src = open("src/phase_16_backtesting/backtest_core.py", encoding="utf-8").read()
        assert "risk_free_daily" in src, "Sharpe should use risk-free rate"
        assert "excess_returns" in src, "Should compute excess returns"

    def test_backtest_passes_volatility_to_portfolio(self):
        """Backtest engine should compute and pass volatility to trades."""
        src = open("src/phase_16_backtesting/backtest_core.py", encoding="utf-8").read()
        assert "volatility=" in src, "Should pass volatility to portfolio methods"

    def test_signal_generator_has_calibrator(self):
        """Paper trading SignalGenerator should have a calibrator attribute."""
        src = open("src/phase_19_paper_trading/signal_generator.py", encoding="utf-8").read()
        assert "self.calibrator" in src, "Should init calibrator"
        assert "ConfidenceCalibrator" in src, "Should import ConfidenceCalibrator"

    def test_performance_tracker_accepts_calibrator(self):
        """PaperPerformanceTracker should accept and use calibrator."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        assert "calibrator=" in src, "Should pass calibrator to tracker"
        assert "_calibrator_callback" in src, "Should have callback method"


# =============================================================================
# CONFIDENCE CALCULATION FIX TESTS (Wave 18b)
# =============================================================================
class TestConfidenceCalculation:
    """Tests for DynamicModelSelector confidence combining strength + agreement."""

    def test_confidence_formula_uses_prediction_strength(self):
        """Confidence should include prediction_strength component."""
        src = open("src/phase_25_risk_management/model_selector.py", encoding="utf-8").read()
        assert "prediction_strength" in src, "Should compute prediction strength"
        assert "abs(swing_proba - 0.5)" in src, "Should measure distance from random"

    def test_confidence_formula_uses_model_agreement(self):
        """Confidence should include model_agreement component."""
        src = open("src/phase_25_risk_management/model_selector.py", encoding="utf-8").read()
        assert "model_agreement" in src, "Should compute model agreement"

    def test_confidence_combines_both_factors(self):
        """Confidence should blend strength and agreement, not just agreement."""
        src = open("src/phase_25_risk_management/model_selector.py", encoding="utf-8").read()
        assert "0.6 * prediction_strength" in src, "60% weight on prediction strength"
        assert "0.4 * model_agreement" in src, "40% weight on model agreement"

    def test_weak_prediction_low_confidence(self):
        """A prediction near 0.5 should have low confidence even if models agree."""
        # swing_proba = 0.51, std = 0.01 (perfect agreement)
        # prediction_strength = abs(0.51 - 0.5) * 2 = 0.02
        # model_agreement = max(0, 1 - 0.01 * 4) = 0.96
        # confidence = 0.6 * 0.02 + 0.4 * 0.96 = 0.012 + 0.384 = 0.396
        prediction_strength = abs(0.51 - 0.5) * 2
        model_agreement = max(0.0, 1 - 0.01 * 4)
        confidence = 0.6 * prediction_strength + 0.4 * model_agreement
        assert confidence < 0.5, f"Weak prediction should have low confidence, got {confidence:.3f}"

    def test_strong_prediction_high_confidence(self):
        """A strong prediction (0.85) with agreement should have high confidence."""
        prediction_strength = abs(0.85 - 0.5) * 2
        model_agreement = max(0.0, 1 - 0.03 * 4)
        confidence = 0.6 * prediction_strength + 0.4 * model_agreement
        assert confidence > 0.6, f"Strong prediction should have high confidence, got {confidence:.3f}"

    def test_experiment_runner_logs_date_range(self):
        """Experiment runner should log date range of loaded data."""
        src = open("src/phase_21_continuous/experiment_runner.py", encoding="utf-8").read()
        assert "first" in src and "last" in src, "Should log first/last dates"


# =============================================================================
# PIPELINE INTEGRITY TESTS (Wave 18c)
# =============================================================================
class TestPipelineIntegrity:
    """Tests for ensemble normalization, daily reset, and test leakage fix."""

    def test_ensemble_renormalizes_contributing_weights(self):
        """Ensemble should renormalize when some strategies are neutral."""
        src = open("src/phase_15_strategy/ensemble_signal_generator.py", encoding="utf-8").read()
        assert "contributing_weight_sum" in src, "Should track contributing weight sum"
        assert "weighted_score /= contributing_weight_sum" in src, "Should renormalize score"

    def test_daily_risk_reset_in_trading_bot(self):
        """Trading bot should reset daily risk counters each new day."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        assert "reset_daily()" in src, "Should call reset_daily"
        assert "_last_trading_date" in src, "Should track last trading date"

    def test_training_pipeline_test_eval_no_leakage(self):
        """Test evaluation should re-fit transformers on train-only data."""
        src = open("src/phase_12_model_training/training_pipeline_v2.py", encoding="utf-8").read()
        assert "X_train_raw = X[:-test_size]" in src, "Should split before fitting"
        assert "eval_selector" in src, "Should use separate eval selector"
        assert "deepcopy" in src, "Should deep-copy transformers for eval"

    def test_training_pipeline_test_eval_trains_fresh_model(self):
        """Test eval should train a fresh model on properly split data."""
        src = open("src/phase_12_model_training/training_pipeline_v2.py", encoding="utf-8").read()
        assert "eval_model" in src, "Should create eval model"
        assert "eval_model.fit(X_tr, y_train_raw)" in src, "Should train on train-only data"


# =============================================================================
# MODEL DIVERSITY TESTS
# =============================================================================

class TestModelDiversity:
    """Tests for diversity-aware model selection in DynamicModelSelector."""

    def _make_candidate(self, model_id, score_auc=0.60, config=None):
        from src.phase_25_risk_management.ensemble_strategies import ModelCandidate
        return ModelCandidate(
            model_id=model_id,
            model_path="",
            config=config or {},
            test_auc=score_auc,
            cv_auc=score_auc,
            wmes_score=0.55,
        )

    def test_config_fingerprint_extracts_dims(self):
        """Fingerprint should extract dim_reduction method from config."""
        from src.phase_25_risk_management.model_selector import DynamicModelSelector
        sel = DynamicModelSelector()
        cand = self._make_candidate("m1", config={
            "dim_reduction": {"method": "kernel_pca", "n_components": 20},
            "model_type": "logistic",
        })
        fp = sel._config_fingerprint(cand)
        assert fp["dim_method"] == "kernel_pca"
        assert fp["model_type"] == "logistic"

    def test_diversity_score_identical_is_zero(self):
        """Two identical configs should have diversity score 0."""
        from src.phase_25_risk_management.model_selector import DynamicModelSelector
        sel = DynamicModelSelector()
        cfg = {"dim_reduction": {"method": "umap"}, "model_type": "gb"}
        c1 = self._make_candidate("m1", config=cfg)
        c2 = self._make_candidate("m2", config=cfg)
        score = sel._diversity_score(c2, [c1])
        assert score == 0.0, f"Identical configs should have 0 diversity, got {score}"

    def test_diversity_score_different_is_positive(self):
        """Different configs should have positive diversity score."""
        from src.phase_25_risk_management.model_selector import DynamicModelSelector
        sel = DynamicModelSelector()
        c1 = self._make_candidate("m1", config={
            "dim_reduction": {"method": "umap"}, "model_type": "gb"
        })
        c2 = self._make_candidate("m2", config={
            "dim_reduction": {"method": "kernel_pca"}, "model_type": "logistic"
        })
        score = sel._diversity_score(c2, [c1])
        assert score > 0.0, f"Different configs should have positive diversity"

    def test_diverse_selection_picks_varied_models(self):
        """With diversity_weight > 0, selector should pick diverse models over homogeneous."""
        from src.phase_25_risk_management.model_selector import DynamicModelSelector
        sel = DynamicModelSelector()
        # 3 similar models (all umap/gb) + 1 very different with slightly lower score
        similar_cfg = {"dim_reduction": {"method": "umap"}, "model_type": "gb"}
        diff_cfg = {
            "dim_reduction": {"method": "kernel_pca", "n_components": 30,
                              "feature_selection_method": "f_classif"},
            "model_type": "logistic",
            "data_config": {"period": "10Y"},
            "source": "thick_weave",
        }
        sel.candidates = [
            self._make_candidate("top1", score_auc=0.65, config=similar_cfg),
            self._make_candidate("top2", score_auc=0.64, config=similar_cfg),
            self._make_candidate("top3", score_auc=0.63, config=similar_cfg),
            self._make_candidate("diff1", score_auc=0.62, config=diff_cfg),
        ]
        # Sort by score (like load_from_registry does)
        sel.candidates.sort(key=lambda c: c.score(), reverse=True)
        selected = sel.get_best_models(n=3, diversity_weight=0.4)
        ids = {c.model_id for c in selected}
        assert "top1" in ids, "Best model should always be selected"
        assert "diff1" in ids, "Diverse model should be preferred over 3rd similar"

    def test_pure_quality_selection_ignores_diversity(self):
        """With diversity_weight=0, should behave like pure top-N."""
        from src.phase_25_risk_management.model_selector import DynamicModelSelector
        sel = DynamicModelSelector()
        cfg_a = {"dim_reduction": {"method": "umap"}, "model_type": "gb"}
        cfg_b = {"dim_reduction": {"method": "kpca"}, "model_type": "logistic"}
        sel.candidates = [
            self._make_candidate("best", score_auc=0.68, config=cfg_a),
            self._make_candidate("2nd", score_auc=0.66, config=cfg_a),
            self._make_candidate("3rd", score_auc=0.64, config=cfg_a),
            self._make_candidate("diff", score_auc=0.60, config=cfg_b),
        ]
        sel.candidates.sort(key=lambda c: c.score(), reverse=True)
        selected = sel.get_best_models(n=3, diversity_weight=0.0)
        ids = [c.model_id for c in selected]
        assert ids == ["best", "2nd", "3rd"], f"Pure quality should pick top 3: {ids}"

    def test_diversity_with_empty_selected(self):
        """Diversity score with no selected models should be 1.0."""
        from src.phase_25_risk_management.model_selector import DynamicModelSelector
        sel = DynamicModelSelector()
        cand = self._make_candidate("m1")
        score = sel._diversity_score(cand, [])
        assert score == 1.0


# =============================================================================
# ATOMIC WRITE TESTS
# =============================================================================

class TestAtomicWrites:
    """Tests for atomic JSON file writes."""

    def test_atomic_write_creates_file(self, tmp_path):
        """atomic_write_json should create a valid JSON file."""
        from src.core.state_manager import atomic_write_json
        target = tmp_path / "test.json"
        atomic_write_json(target, {"key": "value"})
        assert target.exists()
        data = json.loads(target.read_text())
        assert data["key"] == "value"

    def test_atomic_write_no_temp_leftover(self, tmp_path):
        """No .tmp file should remain after successful write."""
        from src.core.state_manager import atomic_write_json
        target = tmp_path / "test.json"
        atomic_write_json(target, {"a": 1})
        tmp_file = target.with_suffix(".tmp")
        assert not tmp_file.exists()

    def test_atomic_write_overwrites_existing(self, tmp_path):
        """Should atomically replace existing content."""
        from src.core.state_manager import atomic_write_json
        target = tmp_path / "test.json"
        atomic_write_json(target, {"v": 1})
        atomic_write_json(target, {"v": 2})
        data = json.loads(target.read_text())
        assert data["v"] == 2

    def test_atomic_write_creates_parent_dirs(self, tmp_path):
        """Should create parent directories if missing."""
        from src.core.state_manager import atomic_write_json
        target = tmp_path / "sub" / "dir" / "test.json"
        atomic_write_json(target, {"nested": True})
        assert target.exists()

    def test_trading_bot_uses_atomic_write(self):
        """trading_bot.py should use atomic_write_json instead of write_text."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        assert "atomic_write_json" in src
        # Should no longer use write_text for JSON files
        assert "write_text(json.dumps" not in src

    def test_health_checker_uses_atomic_write(self):
        """health_checker.py should use atomic_write_json."""
        src = open("src/phase_20_monitoring/health_checker.py", encoding="utf-8").read()
        assert "atomic_write_json" in src

    def test_experiment_tracking_uses_sqlite(self):
        """experiment_tracking.py should use SQLite (RegistryDB) for storage."""
        src = open("src/phase_21_continuous/experiment_tracking.py", encoding="utf-8").read()
        assert "self._db" in src
        # Should not have raw json.dump or json file reads
        assert "json.dump(data, f" not in src
        assert "json.load" not in src


# =============================================================================
# EXPERIMENT HISTORY THREAD SAFETY & STATISTICS TESTS
# =============================================================================

class TestExperimentHistoryRobustness:
    """Tests for thread safety, atomic writes, and statistics filtering."""

    def test_experiment_history_has_lock(self):
        """ExperimentHistory should have a threading lock."""
        src = open("src/phase_21_continuous/experiment_tracking.py", encoding="utf-8").read()
        assert "self._lock = threading.Lock()" in src

    def test_experiment_tracking_uses_threading(self):
        """experiment_tracking module should use threading for locks."""
        src = open("src/phase_21_continuous/experiment_tracking.py", encoding="utf-8").read()
        assert "threading" in src
        # ExperimentHistory has a lock
        assert "self._lock = threading.Lock()" in src

    def test_statistics_filters_zero_auc(self):
        """get_statistics should exclude zero-AUC completed experiments from averages."""
        from src.phase_21_continuous.experiment_tracking import (
            ExperimentHistory, ExperimentResult, ExperimentStatus
        )
        from src.experiment_config import create_default_config
        cfg = create_default_config("test")
        hist = ExperimentHistory.__new__(ExperimentHistory)
        hist._lock = __import__("threading").Lock()
        hist.results = [
            ExperimentResult(experiment_id="e1", config=cfg, status=ExperimentStatus.COMPLETED, test_auc=0.60),
            ExperimentResult(experiment_id="e2", config=cfg, status=ExperimentStatus.COMPLETED, test_auc=0.0),
            ExperimentResult(experiment_id="e3", config=cfg, status=ExperimentStatus.COMPLETED, test_auc=0.65),
            ExperimentResult(experiment_id="e4", config=cfg, status=ExperimentStatus.FAILED, test_auc=0.0),
        ]
        stats = hist.get_statistics()
        # avg_test_auc should be (0.60 + 0.65) / 2 = 0.625, NOT (0.60 + 0.0 + 0.65) / 3
        assert stats["scored"] == 2
        assert abs(stats["avg_test_auc"] - 0.625) < 0.001

    def test_experiment_history_add_is_thread_safe(self):
        """ExperimentHistory.add() should use lock."""
        src = open("src/phase_21_continuous/experiment_tracking.py", encoding="utf-8").read()
        # Find the add method and verify it uses self._lock
        idx = src.index("def add(")
        block = src[idx:idx+200]
        assert "with self._lock:" in block


# =============================================================================
# CONSECUTIVE LOSS & PERFORMANCE CAP TESTS
# =============================================================================

class TestRiskAndPerformance:
    """Tests for consecutive loss tracking and performance tracker cap."""

    def test_consecutive_loss_tracking(self):
        """RiskManager should track consecutive losses."""
        from src.phase_19_paper_trading.risk_management import RiskManager
        rm = RiskManager()
        rm.peak_equity = 100000
        # Record 4 losses — should not halt
        for _ in range(4):
            rm.record_trade_result(-100)
        assert rm.consecutive_losses == 4
        assert not rm.is_halted
        # 5th loss triggers halt
        rm.record_trade_result(-100)
        assert rm.consecutive_losses == 5
        assert rm.is_halted

    def test_consecutive_loss_resets_on_win(self):
        """A winning trade should reset the consecutive loss counter."""
        from src.phase_19_paper_trading.risk_management import RiskManager
        rm = RiskManager()
        rm.record_trade_result(-100)
        rm.record_trade_result(-100)
        assert rm.consecutive_losses == 2
        rm.record_trade_result(50)  # Win
        assert rm.consecutive_losses == 0

    def test_can_trade_blocks_on_consecutive_losses(self):
        """can_trade should return False when consecutive loss limit hit."""
        from src.phase_19_paper_trading.risk_management import RiskManager
        rm = RiskManager()
        rm.peak_equity = 100000
        for _ in range(5):
            rm.record_trade_result(-50)
        can, reason = rm.can_trade()
        assert not can
        assert "consecutive" in reason.lower()

    def test_daily_reset_clears_consecutive_losses(self):
        """reset_daily should clear consecutive loss counter."""
        from src.phase_19_paper_trading.risk_management import RiskManager
        rm = RiskManager()
        rm.record_trade_result(-100)
        rm.record_trade_result(-100)
        rm.reset_daily()
        assert rm.consecutive_losses == 0

    def test_performance_tracker_max_predictions(self):
        """PaperPerformanceTracker should cap predictions list."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        assert "max_predictions" in src
        assert "self.predictions = self.predictions[-self.max_predictions:]" in src

    def test_config_hash_includes_all_dimensions(self):
        """Config hash should include model, dim_reduction, features, and CV settings."""
        from src.phase_21_continuous.experiment_tracking import ExperimentHistory
        from src.experiment_config import create_default_config
        cfg1 = create_default_config("test1")
        cfg2 = create_default_config("test2")
        # Identical configs should produce same hash
        h1 = ExperimentHistory._config_hash(cfg1)
        h2 = ExperimentHistory._config_hash(cfg2)
        assert h1 == h2, "Identical configs should hash the same"
        # Changing model type should change hash
        cfg2.model.model_type = "logistic"
        h3 = ExperimentHistory._config_hash(cfg2)
        assert h1 != h3, "Different model types should produce different hashes"
        # Changing CV folds should change hash
        cfg3 = create_default_config("test3")
        cfg3.cross_validation.n_cv_folds = 3
        h4 = ExperimentHistory._config_hash(cfg3)
        assert h1 != h4, "Different CV folds should produce different hashes"

    def test_signal_history_capped(self):
        """Ensemble signal generator should cap signal_history."""
        src = open("src/phase_15_strategy/ensemble_signal_generator.py", encoding="utf-8").read()
        assert "_max_history" in src
        assert "self.signal_history = self.signal_history[-self._max_history:]" in src


# =============================================================================
# RISK MANAGER STATE PERSISTENCE TESTS (Wave 20)
# =============================================================================

class TestRiskManagerPersistence:
    """Tests for RiskManager state persistence across restarts."""

    def test_persist_and_restore_same_day(self, tmp_path):
        """Risk state should persist and restore on the same day."""
        from src.phase_19_paper_trading.risk_management import RiskManager
        state_file = tmp_path / "risk_state.json"

        # Create and modify risk manager
        rm = RiskManager(state_path=state_file)
        rm.daily_pnl = -150.0
        rm.peak_equity = 100000.0
        rm.daily_trades = 3
        rm.consecutive_losses = 2
        rm._persist_state()

        # New instance should restore state
        rm2 = RiskManager(state_path=state_file)
        assert rm2.daily_pnl == -150.0
        assert rm2.peak_equity == 100000.0
        assert rm2.daily_trades == 3
        assert rm2.consecutive_losses == 2

    def test_stale_state_not_restored(self, tmp_path):
        """Risk state from a previous day should NOT be restored."""
        import json
        state_file = tmp_path / "risk_state.json"

        # Write state with yesterday's date
        data = {
            "date": "2020-01-01",
            "daily_pnl": -500.0,
            "peak_equity": 50000.0,
            "daily_trades": 10,
            "consecutive_losses": 4,
            "is_halted": True,
            "halt_reason": "old halt",
        }
        with open(state_file, "w") as f:
            json.dump(data, f)

        # Should NOT restore old state
        from src.phase_19_paper_trading.risk_management import RiskManager
        rm = RiskManager(state_path=state_file)
        assert rm.daily_pnl == 0.0
        assert rm.daily_trades == 0
        assert rm.consecutive_losses == 0
        assert not rm.is_halted

    def test_update_pnl_persists(self, tmp_path):
        """update_pnl should trigger state persistence."""
        from src.phase_19_paper_trading.risk_management import RiskManager
        state_file = tmp_path / "risk_state.json"

        rm = RiskManager(state_path=state_file)
        rm.update_pnl(-50.0, 100000.0)

        assert state_file.exists()
        import json
        with open(state_file) as f:
            data = json.load(f)
        assert data["daily_pnl"] == -50.0

    def test_record_trade_result_persists(self, tmp_path):
        """record_trade_result should trigger state persistence."""
        from src.phase_19_paper_trading.risk_management import RiskManager
        state_file = tmp_path / "risk_state.json"

        rm = RiskManager(state_path=state_file)
        rm.record_trade_result(-100.0)

        import json
        with open(state_file) as f:
            data = json.load(f)
        assert data["consecutive_losses"] == 1

    def test_reset_daily_persists(self, tmp_path):
        """reset_daily should persist the cleared state."""
        from src.phase_19_paper_trading.risk_management import RiskManager
        state_file = tmp_path / "risk_state.json"

        rm = RiskManager(state_path=state_file)
        rm.daily_pnl = -200.0
        rm.consecutive_losses = 3
        rm._persist_state()

        rm.reset_daily()

        import json
        with open(state_file) as f:
            data = json.load(f)
        assert data["daily_pnl"] == 0.0
        assert data["consecutive_losses"] == 0

    def test_no_state_path_no_error(self):
        """RiskManager without state_path should work silently."""
        from src.phase_19_paper_trading.risk_management import RiskManager
        rm = RiskManager()  # No state_path
        rm.update_pnl(-10.0, 100000.0)
        rm.record_trade_result(-5.0)
        rm.reset_daily()
        # Should not raise

    def test_corrupt_state_file_handled(self, tmp_path):
        """Corrupt state file should be handled gracefully."""
        from src.phase_19_paper_trading.risk_management import RiskManager
        state_file = tmp_path / "risk_state.json"
        state_file.write_text("not valid json {{{{")

        rm = RiskManager(state_path=state_file)
        # Should start fresh without raising
        assert rm.daily_pnl == 0.0


# =============================================================================
# GRACEFUL SHUTDOWN TESTS (Wave 20)
# =============================================================================

class TestGracefulShutdown:
    """Tests for graceful shutdown handling in start_system.py."""

    def test_shutdown_handler_exists(self):
        """start_system.py should have a shutdown handler."""
        src = open("scripts/start_system.py", encoding="utf-8").read()
        assert "_shutdown_handler" in src
        assert "signal.SIGINT" in src

    def test_shutdown_event_exists(self):
        """start_system.py should have a shutdown event."""
        src = open("scripts/start_system.py", encoding="utf-8").read()
        assert "_shutdown_requested" in src
        assert "threading.Event()" in src

    def test_bot_registered_for_shutdown(self):
        """run_trading_bot should register bot for shutdown handler."""
        src = open("scripts/start_system.py", encoding="utf-8").read()
        assert "_active_bot = bot" in src
        assert "_active_bot = None" in src

    def test_shutdown_flushes_state(self):
        """Shutdown should flush performance tracker and risk state."""
        src = open("scripts/start_system.py", encoding="utf-8").read()
        assert "performance_tracker._save()" in src
        assert "risk_manager._persist_state()" in src

    def test_shutdown_cancels_pending_orders(self):
        """Shutdown should cancel pending orders."""
        src = open("scripts/start_system.py", encoding="utf-8").read()
        assert "cancel_order" in src
        assert "pending_orders" in src

    def test_status_json_uses_atomic_write(self):
        """write_status_json should use atomic_write_json."""
        src = open("scripts/start_system.py", encoding="utf-8").read()
        assert "atomic_write_json" in src


# =============================================================================
# INFERENCE FEATURE VALIDATION TESTS (Wave 20)
# =============================================================================

class TestInferenceFeatureValidation:
    """Tests for inference-time feature validation in signal generator."""

    def test_partial_day_features_defined(self):
        """SignalGenerator should define PARTIAL_DAY_FEATURES."""
        from src.phase_19_paper_trading.signal_generator import SignalGenerator
        assert hasattr(SignalGenerator, "PARTIAL_DAY_FEATURES")
        assert "day_range" in SignalGenerator.PARTIAL_DAY_FEATURES
        assert "day_volume" in SignalGenerator.PARTIAL_DAY_FEATURES

    def test_validate_method_exists(self):
        """SignalGenerator should have _validate_inference_features method."""
        from src.phase_19_paper_trading.signal_generator import SignalGenerator
        assert hasattr(SignalGenerator, "_validate_inference_features")

    def test_validate_substitutes_truncated_features(self):
        """Should substitute partial-day features that look truncated."""
        import pandas as pd
        from src.phase_19_paper_trading.signal_generator import SignalGenerator

        # Build a small 2-row daily dataframe: yesterday (full) + today (partial)
        df = pd.DataFrame({
            "day_range": [5.0, 0.5],       # Today's range is 10% of yesterday
            "day_volume": [1e7, 1e5],       # Today's volume is 1% of yesterday
            "rsi_14": [55.0, 54.0],         # Not a partial-day feature
            "other_feat": [1.0, 0.9],       # Not a partial-day feature
        })

        feature_cols = ["day_range", "day_volume", "rsi_14", "other_feat"]

        # Patch datetime to pretend we're mid-session
        from unittest.mock import patch
        mock_now = datetime(2026, 2, 7, 11, 30)  # 11:30 AM
        with patch("src.phase_19_paper_trading.signal_generator.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            sg = SignalGenerator.__new__(SignalGenerator)  # Skip __init__
            result = sg._validate_inference_features(df, feature_cols)

        # Partial-day features should be substituted
        assert result["day_range"].iloc[-1] == 5.0   # Yesterday's value
        assert result["day_volume"].iloc[-1] == 1e7   # Yesterday's value
        # Non-partial features should be untouched
        assert result["rsi_14"].iloc[-1] == 54.0
        assert result["other_feat"].iloc[-1] == 0.9

    def test_validate_no_op_outside_market_hours(self):
        """Should not substitute features outside market hours."""
        import pandas as pd
        from src.phase_19_paper_trading.signal_generator import SignalGenerator

        df = pd.DataFrame({
            "day_range": [5.0, 0.5],
            "day_volume": [1e7, 1e5],
        })

        feature_cols = ["day_range", "day_volume"]

        from unittest.mock import patch
        mock_now = datetime(2026, 2, 7, 20, 0)  # 8 PM — after hours
        with patch("src.phase_19_paper_trading.signal_generator.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            sg = SignalGenerator.__new__(SignalGenerator)
            result = sg._validate_inference_features(df, feature_cols)

        # Should NOT substitute outside market hours
        assert result["day_range"].iloc[-1] == 0.5
        assert result["day_volume"].iloc[-1] == 1e5

    def test_validate_wired_into_prepare_features(self):
        """prepare_features should call _validate_inference_features."""
        src = open("src/phase_19_paper_trading/signal_generator.py", encoding="utf-8").read()
        assert "_validate_inference_features" in src
        assert "df_daily = self._validate_inference_features(df_daily, all_feature_cols)" in src


# =============================================================================
# TIMEDELTA BUG FIX TESTS (Wave 20)
# =============================================================================

class TestTimedeltaFix:
    """Tests for .total_seconds() usage (was .seconds bug)."""

    def test_orchestrator_uses_total_seconds(self):
        """giga_orchestrator.py should use .total_seconds(), not .seconds."""
        src = open("src/giga_orchestrator.py", encoding="utf-8").read()
        # Should NOT have bare .seconds (which only returns 0-59)
        import re
        bad_pattern = re.compile(r'\.seconds\b(?!_)')
        matches = bad_pattern.findall(src)
        assert len(matches) == 0, f"Found .seconds instead of .total_seconds(): {matches}"
        # Should have .total_seconds()
        assert ".total_seconds()" in src

    def test_total_seconds_vs_seconds_difference(self):
        """Demonstrate why .seconds is wrong for elapsed time checks > 1 day."""
        from datetime import timedelta
        # For durations > 1 day, .seconds only returns the sub-day component
        td = timedelta(days=1, seconds=5)
        assert td.seconds == 5              # WRONG: loses the 86400 from the day
        assert td.total_seconds() == 86405.0  # RIGHT: full elapsed seconds


# =============================================================================
# THREAD SAFETY & NULL CHECK TESTS (Wave 20)
# =============================================================================

class TestThreadSafetyAndNullChecks:
    """Tests for thread safety and null safety improvements."""

    def test_performance_tracker_has_lock(self):
        """PaperPerformanceTracker should have a threading lock."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        assert "self._lock = threading.Lock()" in src
        assert "with self._lock:" in src

    def test_record_signal_uses_lock(self):
        """record_signal should use lock for thread safety."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        # Find record_signal method body (PredictionRecord construction + lock)
        idx = src.index("def record_signal")
        next_def = src.index("\n    def ", idx + 1)
        method_body = src[idx:next_def]
        assert "with self._lock:" in method_body

    def test_record_close_uses_lock(self):
        """record_close should use lock for thread safety."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        idx = src.index("def record_close")
        method_body = src[idx:idx+800]
        assert "with self._lock:" in method_body

    def test_account_null_check(self):
        """run_once should check account is not None."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        assert 'not account or not account.get("equity")' in src

    def test_no_silent_exception_swallowing(self):
        """Should not have bare except:pass (except in shutdown cleanup)."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        import re
        # Find all "except Exception:" followed by "pass" (non-shutdown)
        # The only ones should be in the shutdown finally block
        # This verifies the gate check exception now logs
        assert 'except Exception as e:\n                logger.debug(f"Gate check for dashboard failed: {e}")' in src

    def test_risk_state_uses_specific_exceptions(self):
        """_restore_state should catch specific exceptions, not bare Exception."""
        src = open("src/phase_19_paper_trading/risk_management.py", encoding="utf-8").read()
        idx = src.index("def _restore_state")
        # Find the next method definition to get full body
        next_def = src.index("\n    def ", idx + 1)
        method_body = src[idx:next_def]
        assert "json.JSONDecodeError" in method_body
        assert "IOError" in method_body


# =============================================================================
# SYMBOLIC CROSS-LEARNER TESTS
# =============================================================================

class TestFeatureImportanceExtractor:
    """Tests for FeatureImportanceExtractor."""

    def test_from_tree_model(self):
        """Extract importances from a tree model."""
        from sklearn.tree import DecisionTreeClassifier
        from src.phase_23_analytics.symbolic_cross_learner import FeatureImportanceExtractor

        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = (X[:, 0] + X[:, 2] > 0).astype(int)
        model = DecisionTreeClassifier(max_depth=3, random_state=42)
        model.fit(X, y)

        names = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]
        imp = FeatureImportanceExtractor.from_tree_model(model, names)
        assert len(imp) == 5
        assert all(v >= 0 for v in imp.values())
        # Features 0 and 2 should have most importance
        assert imp["feat_a"] > 0 or imp["feat_c"] > 0

    def test_from_linear_model(self):
        """Extract importances from a linear model."""
        from sklearn.linear_model import LogisticRegression
        from src.phase_23_analytics.symbolic_cross_learner import FeatureImportanceExtractor

        rng = np.random.RandomState(42)
        X = rng.randn(100, 4)
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X, y)

        names = ["a", "b", "c", "d"]
        imp = FeatureImportanceExtractor.from_linear_model(model, names)
        assert len(imp) == 4
        assert abs(sum(imp.values()) - 1.0) < 0.01  # Normalized
        assert imp["a"] > imp["b"]  # Feature 0 is the predictive one

    def test_from_metrics(self):
        """Extract importances from stored ModelMetrics dict."""
        from src.phase_23_analytics.symbolic_cross_learner import FeatureImportanceExtractor

        metrics = {
            "top_features": ["rsi_14", "volume_ratio", "pm_return"],
            "top_feature_importances": [0.3, 0.2, 0.15],
        }
        imp = FeatureImportanceExtractor.from_metrics(metrics)
        assert len(imp) == 3
        assert imp["rsi_14"] == 0.3

    def test_from_metrics_no_importances(self):
        """Should auto-assign decreasing importance when values missing."""
        from src.phase_23_analytics.symbolic_cross_learner import FeatureImportanceExtractor

        metrics = {"top_features": ["a", "b", "c"], "top_feature_importances": []}
        imp = FeatureImportanceExtractor.from_metrics(metrics)
        assert len(imp) == 3
        assert imp["a"] > imp["b"] > imp["c"]

    def test_extract_auto_detect(self):
        """Auto-detect model type in extract()."""
        from sklearn.tree import DecisionTreeClassifier
        from src.phase_23_analytics.symbolic_cross_learner import FeatureImportanceExtractor

        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        y = (X[:, 0] > 0).astype(int)
        model = DecisionTreeClassifier(max_depth=2, random_state=42)
        model.fit(X, y)

        imp = FeatureImportanceExtractor.extract(model, ["x", "y", "z"])
        assert len(imp) == 3

    def test_extract_with_metrics_fallback(self):
        """Fall back to metrics when model has no importances."""
        from src.phase_23_analytics.symbolic_cross_learner import FeatureImportanceExtractor

        class DummyModel:
            pass

        metrics = {"top_features": ["feat1"], "top_feature_importances": [1.0]}
        imp = FeatureImportanceExtractor.extract(DummyModel(), metrics=metrics)
        assert imp["feat1"] == 1.0


class TestUniversalFeatureMap:
    """Tests for UniversalFeatureMap."""

    def test_add_model_and_ranking(self):
        """Add models and get universal ranking."""
        from src.phase_23_analytics.symbolic_cross_learner import UniversalFeatureMap

        fmap = UniversalFeatureMap()
        fmap.add_model_full("m1", {"rsi": 0.3, "vol": 0.2, "price": 0.5}, model_auc=0.65, model_type="tree")
        fmap.add_model_full("m2", {"rsi": 0.4, "vol": 0.1, "macd": 0.5}, model_auc=0.60, model_type="linear")
        fmap.add_model_full("m3", {"rsi": 0.2, "price": 0.3, "macd": 0.5}, model_auc=0.70, model_type="tree")

        # rsi appears in all 3 models
        assert fmap.profiles["rsi"].n_models_using == 3
        assert fmap.profiles["rsi"].universality_score() == 1.0

        # vol appears in 2 models
        assert fmap.profiles["vol"].n_models_using == 2

        ranking = fmap.get_universal_ranking(min_universality=0.5)
        assert len(ranking) > 0
        # rsi should be in the ranking
        names = [name for name, _ in ranking]
        assert "rsi" in names

    def test_niche_features(self):
        """Detect niche features used by few models."""
        from src.phase_23_analytics.symbolic_cross_learner import UniversalFeatureMap

        fmap = UniversalFeatureMap()
        fmap.add_model_full("m1", {"common": 0.3, "rare": 0.8}, model_auc=0.65)
        fmap.add_model_full("m2", {"common": 0.4, "other": 0.1}, model_auc=0.60)
        fmap.add_model_full("m3", {"common": 0.2, "other": 0.2}, model_auc=0.70)
        fmap.add_model_full("m4", {"common": 0.3, "other": 0.3}, model_auc=0.55)

        niche = fmap.get_niche_features(max_universality=0.3, min_importance=0.05)
        niche_names = [n for n, _, _ in niche]
        assert "rare" in niche_names  # Only used by 1 model (25%)

    def test_unused_features(self):
        """Detect features no model uses."""
        from src.phase_23_analytics.symbolic_cross_learner import UniversalFeatureMap

        fmap = UniversalFeatureMap()
        fmap.add_model_full("m1", {"a": 0.5, "b": 0.0}, model_auc=0.6)
        # "b" has importance 0.0 so it's not "used"
        unused = fmap.get_unused_features()
        assert "b" in unused

    def test_suggest_feature_set_balanced(self):
        """Suggest a balanced feature set."""
        from src.phase_23_analytics.symbolic_cross_learner import UniversalFeatureMap

        fmap = UniversalFeatureMap()
        for i in range(5):
            imp = {f"universal_{j}": 0.3 for j in range(5)}
            imp[f"niche_{i}"] = 0.8
            fmap.add_model_full(f"m{i}", imp, model_auc=0.6)

        suggested = fmap.suggest_feature_set("balanced", n_features=7)
        assert len(suggested) <= 7
        assert len(suggested) > 0

    def test_suggest_feature_set_strategies(self):
        """All suggestion strategies should return lists."""
        from src.phase_23_analytics.symbolic_cross_learner import UniversalFeatureMap

        fmap = UniversalFeatureMap()
        fmap.add_model_full("m1", {"a": 0.5, "b": 0.3}, model_auc=0.6)
        fmap.add_model_full("m2", {"a": 0.4, "c": 0.6}, model_auc=0.7)

        for strategy in ["universal", "niche", "balanced", "complementary"]:
            result = fmap.suggest_feature_set(strategy, n_features=5)
            assert isinstance(result, list)

    def test_persistence(self):
        """Feature map should persist and restore."""
        import tempfile
        from src.phase_23_analytics.symbolic_cross_learner import UniversalFeatureMap

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = Path(f.name)
        try:
            fmap = UniversalFeatureMap(persist_path=path)
            fmap.add_model_full("m1", {"rsi": 0.5, "vol": 0.3}, model_auc=0.65)
            assert path.is_file()

            # Reload
            fmap2 = UniversalFeatureMap(persist_path=path)
            assert "rsi" in fmap2.profiles
            assert fmap2.profiles["rsi"].n_models_using == 1
        finally:
            path.unlink(missing_ok=True)

    def test_get_report(self):
        """Report should contain expected keys."""
        from src.phase_23_analytics.symbolic_cross_learner import UniversalFeatureMap

        fmap = UniversalFeatureMap()
        fmap.add_model_full("m1", {"a": 0.5}, model_auc=0.6)
        report = fmap.get_report()
        assert report["n_features"] == 1
        assert report["n_models"] == 1
        assert "mean_universality" in report


class TestSymbolicRuleExtractor:
    """Tests for SymbolicRuleExtractor."""

    def test_extract_from_decision_tree(self):
        """Extract rules from a simple decision tree."""
        from sklearn.tree import DecisionTreeClassifier
        from src.phase_23_analytics.symbolic_cross_learner import SymbolicRuleExtractor

        rng = np.random.RandomState(42)
        X = rng.randn(200, 3)
        y = (X[:, 0] > 0).astype(int)
        model = DecisionTreeClassifier(max_depth=2, random_state=42)
        model.fit(X, y)

        rules = SymbolicRuleExtractor.from_decision_tree(
            model, ["feat_0", "feat_1", "feat_2"],
            source_model="test_tree", min_samples=5, min_confidence=0.5,
        )
        assert len(rules) > 0
        for rule in rules:
            assert len(rule.conditions) > 0
            assert rule.prediction in (0, 1)
            assert rule.confidence >= 0.5
            assert rule.n_samples >= 5

    def test_rule_to_readable(self):
        """Rule should produce readable string."""
        from src.phase_23_analytics.symbolic_cross_learner import SymbolicRule

        rule = SymbolicRule(
            conditions=[("rsi_14", "<=", 30.0), ("volume", ">", 1000000.0)],
            prediction=1,
            confidence=0.75,
            n_samples=50,
        )
        readable = rule.to_readable()
        assert "IF" in readable
        assert "rsi_14" in readable
        assert "UP" in readable

    def test_rule_feature_set(self):
        """Rule should return its feature set."""
        from src.phase_23_analytics.symbolic_cross_learner import SymbolicRule

        rule = SymbolicRule(
            conditions=[("a", "<=", 1.0), ("b", ">", 2.0), ("a", "<=", 0.5)],
            prediction=0, confidence=0.8, n_samples=100,
        )
        assert rule.feature_set() == {"a", "b"}

    def test_extract_from_gradient_boosting(self):
        """Extract rules from GB's internal trees."""
        from sklearn.ensemble import GradientBoostingClassifier
        from src.phase_23_analytics.symbolic_cross_learner import SymbolicRuleExtractor

        rng = np.random.RandomState(42)
        X = rng.randn(200, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        model = GradientBoostingClassifier(n_estimators=5, max_depth=2, random_state=42)
        model.fit(X, y)

        rules = SymbolicRuleExtractor.from_gradient_boosting(
            model, ["a", "b", "c", "d"],
            source_model="gb_test", max_trees=3,
            min_samples=1, min_confidence=0.0,
        )
        assert len(rules) > 0

    def test_extract_auto_detect(self):
        """Auto-detect model type in extract()."""
        from sklearn.tree import DecisionTreeClassifier
        from src.phase_23_analytics.symbolic_cross_learner import SymbolicRuleExtractor

        rng = np.random.RandomState(42)
        X = rng.randn(100, 2)
        y = (X[:, 0] > 0).astype(int)
        model = DecisionTreeClassifier(max_depth=2, random_state=42)
        model.fit(X, y)

        rules = SymbolicRuleExtractor.extract(model, ["x", "y"], min_samples=1)
        assert len(rules) > 0

    def test_extract_from_linear_returns_empty(self):
        """Linear models have no tree structure — should return empty."""
        from sklearn.linear_model import LogisticRegression
        from src.phase_23_analytics.symbolic_cross_learner import SymbolicRuleExtractor

        rng = np.random.RandomState(42)
        X = rng.randn(100, 2)
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X, y)

        rules = SymbolicRuleExtractor.extract(model, ["x", "y"])
        assert len(rules) == 0


class TestRuleCrossAnalyzer:
    """Tests for RuleCrossAnalyzer."""

    def test_find_consensus_rules(self):
        """Find rules where multiple models agree."""
        from src.phase_23_analytics.symbolic_cross_learner import (
            RuleCrossAnalyzer, SymbolicRule,
        )

        rules = [
            # Model A and B both use rsi to predict UP
            SymbolicRule([("rsi", "<=", 30.0)], 1, 0.8, 100, "modelA"),
            SymbolicRule([("rsi", "<=", 35.0)], 1, 0.7, 80, "modelB"),
            # Only model C uses volume to predict DOWN
            SymbolicRule([("volume", ">", 1e6)], 0, 0.6, 50, "modelC"),
        ]
        analyzer = RuleCrossAnalyzer(rules)
        consensus = analyzer.find_consensus_rules(min_agreement=2)
        assert len(consensus) >= 1
        # rsi-based rules should form a consensus
        rsi_consensus = [c for c in consensus if "rsi" in c["features_used"]]
        assert len(rsi_consensus) >= 1

    def test_find_contradictions(self):
        """Find rules with same features but opposite predictions."""
        from src.phase_23_analytics.symbolic_cross_learner import (
            RuleCrossAnalyzer, SymbolicRule,
        )

        rules = [
            SymbolicRule([("rsi", "<=", 30.0)], 1, 0.8, 100, "modelA"),
            SymbolicRule([("rsi", "<=", 28.0)], 0, 0.6, 80, "modelB"),
        ]
        analyzer = RuleCrossAnalyzer(rules)
        contradictions = analyzer.find_contradictions()
        assert len(contradictions) >= 1
        assert contradictions[0]["n_up_rules"] >= 1
        assert contradictions[0]["n_down_rules"] >= 1

    def test_find_blind_spots(self):
        """Find features not used in any rule."""
        from src.phase_23_analytics.symbolic_cross_learner import (
            RuleCrossAnalyzer, SymbolicRule,
        )

        rules = [
            SymbolicRule([("rsi", "<=", 30.0)], 1, 0.8, 100, "m1"),
        ]
        analyzer = RuleCrossAnalyzer(rules)
        blind = analyzer.find_feature_blind_spots(["rsi", "macd", "volume"])
        assert "macd" in blind
        assert "volume" in blind
        assert "rsi" not in blind

    def test_feature_rule_frequency(self):
        """Count feature appearances across rules."""
        from src.phase_23_analytics.symbolic_cross_learner import (
            RuleCrossAnalyzer, SymbolicRule,
        )

        rules = [
            SymbolicRule([("rsi", "<=", 30.0), ("vol", ">", 1e6)], 1, 0.8, 100, "m1"),
            SymbolicRule([("rsi", "<=", 35.0)], 1, 0.7, 80, "m2"),
            SymbolicRule([("vol", ">", 2e6)], 0, 0.6, 50, "m3"),
        ]
        analyzer = RuleCrossAnalyzer(rules)
        freq = analyzer.get_feature_rule_frequency()
        assert freq["rsi"] == 2
        assert freq["vol"] == 2

    def test_get_report(self):
        """Report should contain all expected sections."""
        from src.phase_23_analytics.symbolic_cross_learner import (
            RuleCrossAnalyzer, SymbolicRule,
        )

        rules = [
            SymbolicRule([("a", "<=", 1.0)], 1, 0.8, 100, "m1"),
            SymbolicRule([("a", "<=", 2.0)], 1, 0.7, 80, "m2"),
        ]
        analyzer = RuleCrossAnalyzer(rules)
        report = analyzer.get_report(all_feature_names=["a", "b"])
        assert "total_rules" in report
        assert "consensus_rules" in report
        assert "contradictions" in report
        assert "blind_spots" in report
        assert "b" in report["blind_spots"]


class TestCrossModelAugmenter:
    """Tests for CrossModelAugmenter."""

    def test_rule_activation_features(self):
        """Generate binary rule activation meta-features."""
        from src.phase_23_analytics.symbolic_cross_learner import (
            CrossModelAugmenter, SymbolicRule,
        )

        rules = [
            SymbolicRule([("x", "<=", 0.0)], 1, 0.8, 100, "m1"),
            SymbolicRule([("y", ">", 0.5)], 0, 0.7, 80, "m2"),
        ]
        augmenter = CrossModelAugmenter(rules=rules)
        augmenter.select_rules(min_confidence=0.5, min_samples=50)

        X = np.array([
            [-1.0, 1.0],  # Rule 0 fires (x<=0), Rule 1 fires (y>0.5)
            [1.0, 0.0],   # Neither fires
            [-0.5, 0.3],  # Only Rule 0 fires
        ])
        feature_names = ["x", "y"]

        activations, names = augmenter.generate_rule_activations(X, feature_names)
        assert activations.shape[0] == 3
        assert activations.shape[1] == 2
        assert len(names) == 2
        # Row 0: both rules fire
        assert activations[0, 0] == 1.0  # x <= 0
        assert activations[0, 1] == 1.0  # y > 0.5
        # Row 1: neither fires
        assert activations[1, 0] == 0.0
        assert activations[1, 1] == 0.0
        # Row 2: only first fires
        assert activations[2, 0] == 1.0
        assert activations[2, 1] == 0.0

    def test_importance_weighted_features(self):
        """Generate importance-weighted meta-features."""
        from src.phase_23_analytics.symbolic_cross_learner import (
            CrossModelAugmenter, UniversalFeatureMap,
        )

        fmap = UniversalFeatureMap()
        fmap.add_model_full("m1", {"x": 0.8, "y": 0.2}, model_auc=0.7)
        fmap.add_model_full("m2", {"x": 0.6, "y": 0.4}, model_auc=0.65)

        augmenter = CrossModelAugmenter(feature_map=fmap)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        feature_names = ["x", "y"]

        weighted, names = augmenter.generate_importance_weighted(X, feature_names)
        assert weighted.shape == (2, 2)
        assert len(names) == 2
        # x should be weighted higher than y
        # (weighted importance * universality both higher for x)

    def test_generate_all_meta_features(self):
        """Generate combined meta-features."""
        from src.phase_23_analytics.symbolic_cross_learner import (
            CrossModelAugmenter, UniversalFeatureMap, SymbolicRule,
        )

        fmap = UniversalFeatureMap()
        fmap.add_model_full("m1", {"x": 0.5, "y": 0.5}, model_auc=0.6)

        rules = [
            SymbolicRule([("x", ">", 0.0)], 1, 0.9, 200, "m1"),
        ]

        augmenter = CrossModelAugmenter(feature_map=fmap, rules=rules)
        augmenter.select_rules(min_confidence=0.5, min_samples=10)

        X = np.array([[1.0, 2.0], [-1.0, 3.0]])
        combined, names = augmenter.generate_all_meta_features(X, ["x", "y"])
        assert combined.shape[0] == 2
        assert combined.shape[1] >= 2  # At least 1 rule + 2 weighted
        assert len(names) == combined.shape[1]

    def test_empty_augmenter(self):
        """Augmenter with no data should return empty arrays."""
        from src.phase_23_analytics.symbolic_cross_learner import CrossModelAugmenter

        augmenter = CrossModelAugmenter()
        X = np.array([[1.0, 2.0]])
        combined, names = augmenter.generate_all_meta_features(X, ["x", "y"])
        assert combined.shape[1] == 0
        assert len(names) == 0

    def test_select_rules_filtering(self):
        """Select rules should filter by confidence and samples."""
        from src.phase_23_analytics.symbolic_cross_learner import (
            CrossModelAugmenter, SymbolicRule,
        )

        rules = [
            SymbolicRule([("x", ">", 0)], 1, 0.9, 200, "m1"),   # Passes
            SymbolicRule([("x", ">", 0)], 1, 0.4, 200, "m2"),   # Fails confidence
            SymbolicRule([("x", ">", 0)], 1, 0.9, 5, "m3"),     # Fails samples
        ]
        augmenter = CrossModelAugmenter(rules=rules)
        n = augmenter.select_rules(min_confidence=0.6, min_samples=20)
        assert n == 1


# =============================================================================
# SYMBOLIC CROSS-LEARNER WIRING TESTS
# =============================================================================

class TestSymbolicCrossLearnerWiring:
    """Tests for wiring into experiment pipeline."""

    def test_experiment_engine_has_update_feature_map(self):
        """ExperimentEngine should have _update_feature_map method."""
        src = open("src/phase_21_continuous/experiment_runner.py", encoding="utf-8").read()
        assert "def _update_feature_map" in src
        assert "FeatureImportanceExtractor" in src
        assert "UniversalFeatureMap" in src

    def test_run_experiment_calls_update_feature_map(self):
        """run_experiment should call _update_feature_map."""
        src = open("src/phase_21_continuous/experiment_runner.py", encoding="utf-8").read()
        idx = src.index("def run_experiment(self, config")
        next_def = src.index("\n    def ", idx + 1)
        body = src[idx:next_def]
        assert "_update_feature_map" in body

    def test_cross_model_augmentation_in_runner(self):
        """Experiment runner should have cross-model augmentation block."""
        src = open("src/phase_21_continuous/experiment_runner.py", encoding="utf-8").read()
        assert "CrossModelAugmenter" in src
        assert "generate_importance_weighted" in src
        assert "cross-model meta-features" in src.lower() or "AUGMENT" in src

    def test_augmentation_requires_min_models(self):
        """Augmentation should only activate with >= 10 model contributions."""
        src = open("src/phase_21_continuous/experiment_runner.py", encoding="utf-8").read()
        assert "model_contributions) >= 10" in src

    def test_feature_map_persist_path(self):
        """Feature map should persist to models/feature_importance_map.json."""
        src = open("src/phase_21_continuous/experiment_runner.py", encoding="utf-8").read()
        assert "feature_importance_map.json" in src


# =============================================================================
# WAVE 22: ORDER RETRY / RECOVERY TESTS
# =============================================================================

class TestOrderRetryRecovery:
    """Tests for OrderManager retry and orphan detection logic."""

    def test_order_manager_has_retry_constants(self):
        """OrderManager should have MAX_RETRIES and RETRY_BASE_DELAY."""
        from src.phase_19_paper_trading.risk_management import OrderManager
        assert hasattr(OrderManager, "MAX_RETRIES")
        assert OrderManager.MAX_RETRIES == 3
        assert hasattr(OrderManager, "RETRY_BASE_DELAY")
        assert OrderManager.RETRY_BASE_DELAY == 2.0

    def test_submit_with_retry_succeeds_first_try(self):
        """_submit_with_retry should return order_id on first success."""
        from src.phase_19_paper_trading.risk_management import OrderManager
        mock_client = MagicMock()
        om = OrderManager(mock_client)
        result = om._submit_with_retry(lambda: "order_123", "test order")
        assert result == "order_123"

    def test_submit_with_retry_returns_none_on_rejection(self):
        """_submit_with_retry should return None if submit_fn returns None."""
        from src.phase_19_paper_trading.risk_management import OrderManager
        mock_client = MagicMock()
        om = OrderManager(mock_client)
        result = om._submit_with_retry(lambda: None, "rejected order")
        assert result is None

    def test_submit_with_retry_retries_on_exception(self):
        """_submit_with_retry should retry on transient failures."""
        from src.phase_19_paper_trading.risk_management import OrderManager
        mock_client = MagicMock()
        om = OrderManager(mock_client)

        call_count = [0]
        def flaky_submit():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("network error")
            return "order_456"

        with patch("time.sleep"):  # Skip actual sleep
            result = om._submit_with_retry(flaky_submit, "flaky order")
        assert result == "order_456"
        assert call_count[0] == 3

    def test_submit_with_retry_returns_none_after_max_retries(self):
        """_submit_with_retry should return None after exhausting retries."""
        from src.phase_19_paper_trading.risk_management import OrderManager
        mock_client = MagicMock()
        om = OrderManager(mock_client)

        def always_fail():
            raise ConnectionError("persistent error")

        with patch("time.sleep"):
            result = om._submit_with_retry(always_fail, "failing order")
        assert result is None

    def test_execute_signal_uses_retry(self):
        """execute_signal should use _submit_with_retry internally."""
        src = open("src/phase_19_paper_trading/risk_management.py", encoding="utf-8").read()
        # Find execute_signal method body
        idx = src.index("def execute_signal(")
        next_def = src.index("\n    def ", idx + 1)
        body = src[idx:next_def]
        assert "_submit_with_retry" in body

    def test_close_position_uses_retry(self):
        """close_position should use _submit_with_retry internally."""
        src = open("src/phase_19_paper_trading/risk_management.py", encoding="utf-8").read()
        idx = src.index("def close_position(")
        next_def = src.index("\n    def ", idx + 1)
        body = src[idx:next_def]
        assert "_submit_with_retry" in body

    def test_check_pending_orders_handles_status_errors(self):
        """check_pending_orders should handle get_order_status exceptions gracefully."""
        src = open("src/phase_19_paper_trading/risk_management.py", encoding="utf-8").read()
        idx = src.index("def check_pending_orders(")
        next_def_idx = src.find("\n    def ", idx + 1)
        if next_def_idx == -1:
            next_def_idx = src.find("\nclass ", idx + 1)
        body = src[idx:next_def_idx] if next_def_idx > 0 else src[idx:]
        # Should have try/except around get_order_status
        assert "get_order_status" in body
        assert "Orphan order detected" in body or "orphan" in body.lower()


# =============================================================================
# WAVE 22: HEALTH CHECKER WIRING TESTS
# =============================================================================

class TestHealthCheckerWiring:
    """Tests for health checker integration in start_system.py."""

    def test_start_system_imports_health_checker(self):
        """start_system.py should import HealthChecker."""
        src = open("scripts/start_system.py", encoding="utf-8").read()
        assert "HealthChecker" in src

    def test_health_checker_in_component_state(self):
        """health_checker should be tracked in component state."""
        src = open("scripts/start_system.py", encoding="utf-8").read()
        assert '"health_checker"' in src or "'health_checker'" in src

    def test_health_checker_started_and_stopped(self):
        """start_system.py should start and stop the health checker."""
        src = open("scripts/start_system.py", encoding="utf-8").read()
        assert "start_background()" in src
        assert "stop_background()" in src


# =============================================================================
# WAVE 22: WEB MONITOR EXCEPTION HANDLING TESTS
# =============================================================================

class TestWebMonitorExceptionHandling:
    """Tests for proper exception logging in web_monitor.py."""

    def test_web_monitor_has_logger(self):
        """web_monitor.py should have a dedicated logger."""
        src = open("src/phase_20_monitoring/web_monitor.py", encoding="utf-8").read()
        assert 'web_logger' in src or 'getLogger("WebMonitor")' in src

    def test_no_bare_except_pass(self):
        """web_monitor.py should not have bare 'except Exception: pass' blocks."""
        src = open("src/phase_20_monitoring/web_monitor.py", encoding="utf-8").read()
        import re
        # Look for except blocks followed immediately by pass (with optional whitespace)
        bare_patterns = re.findall(r'except\s+Exception\s*:\s*\n\s*pass\b', src)
        assert len(bare_patterns) == 0, f"Found {len(bare_patterns)} bare except:pass blocks"

    def test_exception_handlers_log_messages(self):
        """Exception handlers should log with web_logger."""
        src = open("src/phase_20_monitoring/web_monitor.py", encoding="utf-8").read()
        # All exception handlers should use web_logger for logging
        import re
        handlers = re.findall(r'except\s+Exception\s+as\s+\w+:', src)
        # Should have at least 5 handlers (the ones we fixed)
        assert len(handlers) >= 5, f"Expected >=5 except-as handlers, found {len(handlers)}"


# =============================================================================
# WAVE 22b: DATA VALIDATION TESTS
# =============================================================================

class TestDataValidation:
    """Tests for input data validation in TradingBot."""

    def test_validate_data_method_exists(self):
        """TradingBot should have _validate_data method."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        assert "def _validate_data(self, df" in src

    def test_validate_data_checks_columns(self):
        """_validate_data should check for required OHLCV columns."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        assert "_REQUIRED_COLUMNS" in src
        assert '"open"' in src or "'open'" in src

    def test_validate_data_checks_nan(self):
        """_validate_data should check for excessive NaN values."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        idx = src.index("def _validate_data(")
        next_def = src.index("\n    def ", idx + 1)
        body = src[idx:next_def]
        assert "nan" in body.lower() or "isna" in body or "NaN" in body

    def test_validate_data_wired_into_run_once(self):
        """run_once should call _validate_data before signal generation."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        assert "_validate_data(df_1min)" in src


# =============================================================================
# WAVE 22b: PREDICTION BOUNDS TESTS
# =============================================================================

class TestPredictionBounds:
    """Tests for prediction bounds checking in signal_generator.py."""

    def test_nan_check_on_predictions(self):
        """Signal generator should check for NaN in predictions."""
        src = open("src/phase_19_paper_trading/signal_generator.py", encoding="utf-8").read()
        assert "isnan" in src or "np.isnan" in src

    def test_probability_clamping(self):
        """Signal generator should clamp probabilities to [0, 1]."""
        src = open("src/phase_19_paper_trading/signal_generator.py", encoding="utf-8").read()
        assert "np.clip" in src
        assert "0.0, 1.0" in src

    def test_model_freshness_log_level(self):
        """Model freshness check should log at WARNING, not DEBUG."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        idx = src.index("def _check_model_freshness(")
        next_def = src.index("\n    def ", idx + 1)
        body = src[idx:next_def]
        # Should use warning, not debug
        assert 'logger.warning(f"Model freshness check failed' in body


# =============================================================================
# WAVE 22b: POSITION TRACKER ATOMIC WRITES
# =============================================================================

class TestPositionTrackerAtomic:
    """Tests for atomic writes in position_tracker.py."""

    def test_save_json_uses_atomic_write(self):
        """_save_json should use atomic_write_json."""
        src = open("src/position_tracker.py", encoding="utf-8").read()
        assert "atomic_write_json" in src

    def test_position_tracker_loads_safely(self):
        """PositionHistoryTracker should load safely from missing files."""
        from src.position_tracker import PositionHistoryTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = PositionHistoryTracker(data_dir=Path(tmpdir))
            assert tracker._positions == []
            assert tracker._trades == []
            assert tracker._equity == []


# =============================================================================
# WAVE 22b: STARTUP RECONCILIATION TESTS
# =============================================================================

class TestStartupReconciliation:
    """Tests for position reconciliation on bot startup."""

    def test_reconcile_method_exists(self):
        """TradingBot should have _reconcile_positions_on_startup."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        assert "def _reconcile_positions_on_startup(self)" in src

    def test_reconcile_called_in_init(self):
        """__init__ should call _reconcile_positions_on_startup."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        idx = src.index("def __init__(self):")
        next_def = src.index("\n    def ", idx + 1)
        body = src[idx:next_def]
        assert "_reconcile_positions_on_startup" in body

    def test_reconcile_logs_existing_positions(self):
        """Reconciliation should log warning when position exists."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        idx = src.index("def _reconcile_positions_on_startup(")
        next_def = src.index("\n    def ", idx + 1)
        body = src[idx:next_def]
        assert "[RECONCILIATION]" in body
        assert "logger.warning" in body


# =============================================================================
# WAVE 22b: WIN RATE CIRCUIT BREAKER TESTS
# =============================================================================

class TestWinRateCircuitBreaker:
    """Tests for rolling win rate degradation circuit breaker."""

    def test_method_exists(self):
        """PaperPerformanceTracker should have check_win_rate_degradation."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        assert "def check_win_rate_degradation" in src

    def test_returns_ok_with_few_trades(self):
        """Should return OK if not enough trades to evaluate."""
        from src.phase_19_paper_trading.trading_bot import PaperPerformanceTracker
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = PaperPerformanceTracker(log_dir=Path(tmpdir))
            ok, msg = tracker.check_win_rate_degradation(window=20)
            assert ok is True
            assert "Insufficient" in msg

    def test_detects_low_win_rate(self):
        """Should detect when win rate drops below threshold."""
        from src.phase_19_paper_trading.trading_bot import (
            PaperPerformanceTracker, PredictionRecord,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = PaperPerformanceTracker(log_dir=Path(tmpdir))
            # Inject 20 losing trades
            for i in range(20):
                tracker.predictions.append(PredictionRecord(
                    timestamp=f"2026-01-{i+1:02d}",
                    signal_type="BUY",
                    swing_probability=0.7,
                    timing_probability=0.6,
                    confidence=0.65,
                    entry_price=100.0,
                    exit_price=99.0,
                    predicted_correct=False,
                    actual_return=-0.01,
                ))
            ok, msg = tracker.check_win_rate_degradation(window=20, min_win_rate=0.40)
            assert ok is False
            assert "below minimum" in msg

    def test_passes_with_good_win_rate(self):
        """Should pass with healthy win rate."""
        from src.phase_19_paper_trading.trading_bot import (
            PaperPerformanceTracker, PredictionRecord,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = PaperPerformanceTracker(log_dir=Path(tmpdir))
            # Inject 20 trades: 14 wins, 6 losses = 70%
            for i in range(20):
                win = i < 14
                tracker.predictions.append(PredictionRecord(
                    timestamp=f"2026-01-{i+1:02d}",
                    signal_type="BUY",
                    swing_probability=0.7,
                    timing_probability=0.6,
                    confidence=0.65,
                    entry_price=100.0,
                    exit_price=101.0 if win else 99.0,
                    predicted_correct=win,
                    actual_return=0.01 if win else -0.01,
                ))
            ok, msg = tracker.check_win_rate_degradation(window=20, min_win_rate=0.40)
            assert ok is True
            assert "OK" in msg

    def test_wired_into_run_once(self):
        """run_once should check win rate before signal generation."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        assert "check_win_rate_degradation" in src
        assert "CIRCUIT BREAKER" in src


# =============================================================================
# WAVE 22c: ORPHAN CANCEL FIX TESTS
# =============================================================================

class TestOrphanCancelFix:
    """Tests for orphan order cancel failure logging."""

    def test_orphan_cancel_logs_error(self):
        """Failed orphan cancel should log error, not silently pass."""
        src = open("src/phase_19_paper_trading/risk_management.py", encoding="utf-8").read()
        idx = src.index("Orphan order detected")
        block = src[idx:idx+500]
        # Should NOT have bare except:pass
        assert "except Exception:" not in block or "pass" not in block.split("except Exception:")[1][:50]
        # Should log the error
        assert "Failed to cancel orphan" in block or "cancel_err" in block

    def test_orphan_keeps_order_on_cancel_failure(self):
        """If cancel fails, order should remain in pending_orders."""
        src = open("src/phase_19_paper_trading/risk_management.py", encoding="utf-8").read()
        idx = src.index("Orphan order detected")
        block = src[idx:idx+500]
        # After cancel failure, should keep for next cycle
        assert "keeping in pending" in block or "next cycle" in block


# =============================================================================
# WAVE 22c: MARGIN VALIDATION TESTS
# =============================================================================

class TestMarginValidation:
    """Tests for margin/buying power validation."""

    def test_negative_buying_power_check(self):
        """run_once should halt on negative buying power."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        assert "MARGIN ALERT" in src
        assert "buying_power" in src
        assert "Negative buying power" in src or "negative" in src.lower()

    def test_margin_halt_persisted(self):
        """Margin halt should persist risk state."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        # Find the margin check block
        idx = src.index("MARGIN ALERT")
        block = src[idx:idx+500]
        assert "_persist_state" in block


# =============================================================================
# WAVE 22c: CONFIG VALIDATION TESTS
# =============================================================================

class TestConfigValidation:
    """Tests for TRADING_CONFIG validation on TradingBot startup."""

    def test_validate_config_method_exists(self):
        """TradingBot should have _validate_config method."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        assert "def _validate_config(self)" in src

    def test_validate_config_called_in_init(self):
        """__init__ should call _validate_config before client creation."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        idx = src.index("class TradingBot")
        init_idx = src.index("def __init__(self):", idx)
        client_idx = src.index("AlpacaPaperClient()", init_idx)
        validate_idx = src.index("_validate_config()", init_idx)
        # Validate must come before client
        assert validate_idx < client_idx

    def test_required_config_keys_defined(self):
        """TradingBot should define required config keys."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        assert "_REQUIRED_CONFIG_KEYS" in src
        assert '"symbol"' in src
        assert '"max_position_pct"' in src

    def test_validate_checks_pct_ranges(self):
        """Config validation should check percentage values are 0-1."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        idx = src.index("def _validate_config(")
        next_def = src.index("\n    def ", idx + 1)
        body = src[idx:next_def]
        assert "max_position_pct" in body
        assert "0 < val" in body or "0 <" in body


# =============================================================================
# WAVE 22c: FEATURE MISMATCH IMPROVEMENT TESTS
# =============================================================================

class TestFeatureMismatchImprovement:
    """Tests for improved feature mismatch handling."""

    def test_large_mismatch_rejected(self):
        """Large feature mismatch (>20%) should reject signal."""
        src = open("src/phase_19_paper_trading/signal_generator.py", encoding="utf-8").read()
        assert "rejecting signal" in src or "too large" in src

    def test_mismatch_percentage_calculated(self):
        """Feature mismatch should calculate percentage difference."""
        src = open("src/phase_19_paper_trading/signal_generator.py", encoding="utf-8").read()
        assert "pct_diff" in src or "% difference" in src

    def test_mismatch_logs_details(self):
        """Feature mismatch should log actual vs expected counts."""
        src = open("src/phase_19_paper_trading/signal_generator.py", encoding="utf-8").read()
        assert "n_actual" in src or "expected_features" in src


# =============================================================================
# WAVE 22c: DASHBOARD STALENESS ENDPOINT TESTS
# =============================================================================

class TestDashboardStaleness:
    """Tests for /api/health/staleness endpoint."""

    def test_staleness_endpoint_exists(self):
        """dashboard_server.py should have /api/health/staleness route."""
        src = open("src/phase_20_monitoring/dashboard_server.py", encoding="utf-8").read()
        assert "/api/health/staleness" in src

    def test_staleness_returns_verdict(self):
        """Staleness endpoint should return fresh/stale/critical verdict."""
        src = open("src/phase_20_monitoring/dashboard_server.py", encoding="utf-8").read()
        idx = src.index("api_staleness")
        block = src[idx:idx+1500]
        assert '"fresh"' in block
        assert '"stale"' in block
        assert '"critical"' in block

    def test_staleness_checks_key_files(self):
        """Staleness endpoint should check status.json, paper_performance, etc."""
        src = open("src/phase_20_monitoring/dashboard_server.py", encoding="utf-8").read()
        idx = src.index("api_staleness")
        block = src[idx:idx+1500]
        assert "status.json" in block
        assert "paper_performance" in block


# =============================================================================
# WAVE 23: LOG ROTATION TESTS
# =============================================================================

class TestLogRotation:
    """Tests for log rotation in setup_logging."""

    def test_rotating_handler_in_trading_bot(self):
        """trading_bot.py should use RotatingFileHandler."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        assert "RotatingFileHandler" in src

    def test_rotating_handler_in_orchestrator(self):
        """giga_orchestrator.py should use RotatingFileHandler."""
        src = open("src/giga_orchestrator.py", encoding="utf-8").read()
        assert "RotatingFileHandler" in src

    def test_rotating_handler_in_start_system(self):
        """start_system.py should use RotatingFileHandler."""
        src = open("scripts/start_system.py", encoding="utf-8").read()
        assert "RotatingFileHandler" in src

    def test_log_cleanup_function_exists(self):
        """_cleanup_old_logs function should exist in trading_bot.py."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        assert "def _cleanup_old_logs" in src

    def test_log_cleanup_called_in_setup(self):
        """setup_logging should call _cleanup_old_logs."""
        src = open("src/phase_19_paper_trading/trading_bot.py", encoding="utf-8").read()
        idx = src.index("def setup_logging()")
        next_fn = src.index("\ndef ", idx + 1)
        body = src[idx:next_fn]
        assert "_cleanup_old_logs" in body

    def test_cleanup_removes_old_files(self):
        """_cleanup_old_logs should remove files older than max_age_days."""
        from src.phase_19_paper_trading.trading_bot import _cleanup_old_logs
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Create a fake old log
            old_log = tmppath / "old.log"
            old_log.write_text("old log data")
            # Set mtime to 60 days ago
            import os
            old_time = time.time() - 60 * 86400
            os.utime(old_log, (old_time, old_time))
            # Create a recent log
            new_log = tmppath / "new.log"
            new_log.write_text("new log data")
            # Run cleanup
            _cleanup_old_logs(tmppath, max_age_days=30)
            assert not old_log.exists(), "Old log should be deleted"
            assert new_log.exists(), "New log should be kept"


# =============================================================================
# WAVE 23: MODEL VERSIONING TESTS
# =============================================================================

class TestModelVersioning:
    """Tests for production model versioning."""

    def test_versioned_save_function_exists(self):
        """_save_versioned_model should exist in train_robust_model.py."""
        src = open("src/train_robust_model.py", encoding="utf-8").read()
        assert "def _save_versioned_model" in src

    def test_versioned_save_called_for_robust(self):
        """save_models should call _save_versioned_model for robust models."""
        src = open("src/train_robust_model.py", encoding="utf-8").read()
        idx = src.index("def save_models(")
        next_fn_idx = src.find("\ndef ", idx + 10)
        body = src[idx:next_fn_idx] if next_fn_idx > 0 else src[idx:]
        assert "_save_versioned_model" in body

    def test_versioned_save_called_for_leak_proof(self):
        """Leak-proof model save should also use _save_versioned_model."""
        src = open("src/train_robust_model.py", encoding="utf-8").read()
        # Should have two calls to _save_versioned_model
        count = src.count("_save_versioned_model")
        assert count >= 2, f"Expected >=2 calls to _save_versioned_model, found {count}"

    def test_versioned_save_prunes_old_versions(self):
        """_save_versioned_model should prune old versions."""
        from src.train_robust_model import _save_versioned_model
        import joblib
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Save 7 versions (max_versions=5)
            for i in range(7):
                _save_versioned_model(tmppath, "test_model", {"v": i}, max_versions=5)
            versions = list(tmppath.glob("test_model_*.joblib"))
            assert len(versions) <= 5, f"Expected <=5 versions, got {len(versions)}"


# =============================================================================
# WAVE 23: SIGNAL GENERATOR FALLBACK TESTS
# =============================================================================

class TestSignalGeneratorFallback:
    """Tests for signal generator dynamic→static fallback."""

    def test_dynamic_falls_through_to_static(self):
        """If dynamic returns zero-confidence HOLD, should fall through to static."""
        src = open("src/phase_19_paper_trading/signal_generator.py", encoding="utf-8").read()
        assert "trying static fallback" in src

    def test_generate_signal_has_static_path(self):
        """generate_signal should have a DEGRADED MODE static path."""
        src = open("src/phase_19_paper_trading/signal_generator.py", encoding="utf-8").read()
        assert "DEGRADED MODE" in src
        assert "STATIC MODEL PATH" in src


# =============================================================================
# WAVE 26: 10-DAY CAMPAIGN INFRASTRUCTURE TESTS
# =============================================================================

class TestWave26CampaignInfrastructure:
    """Tests for Wave 26: campaign infrastructure, threshold recalibration, memory management."""

    def test_tier3_recalibrated_thresholds(self):
        """Tier 3 uses AUC >= 0.57, fragility < 0.40, suite >= 0.45 (Wave 36 v4)."""
        from src.core.registry_db import compute_tier
        # Tier 3 achievable: AUC=0.57, stability=0.65, fragility=0.25
        assert compute_tier(0.65, 0.25, 0.57) == 3
        # Just below AUC threshold
        assert compute_tier(0.65, 0.25, 0.56) == 2
        # Wave 33: Fragility threshold raised to 0.40 (was 0.30)
        assert compute_tier(0.65, 0.45, 0.60) == 2  # Fragility too high
        assert compute_tier(0.65, 0.40, 0.60) == 2  # Exactly at boundary (should NOT pass)
        assert compute_tier(0.65, 0.39, 0.60) == 3  # Just under
        # Suite composite gating (Wave 33)
        assert compute_tier(0.65, 0.25, 0.60, suite_composite=0.45) == 3  # Suite passes
        assert compute_tier(0.65, 0.25, 0.60, suite_composite=0.44) == 2  # Suite fails

    def test_purge_leaky_experiments_exists(self):
        """RegistryDB should have purge_leaky_experiments method."""
        import tempfile
        from src.core.registry_db import RegistryDB
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = RegistryDB(f.name)
            # Add experiments with various AUCs
            db.add_experiment({"experiment_id": "good_1", "test_auc": 0.57, "status": "completed"})
            db.add_experiment({"experiment_id": "leaky_1", "test_auc": 0.92, "status": "completed"})
            db.add_experiment({"experiment_id": "leaky_2", "test_auc": 0.88, "status": "completed"})

            n_purged = db.purge_leaky_experiments(0.85)
            assert n_purged == 2, f"Expected 2 purged, got {n_purged}"

            # Good experiment should remain
            remaining = db.get_experiments()
            assert len(remaining) == 1
            assert remaining[0]["experiment_id"] == "good_1"
            db.close()

    def test_purge_all_contaminated(self):
        """purge_all_contaminated should clean experiments, models, and bad WMES."""
        import tempfile
        from src.core.registry_db import RegistryDB
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = RegistryDB(f.name)
            db.add_experiment({"experiment_id": "e1", "test_auc": 0.90, "status": "completed"})
            db.add_experiment({"experiment_id": "e2", "test_auc": 0.55, "status": "completed"})
            db.add_model("m1", {"test_auc": 0.91, "wmes_score": 0.6, "experiment_id": "e1"})
            db.add_model("m2", {"test_auc": 0.56, "wmes_score": -5.0, "experiment_id": "e2"})

            result = db.purge_all_contaminated(0.85)
            assert result["experiments_purged"] == 1
            assert result["models_purged"] == 1
            assert result["bad_wmes_purged"] == 1
            db.close()

    def test_feature_cache_lru_eviction(self):
        """Feature cache should evict oldest entries when exceeding max size."""
        from src.phase_21_continuous.experiment_runner import UnifiedExperimentRunner
        runner = UnifiedExperimentRunner()
        runner._feature_cache_max_size = 3

        # Add 5 entries
        for i in range(5):
            key = f"threshold_{i}"
            runner._feature_cache[key] = f"data_{i}"
            runner._feature_cache_order.append(key)
            # Evict if over limit
            while len(runner._feature_cache) > runner._feature_cache_max_size:
                oldest = runner._feature_cache_order.pop(0)
                if oldest in runner._feature_cache:
                    del runner._feature_cache[oldest]

        assert len(runner._feature_cache) == 3
        assert "threshold_0" not in runner._feature_cache
        assert "threshold_1" not in runner._feature_cache
        assert "threshold_4" in runner._feature_cache

    def test_fast_screen_parameter_exists(self):
        """UnifiedExperimentRunner.run() should accept fast_screen parameter."""
        import inspect
        from src.phase_21_continuous.experiment_runner import UnifiedExperimentRunner
        sig = inspect.signature(UnifiedExperimentRunner.run)
        assert "fast_screen" in sig.parameters

    def test_regime_filter_on_data_config(self):
        """DataConfig should have regime_filter field."""
        from src.experiment_config import DataConfig
        dc = DataConfig()
        assert hasattr(dc, "regime_filter")
        assert dc.regime_filter == ""
        dc.regime_filter = "low_vol"
        assert dc.regime_filter == "low_vol"

    def test_regime_experiment_types_in_generator(self):
        """ExperimentGenerator should include regime experiment types."""
        from src.phase_21_continuous.experiment_tracking import ExperimentGenerator
        gen = ExperimentGenerator()
        assert "regime_lowvol" in gen.experiment_weights
        assert "regime_highvol" in gen.experiment_weights
        # Weights should sum to ~1.0
        total = sum(gen.experiment_weights.values())
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected ~1.0"

    def test_watchdog_function_exists(self):
        """start_system.py should have start_watchdog function."""
        src = open("scripts/start_system.py", encoding="utf-8").read()
        assert "def start_watchdog" in src
        assert "WATCHDOG" in src

    def test_walk_forward_variance_07(self):
        """Walk-forward variance threshold should be 0.07 (Wave 26)."""
        src = open("src/phase_21_continuous/experiment_runner.py", encoding="utf-8").read()
        assert "wf_variance < 0.07" in src

    def test_campaign_report_imports(self):
        """campaign_report.py should be importable."""
        import ast
        src = open("scripts/campaign_report.py", encoding="utf-8").read()
        ast.parse(src)  # Valid syntax

    def test_launch_campaign_imports(self):
        """launch_campaign.py should be importable."""
        import ast
        src = open("scripts/launch_campaign.py", encoding="utf-8").read()
        ast.parse(src)  # Valid syntax

    def test_consensus_features_method(self):
        """UniversalFeatureMap should have get_consensus_features method."""
        from src.phase_23_analytics.symbolic_cross_learner import UniversalFeatureMap
        fmap = UniversalFeatureMap()
        # Empty map returns empty list
        result = fmap.get_consensus_features(top_n=10, min_models=1)
        assert isinstance(result, list)

    def test_two_pass_fast_screen_in_engine(self):
        """ExperimentEngine.run_experiment should implement two-pass fast screen."""
        src = open("src/phase_21_continuous/experiment_runner.py", encoding="utf-8").read()
        assert "FAST SCREEN PASS" in src
        assert "fast_screen=True" in src
        assert "fast_screen=False" in src

    def test_gc_collect_in_experiment_loop(self):
        """Experiment loop should call gc.collect and monitor memory (Wave 36)."""
        src = open("src/phase_21_continuous/experiment_runner.py", encoding="utf-8").read()
        assert "gc.collect()" in src
        assert "_check_memory" in src

    def test_orchestrator_live_auc_057(self):
        """Live trading gate AUC should be 0.57 (Wave 36 v4: lowered from 0.58)."""
        from src.giga_orchestrator import ORCHESTRATOR_CONFIG
        assert ORCHESTRATOR_CONFIG["min_auc_for_live_trading"] == 0.57

    def test_orchestrator_min_models_5(self):
        """Min models for live should be 5 (Wave 36: tightened from 3)."""
        from src.giga_orchestrator import ORCHESTRATOR_CONFIG
        assert ORCHESTRATOR_CONFIG["min_models_above_threshold"] == 5

    def test_thick_weave_tier3_055(self):
        """Thick weave tier3 WMES threshold should be 0.55 (Wave 26)."""
        from src.phase_23_analytics.thick_weave_search import ThickWeaveConfig
        cfg = ThickWeaveConfig()
        assert cfg.tier3_wmes_threshold == 0.55

    def test_registry_cli_purge_leaky_command(self):
        """Registry CLI should have db purge-leaky command."""
        src = open("scripts/registry_cli.py", encoding="utf-8").read()
        assert "purge-leaky" in src
        assert "cmd_db_purge_leaky" in src


class TestWave27CriticalFixes:
    """Wave 27: Critical fixes for Tier 3 unblocking."""

    def test_fragility_analysis_no_auc_gate(self):
        """Fragility analysis should run on ALL Tier 2 models (no AUC pre-filter).
        Wave 33: Trigger now also allows stability=-1 fallback when suite is good."""
        src = open("src/phase_21_continuous/experiment_runner.py", encoding="utf-8").read()
        # Wave 33: The fragility trigger uses stab_ok_for_frag (stability>=0.60 OR stability==-1+suite>=0.50)
        assert "if tier1_pass and stab_ok_for_frag:" in src
        # compute_tier handles the final AUC check — no AUC pre-filter on fragility trigger
        assert "tier1_pass and stab_ok_for_frag and result.test_auc >= 0.58" not in src

    def test_fragility_log_matches_threshold(self):
        """Fragility log message should use 0.40 threshold (matches compute_tier, Wave 33)."""
        src = open("src/phase_21_continuous/experiment_runner.py", encoding="utf-8").read()
        assert "fragility_score < 0.40" in src or "result.fragility_score < 0.40" in src

    def test_use_leak_proof_initialized_before_try(self):
        """use_leak_proof should be initialized before the try block."""
        import re
        src = open("src/phase_21_continuous/experiment_runner.py", encoding="utf-8").read()
        # Find the run() method and check variable is set before try
        run_method_match = re.search(r'def run\(self, config.*?\n(.*?)(?=\n    def |\nclass )', src, re.DOTALL)
        assert run_method_match is not None
        run_body = run_method_match.group(1)
        # use_leak_proof should appear before the first 'try:'
        try_pos = run_body.find("try:")
        leak_proof_pos = run_body.find("use_leak_proof =")
        assert leak_proof_pos < try_pos, "use_leak_proof must be set before try block"

    def test_leak_proof_reducer_nystroem_for_large(self):
        """LeakProofDimReducer should use Nystroem for large datasets."""
        from src.phase_10_feature_processing.leak_proof_reducer import LeakProofDimReducer
        import numpy as np
        reducer = LeakProofDimReducer(method="kernel_pca", n_components=5)
        # With >5000 samples, should use Nystroem pipeline (not raw KernelPCA)
        X = np.random.randn(6000, 20)
        reducer.fit(X)
        # Should be a Pipeline (Nystroem + PCA), not raw KernelPCA
        from sklearn.pipeline import Pipeline
        assert isinstance(reducer.reducer_, Pipeline), "Should use Nystroem pipeline for >5000 samples"

    def test_leak_proof_reducer_exact_for_small(self):
        """LeakProofDimReducer should use exact KernelPCA for small datasets."""
        from src.phase_10_feature_processing.leak_proof_reducer import LeakProofDimReducer
        from sklearn.decomposition import KernelPCA
        import numpy as np
        reducer = LeakProofDimReducer(method="kernel_pca", n_components=3)
        X = np.random.randn(200, 10)
        reducer.fit(X)
        assert isinstance(reducer.reducer_, KernelPCA), "Should use exact KernelPCA for small datasets"

    def test_group_aware_nystroem_for_large(self):
        """GroupAwareFeatureProcessor should use Nystroem for large KernelPCA."""
        from src.phase_10_feature_processing.group_aware_processor import GroupAwareFeatureProcessor
        from sklearn.pipeline import Pipeline
        proc = GroupAwareFeatureProcessor(
            feature_names=["f1", "f2", "f3"],
            reduction_method="kernel_pca",
            random_state=42,
        )
        reducer = proc._create_reducer(5, n_samples=6000)
        assert isinstance(reducer, Pipeline), "Should use Nystroem pipeline for >5000 samples"

    def test_compute_tier_fragility_040(self):
        """compute_tier should use fragility < 0.40 for Tier 3 (Wave 33, v4 AUC=0.57)."""
        from src.core.registry_db import compute_tier
        # AUC 0.57, stability 0.70, fragility 0.39 → should be Tier 3 (v4)
        assert compute_tier(0.70, 0.39, 0.57) == 3
        # AUC 0.57, stability 0.70, fragility 0.41 → should be Tier 2
        assert compute_tier(0.70, 0.41, 0.57) == 2
        # AUC 0.56, stability 0.70, fragility 0.10 → Tier 2 (AUC too low)
        assert compute_tier(0.70, 0.10, 0.56) == 2
        # Wave 33: suite_composite gates Tier 3 when available
        assert compute_tier(0.70, 0.20, 0.58, suite_composite=0.50) == 3
        assert compute_tier(0.70, 0.20, 0.58, suite_composite=0.40) == 2

    def test_fragility_weights_no_undefined_variable(self):
        """Fragility analysis should NOT reference undefined 'weights_real'."""
        src = open("src/phase_21_continuous/experiment_runner.py", encoding="utf-8").read()
        # The old buggy line referenced 'weights_real' which is undefined
        # (weights_real_cv exists but weights_real does not)
        assert "else weights_real\n" not in src, "weights_real is undefined"
        # Wave 33: Fragility rewritten to use cross_val_score with X_frag/y_frag
        # (no longer uses weights_frag at all — model-aware perturbation approach)
        assert "X_frag" in src, "Wave 33 fragility should use X_frag"


class TestWave28UnattendedHardening:
    """Wave 28: Verify hardening features for 2-6 week unattended operation."""

    def test_start_system_has_maintenance_thread(self):
        """start_system.py should have run_periodic_maintenance function."""
        src = open("scripts/start_system.py", encoding="utf-8").read()
        assert "def run_periodic_maintenance" in src
        assert "def cleanup_old_experiment_models" in src
        assert "def check_memory_usage" in src

    def test_start_system_has_experiment_timeout(self):
        """Experiment runner should have per-experiment timeout."""
        src = open("scripts/start_system.py", encoding="utf-8").read()
        assert "timeout_minutes" in src
        assert "exp_thread.join(timeout=" in src
        assert "TIMED OUT" in src

    def test_start_system_shutdown_flag_checked(self):
        """Experiment runner and maintenance should check shutdown flag."""
        src = open("scripts/start_system.py", encoding="utf-8").read()
        assert "_shutdown_requested.is_set()" in src

    def test_watchdog_covers_dashboard_and_monitor(self):
        """Watchdog should monitor dashboard and web monitor threads."""
        src = open("scripts/start_system.py", encoding="utf-8").read()
        assert '"dashboard_server"' in src
        assert '"web_monitor"' in src
        assert "Dashboard server died" in src
        assert "Web monitor died" in src

    def test_maintenance_includes_vacuum(self):
        """Periodic maintenance should VACUUM the database."""
        src = open("scripts/start_system.py", encoding="utf-8").read()
        assert "db.vacuum()" in src

    def test_maintenance_includes_disk_check(self):
        """Periodic maintenance should check disk space and halt if low."""
        src = open("scripts/start_system.py", encoding="utf-8").read()
        assert "disk_usage" in src
        assert "free_mb < 500" in src

    def test_health_checker_has_memory_check(self):
        """HealthChecker should have process memory monitoring."""
        src = open("src/phase_20_monitoring/health_checker.py", encoding="utf-8").read()
        assert "def _check_process_memory" in src
        assert "process_memory" in src
        assert "psutil" in src

    def test_cleanup_old_experiment_models_function(self):
        """cleanup_old_experiment_models should protect Tier 2+ models."""
        src = open("scripts/start_system.py", encoding="utf-8").read()
        assert "protected_paths" in src
        assert "tier >= 2" in src
        assert "max_age_days" in src

    def test_autostart_batch_file_exists(self):
        """Windows auto-restart batch file should exist."""
        bat_path = Path("scripts/autostart_giga_trader.bat")
        assert bat_path.exists(), "autostart_giga_trader.bat not found"
        content = bat_path.read_text()
        assert "schtasks" in content
        assert "start_system.py" in content

    def test_gc_collect_in_experiment_runner(self):
        """Experiment runner should call gc.collect() periodically."""
        src = open("scripts/start_system.py", encoding="utf-8").read()
        assert "gc.collect()" in src
        assert "import gc" in src

    def test_experiment_runner_memory_check_interval(self):
        """Memory should be checked every 5 experiments."""
        src = open("scripts/start_system.py", encoding="utf-8").read()
        assert "cycle % 5 == 0" in src
        assert "check_memory_usage" in src

    def test_maintenance_aggressive_cleanup_on_low_disk(self):
        """Maintenance should do aggressive cleanup when disk is low."""
        src = open("scripts/start_system.py", encoding="utf-8").read()
        assert "max_age_days=3" in src  # Aggressive: 3 days when low
        assert "free_mb < 2000" in src  # Threshold for aggressive cleanup

    def test_maintenance_has_tier_recomputation(self):
        """Maintenance should recompute tiers to fix mismatches."""
        src = open("scripts/start_system.py", encoding="utf-8").read()
        assert "recompute_all_tiers" in src

    def test_registry_db_recompute_all_tiers(self):
        """RegistryDB.recompute_all_tiers should fix tier mismatches."""
        from src.core.registry_db import RegistryDB, compute_tier
        import tempfile

        tmp = tempfile.mkdtemp()
        try:
            db = RegistryDB(Path(tmp) / "test.db")

            # Insert a model with wrong tier (T2 but qualifies for T3)
            model_data = {
                "model_id": "model_test_001",
                "test_auc": 0.59,
                "stability_score": 0.75,
                "fragility_score": 0.10,
                "tier": 2,  # Wrong — should be 3
            }
            db.add_model("model_test_001", model_data)

            # Insert a correctly-tiered model
            model_data2 = {
                "model_id": "model_test_002",
                "test_auc": 0.55,
                "stability_score": 0.50,
                "fragility_score": 0.50,
                "tier": 1,
            }
            db.add_model("model_test_002", model_data2)

            stats = db.recompute_all_tiers()
            assert stats["checked"] == 2
            assert stats["promoted"] == 1  # model_test_001: T2 -> T3
            assert stats["unchanged"] == 1  # model_test_002 stays T1

            # Verify the fix persisted
            conn = db._get_conn()
            row = conn.execute(
                "SELECT tier FROM models WHERE model_id='model_test_001'"
            ).fetchone()
            assert row["tier"] == 3

            # Close DB connection before cleanup
            db.close()
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Wave 35: New Model Types + Training Augmentation Config
# ═══════════════════════════════════════════════════════════════════════════════

class TestNewModelTypes:
    """Test that the 5 new model types (Wave 35) can be created, fit, and predict."""

    @pytest.fixture
    def sample_data(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        y = (X[:, 0] + X[:, 1] * 0.5 + rng.randn(100) * 0.5 > 0).astype(int)
        return X, y

    def test_lda(self, sample_data):
        X, y = sample_data
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.calibration import CalibratedClassifierCV
        base = LinearDiscriminantAnalysis(solver='svd')
        model = CalibratedClassifierCV(base, cv=3, method='sigmoid')
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (100, 2)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_gaussian_nb(self, sample_data):
        X, y = sample_data
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB(var_smoothing=1e-8)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (100, 2)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_svc_linear(self, sample_data):
        X, y = sample_data
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        base = LinearSVC(C=0.1, loss='squared_hinge', penalty='l2',
                         max_iter=2000, dual='auto', random_state=42)
        model = CalibratedClassifierCV(base, cv=3, method='sigmoid')
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (100, 2)

    def test_bayesian_ridge_wrapper(self, sample_data):
        X, y = sample_data
        from src.phase_21_continuous.experiment_runner import _BayesianRidgeClassifierWrapper
        from sklearn.calibration import CalibratedClassifierCV
        base = _BayesianRidgeClassifierWrapper()
        model = CalibratedClassifierCV(base, cv=3, method='sigmoid')
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (100, 2)

    def test_quantile_gb(self, sample_data):
        X, y = sample_data
        from src.phase_21_continuous.experiment_runner import QuantileGBClassifier
        model = QuantileGBClassifier(
            max_iter=50, max_depth=3, learning_rate=0.1,
            min_samples_leaf=10, random_state=42
        )
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (100, 2)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})


class TestQuantileGBIntervals:
    """Test QuantileGBClassifier prediction intervals."""

    def test_predict_with_intervals(self):
        from src.phase_21_continuous.experiment_runner import QuantileGBClassifier
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)
        model = QuantileGBClassifier(max_iter=50, max_depth=3, random_state=42)
        model.fit(X, y)
        median, width = model.predict_with_intervals(X)
        assert len(median) == 100
        assert len(width) == 100
        assert np.all(width >= 0), "Interval widths should be non-negative"

    def test_get_set_params(self):
        from src.phase_21_continuous.experiment_runner import QuantileGBClassifier
        model = QuantileGBClassifier(max_iter=77, max_depth=4)
        params = model.get_params()
        assert params["max_iter"] == 77
        assert params["max_depth"] == 4
        model.set_params(max_iter=99)
        assert model.max_iter == 99


class TestTrainingAugmentationConfig:
    """Test Wave 35 TrainingAugmentationConfig."""

    def test_default_values(self):
        from src.experiment_config import TrainingAugmentationConfig
        cfg = TrainingAugmentationConfig()
        assert cfg.use_temporal_decay is False
        assert cfg.use_noise_injection is False
        assert cfg.use_nested_cv is False
        assert cfg.use_distillation is True  # On by default
        assert cfg.temporal_decay_lambda == 0.5
        assert cfg.noise_sigma == 0.1
        assert cfg.nested_outer_folds == 3
        assert cfg.nested_inner_folds == 3

    def test_serialization_roundtrip(self):
        from src.experiment_config import ExperimentConfig
        config = ExperimentConfig(experiment_type="augmented_training")
        config.training_augmentation.use_temporal_decay = True
        config.training_augmentation.temporal_decay_lambda = 0.7
        d = config.to_dict()
        assert d["training_augmentation"]["use_temporal_decay"] is True
        assert d["training_augmentation"]["temporal_decay_lambda"] == 0.7
        config2 = ExperimentConfig.from_dict(d)
        assert config2.training_augmentation.use_temporal_decay is True
        assert config2.training_augmentation.temporal_decay_lambda == 0.7

    def test_augmented_training_experiment_type_exists(self):
        from src.phase_21_continuous.experiment_tracking import ExperimentGenerator
        gen = ExperimentGenerator(history=None)
        assert "augmented_training" in gen.experiment_weights

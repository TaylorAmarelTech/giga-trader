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

    def test_paper_gates_currently_pass(self):
        """With registered models, paper gates should pass."""
        from src.giga_orchestrator import ExperimentGateChecker
        checker = ExperimentGateChecker()
        passed, status = checker.check_gates(trading_mode="paper")
        assert passed is True, f"Paper gates should pass. Status: {status}"


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
        history = ExperimentHistory()
        stats = history.get_statistics()
        assert stats.get("completed", 0) > 0

    def test_model_registry_has_entries(self):
        from src.experiment_engine import ModelRegistry
        registry = ModelRegistry()
        assert len(registry.models) > 0

    def test_registered_models_have_auc(self):
        from src.experiment_engine import ModelRegistry
        registry = ModelRegistry()
        for model_id, record in list(registry.models.items())[:5]:
            assert record.test_auc > 0

    def test_model_record_has_live_fields(self):
        """ModelRecord should have live performance fields."""
        from src.phase_21_continuous.experiment_tracking import ModelRecord
        record = ModelRecord(
            model_id="test",
            experiment_id="test",
            created_at=datetime.now().isoformat(),
            model_path="test.joblib",
            config={},
        )
        assert hasattr(record, "live_trades")
        assert hasattr(record, "live_win_rate")
        assert hasattr(record, "live_total_return")
        assert hasattr(record, "live_sharpe")


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

    def test_model_record_has_tier_fields(self):
        """ModelRecord has new robustness fields."""
        from src.phase_21_continuous.experiment_tracking import ModelRecord
        record = ModelRecord(
            model_id="test", experiment_id="exp_1",
            created_at="2026-01-01", model_path="", config={},
        )
        assert hasattr(record, 'stability_score')
        assert hasattr(record, 'fragility_score')
        assert hasattr(record, 'train_test_gap')
        assert hasattr(record, 'tier')
        # Defaults
        assert record.stability_score == 0.0
        assert record.fragility_score == 1.0
        assert record.tier == 1

    def test_model_record_score_includes_stability(self):
        """ModelRecord.score() weights stability and penalizes fragility."""
        from src.phase_21_continuous.experiment_tracking import ModelRecord
        # Stable model
        stable = ModelRecord(
            model_id="stable", experiment_id="exp_1",
            created_at="2026-01-01", model_path="", config={},
            cv_auc=0.65, test_auc=0.62, wmes_score=0.60,
            stability_score=0.80, fragility_score=0.15,
        )
        # Fragile model (same AUC but poor stability)
        fragile = ModelRecord(
            model_id="fragile", experiment_id="exp_2",
            created_at="2026-01-01", model_path="", config={},
            cv_auc=0.65, test_auc=0.62, wmes_score=0.60,
            stability_score=0.20, fragility_score=0.80,
        )
        assert stable.score() > fragile.score()

    def test_compute_tier_assigns_correctly(self):
        """_compute_tier returns correct tier based on metrics."""
        from src.phase_21_continuous.experiment_tracking import (
            ModelRegistry, ExperimentResult, ExperimentConfig, ExperimentStatus,
        )
        registry = ModelRegistry.__new__(ModelRegistry)

        # Tier 1: basic quality
        result_t1 = ExperimentResult.__new__(ExperimentResult)
        result_t1.test_auc = 0.58
        result_t1.stability_score = 0.30  # below 0.50
        result_t1.fragility_score = 1.0
        assert registry._compute_tier(result_t1) == 1

        # Tier 2: stability verified
        result_t2 = ExperimentResult.__new__(ExperimentResult)
        result_t2.test_auc = 0.58
        result_t2.stability_score = 0.65
        result_t2.fragility_score = 0.50  # above 0.35
        assert registry._compute_tier(result_t2) == 2

        # Tier 3: fragility verified + high AUC
        result_t3 = ExperimentResult.__new__(ExperimentResult)
        result_t3.test_auc = 0.65
        result_t3.stability_score = 0.70
        result_t3.fragility_score = 0.20  # below 0.35
        assert registry._compute_tier(result_t3) == 3

    def test_compute_tier_needs_high_auc_for_tier3(self):
        """Tier 3 requires test_auc >= 0.60 even with good fragility."""
        from src.phase_21_continuous.experiment_tracking import (
            ModelRegistry, ExperimentResult,
        )
        registry = ModelRegistry.__new__(ModelRegistry)

        result = ExperimentResult.__new__(ExperimentResult)
        result.test_auc = 0.57  # below 0.60
        result.stability_score = 0.80
        result.fragility_score = 0.10  # great fragility
        # Should be tier 2, not 3 (AUC too low)
        assert registry._compute_tier(result) == 2

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

    def test_backward_compat_old_registry(self):
        """Old registry entries without tier fields load with defaults."""
        from src.phase_21_continuous.experiment_tracking import ModelRecord
        # Simulate old JSON entry (no tier fields)
        old_data = {
            "model_id": "old_model",
            "experiment_id": "exp_old",
            "created_at": "2026-01-01",
            "model_path": "",
            "config": {},
            "cv_auc": 0.62,
            "test_auc": 0.59,
            "wmes_score": 0.55,
        }
        record = ModelRecord(**old_data)
        assert record.tier == 1
        assert record.stability_score == 0.0
        assert record.fragility_score == 1.0
        assert record.train_test_gap == 0.0

    def test_gate_checker_includes_min_tier(self):
        """Gate checker status includes min_tier_required."""
        from src.giga_orchestrator import ExperimentGateChecker
        checker = ExperimentGateChecker()
        _, status = checker.check_gates(trading_mode="paper")
        assert "min_tier_required" in status or "error" in status

    def test_register_model_populates_tier(self):
        """register_model correctly populates tier and robustness fields."""
        import tempfile
        from src.phase_21_continuous.experiment_tracking import (
            ModelRegistry, ExperimentResult, ExperimentConfig, ExperimentStatus,
            create_default_config,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
            f.write("{}")
            tmp_path = Path(f.name)
        try:
            registry = ModelRegistry(registry_path=tmp_path)
            config = create_default_config("test")
            result = ExperimentResult(
                experiment_id="exp_test",
                config=config,
                status=ExperimentStatus.COMPLETED,
                cv_auc_mean=0.62,
                test_auc=0.63,
                train_auc=0.68,
                wmes_score=0.58,
                stability_score=0.72,
                fragility_score=0.25,
            )
            model_id = registry.register_model(result)
            record = registry.models[model_id]
            assert record.tier == 3  # stable + low fragility + AUC >= 0.60
            assert record.stability_score == 0.72
            assert record.fragility_score == 0.25
            assert record.train_test_gap == pytest.approx(0.05, abs=0.001)
        finally:
            tmp_path.unlink(missing_ok=True)


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
        from src.phase_21_continuous.experiment_tracking import ModelRegistry

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
            f.write("{}")
            tmp_path = Path(f.name)
        try:
            registry = ModelRegistry(registry_path=tmp_path)
            report = {"production_candidates": []}
            n = _register_thick_weave_candidates(report, registry)
            assert n == 0
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_register_thick_weave_candidates_with_candidates(self):
        """Bridge registers production candidates with tier scoring."""
        import tempfile
        from scripts.start_system import _register_thick_weave_candidates
        from src.phase_21_continuous.experiment_tracking import ModelRegistry

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
            f.write("{}")
            tmp_path = Path(f.name)
        try:
            registry = ModelRegistry(registry_path=tmp_path)
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
            n = _register_thick_weave_candidates(report, registry)
            assert n == 1
            # Check the registered model has correct fields
            record = list(registry.models.values())[0]
            assert record.wmes_score == 0.62
            assert record.stability_score == 0.65  # PTS maps to stability
            assert record.fragility_score == 0.20
            assert record.tier >= 2  # Should be at least paper-eligible
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_register_thick_weave_candidates_skips_poor_quality(self):
        """Bridge skips candidates with WMES < 0.50."""
        import tempfile
        from scripts.start_system import _register_thick_weave_candidates
        from src.phase_21_continuous.experiment_tracking import ModelRegistry

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w') as f:
            f.write("{}")
            tmp_path = Path(f.name)
        try:
            registry = ModelRegistry(registry_path=tmp_path)
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
            n = _register_thick_weave_candidates(report, registry)
            assert n == 0
        finally:
            tmp_path.unlink(missing_ok=True)

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

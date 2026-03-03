"""
Tests for Wave K modules:
  K1: Feature failure isolation (in anti_overfit_integration)
  K2: RegimeAwareStopLoss, FeatureAlignmentChecker
  K3: HealthDrivenPause, SignalDeduplicator, ExecutionQualityTracker
  K4: CorrelationRegimeFeatures
  K5: ConceptDriftDetector, ModelPerformanceTracker
  K6: PnLAttribution, GateAuditLog
  K7: FamaFrenchFeatures, PutCallRatioFeatures, EODReconciliation, MultiHorizonFilter
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daily(n: int = 200) -> pd.DataFrame:
    """Create minimal daily DataFrame for testing."""
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 450.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    return pd.DataFrame({
        "date": dates,
        "open": close + np.random.randn(n) * 0.1,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(1_000_000, 10_000_000, n),
    })


# ===========================================================================
# K2: RegimeAwareStopLoss
# ===========================================================================

class TestRegimeAwareStopLoss:
    def test_fit_and_compute_long(self):
        from src.phase_15_strategy.regime_stop_loss import RegimeAwareStopLoss
        sl = RegimeAwareStopLoss()
        df = _make_daily(100)
        sl.fit(df)
        levels = sl.compute_levels(450.0, "LONG", 20.0)
        assert "stop_loss" in levels
        assert "take_profit" in levels
        assert levels["stop_loss"] < 450.0
        assert levels["take_profit"] > 450.0
        assert levels["regime"] == "NORMAL"

    def test_fit_and_compute_short(self):
        from src.phase_15_strategy.regime_stop_loss import RegimeAwareStopLoss
        sl = RegimeAwareStopLoss()
        df = _make_daily(100)
        sl.fit(df)
        levels = sl.compute_levels(450.0, "SHORT", 20.0)
        assert levels["stop_loss"] > 450.0
        assert levels["take_profit"] < 450.0

    def test_vix_regime_scaling(self):
        from src.phase_15_strategy.regime_stop_loss import RegimeAwareStopLoss
        sl = RegimeAwareStopLoss()
        df = _make_daily(100)
        sl.fit(df)
        low_vol = sl.compute_levels(450.0, "LONG", 10.0)
        high_vol = sl.compute_levels(450.0, "LONG", 30.0)
        # Higher VIX = wider stop
        assert high_vol["stop_pct"] >= low_vol["stop_pct"]

    def test_regime_labels(self):
        from src.phase_15_strategy.regime_stop_loss import RegimeAwareStopLoss
        sl = RegimeAwareStopLoss()
        assert sl._get_regime(10.0) == "LOW_VOL"
        assert sl._get_regime(20.0) == "NORMAL"
        assert sl._get_regime(30.0) == "HIGH_VOL"
        assert sl._get_regime(40.0) == "EXTREME"

    def test_min_max_stop_clamping(self):
        from src.phase_15_strategy.regime_stop_loss import RegimeAwareStopLoss
        sl = RegimeAwareStopLoss(min_stop_pct=0.005, max_stop_pct=0.02)
        df = _make_daily(100)
        sl.fit(df)
        levels = sl.compute_levels(450.0, "LONG", 20.0)
        assert levels["stop_pct"] >= 0.005
        assert levels["stop_pct"] <= 0.02

    def test_close_only_proxy(self):
        from src.phase_15_strategy.regime_stop_loss import RegimeAwareStopLoss
        sl = RegimeAwareStopLoss()
        # DataFrame without high/low
        df = pd.DataFrame({"close": [450 + i * 0.1 for i in range(50)]})
        sl.fit(df)
        levels = sl.compute_levels(455.0, "LONG", 20.0)
        assert levels["stop_loss"] < 455.0


# ===========================================================================
# K2: FeatureAlignmentChecker
# ===========================================================================

class TestFeatureAlignmentChecker:
    def test_aligned(self):
        from src.phase_15_strategy.regime_stop_loss import FeatureAlignmentChecker
        checker = FeatureAlignmentChecker(expected_features=["a", "b", "c"])
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        result = checker.check(df)
        assert result["aligned"] is True
        assert result["severity"] == "OK"
        assert len(result["missing"]) == 0

    def test_missing_warning(self):
        from src.phase_15_strategy.regime_stop_loss import FeatureAlignmentChecker
        feats = [f"f{i}" for i in range(20)]
        checker = FeatureAlignmentChecker(expected_features=feats)
        # Missing 1/20 = 5%
        df = pd.DataFrame({f"f{i}": [1] for i in range(19)})
        result = checker.check(df)
        assert result["severity"] in ("OK", "WARNING")
        assert len(result["missing"]) == 1

    def test_missing_critical(self):
        from src.phase_15_strategy.regime_stop_loss import FeatureAlignmentChecker
        feats = [f"f{i}" for i in range(10)]
        checker = FeatureAlignmentChecker(expected_features=feats, tolerance_missing_pct=0.10)
        # Missing 3/10 = 30%
        df = pd.DataFrame({f"f{i}": [1] for i in range(7)})
        result = checker.check(df)
        assert result["severity"] == "CRITICAL"

    def test_zero_filled_detection(self):
        from src.phase_15_strategy.regime_stop_loss import FeatureAlignmentChecker
        checker = FeatureAlignmentChecker(expected_features=["a", "b"])
        df = pd.DataFrame({"a": [0.0] * 20, "b": [1.0] * 20})
        result = checker.check(df)
        assert "a" in result["zero_filled"]
        assert "b" not in result["zero_filled"]


# ===========================================================================
# K3: HealthDrivenPause
# ===========================================================================

class TestHealthDrivenPause:
    def test_healthy_no_pause(self):
        from src.phase_19_paper_trading.health_trading_pause import HealthDrivenPause
        hp = HealthDrivenPause()
        result = hp.should_pause({"status": "HEALTHY"})
        assert result["paused"] is False

    def test_unhealthy_pause(self):
        from src.phase_19_paper_trading.health_trading_pause import HealthDrivenPause
        hp = HealthDrivenPause()
        result = hp.should_pause({"status": "UNHEALTHY", "reason": "test"})
        assert result["paused"] is True
        assert result["severity"] == "HIGH"

    def test_degraded_pause(self):
        from src.phase_19_paper_trading.health_trading_pause import HealthDrivenPause
        hp = HealthDrivenPause()
        result = hp.should_pause({"status": "DEGRADED", "reason": "test"})
        assert result["paused"] is True
        assert result["severity"] == "MEDIUM"

    def test_escalation(self):
        from src.phase_19_paper_trading.health_trading_pause import HealthDrivenPause
        hp = HealthDrivenPause(escalation_threshold=3)
        for _ in range(3):
            result = hp.should_pause({"status": "UNHEALTHY"})
        assert result["severity"] == "CRITICAL"
        assert result["consecutive_unhealthy"] == 3

    def test_recovery_resets_counter(self):
        from src.phase_19_paper_trading.health_trading_pause import HealthDrivenPause
        hp = HealthDrivenPause()
        hp.should_pause({"status": "UNHEALTHY"})
        hp.should_pause({"status": "UNHEALTHY"})
        result = hp.should_pause({"status": "HEALTHY"})
        assert result["consecutive_unhealthy"] == 0

    def test_summary(self):
        from src.phase_19_paper_trading.health_trading_pause import HealthDrivenPause
        hp = HealthDrivenPause()
        hp.should_pause({"status": "UNHEALTHY"})
        summary = hp.get_pause_summary()
        assert "1/1" in summary


# ===========================================================================
# K3: SignalDeduplicator
# ===========================================================================

class TestSignalDeduplicator:
    def test_no_duplicate_first(self):
        from src.phase_19_paper_trading.signal_dedup import SignalDeduplicator
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            sd = SignalDeduplicator(state_file=path)
            assert sd.is_duplicate("BUY") is False
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_duplicate_detection(self):
        from src.phase_19_paper_trading.signal_dedup import SignalDeduplicator
        from datetime import datetime
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            sd = SignalDeduplicator(state_file=path)
            now = datetime.now().isoformat()
            sd.record_signal("BUY", now, 450.0)
            assert sd.is_duplicate("BUY", min_interval_seconds=3600) is True
            assert sd.is_duplicate("SELL", min_interval_seconds=3600) is False
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_get_last_signal(self):
        from src.phase_19_paper_trading.signal_dedup import SignalDeduplicator
        from datetime import datetime
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            sd = SignalDeduplicator(state_file=path)
            sd.record_signal("SELL", datetime.now().isoformat(), 455.0, {"reason": "test"})
            last = sd.get_last_signal()
            assert last["signal_type"] == "SELL"
            assert last["price"] == 455.0
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_clear(self):
        from src.phase_19_paper_trading.signal_dedup import SignalDeduplicator
        from datetime import datetime
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            sd = SignalDeduplicator(state_file=path)
            sd.record_signal("BUY", datetime.now().isoformat(), 450.0)
            sd.clear()
            assert sd.get_last_signal() is None
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ===========================================================================
# K3: ExecutionQualityTracker
# ===========================================================================

class TestExecutionQualityTracker:
    def test_record_and_fill(self):
        from src.phase_19_paper_trading.execution_quality import ExecutionQualityTracker
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            eq = ExecutionQualityTracker(state_file=path)
            eq.record_expectation("ord1", 450.0, "LONG", "2024-01-01T10:00:00")
            result = eq.record_fill("ord1", 450.10, "2024-01-01T10:00:05")
            assert result is not None
            assert result["slippage_bps"] > 0  # Paid more
            assert result["favorable"] is False  # Unfavorable for LONG
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_favorable_long(self):
        from src.phase_19_paper_trading.execution_quality import ExecutionQualityTracker
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            eq = ExecutionQualityTracker(state_file=path)
            eq.record_expectation("ord2", 450.0, "LONG", "2024-01-01T10:00:00")
            result = eq.record_fill("ord2", 449.90, "2024-01-01T10:00:05")
            assert result["favorable"] is True  # Paid less
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_summary(self):
        from src.phase_19_paper_trading.execution_quality import ExecutionQualityTracker
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            eq = ExecutionQualityTracker(state_file=path)
            for i in range(5):
                eq.record_expectation(f"ord{i}", 450.0, "LONG", "2024-01-01")
                eq.record_fill(f"ord{i}", 450.0 + (i - 2) * 0.05, "2024-01-01")
            summary = eq.get_summary()
            assert summary["total_orders"] == 5
            assert "avg_slippage_bps" in summary
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_unknown_order(self):
        from src.phase_19_paper_trading.execution_quality import ExecutionQualityTracker
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            eq = ExecutionQualityTracker(state_file=path)
            result = eq.record_fill("unknown", 450.0, "2024-01-01")
            assert result is None
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ===========================================================================
# K4: CorrelationRegimeFeatures
# ===========================================================================

class TestCorrelationRegimeFeatures:
    def test_proxy_features(self):
        from src.phase_08_features_breadth.correlation_regime_features import CorrelationRegimeFeatures
        crf = CorrelationRegimeFeatures()
        df = _make_daily(300)
        result = crf.create_correlation_features(df)
        corr_cols = [c for c in result.columns if c.startswith("corr_")]
        assert len(corr_cols) == 12

    def test_no_nan(self):
        from src.phase_08_features_breadth.correlation_regime_features import CorrelationRegimeFeatures
        crf = CorrelationRegimeFeatures()
        df = _make_daily(300)
        result = crf.create_correlation_features(df)
        corr_cols = [c for c in result.columns if c.startswith("corr_")]
        for col in corr_cols:
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_analyze_regime(self):
        from src.phase_08_features_breadth.correlation_regime_features import CorrelationRegimeFeatures
        crf = CorrelationRegimeFeatures()
        df = _make_daily(300)
        df = crf.create_correlation_features(df)
        regime = crf.analyze_current_regime(df)
        assert regime is not None
        assert "regime" in regime


# ===========================================================================
# K5: ConceptDriftDetector
# ===========================================================================

class TestConceptDriftDetector:
    def test_fit_and_no_drift(self):
        from src.phase_20_monitoring.concept_drift_detector import ConceptDriftDetector
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = (X[:, 0] > 0).astype(int)
        dd = ConceptDriftDetector(min_samples=20)
        dd.fit(X[:100], y[:100], ["f1", "f2", "f3", "f4", "f5"])
        result = dd.detect(X[100:], y[100:])
        assert result["severity"] in ("NONE", "MILD")

    def test_drift_detection(self):
        from src.phase_20_monitoring.concept_drift_detector import ConceptDriftDetector
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y_train = (X[:100, 0] > 0).astype(int)
        # Flip the relationship for new data
        y_new = (X[100:, 0] < 0).astype(int)
        dd = ConceptDriftDetector(min_samples=20)
        dd.fit(X[:100], y_train, ["f1", "f2", "f3", "f4", "f5"])
        result = dd.detect(X[100:], y_new)
        assert result["drift_score"] > 0
        assert len(result["drifted_features"]) > 0

    def test_insufficient_data(self):
        from src.phase_20_monitoring.concept_drift_detector import ConceptDriftDetector
        dd = ConceptDriftDetector(min_samples=50)
        dd.fit(np.random.randn(100, 3), np.random.randint(0, 2, 100))
        result = dd.detect(np.random.randn(10, 3), np.random.randint(0, 2, 10))
        assert "INSUFFICIENT_DATA" in result["recommendation"]

    def test_no_baseline(self):
        from src.phase_20_monitoring.concept_drift_detector import ConceptDriftDetector
        dd = ConceptDriftDetector()
        result = dd.detect(np.random.randn(100, 3), np.random.randint(0, 2, 100))
        assert result["recommendation"] == "FIT_BASELINE_FIRST"


# ===========================================================================
# K5: ModelPerformanceTracker
# ===========================================================================

class TestModelPerformanceTracker:
    def test_record_and_compute(self):
        from src.phase_20_monitoring.model_performance_tracker import ModelPerformanceTracker
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            tracker = ModelPerformanceTracker(
                window_size=20, state_file=path, auto_save_interval=100
            )
            tracker.set_baseline(0.75)
            np.random.seed(42)
            for _ in range(30):
                pred = np.random.random()
                actual = int(pred > 0.5)
                tracker.record_prediction(pred, actual)
            result = tracker.compute_rolling_auc()
            assert result["n_predictions"] == 30
            assert result["baseline_auc"] == 0.75
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_insufficient_data(self):
        from src.phase_20_monitoring.model_performance_tracker import ModelPerformanceTracker
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            tracker = ModelPerformanceTracker(state_file=path)
            result = tracker.compute_rolling_auc()
            assert result["alert"] == "INSUFFICIENT_DATA"
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_persistence(self):
        from src.phase_20_monitoring.model_performance_tracker import ModelPerformanceTracker
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            t1 = ModelPerformanceTracker(state_file=path, auto_save_interval=1)
            t1.set_baseline(0.80)
            t1.record_prediction(0.8, 1)
            t1.record_prediction(0.2, 0)
            # Reload from disk
            t2 = ModelPerformanceTracker(state_file=path)
            assert t2._baseline_auc == 0.80
            assert len(t2._predictions) == 2
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ===========================================================================
# K6: PnLAttribution
# ===========================================================================

class TestPnLAttribution:
    def test_record_and_outcome(self, tmp_path):
        from src.phase_20_monitoring.pnl_attribution import PnLAttribution
        path = str(tmp_path / "test_pnl.db")
        pa = PnLAttribution(db_path=path)
        sid = pa.record_signal({
            "timestamp": "2024-01-01T10:00:00",
            "signal_type": "BUY",
            "entry_price": 450.0,
            "swing_proba": 0.72,
            "timing_proba": 0.65,
            "position_size": 0.10,
            "gates_passed": ["macro_calendar", "vol_regime"],
            "gates_blocked": [],
            "regime": "NORMAL",
        })
        assert sid is not None
        pa.record_outcome(sid, 452.0, 0.0044)
        summary = pa.get_attribution_summary(last_n=10)
        assert summary["total_signals"] == 1
        assert summary["win_rate"] == 1.0

    def test_empty_summary(self, tmp_path):
        from src.phase_20_monitoring.pnl_attribution import PnLAttribution
        path = str(tmp_path / "test_pnl2.db")
        pa = PnLAttribution(db_path=path)
        summary = pa.get_attribution_summary()
        assert summary["total_signals"] == 0


# ===========================================================================
# K6: GateAuditLog
# ===========================================================================

class TestGateAuditLog:
    def test_log_and_stats(self, tmp_path):
        from src.phase_20_monitoring.gate_audit_log import GateAuditLog
        path = str(tmp_path / "test_gate.db")
        gal = GateAuditLog(db_path=path)
        gal.log_evaluation("macro_calendar", "PASS", signal_type="BUY")
        gal.log_evaluation("macro_calendar", "BLOCK", reason="FOMC day")
        gal.log_evaluation("vol_regime", "PASS")
        stats = gal.get_gate_stats()
        assert "macro_calendar" in stats
        assert stats["macro_calendar"]["block_count"] == 1
        assert stats["macro_calendar"]["pass_count"] == 1

    def test_block_history(self, tmp_path):
        from src.phase_20_monitoring.gate_audit_log import GateAuditLog
        path = str(tmp_path / "test_gate2.db")
        gal = GateAuditLog(db_path=path)
        gal.log_evaluation("vol_regime", "BLOCK", reason="VIX > 35")
        blocks = gal.get_block_history()
        assert len(blocks) == 1
        assert blocks[0]["gate_name"] == "vol_regime"


# ===========================================================================
# K7: FamaFrenchFeatures
# ===========================================================================

class TestFamaFrenchFeatures:
    def test_proxy_features(self):
        from src.phase_08_features_breadth.fama_french_features import FamaFrenchFeatures
        ff = FamaFrenchFeatures()
        df = _make_daily(200)
        result = ff.create_fama_french_features(df)
        ff_cols = [c for c in result.columns if c.startswith("ff_")]
        assert len(ff_cols) == 8

    def test_no_nan(self):
        from src.phase_08_features_breadth.fama_french_features import FamaFrenchFeatures
        ff = FamaFrenchFeatures()
        df = _make_daily(200)
        result = ff.create_fama_french_features(df)
        ff_cols = [c for c in result.columns if c.startswith("ff_")]
        for col in ff_cols:
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_analyze(self):
        from src.phase_08_features_breadth.fama_french_features import FamaFrenchFeatures
        ff = FamaFrenchFeatures()
        df = _make_daily(200)
        df = ff.create_fama_french_features(df)
        analysis = ff.analyze_current_factors(df)
        assert analysis is not None
        assert "factor_regime" in analysis


# ===========================================================================
# K7: PutCallRatioFeatures
# ===========================================================================

class TestPutCallRatioFeatures:
    def test_proxy_features(self):
        from src.phase_08_features_breadth.put_call_ratio_features import PutCallRatioFeatures
        pcr = PutCallRatioFeatures()
        df = _make_daily(200)
        result = pcr.create_pcr_features(df)
        pcr_cols = [c for c in result.columns if c.startswith("pcr_")]
        assert len(pcr_cols) == 8

    def test_no_nan(self):
        from src.phase_08_features_breadth.put_call_ratio_features import PutCallRatioFeatures
        pcr = PutCallRatioFeatures()
        df = _make_daily(200)
        result = pcr.create_pcr_features(df)
        pcr_cols = [c for c in result.columns if c.startswith("pcr_")]
        for col in pcr_cols:
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_analyze(self):
        from src.phase_08_features_breadth.put_call_ratio_features import PutCallRatioFeatures
        pcr = PutCallRatioFeatures()
        df = _make_daily(200)
        df = pcr.create_pcr_features(df)
        analysis = pcr.analyze_current_pcr(df)
        assert analysis is not None
        assert analysis["data_source"] in ("realized_vol_proxy", "vix_proxy", "none")


# ===========================================================================
# K7: EODReconciliation
# ===========================================================================

class TestEODReconciliation:
    def test_matched(self):
        from src.phase_19_paper_trading.eod_reconciliation import EODReconciliation
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            recon = EODReconciliation(state_file=path)
            result = recon.reconcile({"SPY": 100}, {"SPY": 100})
            assert result["matched"] is True
            assert result["severity"] == "OK"
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_discrepancy(self):
        from src.phase_19_paper_trading.eod_reconciliation import EODReconciliation
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            recon = EODReconciliation(state_file=path)
            result = recon.reconcile({"SPY": 100}, {"SPY": 95})
            assert result["matched"] is False
            assert len(result["discrepancies"]) == 1
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_missing_position(self):
        from src.phase_19_paper_trading.eod_reconciliation import EODReconciliation
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            recon = EODReconciliation(state_file=path)
            result = recon.reconcile({"SPY": 100}, {"SPY": 100, "QQQ": 50})
            assert result["severity"] == "CRITICAL"
            assert "QQQ" in result["alpaca_only"]
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_history(self):
        from src.phase_19_paper_trading.eod_reconciliation import EODReconciliation
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            recon = EODReconciliation(state_file=path)
            recon.reconcile({"SPY": 100}, {"SPY": 100})
            recon.reconcile({"SPY": 200}, {"SPY": 200})
            hist = recon.get_history()
            assert len(hist) == 2
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ===========================================================================
# K7: MultiHorizonFilter
# ===========================================================================

class TestMultiHorizonFilter:
    def test_compute_features(self):
        from src.phase_15_strategy.multi_horizon_filter import MultiHorizonFilter
        mh = MultiHorizonFilter()
        df = _make_daily(200)
        result = mh.compute_horizon_signals(df)
        mh_cols = [c for c in result.columns if c.startswith("mh_")]
        assert len(mh_cols) == 11

    def test_no_nan(self):
        from src.phase_15_strategy.multi_horizon_filter import MultiHorizonFilter
        mh = MultiHorizonFilter()
        df = _make_daily(200)
        result = mh.compute_horizon_signals(df)
        mh_cols = [c for c in result.columns if c.startswith("mh_")]
        for col in mh_cols:
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_should_filter(self):
        from src.phase_15_strategy.multi_horizon_filter import MultiHorizonFilter
        mh = MultiHorizonFilter(agreement_threshold=0.6)
        # All horizons agree on up
        assert mh.should_filter(0.8, "LONG") is False  # Pass
        assert mh.should_filter(0.3, "LONG") is True  # Block
        # All horizons agree on down
        assert mh.should_filter(-0.8, "SHORT") is False  # Pass
        assert mh.should_filter(-0.3, "SHORT") is True  # Block

    def test_evaluate(self):
        from src.phase_15_strategy.multi_horizon_filter import MultiHorizonFilter
        mh = MultiHorizonFilter()
        df = _make_daily(200)
        df = mh.compute_horizon_signals(df)
        result = mh.evaluate(df, "LONG")
        assert "agreement" in result
        assert "should_block" in result
        assert "horizon_details" in result

    def test_missing_close(self):
        from src.phase_15_strategy.multi_horizon_filter import MultiHorizonFilter
        mh = MultiHorizonFilter()
        df = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=50)})
        result = mh.compute_horizon_signals(df)
        mh_cols = [c for c in result.columns if c.startswith("mh_")]
        assert len(mh_cols) == 11


# ===========================================================================
# K4: FEATURE_GROUPS updated
# ===========================================================================

class TestFeatureGroupsUpdated:
    def test_new_groups_exist(self):
        from src.phase_10_feature_processing.group_aware_processor import FEATURE_GROUPS
        assert "correlation_regime" in FEATURE_GROUPS
        assert "fama_french" in FEATURE_GROUPS
        assert "put_call_ratio" in FEATURE_GROUPS
        assert "multi_horizon" in FEATURE_GROUPS
        # Check prefixes
        assert FEATURE_GROUPS["correlation_regime"] == ["corr_"]
        assert FEATURE_GROUPS["fama_french"] == ["ff_"]
        assert FEATURE_GROUPS["put_call_ratio"] == ["pcr_"]
        assert FEATURE_GROUPS["multi_horizon"] == ["mh_"]


# ===========================================================================
# K2: Config flags exist
# ===========================================================================

class TestConfigFlags:
    def test_anti_overfit_config_has_k_flags(self):
        from src.experiment_config import AntiOverfitConfig
        config = AntiOverfitConfig()
        assert hasattr(config, "use_correlation_regime")
        assert hasattr(config, "use_fama_french")
        assert hasattr(config, "use_put_call_ratio")
        assert hasattr(config, "use_multi_horizon")
        assert config.use_correlation_regime is True
        assert config.use_fama_french is True
        assert config.use_put_call_ratio is True
        assert config.use_multi_horizon is True

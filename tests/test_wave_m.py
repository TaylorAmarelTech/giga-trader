"""
Wave M Tests: 8 new data-coverage feature modules + config integration.

Tests cover:
  M1: ExpandedMacroFeatures (xmacro_)
  M2: VVIXFeatures (vvix_)
  M3: SectorRotationFeatures (secrot_)
  M4: FXCarryFeatures (fxc_)
  M5: MoneyMarketFeatures (mmkt_)
  M6: FinancialStressFeatures (fstress_)
  M7: GlobalEquityFeatures (gleq_)
  M8: RetailSentimentFeatures (rflow_)
"""

import numpy as np
import pandas as pd
import pytest


def _make_daily(n: int = 300) -> pd.DataFrame:
    """Create a minimal daily DataFrame for testing."""
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-03", periods=n)
    close = 400 + np.cumsum(np.random.randn(n) * 2)
    volume = np.random.randint(50_000_000, 200_000_000, size=n).astype(float)
    return pd.DataFrame(
        {"close": close, "open": close * 0.999, "high": close * 1.005,
         "low": close * 0.995, "volume": volume},
        index=dates,
    )


# ─────────────────────────────────────────────────────────────────
# M1: ExpandedMacroFeatures
# ─────────────────────────────────────────────────────────────────

class TestExpandedMacroFeatures:
    def test_proxy_features(self):
        from src.phase_08_features_breadth.expanded_macro_features import ExpandedMacroFeatures
        xm = ExpandedMacroFeatures()
        df = _make_daily()
        result = xm.create_expanded_macro_features(df)
        expected = ExpandedMacroFeatures._all_feature_names()
        for feat in expected:
            assert feat in result.columns, f"Missing {feat}"
        assert not result[expected].isna().any().any()

    def test_feature_count(self):
        from src.phase_08_features_breadth.expanded_macro_features import ExpandedMacroFeatures
        assert len(ExpandedMacroFeatures._all_feature_names()) == 8

    def test_no_inf(self):
        from src.phase_08_features_breadth.expanded_macro_features import ExpandedMacroFeatures
        xm = ExpandedMacroFeatures()
        result = xm.create_expanded_macro_features(_make_daily())
        feats = ExpandedMacroFeatures._all_feature_names()
        assert not np.isinf(result[feats].values).any()

    def test_analyze_current(self):
        from src.phase_08_features_breadth.expanded_macro_features import ExpandedMacroFeatures
        xm = ExpandedMacroFeatures()
        analysis = xm.analyze_current_macro(_make_daily())
        assert analysis is not None
        assert len(analysis) == 8


# ─────────────────────────────────────────────────────────────────
# M2: VVIXFeatures
# ─────────────────────────────────────────────────────────────────

class TestVVIXFeatures:
    def test_proxy_features(self):
        from src.phase_08_features_breadth.vvix_features import VVIXFeatures
        vvix = VVIXFeatures()
        df = _make_daily()
        result = vvix.create_vvix_features(df)
        for feat in VVIXFeatures._all_feature_names():
            assert feat in result.columns, f"Missing {feat}"

    def test_feature_count(self):
        from src.phase_08_features_breadth.vvix_features import VVIXFeatures
        assert len(VVIXFeatures._all_feature_names()) == 8

    def test_no_inf(self):
        from src.phase_08_features_breadth.vvix_features import VVIXFeatures
        vvix = VVIXFeatures()
        result = vvix.create_vvix_features(_make_daily())
        feats = VVIXFeatures._all_feature_names()
        assert not np.isinf(result[feats].values).any()

    def test_regime_values(self):
        from src.phase_08_features_breadth.vvix_features import VVIXFeatures
        vvix = VVIXFeatures()
        result = vvix.create_vvix_features(_make_daily())
        vals = result["vvix_regime"].unique()
        assert all(v in [-1.0, 0.0, 1.0] for v in vals)


# ─────────────────────────────────────────────────────────────────
# M3: SectorRotationFeatures
# ─────────────────────────────────────────────────────────────────

class TestSectorRotationFeatures:
    def test_proxy_features(self):
        from src.phase_08_features_breadth.sector_rotation_features import SectorRotationFeatures
        sr = SectorRotationFeatures()
        result = sr.create_sector_rotation_features(_make_daily())
        for feat in SectorRotationFeatures._all_feature_names():
            assert feat in result.columns, f"Missing {feat}"

    def test_feature_count(self):
        from src.phase_08_features_breadth.sector_rotation_features import SectorRotationFeatures
        assert len(SectorRotationFeatures._all_feature_names()) == 8

    def test_no_nan(self):
        from src.phase_08_features_breadth.sector_rotation_features import SectorRotationFeatures
        sr = SectorRotationFeatures()
        result = sr.create_sector_rotation_features(_make_daily())
        feats = SectorRotationFeatures._all_feature_names()
        assert not result[feats].isna().any().any()

    def test_analyze_current(self):
        from src.phase_08_features_breadth.sector_rotation_features import SectorRotationFeatures
        sr = SectorRotationFeatures()
        analysis = sr.analyze_current_rotation(_make_daily())
        assert analysis is not None


# ─────────────────────────────────────────────────────────────────
# M4: FXCarryFeatures
# ─────────────────────────────────────────────────────────────────

class TestFXCarryFeatures:
    def test_proxy_features(self):
        from src.phase_08_features_breadth.fx_carry_features import FXCarryFeatures
        fxc = FXCarryFeatures()
        result = fxc.create_fx_carry_features(_make_daily())
        for feat in FXCarryFeatures._all_feature_names():
            assert feat in result.columns, f"Missing {feat}"

    def test_feature_count(self):
        from src.phase_08_features_breadth.fx_carry_features import FXCarryFeatures
        assert len(FXCarryFeatures._all_feature_names()) == 8

    def test_no_inf(self):
        from src.phase_08_features_breadth.fx_carry_features import FXCarryFeatures
        fxc = FXCarryFeatures()
        result = fxc.create_fx_carry_features(_make_daily())
        feats = FXCarryFeatures._all_feature_names()
        assert not np.isinf(result[feats].values).any()


# ─────────────────────────────────────────────────────────────────
# M5: MoneyMarketFeatures
# ─────────────────────────────────────────────────────────────────

class TestMoneyMarketFeatures:
    def test_proxy_features(self):
        from src.phase_08_features_breadth.money_market_features import MoneyMarketFeatures
        mm = MoneyMarketFeatures()
        result = mm.create_money_market_features(_make_daily())
        for feat in MoneyMarketFeatures._all_feature_names():
            assert feat in result.columns, f"Missing {feat}"

    def test_feature_count(self):
        from src.phase_08_features_breadth.money_market_features import MoneyMarketFeatures
        assert len(MoneyMarketFeatures._all_feature_names()) == 8

    def test_no_nan(self):
        from src.phase_08_features_breadth.money_market_features import MoneyMarketFeatures
        mm = MoneyMarketFeatures()
        result = mm.create_money_market_features(_make_daily())
        feats = MoneyMarketFeatures._all_feature_names()
        assert not result[feats].isna().any().any()

    def test_analyze_current(self):
        from src.phase_08_features_breadth.money_market_features import MoneyMarketFeatures
        mm = MoneyMarketFeatures()
        analysis = mm.analyze_current_money_market(_make_daily())
        assert analysis is not None
        assert len(analysis) == 8


# ─────────────────────────────────────────────────────────────────
# M6: FinancialStressFeatures
# ─────────────────────────────────────────────────────────────────

class TestFinancialStressFeatures:
    def test_proxy_features(self):
        from src.phase_08_features_breadth.financial_stress_features import FinancialStressFeatures
        fs = FinancialStressFeatures()
        result = fs.create_financial_stress_features(_make_daily())
        for feat in FinancialStressFeatures._all_feature_names():
            assert feat in result.columns, f"Missing {feat}"

    def test_feature_count(self):
        from src.phase_08_features_breadth.financial_stress_features import FinancialStressFeatures
        assert len(FinancialStressFeatures._all_feature_names()) == 8

    def test_no_inf(self):
        from src.phase_08_features_breadth.financial_stress_features import FinancialStressFeatures
        fs = FinancialStressFeatures()
        result = fs.create_financial_stress_features(_make_daily())
        feats = FinancialStressFeatures._all_feature_names()
        assert not np.isinf(result[feats].values).any()

    def test_regime_values(self):
        from src.phase_08_features_breadth.financial_stress_features import FinancialStressFeatures
        fs = FinancialStressFeatures()
        result = fs.create_financial_stress_features(_make_daily())
        vals = result["fstress_regime"].unique()
        assert all(v in [-1.0, 0.0, 1.0] for v in vals)


# ─────────────────────────────────────────────────────────────────
# M7: GlobalEquityFeatures
# ─────────────────────────────────────────────────────────────────

class TestGlobalEquityFeatures:
    def test_proxy_features(self):
        from src.phase_08_features_breadth.global_equity_features import GlobalEquityFeatures
        ge = GlobalEquityFeatures()
        result = ge.create_global_equity_features(_make_daily())
        for feat in GlobalEquityFeatures._all_feature_names():
            assert feat in result.columns, f"Missing {feat}"

    def test_feature_count(self):
        from src.phase_08_features_breadth.global_equity_features import GlobalEquityFeatures
        assert len(GlobalEquityFeatures._all_feature_names()) == 8

    def test_no_nan(self):
        from src.phase_08_features_breadth.global_equity_features import GlobalEquityFeatures
        ge = GlobalEquityFeatures()
        result = ge.create_global_equity_features(_make_daily())
        feats = GlobalEquityFeatures._all_feature_names()
        assert not result[feats].isna().any().any()

    def test_analyze_current(self):
        from src.phase_08_features_breadth.global_equity_features import GlobalEquityFeatures
        ge = GlobalEquityFeatures()
        analysis = ge.analyze_current_global(_make_daily())
        assert analysis is not None


# ─────────────────────────────────────────────────────────────────
# M8: RetailSentimentFeatures
# ─────────────────────────────────────────────────────────────────

class TestRetailSentimentFeatures:
    def test_proxy_features(self):
        from src.phase_08_features_breadth.retail_sentiment_features import RetailSentimentFeatures
        rs = RetailSentimentFeatures()
        result = rs.create_retail_sentiment_features(_make_daily())
        for feat in RetailSentimentFeatures._all_feature_names():
            assert feat in result.columns, f"Missing {feat}"

    def test_feature_count(self):
        from src.phase_08_features_breadth.retail_sentiment_features import RetailSentimentFeatures
        assert len(RetailSentimentFeatures._all_feature_names()) == 8

    def test_no_inf(self):
        from src.phase_08_features_breadth.retail_sentiment_features import RetailSentimentFeatures
        rs = RetailSentimentFeatures()
        result = rs.create_retail_sentiment_features(_make_daily())
        feats = RetailSentimentFeatures._all_feature_names()
        assert not np.isinf(result[feats].values).any()

    def test_regime_values(self):
        from src.phase_08_features_breadth.retail_sentiment_features import RetailSentimentFeatures
        rs = RetailSentimentFeatures()
        result = rs.create_retail_sentiment_features(_make_daily())
        vals = result["rflow_regime"].unique()
        assert all(v in [-1.0, 0.0, 1.0] for v in vals)


# ─────────────────────────────────────────────────────────────────
# Config Integration Tests
# ─────────────────────────────────────────────────────────────────

class TestWaveMConfig:
    """Verify all 8 Wave M config flags exist with correct defaults."""

    def test_config_flags_exist(self):
        from src.experiment_config import AntiOverfitConfig
        config = AntiOverfitConfig()
        flags = [
            ("use_expanded_macro", True),
            ("use_vvix", True),
            ("use_sector_rotation", True),
            ("use_fx_carry", True),
            ("use_money_market", True),
            ("use_financial_stress", True),
            ("use_global_equity", True),
            ("use_retail_sentiment", True),
        ]
        for flag, default in flags:
            assert hasattr(config, flag), f"Missing config flag: {flag}"
            assert getattr(config, flag) == default, f"{flag} default should be {default}"

    def test_feature_groups_exist(self):
        from src.phase_10_feature_processing.group_aware_processor import FEATURE_GROUPS
        expected_groups = [
            "expanded_macro", "vvix", "sector_rotation", "fx_carry",
            "money_market", "financial_stress", "global_equity", "retail_sentiment",
        ]
        for group in expected_groups:
            assert group in FEATURE_GROUPS, f"Missing FEATURE_GROUPS: {group}"

    def test_all_modules_default_on(self):
        """All 8 Wave M modules should default to ON (reliable data sources)."""
        from src.experiment_config import AntiOverfitConfig
        config = AntiOverfitConfig()
        wave_m_flags = [
            "use_expanded_macro", "use_vvix", "use_sector_rotation",
            "use_fx_carry", "use_money_market", "use_financial_stress",
            "use_global_equity", "use_retail_sentiment",
        ]
        for flag in wave_m_flags:
            assert getattr(config, flag) is True, f"{flag} should default to True"

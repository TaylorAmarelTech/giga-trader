"""
Tests for Wave L modules (10 new data source feature modules):
  L1: EarningsRevisionFeatures (8 ern_ features)
  L2: ShortInterestFeatures (8 si_ features)
  L3: DollarIndexFeatures (8 dxy_ features)
  L4: InstitutionalFlowFeatures (8 inst_ features)
  L5: GoogleTrendsFeatures (8 gtrend_ features)
  L6: CommoditySignalFeatures (10 cmdty_ features)
  L7: TreasuryAuctionFeatures (6 tauct_ features)
  L8: FedLiquidityFeatures (8 fedliq_ features)
  L9: EarningsCalendarFeatures (6 ecal_ features)
  L10: AnalystRatingFeatures (8 anlst_ features)

Total: ~78 new features, ~38 tests
"""

import numpy as np
import pandas as pd
import pytest


def _make_daily(n: int = 200) -> pd.DataFrame:
    """Create minimal daily DataFrame for testing."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 450.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close + np.random.randn(n) * 0.1,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1_000_000, 10_000_000, n),
        }
    )


# ─────────────────────────────────────────────────────────
# L1: EarningsRevisionFeatures
# ─────────────────────────────────────────────────────────
class TestEarningsRevisionFeatures:
    def test_proxy_features(self):
        from src.phase_08_features_breadth.earnings_revision_features import (
            EarningsRevisionFeatures,
        )

        ern = EarningsRevisionFeatures()
        df = _make_daily(300)
        result = ern.create_earnings_revision_features(df)
        ern_cols = [c for c in result.columns if c.startswith("ern_")]
        assert len(ern_cols) == 8

    def test_no_nan(self):
        from src.phase_08_features_breadth.earnings_revision_features import (
            EarningsRevisionFeatures,
        )

        ern = EarningsRevisionFeatures()
        df = _make_daily(300)
        result = ern.create_earnings_revision_features(df)
        for col in [c for c in result.columns if c.startswith("ern_")]:
            assert result[col].isna().sum() == 0, f"NaN in {col}"
            assert np.isinf(result[col]).sum() == 0, f"Inf in {col}"

    def test_analyze(self):
        from src.phase_08_features_breadth.earnings_revision_features import (
            EarningsRevisionFeatures,
        )

        ern = EarningsRevisionFeatures()
        df = _make_daily(300)
        df = ern.create_earnings_revision_features(df)
        analysis = ern.analyze_current_earnings_revision(df)
        assert analysis is not None
        assert "revision_regime" in analysis

    def test_short_data(self):
        from src.phase_08_features_breadth.earnings_revision_features import (
            EarningsRevisionFeatures,
        )

        ern = EarningsRevisionFeatures()
        df = _make_daily(10)
        result = ern.create_earnings_revision_features(df)
        assert len(result) == 10


# ─────────────────────────────────────────────────────────
# L2: ShortInterestFeatures
# ─────────────────────────────────────────────────────────
class TestShortInterestFeatures:
    def test_proxy_features(self):
        from src.phase_08_features_breadth.short_interest_features import (
            ShortInterestFeatures,
        )

        si = ShortInterestFeatures()
        df = _make_daily(300)
        result = si.create_short_interest_features(df)
        si_cols = [c for c in result.columns if c.startswith("si_")]
        assert len(si_cols) == 8

    def test_no_nan(self):
        from src.phase_08_features_breadth.short_interest_features import (
            ShortInterestFeatures,
        )

        si = ShortInterestFeatures()
        df = _make_daily(300)
        result = si.create_short_interest_features(df)
        for col in [c for c in result.columns if c.startswith("si_")]:
            assert result[col].isna().sum() == 0, f"NaN in {col}"
            assert np.isinf(result[col]).sum() == 0, f"Inf in {col}"

    def test_analyze(self):
        from src.phase_08_features_breadth.short_interest_features import (
            ShortInterestFeatures,
        )

        si = ShortInterestFeatures()
        df = _make_daily(300)
        df = si.create_short_interest_features(df)
        analysis = si.analyze_current_short_interest(df)
        assert analysis is not None
        assert "regime" in analysis


# ─────────────────────────────────────────────────────────
# L3: DollarIndexFeatures
# ─────────────────────────────────────────────────────────
class TestDollarIndexFeatures:
    def test_proxy_features(self):
        from src.phase_08_features_breadth.dollar_index_features import (
            DollarIndexFeatures,
        )

        dxy = DollarIndexFeatures()
        df = _make_daily(300)
        result = dxy.create_dollar_index_features(df)
        dxy_cols = [c for c in result.columns if c.startswith("dxy_")]
        assert len(dxy_cols) == 8

    def test_no_nan(self):
        from src.phase_08_features_breadth.dollar_index_features import (
            DollarIndexFeatures,
        )

        dxy = DollarIndexFeatures()
        df = _make_daily(300)
        result = dxy.create_dollar_index_features(df)
        for col in [c for c in result.columns if c.startswith("dxy_")]:
            assert result[col].isna().sum() == 0, f"NaN in {col}"
            assert np.isinf(result[col]).sum() == 0, f"Inf in {col}"

    def test_analyze(self):
        from src.phase_08_features_breadth.dollar_index_features import (
            DollarIndexFeatures,
        )

        dxy = DollarIndexFeatures()
        df = _make_daily(300)
        df = dxy.create_dollar_index_features(df)
        analysis = dxy.analyze_current_dollar(df)
        assert analysis is not None
        assert "regime" in analysis
        assert analysis["regime"] in ("STRONG_DOLLAR", "WEAK_DOLLAR", "NEUTRAL")

    def test_data_source_default(self):
        from src.phase_08_features_breadth.dollar_index_features import (
            DollarIndexFeatures,
        )

        dxy = DollarIndexFeatures()
        assert dxy._data_source == "none"


# ─────────────────────────────────────────────────────────
# L4: InstitutionalFlowFeatures
# ─────────────────────────────────────────────────────────
class TestInstitutionalFlowFeatures:
    def test_proxy_features(self):
        from src.phase_08_features_breadth.institutional_flow_features import (
            InstitutionalFlowFeatures,
        )

        inst = InstitutionalFlowFeatures()
        df = _make_daily(300)
        result = inst.create_institutional_flow_features(df)
        inst_cols = [c for c in result.columns if c.startswith("inst_")]
        assert len(inst_cols) == 8

    def test_no_nan(self):
        from src.phase_08_features_breadth.institutional_flow_features import (
            InstitutionalFlowFeatures,
        )

        inst = InstitutionalFlowFeatures()
        df = _make_daily(300)
        result = inst.create_institutional_flow_features(df)
        for col in [c for c in result.columns if c.startswith("inst_")]:
            assert result[col].isna().sum() == 0, f"NaN in {col}"
            assert np.isinf(result[col]).sum() == 0, f"Inf in {col}"

    def test_analyze(self):
        from src.phase_08_features_breadth.institutional_flow_features import (
            InstitutionalFlowFeatures,
        )

        inst = InstitutionalFlowFeatures()
        df = _make_daily(300)
        df = inst.create_institutional_flow_features(df)
        analysis = inst.analyze_current_flow(df)
        assert analysis is not None
        assert "regime" in analysis


# ─────────────────────────────────────────────────────────
# L5: GoogleTrendsFeatures
# ─────────────────────────────────────────────────────────
class TestGoogleTrendsFeatures:
    def test_proxy_features(self):
        from src.phase_08_features_breadth.google_trends_features import (
            GoogleTrendsFeatures,
        )

        gtrend = GoogleTrendsFeatures()
        df = _make_daily(300)
        result = gtrend.create_google_trends_features(df)
        gtrend_cols = [c for c in result.columns if c.startswith("gtrend_")]
        assert len(gtrend_cols) == 8

    def test_no_nan(self):
        from src.phase_08_features_breadth.google_trends_features import (
            GoogleTrendsFeatures,
        )

        gtrend = GoogleTrendsFeatures()
        df = _make_daily(300)
        result = gtrend.create_google_trends_features(df)
        for col in [c for c in result.columns if c.startswith("gtrend_")]:
            assert result[col].isna().sum() == 0, f"NaN in {col}"
            assert np.isinf(result[col]).sum() == 0, f"Inf in {col}"

    def test_analyze(self):
        from src.phase_08_features_breadth.google_trends_features import (
            GoogleTrendsFeatures,
        )

        gtrend = GoogleTrendsFeatures()
        df = _make_daily(300)
        df = gtrend.create_google_trends_features(df)
        analysis = gtrend.analyze_current_trends(df)
        assert analysis is not None
        assert "mood" in analysis


# ─────────────────────────────────────────────────────────
# L6: CommoditySignalFeatures
# ─────────────────────────────────────────────────────────
class TestCommoditySignalFeatures:
    def test_proxy_features(self):
        from src.phase_08_features_breadth.commodity_signal_features import (
            CommoditySignalFeatures,
        )

        cmdty = CommoditySignalFeatures()
        df = _make_daily(300)
        result = cmdty.create_commodity_signal_features(df)
        cmdty_cols = [c for c in result.columns if c.startswith("cmdty_")]
        assert len(cmdty_cols) == 10

    def test_no_nan(self):
        from src.phase_08_features_breadth.commodity_signal_features import (
            CommoditySignalFeatures,
        )

        cmdty = CommoditySignalFeatures()
        df = _make_daily(300)
        result = cmdty.create_commodity_signal_features(df)
        for col in [c for c in result.columns if c.startswith("cmdty_")]:
            assert result[col].isna().sum() == 0, f"NaN in {col}"
            assert np.isinf(result[col]).sum() == 0, f"Inf in {col}"

    def test_analyze(self):
        from src.phase_08_features_breadth.commodity_signal_features import (
            CommoditySignalFeatures,
        )

        cmdty = CommoditySignalFeatures()
        df = _make_daily(300)
        df = cmdty.create_commodity_signal_features(df)
        analysis = cmdty.analyze_current_commodity(df)
        assert analysis is not None
        assert "regime" in analysis
        assert analysis["regime"] in ("EXPANSION", "CONTRACTION", "NEUTRAL")

    def test_feature_names(self):
        from src.phase_08_features_breadth.commodity_signal_features import (
            CommoditySignalFeatures,
        )

        names = CommoditySignalFeatures._all_feature_names()
        assert len(names) == 10
        assert all(n.startswith("cmdty_") for n in names)


# ─────────────────────────────────────────────────────────
# L7: TreasuryAuctionFeatures
# ─────────────────────────────────────────────────────────
class TestTreasuryAuctionFeatures:
    def test_proxy_features(self):
        from src.phase_08_features_breadth.treasury_auction_features import (
            TreasuryAuctionFeatures,
        )

        tauct = TreasuryAuctionFeatures()
        df = _make_daily(300)
        result = tauct.create_treasury_auction_features(df)
        tauct_cols = [c for c in result.columns if c.startswith("tauct_")]
        assert len(tauct_cols) == 6

    def test_no_nan(self):
        from src.phase_08_features_breadth.treasury_auction_features import (
            TreasuryAuctionFeatures,
        )

        tauct = TreasuryAuctionFeatures()
        df = _make_daily(300)
        result = tauct.create_treasury_auction_features(df)
        for col in [c for c in result.columns if c.startswith("tauct_")]:
            assert result[col].isna().sum() == 0, f"NaN in {col}"
            assert np.isinf(result[col]).sum() == 0, f"Inf in {col}"

    def test_analyze(self):
        from src.phase_08_features_breadth.treasury_auction_features import (
            TreasuryAuctionFeatures,
        )

        tauct = TreasuryAuctionFeatures()
        df = _make_daily(300)
        df = tauct.create_treasury_auction_features(df)
        analysis = tauct.analyze_current_auction(df)
        assert analysis is not None
        assert "quality" in analysis


# ─────────────────────────────────────────────────────────
# L8: FedLiquidityFeatures
# ─────────────────────────────────────────────────────────
class TestFedLiquidityFeatures:
    def test_proxy_features(self):
        from src.phase_08_features_breadth.fed_liquidity_features import (
            FedLiquidityFeatures,
        )

        fedliq = FedLiquidityFeatures()
        df = _make_daily(300)
        result = fedliq.create_fed_liquidity_features(df)
        fedliq_cols = [c for c in result.columns if c.startswith("fedliq_")]
        assert len(fedliq_cols) == 8

    def test_no_nan(self):
        from src.phase_08_features_breadth.fed_liquidity_features import (
            FedLiquidityFeatures,
        )

        fedliq = FedLiquidityFeatures()
        df = _make_daily(300)
        result = fedliq.create_fed_liquidity_features(df)
        for col in [c for c in result.columns if c.startswith("fedliq_")]:
            assert result[col].isna().sum() == 0, f"NaN in {col}"
            assert np.isinf(result[col]).sum() == 0, f"Inf in {col}"

    def test_analyze(self):
        from src.phase_08_features_breadth.fed_liquidity_features import (
            FedLiquidityFeatures,
        )

        fedliq = FedLiquidityFeatures()
        df = _make_daily(300)
        df = fedliq.create_fed_liquidity_features(df)
        analysis = fedliq.analyze_current_liquidity(df)
        assert analysis is not None
        assert "regime" in analysis
        assert analysis["regime"] in ("EASING", "TIGHTENING", "NEUTRAL")

    def test_fred_series_defined(self):
        from src.phase_08_features_breadth.fed_liquidity_features import (
            FedLiquidityFeatures,
        )

        assert "WALCL" in FedLiquidityFeatures.FRED_SERIES
        assert "RRPONTSYD" in FedLiquidityFeatures.FRED_SERIES


# ─────────────────────────────────────────────────────────
# L9: EarningsCalendarFeatures
# ─────────────────────────────────────────────────────────
class TestEarningsCalendarFeatures:
    def test_proxy_features(self):
        from src.phase_08_features_breadth.earnings_calendar_features import (
            EarningsCalendarFeatures,
        )

        ecal = EarningsCalendarFeatures()
        df = _make_daily(300)
        result = ecal.create_earnings_calendar_features(df)
        ecal_cols = [c for c in result.columns if c.startswith("ecal_")]
        assert len(ecal_cols) == 6

    def test_no_nan(self):
        from src.phase_08_features_breadth.earnings_calendar_features import (
            EarningsCalendarFeatures,
        )

        ecal = EarningsCalendarFeatures()
        df = _make_daily(300)
        result = ecal.create_earnings_calendar_features(df)
        for col in [c for c in result.columns if c.startswith("ecal_")]:
            assert result[col].isna().sum() == 0, f"NaN in {col}"
            assert np.isinf(result[col]).sum() == 0, f"Inf in {col}"

    def test_analyze(self):
        from src.phase_08_features_breadth.earnings_calendar_features import (
            EarningsCalendarFeatures,
        )

        ecal = EarningsCalendarFeatures()
        df = _make_daily(300)
        df = ecal.create_earnings_calendar_features(df)
        analysis = ecal.analyze_current_calendar(df)
        assert analysis is not None
        assert "in_earnings_season" in analysis

    def test_peak_season_dates(self):
        from src.phase_08_features_breadth.earnings_calendar_features import (
            EarningsCalendarFeatures,
        )

        # January week 3 should be peak season
        assert (1, 3) in EarningsCalendarFeatures.PEAK_MONTHS_WEEKS
        # March week 2 should NOT be peak season
        assert (3, 2) not in EarningsCalendarFeatures.PEAK_MONTHS_WEEKS


# ─────────────────────────────────────────────────────────
# L10: AnalystRatingFeatures
# ─────────────────────────────────────────────────────────
class TestAnalystRatingFeatures:
    def test_proxy_features(self):
        from src.phase_08_features_breadth.analyst_rating_features import (
            AnalystRatingFeatures,
        )

        anlst = AnalystRatingFeatures()
        df = _make_daily(300)
        result = anlst.create_analyst_rating_features(df)
        anlst_cols = [c for c in result.columns if c.startswith("anlst_")]
        assert len(anlst_cols) == 8

    def test_no_nan(self):
        from src.phase_08_features_breadth.analyst_rating_features import (
            AnalystRatingFeatures,
        )

        anlst = AnalystRatingFeatures()
        df = _make_daily(300)
        result = anlst.create_analyst_rating_features(df)
        for col in [c for c in result.columns if c.startswith("anlst_")]:
            assert result[col].isna().sum() == 0, f"NaN in {col}"
            assert np.isinf(result[col]).sum() == 0, f"Inf in {col}"

    def test_analyze(self):
        from src.phase_08_features_breadth.analyst_rating_features import (
            AnalystRatingFeatures,
        )

        anlst = AnalystRatingFeatures()
        df = _make_daily(300)
        df = anlst.create_analyst_rating_features(df)
        analysis = anlst.analyze_current_ratings(df)
        assert analysis is not None
        assert "consensus" in analysis
        assert analysis["consensus"] in ("STRONG_BUY", "BUY", "HOLD", "SELL")


# ─────────────────────────────────────────────────────────
# Integration: Config flags exist
# ─────────────────────────────────────────────────────────
class TestWaveLConfig:
    def test_config_flags_exist(self):
        from src.experiment_config import AntiOverfitConfig

        config = AntiOverfitConfig()
        assert hasattr(config, "use_earnings_revision")
        assert hasattr(config, "use_short_interest")
        assert hasattr(config, "use_dollar_index")
        assert hasattr(config, "use_institutional_flow")
        assert hasattr(config, "use_google_trends")
        assert hasattr(config, "use_commodity_signals")
        assert hasattr(config, "use_treasury_auction")
        assert hasattr(config, "use_fed_liquidity")
        assert hasattr(config, "use_earnings_calendar")
        assert hasattr(config, "use_analyst_rating")

    def test_default_values(self):
        from src.experiment_config import AntiOverfitConfig

        config = AntiOverfitConfig()
        # Default ON
        assert config.use_earnings_revision is True
        assert config.use_dollar_index is True
        assert config.use_commodity_signals is True
        assert config.use_fed_liquidity is True
        assert config.use_earnings_calendar is True
        assert config.use_analyst_rating is True
        # Default OFF
        assert config.use_short_interest is False
        assert config.use_institutional_flow is False
        assert config.use_google_trends is False
        assert config.use_treasury_auction is False

    def test_feature_groups_registered(self):
        from src.phase_10_feature_processing.group_aware_processor import (
            FEATURE_GROUPS,
        )

        expected = [
            "earnings_revision",
            "short_interest",
            "dollar_index",
            "institutional_flow",
            "google_trends",
            "commodity_signal",
            "treasury_auction",
            "fed_liquidity",
            "earnings_calendar",
            "analyst_rating",
        ]
        for group in expected:
            assert group in FEATURE_GROUPS, f"Missing FEATURE_GROUP: {group}"

"""
GIGA TRADER - Anti-Overfit Integration
========================================
Main integration function that combines all anti-overfitting measures:
  1. Component Streak Breadth Features
  2. Cross-Asset Features
  3. MAG Market Breadth Features (MAG3/5/6/7/10/15)
  4. Sector Breadth Features (S&P 500 Sectors)
  5. Volatility Regime Features (VXX-based)
  6. Calendar & Event Features (FOMC, opex, NFP, CPI, GDP)
  7. Sentiment Features (VIX fear/greed, cross-asset flows, optional news)
  8. Economic Indicator Features (yields, VIX, credit spreads)
  9. Event Recency Features (days since last drop, rally, reversal, streak)
  10. Synthetic SPY Universes
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from src.phase_08_features_breadth.streak_features import ComponentStreakFeatures
from src.phase_08_features_breadth.cross_asset_features import CrossAssetFeatures
from src.phase_08_features_breadth.mag7_breadth import Mag7BreadthFeatures
from src.phase_08_features_breadth.sector_breadth import SectorBreadthFeatures
from src.phase_08_features_breadth.volatility_regime import VolatilityRegimeFeatures
from src.phase_03_synthetic_data.synthetic_universe import SyntheticSPYGenerator
from src.phase_08_features_breadth.economic_features import EconomicFeatures
from src.phase_08_features_breadth.sentiment_features import SentimentFeatures
from src.phase_08_features_breadth.fear_greed_features import FearGreedFeatures
from src.phase_08_features_breadth.reddit_sentiment_features import RedditSentimentFeatures
from src.phase_08_features_breadth.crypto_sentiment_features import CryptoSentimentFeatures
from src.phase_08_features_breadth.gamma_exposure_features import GammaExposureFeatures
from src.phase_08_features_breadth.finnhub_social_features import FinnhubSocialFeatures
from src.phase_08_features_breadth.dark_pool_features import DarkPoolFeatures
from src.phase_08_features_breadth.options_features import OptionsFeatures
from src.phase_08_features_breadth.event_recency_features import EventRecencyFeatures
from src.phase_08_features_breadth.block_structure_features import BlockStructureFeatures
from src.phase_08_features_breadth.amihud_features import AmihudFeatures
from src.phase_08_features_breadth.range_vol_features import RangeVolFeatures
from src.phase_08_features_breadth.entropy_features import EntropyFeatures
from src.phase_08_features_breadth.hurst_features import HurstFeatures
from src.phase_08_features_breadth.nmi_features import NMIFeatures
from src.phase_08_features_breadth.absorption_ratio_features import AbsorptionRatioFeatures
from src.phase_08_features_breadth.drift_features import DriftFeatures
from src.phase_08_features_breadth.changepoint_features import ChangepointFeatures
from src.phase_08_features_breadth.hmm_features import HMMFeatures
from src.phase_08_features_breadth.vpin_features import VPINFeatures
from src.phase_08_features_breadth.intraday_momentum_features import IntradayMomentumFeatures
from src.phase_08_features_breadth.futures_basis_features import FuturesBasisFeatures
from src.phase_08_features_breadth.congressional_features import CongressionalFeatures
from src.phase_08_features_breadth.insider_aggregate_features import InsiderAggregateFeatures
from src.phase_08_features_breadth.etf_flow_features import ETFFlowFeatures
from src.phase_08_features_breadth.wavelet_features import WaveletFeatures
from src.phase_08_features_breadth.sax_features import SAXFeatures
from src.phase_08_features_breadth.transfer_entropy_features import TransferEntropyFeatures
from src.phase_08_features_breadth.mfdfa_features import MFDFAFeatures
from src.phase_08_features_breadth.rqa_features import RQAFeatures
from src.phase_08_features_breadth.copula_features import CopulaFeatures
from src.phase_08_features_breadth.network_features import NetworkFeatures
from src.phase_08_features_breadth.path_signature_features import PathSignatureFeatures
from src.phase_08_features_breadth.wavelet_scattering_features import WaveletScatteringFeatures
from src.phase_14_robustness.wasserstein_regime import WassersteinRegimeDetector
from src.phase_08_features_breadth.market_structure_features import MarketStructureFeatures
from src.phase_08_features_breadth.time_series_model_features import TimeSeriesModelFeatures
from src.phase_08_features_breadth.har_rv_features import HARRVFeatures
from src.phase_08_features_breadth.l_moments_features import LMomentsFeatures
from src.phase_08_features_breadth.multiscale_entropy_features import MultiscaleEntropyFeatures
from src.phase_08_features_breadth.rv_signature_features import RVSignaturePlotFeatures
from src.phase_08_features_breadth.tda_features import TDAHomologyFeatures
from src.phase_08_features_breadth.credit_spread_features import CreditSpreadFeatures
from src.phase_08_features_breadth.yield_curve_features import YieldCurveFeatures
from src.phase_08_features_breadth.vol_term_structure_features import VolTermStructureFeatures
from src.phase_08_features_breadth.macro_surprise_features import MacroSurpriseFeatures
from src.phase_08_features_breadth.cross_asset_momentum_features import CrossAssetMomentumFeatures
from src.phase_08_features_breadth.skew_kurtosis_features import SkewKurtosisFeatures
from src.phase_08_features_breadth.seasonality_features import SeasonalityFeatures
from src.phase_08_features_breadth.order_flow_imbalance_features import OrderFlowImbalanceFeatures
from src.phase_08_features_breadth.correlation_regime_features import CorrelationRegimeFeatures
from src.phase_08_features_breadth.fama_french_features import FamaFrenchFeatures
from src.phase_08_features_breadth.put_call_ratio_features import PutCallRatioFeatures
from src.phase_09_features_calendar.calendar_features import CalendarFeatureGenerator
from src.phase_15_strategy.multi_horizon_filter import MultiHorizonFilter
from src.phase_08_features_breadth.earnings_revision_features import EarningsRevisionFeatures
from src.phase_08_features_breadth.short_interest_features import ShortInterestFeatures
from src.phase_08_features_breadth.dollar_index_features import DollarIndexFeatures
from src.phase_08_features_breadth.institutional_flow_features import InstitutionalFlowFeatures
from src.phase_08_features_breadth.google_trends_features import GoogleTrendsFeatures
from src.phase_08_features_breadth.commodity_signal_features import CommoditySignalFeatures
from src.phase_08_features_breadth.treasury_auction_features import TreasuryAuctionFeatures
from src.phase_08_features_breadth.fed_liquidity_features import FedLiquidityFeatures
from src.phase_08_features_breadth.earnings_calendar_features import EarningsCalendarFeatures
from src.phase_08_features_breadth.analyst_rating_features import AnalystRatingFeatures
from src.phase_08_features_breadth.expanded_macro_features import ExpandedMacroFeatures
from src.phase_08_features_breadth.vvix_features import VVIXFeatures
from src.phase_08_features_breadth.sector_rotation_features import SectorRotationFeatures
from src.phase_08_features_breadth.fx_carry_features import FXCarryFeatures
from src.phase_08_features_breadth.money_market_features import MoneyMarketFeatures
from src.phase_08_features_breadth.financial_stress_features import FinancialStressFeatures
from src.phase_08_features_breadth.global_equity_features import GlobalEquityFeatures
from src.phase_08_features_breadth.retail_sentiment_features import RetailSentimentFeatures
from src.core.system_resources import maybe_gc as _maybe_gc


def integrate_anti_overfit(
    df_daily: pd.DataFrame,
    spy_1min: pd.DataFrame = None,
    use_synthetic: bool = True,
    use_cross_assets: bool = True,
    use_breadth_streaks: bool = True,
    use_mag_breadth: bool = True,  # MAG3/5/6/7/10/15 features
    use_sector_breadth: bool = True,  # Sector rotation features
    use_vol_regime: bool = True,  # Volatility regime features
    use_economic_features: bool = True,  # Economic indicators (yields, VIX, credit)
    use_calendar_features: bool = True,  # FOMC, opex, NFP/CPI/GDP event features
    use_sentiment_features: bool = True,  # VIX fear/greed, cross-asset flows, news
    validate_ohlc: bool = True,  # OHLC data validation (Wave F1.1)
    use_fear_greed: bool = True,  # CNN Fear & Greed Index features
    use_reddit_sentiment: bool = True,  # ApeWisdom Reddit sentiment
    use_crypto_sentiment: bool = True,  # Alternative.me Crypto Fear & Greed
    use_gamma_exposure: bool = True,  # GEX proxy from VIX term structure
    use_finnhub_social: bool = True,  # Finnhub social sentiment (requires key)
    use_dark_pool: bool = True,  # FINRA dark pool short sale volume
    use_options_features: bool = True,  # Options IV/SKEW features (VIX rank, SKEW Index)
    use_event_recency: bool = True,  # Days-since event recency features (dts_*)
    use_block_structure: bool = True,  # Multi-day block structure features (blk_*)
    use_amihud_features: bool = True,  # Amihud illiquidity ratio (liq_*)
    use_range_vol_features: bool = True,  # Range-based vol estimators (rvol_*)
    use_entropy_features: bool = True,  # Shannon/permutation/sample entropy (ent_*)
    use_hurst_features: bool = True,  # R/S Hurst exponent (hurst_*)
    use_nmi_features: bool = True,  # NMI market efficiency (nmi_*)
    use_absorption_ratio: bool = True,  # PCA-based systemic risk (ar_*)
    use_drift_features: bool = True,  # ADWIN drift detection (drift_*)
    use_changepoint_features: bool = True,  # Bayesian changepoint detection (cpd_*)
    use_hmm_features: bool = True,  # HMM regime states (hmm_*)
    use_vpin_features: bool = True,  # VPIN order flow toxicity (vpin_*)
    use_intraday_momentum: bool = True,  # Intraday momentum (imom_*)
    use_futures_basis: bool = True,  # Futures-spot basis (basis_*)
    use_congressional_features: bool = True,  # Congressional trading proxy (congress_*)
    use_insider_aggregate: bool = True,  # Insider aggregate proxy (insider_agg_*)
    use_etf_flow: bool = True,  # ETF fund flow proxy (etf_flow_*)
    use_wavelet_features: bool = True,  # Wavelet decomposition (wav_*)
    use_sax_features: bool = True,  # SAX patterns (sax_*)
    use_transfer_entropy: bool = True,  # Transfer entropy (te_*)
    use_mfdfa_features: bool = True,  # MFDFA fractal analysis (mfdfa_*)
    use_rqa_features: bool = True,  # RQA recurrence analysis (rqa_*)
    use_copula_features: bool = True,  # Copula tail dependence (copula_*)
    use_network_centrality: bool = True,  # Correlation network centrality (netw_*)
    use_path_signatures: bool = True,  # Path signature features (psig_*)
    use_wavelet_scattering: bool = True,  # Wavelet scattering features (wscat_*)
    use_wasserstein_regime: bool = True,  # Wasserstein regime detection (wreg_*)
    use_market_structure: bool = True,  # Market structure compression/inflection (mstr_*)
    use_time_series_models: bool = False,  # ARIMA/Chronos time series model features (tsm_*)
    use_catch22: bool = False,  # catch22 canonical time series features (tsm_c22_*)
    use_har_rv: bool = True,  # HAR-RV volatility features (harv_*)
    use_l_moments: bool = True,  # L-Moments distributional features (lmom_*)
    use_multiscale_entropy: bool = True,  # Multiscale sample entropy (mse_*)
    use_rv_signature_plot: bool = False,  # RV signature plot features (rvsp_*)
    use_tda_homology: bool = False,  # TDA persistent homology (tda_*)
    use_credit_spread_features: bool = True,  # Credit spread features (cred_*)
    use_yield_curve_features: bool = True,  # Yield curve features (yc_*)
    use_vol_term_structure_features: bool = True,  # Vol term structure features (vts_*)
    use_macro_surprise_features: bool = True,  # Macro surprise features (msurp_*)
    use_cross_asset_momentum: bool = True,  # Cross-asset momentum features (xmom_*)
    use_skew_kurtosis_features: bool = True,  # Skew/kurtosis features (skku_*)
    use_seasonality_features: bool = True,  # Seasonality features (seas_*)
    use_order_flow_imbalance: bool = True,  # Order flow imbalance features (ofi_*)
    use_correlation_regime: bool = True,  # Cross-asset correlation regime features (corr_*)
    use_fama_french: bool = True,  # Fama-French factor exposure features (ff_*)
    use_put_call_ratio: bool = True,  # Put-call ratio features (pcr_*)
    use_multi_horizon: bool = True,  # Multi-horizon ensemble filter features (mh_*)
    use_earnings_revision: bool = True,  # Earnings estimate revision features (ern_*)
    use_short_interest: bool = False,  # Short interest features (si_*) -- sparse
    use_dollar_index: bool = True,  # Dollar index features (dxy_*)
    use_institutional_flow: bool = False,  # Institutional flow features (inst_*) -- sparse
    use_google_trends: bool = False,  # Google Trends features (gtrend_*) -- rate limited
    use_commodity_signals: bool = True,  # Commodity signal features (cmdty_*)
    use_treasury_auction: bool = False,  # Treasury auction features (tauct_*) -- limited
    use_fed_liquidity: bool = True,  # Fed liquidity features (fedliq_*)
    use_earnings_calendar: bool = True,  # Earnings calendar features (ecal_*)
    use_analyst_rating: bool = True,  # Analyst rating features (anlst_*)
    use_expanded_macro: bool = True,  # Expanded FRED macro features (xmacro_*)
    use_vvix: bool = True,  # VVIX vol-of-vol features (vvix_*)
    use_sector_rotation: bool = True,  # Sector rotation rank features (secrot_*)
    use_fx_carry: bool = True,  # FX carry & currency features (fxc_*)
    use_money_market: bool = True,  # Money market rate features (mmkt_*)
    use_financial_stress: bool = True,  # Financial stress index features (fstress_*)
    use_global_equity: bool = True,  # Global equity ETF features (gleq_*)
    use_retail_sentiment: bool = True,  # Retail sentiment proxy features (rflow_*)
    synthetic_weight: float = 0.4,  # Weight for synthetic data (real = 1 - synthetic)
    use_bear_universes: bool = True,  # Bear market synthetic series
    bear_mean_shift_bps: Optional[List[int]] = None,
    bear_vol_amplify_factor: float = 1.5,
    bear_vol_dampen_factor: float = 0.7,
    use_multiscale_bootstrap: bool = True,  # Multi-timescale regime bootstrap
    resource_config=None,  # ResourceConfig for memory-aware scaling
) -> Tuple[pd.DataFrame, Dict]:
    """
    Integrate all anti-overfitting measures.

    Args:
        df_daily: Daily feature dataframe
        spy_1min: 1-minute SPY data (optional)
        use_synthetic: Generate synthetic SPY universes
        use_cross_assets: Add TLT, QQQ, GLD features
        use_breadth_streaks: Add component streak breadth features
        use_mag_breadth: Add MAG3/5/6/7/10/15 market breadth features
        use_sector_breadth: Add sector rotation and breadth features
        use_vol_regime: Add volatility regime features (VXX-based)
        synthetic_weight: Weight for synthetic data (0.4 = 40% synthetic)

    Returns:
        Augmented dataframe and metadata
    """
    print("\n" + "=" * 70)
    print("ANTI-OVERFITTING INTEGRATION")
    print("=" * 70)

    # Convert date column to pd.Timestamp for consistent merging
    # (spy_daily may have datetime.date objects which don't merge with Timestamps)
    df_daily = df_daily.copy()
    try:
        df_daily["date"] = pd.to_datetime(df_daily["date"], errors='coerce')
        # Remove any rows with invalid dates
        df_daily = df_daily.dropna(subset=["date"])
    except Exception as e:
        print(f"[ANTI-OVERFIT] Warning: Date conversion issue: {e}")

    metadata = {}

    # Safely get date range with validation
    try:
        start_date = pd.to_datetime(df_daily["date"].min())
        end_date = pd.to_datetime(df_daily["date"].max())

        # Validate dates are reasonable (not too far in past or future)
        min_valid_date = pd.Timestamp("2000-01-01")
        max_valid_date = pd.Timestamp.now() + pd.Timedelta(days=1)

        if start_date < min_valid_date or end_date > max_valid_date:
            print(f"[ANTI-OVERFIT] Warning: Date range seems invalid: {start_date} to {end_date}")
            print("[ANTI-OVERFIT] Clamping to valid range")
            start_date = max(start_date, min_valid_date)
            end_date = min(end_date, max_valid_date)
    except Exception as e:
        print(f"[ANTI-OVERFIT] Error getting date range: {e}")
        # Default to last 5 years if date extraction fails
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=365 * 5)

    # 0. OHLC Data Validation (must run before any feature engineering)
    if validate_ohlc:
        try:
            from src.phase_02_preprocessing.ohlc_validator import OHLCValidator
            validator = OHLCValidator(max_daily_pct_change=0.50, auto_fix=True)
            df_daily, ohlc_stats = validator.validate(df_daily)
            n_fixed = ohlc_stats.get("rows_fixed", 0)
            n_dropped = ohlc_stats.get("rows_dropped", 0)
            if n_fixed > 0 or n_dropped > 0:
                print(f"  [OHLC] Validated: {n_fixed} rows fixed, {n_dropped} rows dropped")
            metadata["ohlc_validated"] = True
            metadata["ohlc_stats"] = ohlc_stats
        except Exception as e:
            print(f"  [OHLC] Warning: OHLC validation failed: {e}")
            metadata["ohlc_validated"] = False

    # 1. Component Streak Breadth Features
    if use_breadth_streaks:
        try:
            streak_features = ComponentStreakFeatures(max_streak=7)
            component_prices = streak_features.download_component_data(start_date, end_date)

            if not component_prices.empty:
                df_daily = streak_features.compute_streak_features(component_prices, df_daily)
                metadata["streak_features"] = True
                metadata["components_tracked"] = len(component_prices.columns)

                # Analyze current breadth
                signal = streak_features.analyze_breadth_signal(df_daily)
                if signal:
                    print(f"  Current breadth: {signal.get('interpretation', 'N/A')}")
        except Exception as e:
            print(f"  [BREADTH_STREAKS] Warning: Breadth streak features failed: {e}")
            metadata["streak_features"] = False

    # 2. Cross-Asset Features
    if use_cross_assets:
        try:
            cross_assets = CrossAssetFeatures()
            cross_data = cross_assets.download_cross_assets(start_date, end_date)

            if not cross_data.empty:
                df_daily = cross_assets.create_cross_asset_features(cross_data, df_daily)
                metadata["cross_assets"] = list(cross_data.columns)
        except Exception as e:
            print(f"  [CROSS_ASSETS] Warning: Cross-asset features failed: {e}")
            metadata["cross_assets"] = False

    # 3. MAG Market Breadth Features (MAG3/5/6/7/10/15)
    if use_mag_breadth:
        try:
            mag_breadth = Mag7BreadthFeatures()
            mag_data = mag_breadth.download_mag_data(start_date, end_date)

            if not mag_data.empty:
                df_daily = mag_breadth.create_mag_features(df_daily)
                metadata["mag_features"] = True
                metadata["mag_tickers"] = mag_breadth.ALL_MAG_TICKERS
                metadata["mag_groups"] = ["MAG3", "MAG5", "MAG6", "MAG7", "MAG10", "MAG15"]

                # Analyze current MAG leadership
                mag_signal = mag_breadth.analyze_mag_leadership(df_daily)
                if mag_signal:
                    print(f"  MAG7 Leadership: {mag_signal.get('interpretation', 'N/A')}")
                    print(f"    Bias: {mag_signal.get('bias', 'N/A')}")
                    print(f"    Tech Rotation: {mag_signal.get('rotation', 'N/A')}")
                    if mag_signal.get('breadth_warning') != 'NONE':
                        print(f"    Breadth Warning: {mag_signal.get('breadth_warning')}")
        except Exception as e:
            print(f"  [MAG_BREADTH] Warning: MAG breadth features failed: {e}")
            metadata["mag_features"] = False

    # 4. Sector Breadth Features (S&P 500 Sectors)
    if use_sector_breadth:
        try:
            sector_breadth = SectorBreadthFeatures()
            sector_data = sector_breadth.download_sector_data(start_date, end_date)

            if not sector_data.empty:
                df_daily = sector_breadth.create_sector_features(df_daily)
                metadata["sector_features"] = True
                metadata["sector_etfs"] = list(sector_breadth.SECTOR_ETFS.keys())

                # Analyze current sector rotation
                sector_signal = sector_breadth.analyze_sector_rotation(df_daily)
                if sector_signal:
                    print(f"  Sector Rotation: {sector_signal.get('rotation', 'N/A')}")
                    print(f"    Market Phase: {sector_signal.get('market_phase', 'N/A')}")
                    print(f"    Risk Sentiment: {sector_signal.get('risk_sentiment', 'N/A')}")
        except Exception as e:
            print(f"  [SECTOR_BREADTH] Warning: Sector breadth features failed: {e}")
            metadata["sector_features"] = False

    # 5. Volatility Regime Features (VXX-based)
    if use_vol_regime:
        try:
            vol_regime = VolatilityRegimeFeatures()
            vol_data = vol_regime.download_vol_data(start_date, end_date)

            if not vol_data.empty:
                df_daily = vol_regime.create_vol_features(df_daily)
                metadata["vol_regime_features"] = True

                # Analyze current volatility regime
                vol_signal = vol_regime.analyze_vol_regime(df_daily)
                if vol_signal:
                    print(f"  Volatility Regime: {vol_signal.get('regime', 'N/A')}")
                    print(f"    Market Condition: {vol_signal.get('market_condition', 'N/A')}")
                    print(f"    VXX Percentile: {vol_signal.get('vxx_percentile', 0):.1%}")
        except Exception as e:
            print(f"  [VOL_REGIME] Warning: Volatility regime features failed: {e}")
            metadata["vol_regime_features"] = False

    # 6. Calendar & Event Features (FOMC, opex, NFP, CPI, GDP)
    if use_calendar_features:
        try:
            cal_gen = CalendarFeatureGenerator(
                include_fomc=True,
                include_opex=True,
                include_economic=True,
                include_calendar_basics=True,
            )
            n_before = len(df_daily.columns)
            df_daily = cal_gen.create_all_features(df_daily)
            n_cal_features = len(df_daily.columns) - n_before
            metadata["calendar_features"] = True
            metadata["n_calendar_features"] = n_cal_features
            print(f"  [CALENDAR] Added {n_cal_features} calendar/event features "
                  f"(FOMC, opex, NFP, CPI, PMI, GDP)")

            # Log upcoming events
            feature_names = cal_gen.get_feature_names()
            fomc_cols = [c for c in df_daily.columns if c.startswith("fomc_")]
            opex_cols = [c for c in df_daily.columns if c.startswith("opex_")]
            econ_event_cols = [c for c in df_daily.columns if c.startswith("econ_is_")]
            cal_cols = [c for c in df_daily.columns if c.startswith("cal_")]
            print(f"    FOMC: {len(fomc_cols)} features | Opex: {len(opex_cols)} | "
                  f"Economic events: {len(econ_event_cols)} | Calendar: {len(cal_cols)}")
        except Exception as e:
            print(f"  [CALENDAR] Warning: Calendar feature generation failed: {e}")
            metadata["calendar_features"] = False

    # 7. Sentiment Features (VIX fear/greed, cross-asset flows, optional news)
    if use_sentiment_features:
        try:
            sent_engine = SentimentFeatures()
            sent_data = sent_engine.download_sentiment_data(start_date, end_date)

            if not sent_data.empty:
                df_daily = sent_engine.create_sentiment_features(df_daily)
                sent_cols = [c for c in df_daily.columns if c.startswith("sent_")]
                metadata["sentiment_features"] = True
                metadata["n_sentiment_features"] = len(sent_cols)
                print(f"  [SENTIMENT] Added {len(sent_cols)} sentiment features")

                # Analyze current sentiment conditions
                conditions = sent_engine.analyze_current_sentiment(df_daily)
                if conditions:
                    regime = conditions.get("sentiment_regime", "N/A")
                    appetite = conditions.get("risk_appetite", "N/A")
                    print(f"    Sentiment Regime: {regime} | Risk Appetite: {appetite}")
            else:
                metadata["sentiment_features"] = False
        except Exception as e:
            print(f"  [SENTIMENT] Warning: Sentiment feature generation failed: {e}")
            metadata["sentiment_features"] = False

    # 8. Economic Indicator Features (yields, VIX, credit spreads)
    if use_economic_features:
        try:
            econ_features = EconomicFeatures()
            econ_data = econ_features.download_economic_data(start_date, end_date)

            if not econ_data.empty:
                df_daily = econ_features.create_economic_features(df_daily)
                metadata["economic_features"] = True
                metadata["economic_sources"] = list(econ_data.columns)

                # Analyze current conditions
                econ_signal = econ_features.analyze_current_conditions(df_daily)
                if econ_signal:
                    if "vix_regime" in econ_signal:
                        print(f"  VIX Regime: {econ_signal['vix_regime']} "
                              f"(level={econ_signal.get('vix_level', 0):.1f})")
                    if "yield_curve_signal" in econ_signal:
                        print(f"  Yield Curve: {econ_signal['yield_curve_signal']} "
                              f"(10Y-5Y={econ_signal.get('yield_curve_10_5', 0):.2f})")
                    if "credit_signal" in econ_signal:
                        print(f"  Credit: {econ_signal['credit_signal']}")
        except Exception as e:
            print(f"  [ECONOMIC] Warning: Economic features failed: {e}")
            metadata["economic_features"] = False

    _maybe_gc(resource_config, "steps 0-8")

    # 10. CNN Fear & Greed Index Features
    if use_fear_greed:
        try:
            fg_features = FearGreedFeatures()
            fg_data = fg_features.download_fear_greed_data(start_date, end_date)

            if not fg_data.empty:
                df_daily = fg_features.create_fear_greed_features(df_daily)
                metadata["fear_greed_features"] = True

                # Analyze current conditions
                fg_signal = fg_features.analyze_current_fear_greed(df_daily)
                if fg_signal:
                    print(f"  Fear & Greed: {fg_signal.get('fear_greed_regime', 'N/A')} "
                          f"(score={fg_signal.get('fear_greed_score', 0):.0f})")
            else:
                metadata["fear_greed_features"] = False
        except Exception as e:
            print(f"  [FEAR_GREED] Warning: Fear & Greed feature generation failed: {e}")
            metadata["fear_greed_features"] = False

    # 11. Reddit Sentiment Features (ApeWisdom)
    if use_reddit_sentiment:
        try:
            reddit_features = RedditSentimentFeatures()
            reddit_data = reddit_features.download_reddit_data(start_date, end_date)

            if not reddit_data.empty:
                df_daily = reddit_features.create_reddit_features(df_daily)
                metadata["reddit_sentiment_features"] = True
            else:
                metadata["reddit_sentiment_features"] = False
        except Exception as e:
            print(f"  [REDDIT] Warning: Reddit sentiment failed: {e}")
            metadata["reddit_sentiment_features"] = False

    # 12. Crypto Fear & Greed Features (Alternative.me)
    if use_crypto_sentiment:
        try:
            crypto_features = CryptoSentimentFeatures()
            crypto_data = crypto_features.download_crypto_data(start_date, end_date)

            if not crypto_data.empty:
                df_daily = crypto_features.create_crypto_features(df_daily)
                metadata["crypto_sentiment_features"] = True

                crypto_signal = crypto_features.analyze_current_crypto(df_daily)
                if crypto_signal:
                    print(f"  Crypto Sentiment: {crypto_signal.get('crypto_regime', 'N/A')} "
                          f"(score={crypto_signal.get('crypto_fg_score', 0):.0f})")
            else:
                metadata["crypto_sentiment_features"] = False
        except Exception as e:
            print(f"  [CRYPTO] Warning: Crypto sentiment failed: {e}")
            metadata["crypto_sentiment_features"] = False

    # 13. Gamma Exposure (GEX) Proxy Features
    if use_gamma_exposure:
        try:
            gex_features = GammaExposureFeatures()
            gex_data = gex_features.download_gex_data(start_date, end_date)

            if not gex_data.empty:
                df_daily = gex_features.create_gex_features(df_daily)
                metadata["gex_features"] = True

                gex_signal = gex_features.analyze_current_gex(df_daily)
                if gex_signal:
                    print(f"  GEX Proxy: {gex_signal.get('gex_regime', 'N/A')} "
                          f"({gex_signal.get('market_behavior', 'N/A')})")
            else:
                metadata["gex_features"] = False
        except Exception as e:
            print(f"  [GEX] Warning: GEX feature generation failed: {e}")
            metadata["gex_features"] = False

    # 14. Finnhub Social Sentiment Features
    if use_finnhub_social:
        try:
            fh_features = FinnhubSocialFeatures()
            fh_data = fh_features.download_finnhub_social_data(start_date, end_date)

            if not fh_data.empty:
                df_daily = fh_features.create_finnhub_social_features(df_daily)
                metadata["finnhub_social_features"] = True

                fh_signal = fh_features.analyze_current_finnhub_social(df_daily)
                if fh_signal:
                    print(f"  Finnhub Social: {fh_signal.get('social_sentiment', 'N/A')} "
                          f"(score={fh_signal.get('social_score', 0):.2f})")
            else:
                metadata["finnhub_social_features"] = False
        except Exception as e:
            print(f"  [FINNHUB] Warning: Finnhub social feature generation failed: {e}")
            metadata["finnhub_social_features"] = False

    # 15. FINRA Dark Pool / Short Sale Volume Features
    if use_dark_pool:
        try:
            dp_features = DarkPoolFeatures()
            dp_data = dp_features.download_dark_pool_data(start_date, end_date)

            if not dp_data.empty:
                df_daily = dp_features.create_dark_pool_features(df_daily)
                metadata["dark_pool_features"] = True

                dp_signal = dp_features.analyze_current_dark_pool(df_daily)
                if dp_signal:
                    print(f"  Dark Pool: {dp_signal.get('sentiment', 'N/A')} "
                          f"(short ratio={dp_signal.get('short_volume_ratio', 0):.2f})")
            else:
                metadata["dark_pool_features"] = False
        except Exception as e:
            print(f"  [DARK_POOL] Warning: Dark pool feature generation failed: {e}")
            metadata["dark_pool_features"] = False

    # 16. Options IV/SKEW Features (VIX rank, CBOE SKEW Index, vol-of-vol)
    if use_options_features:
        try:
            opt_features = OptionsFeatures()
            opt_data = opt_features.download_options_data(start_date, end_date)

            if not opt_data.empty:
                df_daily = opt_features.create_options_features(df_daily)
                metadata["options_features"] = True

                opt_signal = opt_features.analyze_current_options(df_daily)
                if opt_signal:
                    print(f"  Options IV: rank={opt_signal.get('iv_rank', 0):.0f}% "
                          f"skew={opt_signal.get('skew_regime', 'N/A')}")
            else:
                metadata["options_features"] = False
        except Exception as e:
            print(f"  [OPTIONS] Warning: Options feature generation failed: {e}")
            metadata["options_features"] = False

    # 17. Event Recency Features (days since last drop, rally, reversal, etc.)
    if use_event_recency:
        try:
            recency = EventRecencyFeatures()
            n_before = len(df_daily.columns)
            df_daily = recency.create_event_recency_features(df_daily)
            n_recency = len(df_daily.columns) - n_before
            metadata["event_recency_features"] = True
            metadata["n_event_recency_features"] = n_recency

            # Analyze current conditions
            recency_signal = recency.analyze_current_recency(df_daily)
            if recency_signal:
                regime = recency_signal.get("stress_regime", "N/A")
                trend = recency_signal.get("trend_regime", "N/A")
                print(f"  Event Recency: stress={regime} trend={trend} "
                      f"(last -1%: {recency_signal.get('days_since_1pct_drop', '?')}d, "
                      f"last -2%: {recency_signal.get('days_since_2pct_drop', '?')}d)")
        except Exception as e:
            print(f"  [EVENT_RECENCY] Warning: Event recency features failed: {e}")
            metadata["event_recency_features"] = False

    # 18. Block Structure Features (multi-day 3d/5d blocks, cascades, texture)
    if use_block_structure:
        try:
            blk = BlockStructureFeatures()
            n_before = len(df_daily.columns)
            df_daily = blk.create_block_structure_features(df_daily)
            n_blk = len(df_daily.columns) - n_before
            metadata["block_structure_features"] = True
            metadata["n_block_structure_features"] = n_blk

            blk_signal = blk.analyze_current_structure(df_daily)
            if blk_signal:
                cascade = blk_signal.get("cascade_regime", "N/A")
                trend = blk_signal.get("block_trend", "N/A")
                texture = blk_signal.get("texture_regime", "N/A")
                print(f"  Block Structure: cascade={cascade} trend={trend} "
                      f"texture={texture}")
        except Exception as e:
            print(f"  [BLOCK_STRUCTURE] Warning: Block structure features failed: {e}")
            metadata["block_structure_features"] = False

    _maybe_gc(resource_config, "steps 10-18")

    # 19. Amihud Illiquidity Features (liq_ prefix)
    if use_amihud_features:
        try:
            amihud = AmihudFeatures()
            n_before = len(df_daily.columns)
            df_daily = amihud.create_amihud_features(df_daily)
            n_amihud = len(df_daily.columns) - n_before
            metadata["amihud_features"] = True
            metadata["n_amihud_features"] = n_amihud
            print(f"  [AMIHUD] Added {n_amihud} illiquidity features")

            amihud_signal = amihud.analyze_current_liquidity(df_daily)
            if amihud_signal:
                print(f"    Liquidity Regime: {amihud_signal.get('liquidity_regime', 'N/A')}")
        except Exception as e:
            print(f"  [AMIHUD] Warning: Amihud feature generation failed: {e}")
            metadata["amihud_features"] = False

    # 20. Range-Based Volatility Features (rvol_ prefix)
    if use_range_vol_features:
        try:
            rvol = RangeVolFeatures()
            n_before = len(df_daily.columns)
            df_daily = rvol.create_range_vol_features(df_daily)
            n_rvol = len(df_daily.columns) - n_before
            metadata["range_vol_features"] = True
            metadata["n_range_vol_features"] = n_rvol
            print(f"  [RANGE_VOL] Added {n_rvol} range-based volatility features")

            rvol_signal = rvol.analyze_current_volatility(df_daily)
            if rvol_signal:
                print(f"    Vol Regime: {rvol_signal.get('vol_regime', 'N/A')} "
                      f"(YZ20d={rvol_signal.get('yz_vol_20d', 0):.3f})")
        except Exception as e:
            print(f"  [RANGE_VOL] Warning: Range vol feature generation failed: {e}")
            metadata["range_vol_features"] = False

    # 21. Entropy Features (ent_ prefix)
    if use_entropy_features:
        try:
            entropy = EntropyFeatures()
            n_before = len(df_daily.columns)
            df_daily = entropy.create_entropy_features(df_daily)
            n_ent = len(df_daily.columns) - n_before
            metadata["entropy_features"] = True
            metadata["n_entropy_features"] = n_ent
            print(f"  [ENTROPY] Added {n_ent} entropy features")
        except Exception as e:
            print(f"  [ENTROPY] Warning: Entropy feature generation failed: {e}")
            metadata["entropy_features"] = False

    # 22. Hurst Exponent Features (hurst_ prefix)
    if use_hurst_features:
        try:
            hurst = HurstFeatures()
            n_before = len(df_daily.columns)
            df_daily = hurst.create_hurst_features(df_daily)
            n_hurst = len(df_daily.columns) - n_before
            metadata["hurst_features"] = True
            metadata["n_hurst_features"] = n_hurst
            print(f"  [HURST] Added {n_hurst} Hurst exponent features")

            hurst_signal = hurst.analyze_current_hurst(df_daily)
            if hurst_signal:
                print(f"    Hurst Regime: {hurst_signal.get('hurst_regime', 'N/A')}")
        except Exception as e:
            print(f"  [HURST] Warning: Hurst feature generation failed: {e}")
            metadata["hurst_features"] = False

    # 23. NMI Market Efficiency Features (nmi_ prefix)
    if use_nmi_features:
        try:
            nmi = NMIFeatures()
            n_before = len(df_daily.columns)
            df_daily = nmi.create_nmi_features(df_daily)
            n_nmi = len(df_daily.columns) - n_before
            metadata["nmi_features"] = True
            metadata["n_nmi_features"] = n_nmi
            print(f"  [NMI] Added {n_nmi} market efficiency features")

            nmi_signal = nmi.analyze_current_efficiency(df_daily)
            if nmi_signal:
                print(f"    Efficiency: {nmi_signal.get('efficiency_regime', 'N/A')}")
        except Exception as e:
            print(f"  [NMI] Warning: NMI feature generation failed: {e}")
            metadata["nmi_features"] = False

    # 24. Absorption Ratio Features (ar_ prefix)
    if use_absorption_ratio:
        try:
            ar = AbsorptionRatioFeatures()
            n_before = len(df_daily.columns)
            df_daily = ar.create_absorption_ratio_features(df_daily)
            n_ar = len(df_daily.columns) - n_before
            metadata["absorption_ratio_features"] = True
            metadata["n_absorption_ratio_features"] = n_ar
            print(f"  [ABSORPTION_RATIO] Added {n_ar} systemic risk features")

            ar_signal = ar.analyze_current_absorption(df_daily)
            if ar_signal:
                print(f"    Systemic Risk: {ar_signal.get('ar_regime', 'N/A')}")
        except Exception as e:
            print(f"  [ABSORPTION_RATIO] Warning: Absorption ratio features failed: {e}")
            metadata["absorption_ratio_features"] = False

    # 25. Drift Detection Features (drift_ prefix)
    if use_drift_features:
        try:
            drift = DriftFeatures()
            n_before = len(df_daily.columns)
            df_daily = drift.create_drift_features(df_daily)
            n_drift = len(df_daily.columns) - n_before
            metadata["drift_features"] = True
            metadata["n_drift_features"] = n_drift
            print(f"  [DRIFT] Added {n_drift} drift detection features")

            drift_signal = drift.analyze_current_drift(df_daily)
            if drift_signal:
                print(f"    Drift Regime: {drift_signal.get('drift_regime', 'N/A')}")
        except Exception as e:
            print(f"  [DRIFT] Warning: Drift feature generation failed: {e}")
            metadata["drift_features"] = False

    # 26. Changepoint Detection Features (cpd_ prefix)
    if use_changepoint_features:
        try:
            cpd = ChangepointFeatures()
            n_before = len(df_daily.columns)
            df_daily = cpd.create_changepoint_features(df_daily)
            n_cpd = len(df_daily.columns) - n_before
            metadata["changepoint_features"] = True
            metadata["n_changepoint_features"] = n_cpd
            print(f"  [CHANGEPOINT] Added {n_cpd} changepoint detection features")

            cpd_signal = cpd.analyze_current_changepoint(df_daily)
            if cpd_signal:
                print(f"    Changepoint: run_length={cpd_signal.get('run_length', 'N/A')} "
                      f"regime={cpd_signal.get('regime', 'N/A')}")
        except Exception as e:
            print(f"  [CHANGEPOINT] Warning: Changepoint feature generation failed: {e}")
            metadata["changepoint_features"] = False

    # 27. HMM Regime Features (hmm_ prefix)
    if use_hmm_features:
        try:
            hmm = HMMFeatures()
            n_before = len(df_daily.columns)
            df_daily = hmm.create_hmm_features(df_daily)
            n_hmm = len(df_daily.columns) - n_before
            metadata["hmm_features"] = True
            metadata["n_hmm_features"] = n_hmm
            print(f"  [HMM] Added {n_hmm} regime features")

            hmm_signal = hmm.analyze_current_regime(df_daily)
            if hmm_signal:
                print(f"    HMM Regime: {hmm_signal.get('hmm_regime', 'N/A')}")
        except Exception as e:
            print(f"  [HMM] Warning: HMM feature generation failed: {e}")
            metadata["hmm_features"] = False

    # 28. VPIN Order Flow Toxicity Features (vpin_ prefix)
    if use_vpin_features:
        try:
            vpin = VPINFeatures()
            n_before = len(df_daily.columns)
            df_daily = vpin.create_vpin_features(df_daily)
            n_vpin = len(df_daily.columns) - n_before
            metadata["vpin_features"] = True
            metadata["n_vpin_features"] = n_vpin
            print(f"  [VPIN] Added {n_vpin} order flow toxicity features")

            vpin_signal = vpin.analyze_current_vpin(df_daily)
            if vpin_signal:
                print(f"    VPIN Regime: {vpin_signal.get('vpin_regime', 'N/A')}")
        except Exception as e:
            print(f"  [VPIN] Warning: VPIN feature generation failed: {e}")
            metadata["vpin_features"] = False

    # 29. Intraday Momentum Features (imom_ prefix)
    if use_intraday_momentum:
        try:
            imom = IntradayMomentumFeatures()
            n_before = len(df_daily.columns)
            df_daily = imom.create_intraday_momentum_features(df_daily)
            n_imom = len(df_daily.columns) - n_before
            metadata["intraday_momentum_features"] = True
            metadata["n_intraday_momentum_features"] = n_imom
            print(f"  [INTRADAY_MOM] Added {n_imom} intraday momentum features")

            imom_signal = imom.analyze_current_momentum(df_daily)
            if imom_signal:
                print(f"    Momentum: {imom_signal.get('momentum_regime', 'N/A')}")
        except Exception as e:
            print(f"  [INTRADAY_MOM] Warning: Intraday momentum features failed: {e}")
            metadata["intraday_momentum_features"] = False

    # 30. Futures-Spot Basis Features (basis_ prefix)
    if use_futures_basis:
        try:
            basis = FuturesBasisFeatures()
            basis.download_futures_data(start_date, end_date)
            n_before = len(df_daily.columns)
            df_daily = basis.create_futures_basis_features(df_daily)
            n_basis = len(df_daily.columns) - n_before
            metadata["futures_basis_features"] = True
            metadata["n_futures_basis_features"] = n_basis
            print(f"  [FUTURES_BASIS] Added {n_basis} futures basis features")

            basis_signal = basis.analyze_current_basis(df_daily)
            if basis_signal:
                print(f"    Basis Regime: {basis_signal.get('basis_regime', 'N/A')}")
        except Exception as e:
            print(f"  [FUTURES_BASIS] Warning: Futures basis features failed: {e}")
            metadata["futures_basis_features"] = False

    # 31. Congressional Trading Features (congress_ prefix)
    if use_congressional_features:
        try:
            congress = CongressionalFeatures()
            n_before = len(df_daily.columns)
            df_daily = congress.create_congressional_features(df_daily)
            n_congress = len(df_daily.columns) - n_before
            metadata["congressional_features"] = True
            metadata["n_congressional_features"] = n_congress
            print(f"  [CONGRESSIONAL] Added {n_congress} smart-money proxy features")

            congress_signal = congress.analyze_current_congressional(df_daily)
            if congress_signal:
                print(f"    Congressional: {congress_signal.get('congressional_regime', 'N/A')}")
        except Exception as e:
            print(f"  [CONGRESSIONAL] Warning: Congressional features failed: {e}")
            metadata["congressional_features"] = False

    _maybe_gc(resource_config, "steps 19-31")

    # 32. Insider Aggregate Features (insider_agg_ prefix)
    if use_insider_aggregate:
        try:
            insider = InsiderAggregateFeatures()
            n_before = len(df_daily.columns)
            df_daily = insider.create_insider_aggregate_features(df_daily)
            n_insider = len(df_daily.columns) - n_before
            metadata["insider_aggregate_features"] = True
            metadata["n_insider_aggregate_features"] = n_insider
            print(f"  [INSIDER_AGG] Added {n_insider} insider aggregate features")

            insider_signal = insider.analyze_current_insider(df_daily)
            if insider_signal:
                print(f"    Insider: {insider_signal.get('insider_regime', 'N/A')}")
        except Exception as e:
            print(f"  [INSIDER_AGG] Warning: Insider aggregate features failed: {e}")
            metadata["insider_aggregate_features"] = False

    # 33. ETF Fund Flow Features (etf_flow_ prefix)
    if use_etf_flow:
        try:
            etf = ETFFlowFeatures()
            n_before = len(df_daily.columns)
            df_daily = etf.create_etf_flow_features(df_daily)
            n_etf = len(df_daily.columns) - n_before
            metadata["etf_flow_features"] = True
            metadata["n_etf_flow_features"] = n_etf
            print(f"  [ETF_FLOW] Added {n_etf} fund flow proxy features")

            etf_signal = etf.analyze_current_flows(df_daily)
            if etf_signal:
                print(f"    ETF Flow: {etf_signal.get('flow_regime', 'N/A')}")
        except Exception as e:
            print(f"  [ETF_FLOW] Warning: ETF flow features failed: {e}")
            metadata["etf_flow_features"] = False

    # 34. Wavelet Decomposition Features (wav_ prefix)
    if use_wavelet_features:
        try:
            wav = WaveletFeatures()
            n_before = len(df_daily.columns)
            df_daily = wav.create_wavelet_features(df_daily)
            n_wav = len(df_daily.columns) - n_before
            metadata["wavelet_features"] = True
            metadata["n_wavelet_features"] = n_wav
            print(f"  [WAVELET] Added {n_wav} multi-resolution features")

            wav_signal = wav.analyze_current_wavelet(df_daily)
            if wav_signal:
                print(f"    Wavelet Regime: {wav_signal.get('wavelet_regime', 'N/A')}")
        except Exception as e:
            print(f"  [WAVELET] Warning: Wavelet features failed: {e}")
            metadata["wavelet_features"] = False

    # 35. SAX Pattern Features (sax_ prefix)
    if use_sax_features:
        try:
            sax = SAXFeatures()
            n_before = len(df_daily.columns)
            df_daily = sax.create_sax_features(df_daily)
            n_sax = len(df_daily.columns) - n_before
            metadata["sax_features"] = True
            metadata["n_sax_features"] = n_sax
            print(f"  [SAX] Added {n_sax} pattern features")

            sax_signal = sax.analyze_current_pattern(df_daily)
            if sax_signal:
                print(f"    SAX Pattern: {sax_signal.get('pattern_regime', 'N/A')} "
                      f"novelty={sax_signal.get('novelty_level', 'N/A')}")
        except Exception as e:
            print(f"  [SAX] Warning: SAX features failed: {e}")
            metadata["sax_features"] = False

    # 36. Transfer Entropy Features (te_ prefix)
    if use_transfer_entropy:
        try:
            te = TransferEntropyFeatures()
            n_before = len(df_daily.columns)
            df_daily = te.create_transfer_entropy_features(df_daily)
            n_te = len(df_daily.columns) - n_before
            metadata["transfer_entropy_features"] = True
            metadata["n_transfer_entropy_features"] = n_te
            print(f"  [TRANSFER_ENTROPY] Added {n_te} information flow features")

            te_signal = te.analyze_current_te(df_daily)
            if te_signal:
                print(f"    Dominant Source: {te_signal.get('dominant_source', 'N/A')} "
                      f"flow={te_signal.get('information_flow', 'N/A')}")
        except Exception as e:
            print(f"  [TRANSFER_ENTROPY] Warning: Transfer entropy features failed: {e}")
            metadata["transfer_entropy_features"] = False

    # 37. MFDFA Features (mfdfa_ prefix)
    if use_mfdfa_features:
        try:
            mfdfa = MFDFAFeatures()
            n_before = len(df_daily.columns)
            df_daily = mfdfa.create_mfdfa_features(df_daily)
            n_mfdfa = len(df_daily.columns) - n_before
            metadata["mfdfa_features"] = True
            metadata["n_mfdfa_features"] = n_mfdfa
            print(f"  [MFDFA] Added {n_mfdfa} fractal features")

            mfdfa_signal = mfdfa.analyze_current_mfdfa(df_daily)
            if mfdfa_signal:
                print(f"    Fractal Regime: {mfdfa_signal.get('fractal_regime', 'N/A')}")
        except Exception as e:
            print(f"  [MFDFA] Warning: MFDFA features failed: {e}")
            metadata["mfdfa_features"] = False

    # 38. RQA Features (rqa_ prefix)
    if use_rqa_features:
        try:
            rqa = RQAFeatures()
            n_before = len(df_daily.columns)
            df_daily = rqa.create_rqa_features(df_daily)
            n_rqa = len(df_daily.columns) - n_before
            metadata["rqa_features"] = True
            metadata["n_rqa_features"] = n_rqa
            print(f"  [RQA] Added {n_rqa} recurrence analysis features")

            rqa_signal = rqa.analyze_current_rqa(df_daily)
            if rqa_signal:
                print(f"    RQA Regime: {rqa_signal.get('rqa_regime', 'N/A')}")
        except Exception as e:
            print(f"  [RQA] Warning: RQA features failed: {e}")
            metadata["rqa_features"] = False

    # 39. Copula Tail Dependence Features (copula_ prefix)
    if use_copula_features:
        try:
            copula = CopulaFeatures()
            n_before = len(df_daily.columns)
            df_daily = copula.create_copula_features(df_daily)
            n_copula = len(df_daily.columns) - n_before
            metadata["copula_features"] = True
            metadata["n_copula_features"] = n_copula
            print(f"  [COPULA] Added {n_copula} tail dependence features")

            copula_signal = copula.analyze_current_copula(df_daily)
            if copula_signal:
                print(f"    Tail Regime: {copula_signal.get('tail_regime', 'N/A')} "
                      f"lower={copula_signal.get('copula_lower_tail', 0.0):.3f} "
                      f"upper={copula_signal.get('copula_upper_tail', 0.0):.3f}")
        except Exception as e:
            print(f"  [COPULA] Warning: Copula tail dependence features failed: {e}")
            metadata["copula_features"] = False

    # 40. Correlation Network Centrality Features (netw_ prefix)
    if use_network_centrality:
        try:
            netw = NetworkFeatures()
            n_before = len(df_daily.columns)
            df_daily = netw.create_network_features(df_daily)
            n_netw = len(df_daily.columns) - n_before
            metadata["network_features"] = True
            metadata["n_network_features"] = n_netw
            print(f"  [NETWORK] Added {n_netw} correlation network features")

            netw_signal = netw.analyze_current_network(df_daily)
            if netw_signal:
                print(f"    Network: {netw_signal.get('network_regime', 'N/A')}")
        except Exception as e:
            print(f"  [NETWORK] Warning: Network features failed: {e}")
            metadata["network_features"] = False

    _maybe_gc(resource_config, "steps 32-40")

    # 41. Path Signature Features (psig_ prefix)
    if use_path_signatures:
        try:
            psig = PathSignatureFeatures()
            n_before = len(df_daily.columns)
            df_daily = psig.create_path_signature_features(df_daily)
            n_psig = len(df_daily.columns) - n_before
            metadata["path_signature_features"] = True
            metadata["n_path_signature_features"] = n_psig
            print(f"  [PSIG] Added {n_psig} path signature features")
        except Exception as e:
            print(f"  [PSIG] Warning: Path signature features failed: {e}")
            metadata["path_signature_features"] = False

    # 42. Wavelet Scattering Features (wscat_ prefix)
    if use_wavelet_scattering:
        try:
            wscat = WaveletScatteringFeatures()
            n_before = len(df_daily.columns)
            df_daily = wscat.create_wavelet_scattering_features(df_daily)
            n_wscat = len(df_daily.columns) - n_before
            metadata["wavelet_scattering_features"] = True
            metadata["n_wavelet_scattering_features"] = n_wscat
            print(f"  [WSCAT] Added {n_wscat} wavelet scattering features")

            wscat_signal = wscat.analyze_current_scattering(df_daily)
            if wscat_signal:
                print(f"    Scattering: regime={wscat_signal.get('regime', 'N/A')}")
        except Exception as e:
            print(f"  [WSCAT] Warning: Wavelet scattering features failed: {e}")
            metadata["wavelet_scattering_features"] = False

    # 43. Wasserstein Regime Detection Features (wreg_ prefix)
    if use_wasserstein_regime:
        try:
            wreg = WassersteinRegimeDetector()
            n_before = len(df_daily.columns)
            df_daily = wreg.create_wasserstein_features(df_daily)
            n_wreg = len(df_daily.columns) - n_before
            metadata["wasserstein_regime_features"] = True
            metadata["n_wasserstein_regime_features"] = n_wreg
            print(f"  [WREG] Added {n_wreg} Wasserstein regime features")
        except Exception as e:
            print(f"  [WREG] Warning: Wasserstein regime features failed: {e}")
            metadata["wasserstein_regime_features"] = False

    # 44. Market Structure Features (mstr_ prefix)
    if use_market_structure:
        try:
            mstr = MarketStructureFeatures()
            n_before = len(df_daily.columns)
            df_daily = mstr.create_market_structure_features(df_daily)
            n_mstr = len(df_daily.columns) - n_before
            metadata["market_structure_features"] = True
            metadata["n_market_structure_features"] = n_mstr
            print(f"  [MSTR] Added {n_mstr} market structure features")

            mstr_signal = mstr.analyze_current_structure(df_daily)
            if mstr_signal:
                print(f"    Squeeze: {mstr_signal.get('squeeze_on', False)}, "
                      f"compression_energy: {mstr_signal.get('compression_energy', 0.0):.3f}")
        except Exception as e:
            print(f"  [MSTR] Warning: Market structure features failed: {e}")
            metadata["market_structure_features"] = False

    # 45. Time Series Model Features (tsm_ prefix)
    if use_time_series_models:
        try:
            tsm = TimeSeriesModelFeatures(
                use_chronos=True,
                use_catch22=use_catch22,
            )
            n_before = len(df_daily.columns)
            df_daily = tsm.create_time_series_model_features(df_daily)
            n_tsm = len(df_daily.columns) - n_before
            metadata["time_series_model_features"] = True
            metadata["n_time_series_model_features"] = n_tsm
            print(f"  [TSM] Added {n_tsm} time series model features")

            tsm_signal = tsm.analyze_current_ts(df_daily)
            if tsm_signal:
                print(f"    ARIMA residual: {tsm_signal.get('arima_residual', 0.0):.5f}, "
                      f"Chronos: {tsm_signal.get('chronos_available', False)}")
        except Exception as e:
            print(f"  [TSM] Warning: Time series model features failed: {e}")
            metadata["time_series_model_features"] = False

    # 46. HAR-RV Features (harv_ prefix)
    if use_har_rv:
        try:
            harv = HARRVFeatures()
            n_before = len(df_daily.columns)
            df_daily = harv.create_har_rv_features(df_daily)
            n_harv = len(df_daily.columns) - n_before
            metadata["har_rv_features"] = True
            metadata["n_har_rv_features"] = n_harv
            print(f"  [HARV] Added {n_harv} HAR-RV features")

            harv_signal = harv.analyze_current_harv(df_daily)
            if harv_signal:
                print(f"    Vol regime: {harv_signal.get('vol_regime', 'N/A')}, "
                      f"residual_z: {harv_signal.get('residual_z', 0.0):.3f}")
        except Exception as e:
            print(f"  [HARV] Warning: HAR-RV features failed: {e}")
            metadata["har_rv_features"] = False

    # 47. L-Moments Features (lmom_ prefix)
    if use_l_moments:
        try:
            lmom = LMomentsFeatures()
            n_before = len(df_daily.columns)
            df_daily = lmom.create_l_moments_features(df_daily)
            n_lmom = len(df_daily.columns) - n_before
            metadata["l_moments_features"] = True
            metadata["n_l_moments_features"] = n_lmom
            print(f"  [LMOM] Added {n_lmom} L-Moments features")

            lmom_signal = lmom.analyze_current_lmoments(df_daily)
            if lmom_signal:
                print(f"    Distribution: {lmom_signal.get('distribution_regime', 'N/A')}, "
                      f"L-skew: {lmom_signal.get('l_skewness', 0.0):.4f}")
        except Exception as e:
            print(f"  [LMOM] Warning: L-Moments features failed: {e}")
            metadata["l_moments_features"] = False

    # 48. Multiscale Sample Entropy Features (mse_ prefix)
    if use_multiscale_entropy:
        try:
            mse = MultiscaleEntropyFeatures()
            n_before = len(df_daily.columns)
            df_daily = mse.create_multiscale_entropy_features(df_daily)
            n_mse = len(df_daily.columns) - n_before
            metadata["multiscale_entropy_features"] = True
            metadata["n_multiscale_entropy_features"] = n_mse
            print(f"  [MSE] Added {n_mse} multiscale entropy features")

            mse_signal = mse.analyze_current_entropy(df_daily)
            if mse_signal:
                print(f"    Entropy regime: {mse_signal.get('entropy_regime', 'N/A')}, "
                      f"complexity: {mse_signal.get('complexity', 0.0):.4f}")
        except Exception as e:
            print(f"  [MSE] Warning: Multiscale entropy features failed: {e}")
            metadata["multiscale_entropy_features"] = False

    # 49. RV Signature Plot Features (rvsp_ prefix)
    if use_rv_signature_plot:
        try:
            rvsp = RVSignaturePlotFeatures()
            n_before = len(df_daily.columns)
            df_daily = rvsp.create_rv_signature_features(df_daily)
            n_rvsp = len(df_daily.columns) - n_before
            metadata["rv_signature_features"] = True
            metadata["n_rv_signature_features"] = n_rvsp
            print(f"  [RVSP] Added {n_rvsp} RV signature plot features")
        except Exception as e:
            print(f"  [RVSP] Warning: RV signature plot features failed: {e}")
            metadata["rv_signature_features"] = False

    # 50. TDA Persistent Homology Features (tda_ prefix)
    if use_tda_homology:
        try:
            tda = TDAHomologyFeatures()
            n_before = len(df_daily.columns)
            df_daily = tda.create_tda_features(df_daily)
            n_tda = len(df_daily.columns) - n_before
            metadata["tda_homology_features"] = True
            metadata["n_tda_homology_features"] = n_tda
            print(f"  [TDA] Added {n_tda} persistent homology features")

            tda_signal = tda.analyze_current_topology(df_daily)
            if tda_signal:
                print(f"    Topology: {tda_signal.get('topology', 'N/A')}")
        except Exception as e:
            print(f"  [TDA] Warning: TDA features failed: {e}")
            metadata["tda_homology_features"] = False

    _maybe_gc(resource_config, "steps 41-50")

    # 51. Credit Spread Features (cred_ prefix)
    if use_credit_spread_features:
        try:
            cs = CreditSpreadFeatures()
            n_before = len(df_daily.columns)
            df_daily = cs.create_credit_spread_features(df_daily)
            n_cs = len(df_daily.columns) - n_before
            metadata["credit_spread_features"] = True
            metadata["n_credit_spread_features"] = n_cs
            print(f"  [CREDIT] Added {n_cs} credit spread features")
        except Exception as e:
            print(f"  [CREDIT] Warning: Credit spread features failed: {e}")
            metadata["credit_spread_features"] = False

    # 52. Yield Curve Features (yc_ prefix)
    if use_yield_curve_features:
        try:
            yc = YieldCurveFeatures()
            n_before = len(df_daily.columns)
            df_daily = yc.create_yield_curve_features(df_daily)
            n_yc = len(df_daily.columns) - n_before
            metadata["yield_curve_features"] = True
            metadata["n_yield_curve_features"] = n_yc
            print(f"  [YIELD] Added {n_yc} yield curve features")
        except Exception as e:
            print(f"  [YIELD] Warning: Yield curve features failed: {e}")
            metadata["yield_curve_features"] = False

    # 53. Volatility Term Structure Features (vts_ prefix)
    if use_vol_term_structure_features:
        try:
            vts = VolTermStructureFeatures()
            n_before = len(df_daily.columns)
            df_daily = vts.create_vol_term_structure_features(df_daily)
            n_vts = len(df_daily.columns) - n_before
            metadata["vol_term_structure_features"] = True
            metadata["n_vol_term_structure_features"] = n_vts
            print(f"  [VTS] Added {n_vts} vol term structure features")
        except Exception as e:
            print(f"  [VTS] Warning: Vol term structure features failed: {e}")
            metadata["vol_term_structure_features"] = False

    # 54. Macro Surprise Features (msurp_ prefix)
    if use_macro_surprise_features:
        try:
            ms = MacroSurpriseFeatures()
            n_before = len(df_daily.columns)
            df_daily = ms.create_macro_surprise_features(df_daily)
            n_ms = len(df_daily.columns) - n_before
            metadata["macro_surprise_features"] = True
            metadata["n_macro_surprise_features"] = n_ms
            print(f"  [MSURP] Added {n_ms} macro surprise features")
        except Exception as e:
            print(f"  [MSURP] Warning: Macro surprise features failed: {e}")
            metadata["macro_surprise_features"] = False

    # 55. Cross-Asset Momentum Features (xmom_ prefix)
    if use_cross_asset_momentum:
        try:
            xmom = CrossAssetMomentumFeatures()
            n_before = len(df_daily.columns)
            df_daily = xmom.create_cross_asset_momentum_features(df_daily)
            n_xmom = len(df_daily.columns) - n_before
            metadata["cross_asset_momentum_features"] = True
            metadata["n_cross_asset_momentum_features"] = n_xmom
            print(f"  [XMOM] Added {n_xmom} cross-asset momentum features")
        except Exception as e:
            print(f"  [XMOM] Warning: Cross-asset momentum features failed: {e}")
            metadata["cross_asset_momentum_features"] = False

    # 56. Skew/Kurtosis Features (skku_ prefix)
    if use_skew_kurtosis_features:
        try:
            skku = SkewKurtosisFeatures()
            n_before = len(df_daily.columns)
            df_daily = skku.create_skew_kurtosis_features(df_daily)
            n_skku = len(df_daily.columns) - n_before
            metadata["skew_kurtosis_features"] = True
            metadata["n_skew_kurtosis_features"] = n_skku
            print(f"  [SKKU] Added {n_skku} skew/kurtosis features")
        except Exception as e:
            print(f"  [SKKU] Warning: Skew/kurtosis features failed: {e}")
            metadata["skew_kurtosis_features"] = False

    # 57. Seasonality Features (seas_ prefix)
    if use_seasonality_features:
        try:
            seas = SeasonalityFeatures()
            n_before = len(df_daily.columns)
            df_daily = seas.create_seasonality_features(df_daily)
            n_seas = len(df_daily.columns) - n_before
            metadata["seasonality_features"] = True
            metadata["n_seasonality_features"] = n_seas
            print(f"  [SEAS] Added {n_seas} seasonality features")
        except Exception as e:
            print(f"  [SEAS] Warning: Seasonality features failed: {e}")
            metadata["seasonality_features"] = False

    # 58. Order Flow Imbalance Features (ofi_ prefix)
    if use_order_flow_imbalance:
        try:
            ofi = OrderFlowImbalanceFeatures()
            n_before = len(df_daily.columns)
            df_daily = ofi.create_order_flow_imbalance_features(df_daily)
            n_ofi = len(df_daily.columns) - n_before
            metadata["order_flow_imbalance_features"] = True
            metadata["n_order_flow_imbalance_features"] = n_ofi
            print(f"  [OFI] Added {n_ofi} order flow imbalance features")
        except Exception as e:
            print(f"  [OFI] Warning: Order flow imbalance features failed: {e}")
            metadata["order_flow_imbalance_features"] = False

    _maybe_gc(resource_config, "steps 51-58")

    # 59. Correlation Regime Features (corr_ prefix)
    if use_correlation_regime:
        try:
            corr_regime = CorrelationRegimeFeatures()
            corr_data = corr_regime.download_correlation_data(str(start_date)[:10], str(end_date)[:10])
            n_before = len(df_daily.columns)
            df_daily = corr_regime.create_correlation_features(df_daily)
            n_corr = len(df_daily.columns) - n_before
            metadata["correlation_regime_features"] = True
            metadata["n_correlation_regime_features"] = n_corr
            print(f"  [CORR_REGIME] Added {n_corr} correlation regime features")

            corr_signal = corr_regime.analyze_current_regime(df_daily)
            if corr_signal:
                print(f"    Correlation Regime: {corr_signal.get('regime', 'N/A')} "
                      f"(SPY-TLT={corr_signal.get('spy_tlt_corr', 0):.3f})")
        except Exception as e:
            print(f"  [CORR_REGIME] Warning: Correlation regime features failed: {e}")
            metadata["correlation_regime_features"] = False

    # 60. Fama-French Factor Exposure Features (ff_ prefix)
    if use_fama_french:
        try:
            ff = FamaFrenchFeatures()
            ff.download_factor_data(str(start_date)[:10], str(end_date)[:10])
            n_before = len(df_daily.columns)
            df_daily = ff.create_fama_french_features(df_daily)
            n_ff = len(df_daily.columns) - n_before
            metadata["fama_french_features"] = True
            metadata["n_fama_french_features"] = n_ff
            print(f"  [FAMA_FRENCH] Added {n_ff} factor exposure features")

            ff_signal = ff.analyze_current_factors(df_daily)
            if ff_signal:
                print(f"    Factor Regime: {ff_signal.get('factor_regime', 'N/A')}")
        except Exception as e:
            print(f"  [FAMA_FRENCH] Warning: Fama-French features failed: {e}")
            metadata["fama_french_features"] = False

    # 61. Put-Call Ratio Features (pcr_ prefix)
    if use_put_call_ratio:
        try:
            pcr = PutCallRatioFeatures()
            pcr.download_pcr_data(str(start_date)[:10], str(end_date)[:10])
            n_before = len(df_daily.columns)
            df_daily = pcr.create_pcr_features(df_daily)
            n_pcr = len(df_daily.columns) - n_before
            metadata["put_call_ratio_features"] = True
            metadata["n_put_call_ratio_features"] = n_pcr
            print(f"  [PCR] Added {n_pcr} put-call ratio features")

            pcr_signal = pcr.analyze_current_pcr(df_daily)
            if pcr_signal:
                print(f"    PCR Regime: {pcr_signal.get('regime', 'N/A')} "
                      f"(ratio={pcr_signal.get('pcr_ratio', 0):.3f}, "
                      f"source={pcr_signal.get('data_source', 'N/A')})")
        except Exception as e:
            print(f"  [PCR] Warning: Put-call ratio features failed: {e}")
            metadata["put_call_ratio_features"] = False

    # 62. Multi-Horizon Ensemble Features (mh_ prefix)
    if use_multi_horizon:
        try:
            mh = MultiHorizonFilter()
            n_before = len(df_daily.columns)
            df_daily = mh.compute_horizon_signals(df_daily)
            n_mh = len(df_daily.columns) - n_before
            metadata["multi_horizon_features"] = True
            metadata["n_multi_horizon_features"] = n_mh
            print(f"  [MULTI_HORIZON] Added {n_mh} multi-horizon features")
        except Exception as e:
            print(f"  [MULTI_HORIZON] Warning: Multi-horizon features failed: {e}")
            metadata["multi_horizon_features"] = False

    _maybe_gc(resource_config, "steps 59-62")

    # ─────────────────────────────────────────────────────────
    # 63. Earnings Revision Features (ern_ prefix)
    # ─────────────────────────────────────────────────────────
    if use_earnings_revision:
        try:
            ern = EarningsRevisionFeatures()
            ern.download_earnings_data(start_date, end_date)
            n_before = len(df_daily.columns)
            df_daily = ern.create_earnings_revision_features(df_daily)
            n_ern = len(df_daily.columns) - n_before
            metadata["earnings_revision_features"] = True
            metadata["n_earnings_revision_features"] = n_ern
            print(f"  [ERN] Added {n_ern} earnings revision features")

            ern_signal = ern.analyze_current_earnings_revision(df_daily)
            if ern_signal:
                print(f"    Revision Regime: {ern_signal.get('revision_regime', 'N/A')}")
        except Exception as e:
            print(f"  [ERN] Warning: Earnings revision features failed: {e}")
            metadata["earnings_revision_features"] = False

    # ─────────────────────────────────────────────────────────
    # 64. Short Interest Features (si_ prefix)
    # ─────────────────────────────────────────────────────────
    if use_short_interest:
        try:
            si = ShortInterestFeatures()
            si.download_short_interest_data(start_date, end_date)
            n_before = len(df_daily.columns)
            df_daily = si.create_short_interest_features(df_daily)
            n_si = len(df_daily.columns) - n_before
            metadata["short_interest_features"] = True
            metadata["n_short_interest_features"] = n_si
            print(f"  [SI] Added {n_si} short interest features")
        except Exception as e:
            print(f"  [SI] Warning: Short interest features failed: {e}")
            metadata["short_interest_features"] = False

    # ─────────────────────────────────────────────────────────
    # 65. Dollar Index Features (dxy_ prefix)
    # ─────────────────────────────────────────────────────────
    if use_dollar_index:
        try:
            dxy = DollarIndexFeatures()
            dxy.download_dollar_data(str(start_date)[:10], str(end_date)[:10])
            n_before = len(df_daily.columns)
            df_daily = dxy.create_dollar_index_features(df_daily)
            n_dxy = len(df_daily.columns) - n_before
            metadata["dollar_index_features"] = True
            metadata["n_dollar_index_features"] = n_dxy
            print(f"  [DXY] Added {n_dxy} dollar index features")

            dxy_signal = dxy.analyze_current_dollar(df_daily)
            if dxy_signal:
                print(f"    Dollar Regime: {dxy_signal.get('regime', 'N/A')} "
                      f"(z={dxy_signal.get('z_score', 0):.3f})")
        except Exception as e:
            print(f"  [DXY] Warning: Dollar index features failed: {e}")
            metadata["dollar_index_features"] = False

    # ─────────────────────────────────────────────────────────
    # 66. Institutional Flow Features (inst_ prefix)
    # ─────────────────────────────────────────────────────────
    if use_institutional_flow:
        try:
            inst = InstitutionalFlowFeatures()
            inst.download_institutional_data(start_date, end_date)
            n_before = len(df_daily.columns)
            df_daily = inst.create_institutional_flow_features(df_daily)
            n_inst = len(df_daily.columns) - n_before
            metadata["institutional_flow_features"] = True
            metadata["n_institutional_flow_features"] = n_inst
            print(f"  [INST] Added {n_inst} institutional flow features")
        except Exception as e:
            print(f"  [INST] Warning: Institutional flow features failed: {e}")
            metadata["institutional_flow_features"] = False

    # ─────────────────────────────────────────────────────────
    # 67. Google Trends Features (gtrend_ prefix)
    # ─────────────────────────────────────────────────────────
    if use_google_trends:
        try:
            gtrend = GoogleTrendsFeatures()
            gtrend.download_trends_data(start_date, end_date)
            n_before = len(df_daily.columns)
            df_daily = gtrend.create_google_trends_features(df_daily)
            n_gtrend = len(df_daily.columns) - n_before
            metadata["google_trends_features"] = True
            metadata["n_google_trends_features"] = n_gtrend
            print(f"  [GTREND] Added {n_gtrend} Google Trends features")
        except Exception as e:
            print(f"  [GTREND] Warning: Google Trends features failed: {e}")
            metadata["google_trends_features"] = False

    _maybe_gc(resource_config, "steps 63-67")

    # ─────────────────────────────────────────────────────────
    # 68. Commodity Signal Features (cmdty_ prefix)
    # ─────────────────────────────────────────────────────────
    if use_commodity_signals:
        try:
            cmdty = CommoditySignalFeatures()
            cmdty.download_commodity_data(str(start_date)[:10], str(end_date)[:10])
            n_before = len(df_daily.columns)
            df_daily = cmdty.create_commodity_signal_features(df_daily)
            n_cmdty = len(df_daily.columns) - n_before
            metadata["commodity_signal_features"] = True
            metadata["n_commodity_signal_features"] = n_cmdty
            print(f"  [CMDTY] Added {n_cmdty} commodity signal features")

            cmdty_signal = cmdty.analyze_current_commodity(df_daily)
            if cmdty_signal:
                print(f"    Commodity Regime: {cmdty_signal.get('regime', 'N/A')}")
        except Exception as e:
            print(f"  [CMDTY] Warning: Commodity signal features failed: {e}")
            metadata["commodity_signal_features"] = False

    # ─────────────────────────────────────────────────────────
    # 69. Treasury Auction Features (tauct_ prefix)
    # ─────────────────────────────────────────────────────────
    if use_treasury_auction:
        try:
            tauct = TreasuryAuctionFeatures()
            tauct.download_auction_data(start_date, end_date)
            n_before = len(df_daily.columns)
            df_daily = tauct.create_treasury_auction_features(df_daily)
            n_tauct = len(df_daily.columns) - n_before
            metadata["treasury_auction_features"] = True
            metadata["n_treasury_auction_features"] = n_tauct
            print(f"  [TAUCT] Added {n_tauct} treasury auction features")
        except Exception as e:
            print(f"  [TAUCT] Warning: Treasury auction features failed: {e}")
            metadata["treasury_auction_features"] = False

    # ─────────────────────────────────────────────────────────
    # 70. Fed Liquidity Features (fedliq_ prefix)
    # ─────────────────────────────────────────────────────────
    if use_fed_liquidity:
        try:
            fedliq = FedLiquidityFeatures()
            fedliq.download_liquidity_data(start_date, end_date)
            n_before = len(df_daily.columns)
            df_daily = fedliq.create_fed_liquidity_features(df_daily)
            n_fedliq = len(df_daily.columns) - n_before
            metadata["fed_liquidity_features"] = True
            metadata["n_fed_liquidity_features"] = n_fedliq
            print(f"  [FEDLIQ] Added {n_fedliq} Fed liquidity features")

            fedliq_signal = fedliq.analyze_current_liquidity(df_daily)
            if fedliq_signal:
                print(f"    Liquidity Regime: {fedliq_signal.get('regime', 'N/A')} "
                      f"(z={fedliq_signal.get('liquidity_z', 0):.3f})")
        except Exception as e:
            print(f"  [FEDLIQ] Warning: Fed liquidity features failed: {e}")
            metadata["fed_liquidity_features"] = False

    # ─────────────────────────────────────────────────────────
    # 71. Earnings Calendar Features (ecal_ prefix)
    # ─────────────────────────────────────────────────────────
    if use_earnings_calendar:
        try:
            ecal = EarningsCalendarFeatures()
            ecal.download_calendar_data(start_date, end_date)
            n_before = len(df_daily.columns)
            df_daily = ecal.create_earnings_calendar_features(df_daily)
            n_ecal = len(df_daily.columns) - n_before
            metadata["earnings_calendar_features"] = True
            metadata["n_earnings_calendar_features"] = n_ecal
            print(f"  [ECAL] Added {n_ecal} earnings calendar features")
        except Exception as e:
            print(f"  [ECAL] Warning: Earnings calendar features failed: {e}")
            metadata["earnings_calendar_features"] = False

    # ─────────────────────────────────────────────────────────
    # 72. Analyst Rating Features (anlst_ prefix)
    # ─────────────────────────────────────────────────────────
    if use_analyst_rating:
        try:
            anlst = AnalystRatingFeatures()
            anlst.download_rating_data(start_date, end_date)
            n_before = len(df_daily.columns)
            df_daily = anlst.create_analyst_rating_features(df_daily)
            n_anlst = len(df_daily.columns) - n_before
            metadata["analyst_rating_features"] = True
            metadata["n_analyst_rating_features"] = n_anlst
            print(f"  [ANLST] Added {n_anlst} analyst rating features")

            anlst_signal = anlst.analyze_current_ratings(df_daily)
            if anlst_signal:
                print(f"    Consensus: {anlst_signal.get('consensus', 'N/A')}")
        except Exception as e:
            print(f"  [ANLST] Warning: Analyst rating features failed: {e}")
            metadata["analyst_rating_features"] = False

    _maybe_gc(resource_config, "steps 68-72")

    # ─────────────────────────────────────────────────────────
    # 73. Expanded Macro Features (xmacro_ prefix)
    # ─────────────────────────────────────────────────────────
    if use_expanded_macro:
        try:
            xmacro = ExpandedMacroFeatures()
            xmacro.download_macro_data(start_date, end_date)
            n_before = len(df_daily.columns)
            df_daily = xmacro.create_expanded_macro_features(df_daily)
            n_xmacro = len(df_daily.columns) - n_before
            metadata["expanded_macro_features"] = True
            metadata["n_expanded_macro_features"] = n_xmacro
            print(f"  [XMACRO] Added {n_xmacro} expanded macro features")
        except Exception as e:
            print(f"  [XMACRO] Warning: Expanded macro features failed: {e}")
            metadata["expanded_macro_features"] = False

    # ─────────────────────────────────────────────────────────
    # 74. VVIX Features (vvix_ prefix)
    # ─────────────────────────────────────────────────────────
    if use_vvix:
        try:
            vvix = VVIXFeatures()
            vvix.download_vvix_data(start_date, end_date)
            n_before = len(df_daily.columns)
            df_daily = vvix.create_vvix_features(df_daily)
            n_vvix = len(df_daily.columns) - n_before
            metadata["vvix_features"] = True
            metadata["n_vvix_features"] = n_vvix
            print(f"  [VVIX] Added {n_vvix} VVIX features")
        except Exception as e:
            print(f"  [VVIX] Warning: VVIX features failed: {e}")
            metadata["vvix_features"] = False

    # ─────────────────────────────────────────────────────────
    # 75. Sector Rotation Features (secrot_ prefix)
    # ─────────────────────────────────────────────────────────
    if use_sector_rotation:
        try:
            secrot = SectorRotationFeatures()
            secrot.download_sector_data(start_date, end_date)
            n_before = len(df_daily.columns)
            df_daily = secrot.create_sector_rotation_features(df_daily)
            n_secrot = len(df_daily.columns) - n_before
            metadata["sector_rotation_features"] = True
            metadata["n_sector_rotation_features"] = n_secrot
            print(f"  [SECROT] Added {n_secrot} sector rotation features")
        except Exception as e:
            print(f"  [SECROT] Warning: Sector rotation features failed: {e}")
            metadata["sector_rotation_features"] = False

    # ─────────────────────────────────────────────────────────
    # 76. FX Carry Features (fxc_ prefix)
    # ─────────────────────────────────────────────────────────
    if use_fx_carry:
        try:
            fxc = FXCarryFeatures()
            fxc.download_fx_data(start_date, end_date)
            n_before = len(df_daily.columns)
            df_daily = fxc.create_fx_carry_features(df_daily)
            n_fxc = len(df_daily.columns) - n_before
            metadata["fx_carry_features"] = True
            metadata["n_fx_carry_features"] = n_fxc
            print(f"  [FXC] Added {n_fxc} FX carry features")
        except Exception as e:
            print(f"  [FXC] Warning: FX carry features failed: {e}")
            metadata["fx_carry_features"] = False

    _maybe_gc(resource_config, "steps 73-76")

    # ─────────────────────────────────────────────────────────
    # 77. Money Market Features (mmkt_ prefix)
    # ─────────────────────────────────────────────────────────
    if use_money_market:
        try:
            mmkt = MoneyMarketFeatures()
            mmkt.download_money_market_data(start_date, end_date)
            n_before = len(df_daily.columns)
            df_daily = mmkt.create_money_market_features(df_daily)
            n_mmkt = len(df_daily.columns) - n_before
            metadata["money_market_features"] = True
            metadata["n_money_market_features"] = n_mmkt
            print(f"  [MMKT] Added {n_mmkt} money market features")
        except Exception as e:
            print(f"  [MMKT] Warning: Money market features failed: {e}")
            metadata["money_market_features"] = False

    # ─────────────────────────────────────────────────────────
    # 78. Financial Stress Features (fstress_ prefix)
    # ─────────────────────────────────────────────────────────
    if use_financial_stress:
        try:
            fstress = FinancialStressFeatures()
            fstress.download_stress_data(start_date, end_date)
            n_before = len(df_daily.columns)
            df_daily = fstress.create_financial_stress_features(df_daily)
            n_fstress = len(df_daily.columns) - n_before
            metadata["financial_stress_features"] = True
            metadata["n_financial_stress_features"] = n_fstress
            print(f"  [FSTRESS] Added {n_fstress} financial stress features")
        except Exception as e:
            print(f"  [FSTRESS] Warning: Financial stress features failed: {e}")
            metadata["financial_stress_features"] = False

    # ─────────────────────────────────────────────────────────
    # 79. Global Equity Features (gleq_ prefix)
    # ─────────────────────────────────────────────────────────
    if use_global_equity:
        try:
            gleq = GlobalEquityFeatures()
            gleq.download_global_data(start_date, end_date)
            n_before = len(df_daily.columns)
            df_daily = gleq.create_global_equity_features(df_daily)
            n_gleq = len(df_daily.columns) - n_before
            metadata["global_equity_features"] = True
            metadata["n_global_equity_features"] = n_gleq
            print(f"  [GLEQ] Added {n_gleq} global equity features")
        except Exception as e:
            print(f"  [GLEQ] Warning: Global equity features failed: {e}")
            metadata["global_equity_features"] = False

    # ─────────────────────────────────────────────────────────
    # 80. Retail Sentiment Features (rflow_ prefix)
    # ─────────────────────────────────────────────────────────
    if use_retail_sentiment:
        try:
            rflow = RetailSentimentFeatures()
            rflow.download_retail_data(start_date, end_date)
            n_before = len(df_daily.columns)
            df_daily = rflow.create_retail_sentiment_features(df_daily)
            n_rflow = len(df_daily.columns) - n_before
            metadata["retail_sentiment_features"] = True
            metadata["n_retail_sentiment_features"] = n_rflow
            print(f"  [RFLOW] Added {n_rflow} retail sentiment features")
        except Exception as e:
            print(f"  [RFLOW] Warning: Retail sentiment features failed: {e}")
            metadata["retail_sentiment_features"] = False

    _maybe_gc(resource_config, "steps 77-80")

    # 9. Synthetic SPY Universes (do last since it multiplies data)
    _n_universes = 20
    if resource_config is not None:
        _n_universes = resource_config.n_synthetic_universes
    if use_synthetic:
        real_weight = 1 - synthetic_weight
        synth_gen = SyntheticSPYGenerator(
            n_universes=_n_universes,
            real_weight=real_weight,
            use_bear_universes=use_bear_universes,
            bear_mean_shift_bps=bear_mean_shift_bps,
            bear_vol_amplify_factor=bear_vol_amplify_factor,
            bear_vol_dampen_factor=bear_vol_dampen_factor,
            use_multiscale_bootstrap=use_multiscale_bootstrap,
        )

        # Reuse component prices if available, otherwise download
        if "component_prices" not in dir() or component_prices.empty:
            component_prices = synth_gen.download_spy_components(start_date, end_date)

        if not component_prices.empty:
            # Generate filter-based universes
            universes = synth_gen.generate_universes(component_prices, df_daily)

            # Generate "SPY minus component" universes (new feature)
            component_universes = synth_gen.generate_spy_minus_component_universes(
                component_prices, df_daily
            )
            universes.extend(component_universes)

            df_daily = synth_gen.create_augmented_dataset(df_daily, universes)
            n_bear = sum(
                1 for u in universes
                if len(u) > 0 and "universe_type" in u.columns
                and str(u["universe_type"].iloc[0]).startswith("bear_")
            )
            metadata["n_universes"] = len(universes)
            metadata["n_bear_universes"] = n_bear
            metadata["real_weight"] = real_weight
            metadata["synthetic_weight"] = synthetic_weight
            metadata["component_modified_universes"] = len(component_universes)

    print(f"\n[ANTI-OVERFIT] Final dataset: {len(df_daily)} samples")
    print(f"  Metadata: {metadata}")

    return df_daily, metadata

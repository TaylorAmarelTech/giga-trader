"""
GIGA TRADER - Anti-Overfit Integration
========================================
Main integration function that combines all anti-overfitting measures:
  0. OHLC Data Validation
  1-8. Core breadth/cross-asset/macro/calendar/sentiment/economic features
  10-86. Registry-driven feature modules (77 steps)
  9. Synthetic SPY Universes (last, multiplies data)

Architecture:
  Steps 0-8 are "special" (custom download/analyze patterns) and kept inline.
  Steps 10-86 use a declarative _FEATURE_STEPS registry + generic _run_feature_step().
"""

import importlib
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- Imports for inline steps 0-8 and synthetic universes ---
from src.phase_08_features_breadth.streak_features import ComponentStreakFeatures
from src.phase_08_features_breadth.cross_asset_features import CrossAssetFeatures
from src.phase_08_features_breadth.mag7_breadth import Mag7BreadthFeatures
from src.phase_08_features_breadth.sector_breadth import SectorBreadthFeatures
from src.phase_08_features_breadth.volatility_regime import VolatilityRegimeFeatures
from src.phase_03_synthetic_data.synthetic_universe import SyntheticSPYGenerator
from src.phase_08_features_breadth.economic_features import EconomicFeatures
from src.phase_08_features_breadth.sentiment_features import SentimentFeatures
from src.phase_09_features_calendar.calendar_features import CalendarFeatureGenerator
from src.core.system_resources import maybe_gc as _maybe_gc


# ---------------------------------------------------------------------------
# Registry dataclass
# ---------------------------------------------------------------------------

@dataclass
class _FeatureStep:
    """Declarative descriptor for a single feature-engineering step."""
    flag: str              # kwarg name, e.g. "use_amihud_features"
    label: str             # log label, e.g. "AMIHUD"
    meta_key: str          # metadata key, e.g. "amihud_features"
    cls_path: str          # dotted import path e.g. "src.phase_08_features_breadth.amihud_features.AmihudFeatures"
    create_method: str     # e.g. "create_amihud_features"
    download_method: Optional[str] = None   # e.g. "download_futures_data" (None = no download)
    gc_after: Optional[str] = None          # GC checkpoint name (e.g. "steps 19-31")
    cls_kwargs: Optional[Dict] = None       # extra constructor kwargs (e.g. {"use_chronos": True})
    date_as_str: bool = False               # pass dates as str(d)[:10] to download method


# ---------------------------------------------------------------------------
# Feature step registry (steps 10-86)
# ---------------------------------------------------------------------------

_P08 = "src.phase_08_features_breadth"

_FEATURE_STEPS: List[_FeatureStep] = [
    # --- Steps 10-18 (gc after 18) ---
    _FeatureStep("use_fear_greed", "FEAR_GREED", "fear_greed_features",
                 f"{_P08}.fear_greed_features.FearGreedFeatures",
                 "create_fear_greed_features", "download_fear_greed_data"),
    _FeatureStep("use_reddit_sentiment", "REDDIT", "reddit_sentiment_features",
                 f"{_P08}.reddit_sentiment_features.RedditSentimentFeatures",
                 "create_reddit_features", "download_reddit_data"),
    _FeatureStep("use_crypto_sentiment", "CRYPTO", "crypto_sentiment_features",
                 f"{_P08}.crypto_sentiment_features.CryptoSentimentFeatures",
                 "create_crypto_features", "download_crypto_data"),
    _FeatureStep("use_gamma_exposure", "GEX", "gex_features",
                 f"{_P08}.gamma_exposure_features.GammaExposureFeatures",
                 "create_gex_features", "download_gex_data"),
    _FeatureStep("use_finnhub_social", "FINNHUB", "finnhub_social_features",
                 f"{_P08}.finnhub_social_features.FinnhubSocialFeatures",
                 "create_finnhub_social_features", "download_finnhub_social_data"),
    _FeatureStep("use_dark_pool", "DARK_POOL", "dark_pool_features",
                 f"{_P08}.dark_pool_features.DarkPoolFeatures",
                 "create_dark_pool_features", "download_dark_pool_data"),
    _FeatureStep("use_options_features", "OPTIONS", "options_features",
                 f"{_P08}.options_features.OptionsFeatures",
                 "create_options_features", "download_options_data"),
    _FeatureStep("use_event_recency", "EVENT_RECENCY", "event_recency_features",
                 f"{_P08}.event_recency_features.EventRecencyFeatures",
                 "create_event_recency_features"),
    _FeatureStep("use_block_structure", "BLOCK_STRUCTURE", "block_structure_features",
                 f"{_P08}.block_structure_features.BlockStructureFeatures",
                 "create_block_structure_features", gc_after="steps 10-18"),

    # --- Steps 19-31 (gc after 31) ---
    _FeatureStep("use_amihud_features", "AMIHUD", "amihud_features",
                 f"{_P08}.amihud_features.AmihudFeatures",
                 "create_amihud_features"),
    _FeatureStep("use_range_vol_features", "RANGE_VOL", "range_vol_features",
                 f"{_P08}.range_vol_features.RangeVolFeatures",
                 "create_range_vol_features"),
    _FeatureStep("use_entropy_features", "ENTROPY", "entropy_features",
                 f"{_P08}.entropy_features.EntropyFeatures",
                 "create_entropy_features"),
    _FeatureStep("use_hurst_features", "HURST", "hurst_features",
                 f"{_P08}.hurst_features.HurstFeatures",
                 "create_hurst_features"),
    _FeatureStep("use_nmi_features", "NMI", "nmi_features",
                 f"{_P08}.nmi_features.NMIFeatures",
                 "create_nmi_features"),
    _FeatureStep("use_absorption_ratio", "ABSORPTION_RATIO", "absorption_ratio_features",
                 f"{_P08}.absorption_ratio_features.AbsorptionRatioFeatures",
                 "create_absorption_ratio_features"),
    _FeatureStep("use_drift_features", "DRIFT", "drift_features",
                 f"{_P08}.drift_features.DriftFeatures",
                 "create_drift_features"),
    _FeatureStep("use_changepoint_features", "CHANGEPOINT", "changepoint_features",
                 f"{_P08}.changepoint_features.ChangepointFeatures",
                 "create_changepoint_features"),
    _FeatureStep("use_hmm_features", "HMM", "hmm_features",
                 f"{_P08}.hmm_features.HMMFeatures",
                 "create_hmm_features"),
    _FeatureStep("use_vpin_features", "VPIN", "vpin_features",
                 f"{_P08}.vpin_features.VPINFeatures",
                 "create_vpin_features"),
    _FeatureStep("use_intraday_momentum", "INTRADAY_MOM", "intraday_momentum_features",
                 f"{_P08}.intraday_momentum_features.IntradayMomentumFeatures",
                 "create_intraday_momentum_features"),
    _FeatureStep("use_futures_basis", "FUTURES_BASIS", "futures_basis_features",
                 f"{_P08}.futures_basis_features.FuturesBasisFeatures",
                 "create_futures_basis_features", "download_futures_data"),
    _FeatureStep("use_congressional_features", "CONGRESSIONAL", "congressional_features",
                 f"{_P08}.congressional_features.CongressionalFeatures",
                 "create_congressional_features", gc_after="steps 19-31"),

    # --- Steps 32-40 (gc after 40) ---
    _FeatureStep("use_insider_aggregate", "INSIDER_AGG", "insider_aggregate_features",
                 f"{_P08}.insider_aggregate_features.InsiderAggregateFeatures",
                 "create_insider_aggregate_features"),
    _FeatureStep("use_etf_flow", "ETF_FLOW", "etf_flow_features",
                 f"{_P08}.etf_flow_features.ETFFlowFeatures",
                 "create_etf_flow_features"),
    _FeatureStep("use_wavelet_features", "WAVELET", "wavelet_features",
                 f"{_P08}.wavelet_features.WaveletFeatures",
                 "create_wavelet_features"),
    _FeatureStep("use_sax_features", "SAX", "sax_features",
                 f"{_P08}.sax_features.SAXFeatures",
                 "create_sax_features"),
    _FeatureStep("use_transfer_entropy", "TRANSFER_ENTROPY", "transfer_entropy_features",
                 f"{_P08}.transfer_entropy_features.TransferEntropyFeatures",
                 "create_transfer_entropy_features"),
    _FeatureStep("use_mfdfa_features", "MFDFA", "mfdfa_features",
                 f"{_P08}.mfdfa_features.MFDFAFeatures",
                 "create_mfdfa_features"),
    _FeatureStep("use_rqa_features", "RQA", "rqa_features",
                 f"{_P08}.rqa_features.RQAFeatures",
                 "create_rqa_features"),
    _FeatureStep("use_copula_features", "COPULA", "copula_features",
                 f"{_P08}.copula_features.CopulaFeatures",
                 "create_copula_features"),
    _FeatureStep("use_network_centrality", "NETWORK", "network_features",
                 f"{_P08}.network_features.NetworkFeatures",
                 "create_network_features", gc_after="steps 32-40"),

    # --- Steps 41-50 (gc after 50) ---
    _FeatureStep("use_path_signatures", "PSIG", "path_signature_features",
                 f"{_P08}.path_signature_features.PathSignatureFeatures",
                 "create_path_signature_features"),
    _FeatureStep("use_wavelet_scattering", "WSCAT", "wavelet_scattering_features",
                 f"{_P08}.wavelet_scattering_features.WaveletScatteringFeatures",
                 "create_wavelet_scattering_features"),
    _FeatureStep("use_wasserstein_regime", "WREG", "wasserstein_regime_features",
                 "src.phase_14_robustness.wasserstein_regime.WassersteinRegimeDetector",
                 "create_wasserstein_features"),
    _FeatureStep("use_market_structure", "MSTR", "market_structure_features",
                 f"{_P08}.market_structure_features.MarketStructureFeatures",
                 "create_market_structure_features"),
    # Step 45: special constructor args — cls_kwargs populated dynamically
    _FeatureStep("use_time_series_models", "TSM", "time_series_model_features",
                 f"{_P08}.time_series_model_features.TimeSeriesModelFeatures",
                 "create_time_series_model_features",
                 cls_kwargs={"use_chronos": True}),
    _FeatureStep("use_har_rv", "HARV", "har_rv_features",
                 f"{_P08}.har_rv_features.HARRVFeatures",
                 "create_har_rv_features"),
    _FeatureStep("use_l_moments", "LMOM", "l_moments_features",
                 f"{_P08}.l_moments_features.LMomentsFeatures",
                 "create_l_moments_features"),
    _FeatureStep("use_multiscale_entropy", "MSE", "multiscale_entropy_features",
                 f"{_P08}.multiscale_entropy_features.MultiscaleEntropyFeatures",
                 "create_multiscale_entropy_features"),
    _FeatureStep("use_rv_signature_plot", "RVSP", "rv_signature_features",
                 f"{_P08}.rv_signature_features.RVSignaturePlotFeatures",
                 "create_rv_signature_features"),
    _FeatureStep("use_tda_homology", "TDA", "tda_homology_features",
                 f"{_P08}.tda_features.TDAHomologyFeatures",
                 "create_tda_features", gc_after="steps 41-50"),

    # --- Steps 51-58 (gc after 58) ---
    _FeatureStep("use_credit_spread_features", "CREDIT", "credit_spread_features",
                 f"{_P08}.credit_spread_features.CreditSpreadFeatures",
                 "create_credit_spread_features"),
    _FeatureStep("use_yield_curve_features", "YIELD", "yield_curve_features",
                 f"{_P08}.yield_curve_features.YieldCurveFeatures",
                 "create_yield_curve_features"),
    _FeatureStep("use_vol_term_structure_features", "VTS", "vol_term_structure_features",
                 f"{_P08}.vol_term_structure_features.VolTermStructureFeatures",
                 "create_vol_term_structure_features"),
    _FeatureStep("use_macro_surprise_features", "MSURP", "macro_surprise_features",
                 f"{_P08}.macro_surprise_features.MacroSurpriseFeatures",
                 "create_macro_surprise_features"),
    _FeatureStep("use_cross_asset_momentum", "XMOM", "cross_asset_momentum_features",
                 f"{_P08}.cross_asset_momentum_features.CrossAssetMomentumFeatures",
                 "create_cross_asset_momentum_features"),
    _FeatureStep("use_skew_kurtosis_features", "SKKU", "skew_kurtosis_features",
                 f"{_P08}.skew_kurtosis_features.SkewKurtosisFeatures",
                 "create_skew_kurtosis_features"),
    _FeatureStep("use_seasonality_features", "SEAS", "seasonality_features",
                 f"{_P08}.seasonality_features.SeasonalityFeatures",
                 "create_seasonality_features"),
    _FeatureStep("use_order_flow_imbalance", "OFI", "order_flow_imbalance_features",
                 f"{_P08}.order_flow_imbalance_features.OrderFlowImbalanceFeatures",
                 "create_order_flow_imbalance_features", gc_after="steps 51-58"),

    # --- Steps 59-62 (gc after 62) ---
    _FeatureStep("use_correlation_regime", "CORR_REGIME", "correlation_regime_features",
                 f"{_P08}.correlation_regime_features.CorrelationRegimeFeatures",
                 "create_correlation_features", "download_correlation_data",
                 date_as_str=True),
    _FeatureStep("use_fama_french", "FAMA_FRENCH", "fama_french_features",
                 f"{_P08}.fama_french_features.FamaFrenchFeatures",
                 "create_fama_french_features", "download_factor_data",
                 date_as_str=True),
    _FeatureStep("use_put_call_ratio", "PCR", "put_call_ratio_features",
                 f"{_P08}.put_call_ratio_features.PutCallRatioFeatures",
                 "create_pcr_features", "download_pcr_data",
                 date_as_str=True),
    _FeatureStep("use_multi_horizon", "MULTI_HORIZON", "multi_horizon_features",
                 "src.phase_15_strategy.multi_horizon_filter.MultiHorizonFilter",
                 "compute_horizon_signals", gc_after="steps 59-62"),

    # --- Steps 63-67 (gc after 67) ---
    _FeatureStep("use_earnings_revision", "ERN", "earnings_revision_features",
                 f"{_P08}.earnings_revision_features.EarningsRevisionFeatures",
                 "create_earnings_revision_features", "download_earnings_data"),
    _FeatureStep("use_short_interest", "SI", "short_interest_features",
                 f"{_P08}.short_interest_features.ShortInterestFeatures",
                 "create_short_interest_features", "download_short_interest_data"),
    _FeatureStep("use_dollar_index", "DXY", "dollar_index_features",
                 f"{_P08}.dollar_index_features.DollarIndexFeatures",
                 "create_dollar_index_features", "download_dollar_data",
                 date_as_str=True),
    _FeatureStep("use_institutional_flow", "INST", "institutional_flow_features",
                 f"{_P08}.institutional_flow_features.InstitutionalFlowFeatures",
                 "create_institutional_flow_features", "download_institutional_data"),
    _FeatureStep("use_google_trends", "GTREND", "google_trends_features",
                 f"{_P08}.google_trends_features.GoogleTrendsFeatures",
                 "create_google_trends_features", "download_trends_data",
                 gc_after="steps 63-67"),

    # --- Steps 68-72 (gc after 72) ---
    _FeatureStep("use_commodity_signals", "CMDTY", "commodity_signal_features",
                 f"{_P08}.commodity_signal_features.CommoditySignalFeatures",
                 "create_commodity_signal_features", "download_commodity_data",
                 date_as_str=True),
    _FeatureStep("use_treasury_auction", "TAUCT", "treasury_auction_features",
                 f"{_P08}.treasury_auction_features.TreasuryAuctionFeatures",
                 "create_treasury_auction_features", "download_auction_data"),
    _FeatureStep("use_fed_liquidity", "FEDLIQ", "fed_liquidity_features",
                 f"{_P08}.fed_liquidity_features.FedLiquidityFeatures",
                 "create_fed_liquidity_features", "download_liquidity_data"),
    _FeatureStep("use_earnings_calendar", "ECAL", "earnings_calendar_features",
                 f"{_P08}.earnings_calendar_features.EarningsCalendarFeatures",
                 "create_earnings_calendar_features", "download_calendar_data"),
    _FeatureStep("use_analyst_rating", "ANLST", "analyst_rating_features",
                 f"{_P08}.analyst_rating_features.AnalystRatingFeatures",
                 "create_analyst_rating_features", "download_rating_data",
                 gc_after="steps 68-72"),

    # --- Steps 73-76 (gc after 76) ---
    _FeatureStep("use_expanded_macro", "XMACRO", "expanded_macro_features",
                 f"{_P08}.expanded_macro_features.ExpandedMacroFeatures",
                 "create_expanded_macro_features", "download_macro_data"),
    _FeatureStep("use_vvix", "VVIX", "vvix_features",
                 f"{_P08}.vvix_features.VVIXFeatures",
                 "create_vvix_features", "download_vvix_data"),
    _FeatureStep("use_sector_rotation", "SECROT", "sector_rotation_features",
                 f"{_P08}.sector_rotation_features.SectorRotationFeatures",
                 "create_sector_rotation_features", "download_sector_data"),
    _FeatureStep("use_fx_carry", "FXC", "fx_carry_features",
                 f"{_P08}.fx_carry_features.FXCarryFeatures",
                 "create_fx_carry_features", "download_fx_data",
                 gc_after="steps 73-76"),

    # --- Steps 77-80 (gc after 80) ---
    _FeatureStep("use_money_market", "MMKT", "money_market_features",
                 f"{_P08}.money_market_features.MoneyMarketFeatures",
                 "create_money_market_features", "download_money_market_data"),
    _FeatureStep("use_financial_stress", "FSTRESS", "financial_stress_features",
                 f"{_P08}.financial_stress_features.FinancialStressFeatures",
                 "create_financial_stress_features", "download_stress_data"),
    _FeatureStep("use_global_equity", "GLEQ", "global_equity_features",
                 f"{_P08}.global_equity_features.GlobalEquityFeatures",
                 "create_global_equity_features", "download_global_data"),
    _FeatureStep("use_retail_sentiment", "RFLOW", "retail_sentiment_features",
                 f"{_P08}.retail_sentiment_features.RetailSentimentFeatures",
                 "create_retail_sentiment_features", "download_retail_data",
                 gc_after="steps 77-80"),

    # --- Steps 81-86 (gc after 86) ---
    _FeatureStep("use_cboe_pcr", "CBOE_PCR", "cboe_pcr_features",
                 f"{_P08}.cboe_pcr_features.CBOEPutCallFeatures",
                 "create_cboe_pcr_features", "download_cboe_data"),
    _FeatureStep("use_stocktwits", "STWIT", "stocktwits_features",
                 f"{_P08}.stocktwits_features.StockTwitsSentimentFeatures",
                 "create_stocktwits_features", "download_stocktwits_data"),
    _FeatureStep("use_alpaca_news", "ANEWS", "alpaca_news_features",
                 f"{_P08}.alpaca_news_features.AlpacaNewsFeatures",
                 "create_alpaca_news_features", "download_alpaca_news"),
    _FeatureStep("use_gnews_headlines", "GNEWS", "gnews_headline_features",
                 f"{_P08}.gnews_headline_features.GNewsHeadlineFeatures",
                 "create_gnews_features", "download_gnews_data"),
    _FeatureStep("use_finbert_nlp", "NLP", "finbert_nlp_features",
                 f"{_P08}.finbert_nlp_features.FinBERTNLPFeatures",
                 "create_finbert_features", "download_nlp_data"),
    _FeatureStep("use_wsb_sentiment", "WSB", "wsb_sentiment_features",
                 f"{_P08}.wsb_sentiment_features.WSBSentimentFeatures",
                 "create_wsb_features", "download_wsb_data",
                 gc_after="steps 81-86"),

    # --- Step 87 (Wave PQ: causal discovery features) ---
    _FeatureStep("use_causal_features", "CAUSAL", "causal_features",
                 f"{_P08}.causal_features.CausalFeatureSelector",
                 "create_causal_features",
                 gc_after="step 87"),

    # --- Steps 88-90 (Wave PQ2: research-driven features) ---
    _FeatureStep("use_kronos_features", "KRONOS", "kronos_features",
                 f"{_P08}.kronos_features.KronosFeatures",
                 "create_kronos_features"),
    _FeatureStep("use_graph_attention", "GAT", "graph_attention_features",
                 f"{_P08}.graph_attention_features.GraphAttentionFeatures",
                 "create_graph_attention_features", "download_cross_asset_data",
                 date_as_str=True),
    _FeatureStep("use_patch_temporal", "PTST", "patch_temporal_features",
                 f"{_P08}.patch_temporal_features.PatchTemporalFeatures",
                 "create_patch_temporal_features",
                 gc_after="steps 88-90"),
]


# ---------------------------------------------------------------------------
# Generic step runner
# ---------------------------------------------------------------------------

def _run_feature_step(
    step: _FeatureStep,
    df_daily: pd.DataFrame,
    start_date,
    end_date,
    metadata: Dict,
    resource_config,
    flags: Dict,
) -> pd.DataFrame:
    """Run a single registry-based feature step with try/except isolation."""
    try:
        # Lazy import via importlib
        module_path, class_name = step.cls_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)

        # Build constructor kwargs (may be augmented per-step)
        kwargs = dict(step.cls_kwargs) if step.cls_kwargs else {}

        # Special: step 45 (TSM) needs use_catch22 from outer flags
        if step.flag == "use_time_series_models":
            kwargs["use_catch22"] = flags.get("use_catch22", False)

        feat = cls(**kwargs)

        # Optional download phase
        if step.download_method:
            dl_method = getattr(feat, step.download_method)
            if step.date_as_str:
                dl_method(str(start_date)[:10], str(end_date)[:10])
            else:
                dl_method(start_date, end_date)

        # Create features
        n_before = len(df_daily.columns)
        df_daily = getattr(feat, step.create_method)(df_daily)
        n_added = len(df_daily.columns) - n_before
        metadata[step.meta_key] = True
        metadata[f"n_{step.meta_key}"] = n_added
        print(f"  [{step.label}] Added {n_added} features")

    except Exception as e:
        print(f"  [{step.label}] Warning: {step.meta_key} failed: {e}")
        metadata[step.meta_key] = False

    return df_daily


# ---------------------------------------------------------------------------
# Main integration function (signature preserved for backward compatibility)
# ---------------------------------------------------------------------------

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
    use_cboe_pcr: bool = True,  # CBOE direct put/call ratio features (cboe_*)
    use_stocktwits: bool = True,  # StockTwits social sentiment features (stwit_*)
    use_alpaca_news: bool = True,  # Alpaca/Benzinga news sentiment features (anews_*)
    use_gnews_headlines: bool = True,  # Google News headline sentiment features (gnews_*)
    use_finbert_nlp: bool = False,  # FinBERT local NLP features (nlp_*) -- heavy deps
    use_wsb_sentiment: bool = False,  # Reddit WSB PRAW sentiment features (wsb_*) -- needs OAuth
    use_causal_features: bool = True,  # Causal feature selection features (causal_*)
    use_kronos_features: bool = True,  # Kronos random projection features (kron_*)
    use_graph_attention: bool = True,  # Graph attention cross-asset features (gat_*)
    use_patch_temporal: bool = True,  # PatchTST temporal features (ptst_*)
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

    # ------------------------------------------------------------------
    # Inline steps 0-8 (special download/analyze patterns)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Registry-driven steps 10-86
    # ------------------------------------------------------------------
    # Collect all use_* kwargs into a flags dict for the generic runner.
    flags = {
        "use_fear_greed": use_fear_greed,
        "use_reddit_sentiment": use_reddit_sentiment,
        "use_crypto_sentiment": use_crypto_sentiment,
        "use_gamma_exposure": use_gamma_exposure,
        "use_finnhub_social": use_finnhub_social,
        "use_dark_pool": use_dark_pool,
        "use_options_features": use_options_features,
        "use_event_recency": use_event_recency,
        "use_block_structure": use_block_structure,
        "use_amihud_features": use_amihud_features,
        "use_range_vol_features": use_range_vol_features,
        "use_entropy_features": use_entropy_features,
        "use_hurst_features": use_hurst_features,
        "use_nmi_features": use_nmi_features,
        "use_absorption_ratio": use_absorption_ratio,
        "use_drift_features": use_drift_features,
        "use_changepoint_features": use_changepoint_features,
        "use_hmm_features": use_hmm_features,
        "use_vpin_features": use_vpin_features,
        "use_intraday_momentum": use_intraday_momentum,
        "use_futures_basis": use_futures_basis,
        "use_congressional_features": use_congressional_features,
        "use_insider_aggregate": use_insider_aggregate,
        "use_etf_flow": use_etf_flow,
        "use_wavelet_features": use_wavelet_features,
        "use_sax_features": use_sax_features,
        "use_transfer_entropy": use_transfer_entropy,
        "use_mfdfa_features": use_mfdfa_features,
        "use_rqa_features": use_rqa_features,
        "use_copula_features": use_copula_features,
        "use_network_centrality": use_network_centrality,
        "use_path_signatures": use_path_signatures,
        "use_wavelet_scattering": use_wavelet_scattering,
        "use_wasserstein_regime": use_wasserstein_regime,
        "use_market_structure": use_market_structure,
        "use_time_series_models": use_time_series_models,
        "use_catch22": use_catch22,
        "use_har_rv": use_har_rv,
        "use_l_moments": use_l_moments,
        "use_multiscale_entropy": use_multiscale_entropy,
        "use_rv_signature_plot": use_rv_signature_plot,
        "use_tda_homology": use_tda_homology,
        "use_credit_spread_features": use_credit_spread_features,
        "use_yield_curve_features": use_yield_curve_features,
        "use_vol_term_structure_features": use_vol_term_structure_features,
        "use_macro_surprise_features": use_macro_surprise_features,
        "use_cross_asset_momentum": use_cross_asset_momentum,
        "use_skew_kurtosis_features": use_skew_kurtosis_features,
        "use_seasonality_features": use_seasonality_features,
        "use_order_flow_imbalance": use_order_flow_imbalance,
        "use_correlation_regime": use_correlation_regime,
        "use_fama_french": use_fama_french,
        "use_put_call_ratio": use_put_call_ratio,
        "use_multi_horizon": use_multi_horizon,
        "use_earnings_revision": use_earnings_revision,
        "use_short_interest": use_short_interest,
        "use_dollar_index": use_dollar_index,
        "use_institutional_flow": use_institutional_flow,
        "use_google_trends": use_google_trends,
        "use_commodity_signals": use_commodity_signals,
        "use_treasury_auction": use_treasury_auction,
        "use_fed_liquidity": use_fed_liquidity,
        "use_earnings_calendar": use_earnings_calendar,
        "use_analyst_rating": use_analyst_rating,
        "use_expanded_macro": use_expanded_macro,
        "use_vvix": use_vvix,
        "use_sector_rotation": use_sector_rotation,
        "use_fx_carry": use_fx_carry,
        "use_money_market": use_money_market,
        "use_financial_stress": use_financial_stress,
        "use_global_equity": use_global_equity,
        "use_retail_sentiment": use_retail_sentiment,
        "use_cboe_pcr": use_cboe_pcr,
        "use_stocktwits": use_stocktwits,
        "use_alpaca_news": use_alpaca_news,
        "use_gnews_headlines": use_gnews_headlines,
        "use_finbert_nlp": use_finbert_nlp,
        "use_wsb_sentiment": use_wsb_sentiment,
    }

    for step in _FEATURE_STEPS:
        if not flags.get(step.flag, True):
            continue
        df_daily = _run_feature_step(
            step, df_daily, start_date, end_date, metadata, resource_config, flags,
        )
        if step.gc_after:
            _maybe_gc(resource_config, step.gc_after)

    # ------------------------------------------------------------------
    # 9. Synthetic SPY Universes (do last since it multiplies data)
    # ------------------------------------------------------------------
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

# Giga Trader — System Architecture

> Comprehensive architecture documentation for the 27-phase ML trading pipeline.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Pipeline Flow](#pipeline-flow)
3. [Data Flow Diagram](#data-flow-diagram)
4. [Feature Engineering Pipeline](#feature-engineering-pipeline)
5. [Model Training Architecture](#model-training-architecture)
6. [Robustness Validation Framework](#robustness-validation-framework)
7. [Trading Execution Architecture](#trading-execution-architecture)
8. [Resource Scaling System](#resource-scaling-system)
9. [Configuration Architecture](#configuration-architecture)
10. [Storage and Persistence](#storage-and-persistence)

---

## High-Level Architecture

The system follows a linear pipeline with feedback loops. Each phase produces artifacts consumed by downstream phases, with a central configuration schema (ExperimentConfig) governing behavior.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GIGA TRADER SYSTEM                                │
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   CONFIG      │    │   REGISTRY   │    │  RESOURCES   │                  │
│  │              │    │              │    │              │                  │
│  │ 13 dataclass │◄──►│ SQLite DB    │◄──►│ Auto-detect  │                  │
│  │ sections     │    │ experiments  │    │ RAM/CPU/GPU  │                  │
│  │ 98+ flags    │    │ models       │    │ 4 tiers      │                  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                  │
│         │                   │                   │                          │
│         ▼                   ▼                   ▼                          │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                    PIPELINE ENGINE                               │       │
│  │                                                                 │       │
│  │  Phase 01-02     Phase 03-09     Phase 10-14    Phase 15-19    │       │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │       │
│  │  │ Data     │──►│ Features │──►│ Model    │──►│ Strategy │   │       │
│  │  │ Ingest   │   │ Engineer │   │ Train &  │   │ & Trade  │   │       │
│  │  │          │   │          │   │ Validate │   │          │   │       │
│  │  └──────────┘   └──────────┘   └──────────┘   └──────────┘   │       │
│  │                                                                 │       │
│  │  Phase 20-21     Phase 22-23     Phase 24-27                   │       │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐                   │       │
│  │  │ Monitor  │──►│ Automate │──►│ Advanced │                   │       │
│  │  │ & Alert  │   │ & Search │   │ & Live   │                   │       │
│  │  └──────────┘   └──────────┘   └──────────┘                   │       │
│  └─────────────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Flow

The 27 phases execute in dependency order. Each phase has an `__init__.py` re-exporting its public API and individual modules handling specific concerns.

```
PHASE 01: Data Acquisition
    │   alpaca_client.py ─── data_manager.py ─── historical_constituents.py
    │
    ▼
PHASE 02: Preprocessing
    │   ohlc_validator.py ─── information_bars.py ─── bar_resampler.py
    │
    ▼
PHASE 03-04: Synthetic Data & Probabilistic
    │   bear_universes.py ─── multiscale_bootstrap.py
    │
    ▼
PHASE 05: Target Creation
    │   target_generator.py ─── cusum_filter.py ─── triple_barrier.py
    │
    ▼
PHASE 06-09: Feature Engineering (200+ features)
    │
    │   Phase 06: Intraday Features
    │     intraday_patterns.py ─── extended_hours.py
    │
    │   Phase 07: Daily Features
    │     technical_indicators.py ─── daily_features.py
    │
    │   Phase 08: Feature Breadth (48 modules)           ◄── Largest phase
    │     economic ─── sentiment ─── microstructure
    │     entropy ─── wavelets ─── network ─── options
    │     dark_pool ─── path_signatures ─── HMM ─── ...
    │
    │   Phase 09: Calendar Features
    │     calendar_features.py ─── feature_researcher.py
    │
    ▼
PHASE 10: Feature Processing
    │   group_aware_processor.py ─── feature_neutralizer.py
    │   interaction_discovery.py
    │
    │   Dimensionality Reduction Methods:
    │   ┌──────────┬──────────┬──────────┬──────────┐
    │   │ Mutual   │ Kernel   │   ICA    │ K-Medoids│
    │   │ Info     │ PCA      │          │          │
    │   │ (20 feat)│(12 comp) │(8 comp)  │(10 clust)│
    │   └────┬─────┴────┬─────┴────┬─────┴────┬─────┘
    │        └──────────┴──────────┴──────────┘
    │                   │
    │                   ▼
    │           ENSEMBLE+ (50 dims)
    │
    ▼
PHASE 11: Cross-Validation
    │   leak_proof_cv.py ─── cpcv.py ─── walk_forward.py
    │
    │   ┌─────────────────────────────────────────────────┐
    │   │  TRAIN         PURGE    TEST      EMBARGO       │
    │   │  ████████████  ░░░░░    ████      ▒▒▒▒▒▒       │
    │   │                5 days              2 days        │
    │   └─────────────────────────────────────────────────┘
    │
    ▼
PHASE 12: Model Training
    │   Logistic (L1/L2) ─── Gradient Boosting ─── XGBoost
    │   CatBoost ─── Stacking Ensemble ─── Quantile Forest
    │   Regime Router (wraps any base model)
    │
    │   Optimization: Optuna Bayesian HPO (TPE sampler)
    │
    ▼
PHASE 13-14: Validation & Robustness
    │
    │   anti_overfit_integration.py (50 feature steps)
    │
    │   AdvancedStabilitySuite (22 methods):
    │   ┌────────────────────────────────────────────────────┐
    │   │  Fast:     hp_perturbation, bootstrap, noise,      │
    │   │           permutation_importance, sfi               │
    │   │                                                    │
    │   │  Medium:  shap, walk_forward, regime_stability,    │
    │   │           conformal_prediction, disagreement        │
    │   │                                                    │
    │   │  Expensive: cpcv, stability_selection, rashomon,   │
    │   │            adversarial_overfitting                  │
    │   │                                                    │
    │   │  External: label_noise, fi_stability, knockoff_gate│
    │   └────────────────────────────────────────────────────┘
    │
    ▼
PHASE 15: Strategy Generation
    │   signal_generator.py ─── meta_labeler.py
    │   conformal_sizer.py ─── cvar_position_sizer.py
    │   thompson_selector.py ─── bayesian_averaging.py
    │   dynamic_weights.py ─── isotonic_calibrator.py
    │   ensemble_disagreement.py ─── regime_router.py
    │
    ▼
PHASE 16-17: Backtesting & Monte Carlo
    │   backtest_engine.py ─── portfolio_tracker.py
    │   monte_carlo.py
    │
    ▼
PHASE 18: Persistence & Registry
    │   registry_db.py (SQLite) ─── grid_search_generator.py
    │   registry_configs.py ─── registry_enums.py
    │
    ▼
PHASE 19: Paper Trading
    │   trading_bot.py ─── signal_generator.py
    │   trading_gates.py (5 gates):
    │   ┌────────────────────────────────────────────────────┐
    │   │  AAII Gate ─── COT Gate ─── EDGAR Gate             │
    │   │  Insider Gate ─── Staleness Checker                │
    │   └────────────────────────────────────────────────────┘
    │   circuit_breaker.py ─── vol_targeting.py
    │
    ▼
PHASE 20-21: Monitoring & Continuous Learning
    │   health_checker.py ─── feature_drift_monitor.py
    │   dashboard_server.py ─── web_monitor.py
    │   experiment_runner.py ─── online_updater.py
    │
    ▼
PHASE 22-27: Advanced Analytics & Live Trading
    thick_weave_search.py ─── temporal_cascades
    risk_management ─── live_trading.py
```

---

## Data Flow Diagram

```
                    EXTERNAL DATA SOURCES
    ┌──────────┬──────────┬──────────┬──────────┐
    │ Alpaca   │ yfinance │ FRED     │ Finnhub  │
    │ (1-min)  │ (daily)  │ (econ)   │ (news)   │
    └────┬─────┴────┬─────┴────┬─────┴────┬─────┘
         │          │          │          │
         ▼          ▼          ▼          ▼
    ┌─────────────────────────────────────────────┐
    │           RAW DATA LAYER                     │
    │                                             │
    │  SPY 1-min bars ── Component daily data     │
    │  Extended hours ── Economic indicators      │
    │  News articles ── Cross-asset ETFs          │
    │                                             │
    │  Storage: data/raw/ (CSV, Parquet)          │
    └─────────────────────┬───────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────┐
    │         PROCESSED DATA LAYER                 │
    │                                             │
    │  OHLC validated ── Resampled bars           │
    │  Merged extended hours ── Cleaned NaNs      │
    │                                             │
    │  Storage: data/processed/                   │
    └─────────────────────┬───────────────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
    ┌──────────────┐ ┌──────────┐ ┌──────────────┐
    │ FEATURES     │ │ TARGETS  │ │ SYNTHETIC    │
    │              │ │          │ │              │
    │ 200+ raw     │ │ target_up│ │ 10-30 alt    │
    │ features     │ │ timing   │ │ SPY universes│
    │ per sample   │ │ soft_lbl │ │              │
    └──────┬───────┘ └────┬─────┘ └──────┬───────┘
           │              │              │
           ▼              ▼              ▼
    ┌─────────────────────────────────────────────┐
    │         TRAINING DATA ASSEMBLY               │
    │                                             │
    │  Real samples (70% weight)                  │
    │  + Synthetic samples (30% weight)           │
    │  = Augmented training set                   │
    │                                             │
    │  Features: 200+ raw → 30-50 reduced         │
    └─────────────────────┬───────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────┐
    │              MODEL ARTIFACTS                 │
    │                                             │
    │  models/production/                         │
    │    spy_robust_models.joblib                 │
    │    conformal_sizer.joblib                   │
    │    thompson_state.json                      │
    │                                             │
    │  data/giga_trader.db (SQLite registry)      │
    │    experiments table                        │
    │    models table                             │
    │    model_entries table                      │
    └─────────────────────────────────────────────┘
```

---

## Feature Engineering Pipeline

The anti-overfitting integration module (`anti_overfit_integration.py`) orchestrates all 50 feature engineering steps:

```
integrate_anti_overfit(df, config) → (augmented_df, metadata)

Steps 0-8: Core Features
    ├── Step 0:  OHLC Validation
    ├── Step 1:  Economic Features (~120 features: VIX, yields, commodities)
    ├── Step 2:  Calendar Features (29 features: day of week, FOMC, expiry)
    ├── Step 3:  Sentiment Features (12 features: VIX cross-asset signals)
    ├── Step 4:  Fear & Greed Index
    ├── Step 5:  Options Features (15 opt_ features)
    ├── Step 6:  Dark Pool Features
    ├── Step 7:  Block Structure Features (54 blk_ features)
    └── Step 8:  Event Recency Features (88 dts_ features)
    ─── [GC checkpoint if memory pressure] ───

Steps 10-18: Alternative Data Features
    ├── Step 10: Fear & Greed Index
    ├── Step 11: Crypto Sentiment
    ├── Step 12: Reddit Sentiment
    ├── Step 13: Finnhub Social
    ├── Step 14: Congressional Trading
    ├── Step 15: ETF Flows
    ├── Step 16: Insider Aggregate
    ├── Step 17: Futures Basis
    └── Step 18: Gamma Exposure
    ─── [GC checkpoint if memory pressure] ───

Steps 19-31: Quantitative Features
    ├── Step 19: Amihud Illiquidity
    ├── Step 20: Entropy (sample, spectral, approximate)
    ├── Step 21: Hurst Exponent
    ├── Step 22: NMI (Normalized Mutual Information)
    ├── Step 23: Absorption Ratio
    ├── Step 24: Drift Detection
    ├── Step 25: Changepoint Detection
    ├── Step 26: HMM Regime Features
    ├── Step 27: VPIN (Volume-Synchronized PIN)
    ├── Step 28: Intraday Momentum
    ├── Step 29: Range Volatility
    └── Step 30-31: Wavelets, SAX, Transfer Entropy
    ─── [GC checkpoint if memory pressure] ───

Steps 32-40: Advanced Statistical Features
    ├── Step 32: Insider Aggregate
    ├── Step 33: MFDFA (Multifractal DFA)
    ├── Step 34: RQA (Recurrence Quantification)
    ├── Step 35: Copula Dependence
    ├── Step 36: Network Features
    ├── Step 37: Wavelet Scattering
    ├── Step 38: Wasserstein Regime
    ├── Step 39: Market Structure
    └── Step 40: Time Series Models
    ─── [GC checkpoint if memory pressure] ───

Steps 41-50: Research-Driven Features
    ├── Step 41: Path Signatures (17 psig_ features)
    ├── Step 42: Wavelet Scattering (12 wscat_ features)
    ├── Step 43: Wasserstein Regime (8 wreg_ features)
    ├── Step 44: Market Structure (18 mstr_ features)
    ├── Step 45: Time Series Models (15 tsm_ features)
    ├── Step 46: HAR-RV (4 harv_ features)
    ├── Step 47: L-Moments (4 lmom_ features)
    ├── Step 48: Multiscale Entropy (3 mse_ features)
    ├── Step 49: RV Signature Plot (3 rvsp_ features)
    └── Step 50: TDA Homology (5 tda_ features)
    ─── [GC checkpoint if memory pressure] ───

Step FINAL: Synthetic SPY Universe Generation
    └── Generate 10-30 alternative SPY histories
        Weight: 70% real + 30% synthetic
```

---

## Model Training Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MODEL TRAINING PIPELINE                         │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                 MODEL TYPE SELECTION                           │  │
│  │                                                               │  │
│  │  REGULARIZED LINEAR          TREE-BASED          ENSEMBLE     │  │
│  │  ┌────────────────┐   ┌────────────────┐   ┌──────────────┐  │  │
│  │  │ Logistic L1    │   │ Gradient Boost │   │ Stacking     │  │  │
│  │  │ Logistic L2    │   │ XGBoost        │   │ Quantile RF  │  │  │
│  │  │ Elastic Net    │   │ CatBoost       │   │ BMA Ensemble │  │  │
│  │  │ C: 0.001-10    │   │ depth: 2-5     │   │              │  │  │
│  │  └────────────────┘   │ trees: 30-150  │   └──────────────┘  │  │
│  │                        └────────────────┘                     │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              OPTUNA BAYESIAN OPTIMIZATION                     │  │
│  │                                                               │  │
│  │  Sampler: TPE (Tree-Parzen Estimator)                        │  │
│  │  Trials: 10-100 (scaled by resource tier)                    │  │
│  │  Pruning: MedianPruner (early stopping of bad trials)        │  │
│  │                                                               │  │
│  │  Search Space:                                                │  │
│  │    C ∈ [0.01, 10.0]         (log scale)                      │  │
│  │    n_estimators ∈ [30, 150]                                   │  │
│  │    max_depth ∈ [2, 5]       (NEVER > 5 per Edge 1)           │  │
│  │    learning_rate ∈ [0.01, 0.3]  (log scale)                  │  │
│  │    min_samples_leaf ∈ [20, 100]                               │  │
│  │    subsample ∈ [0.6, 1.0]                                     │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              OPTIONAL WRAPPERS                                │  │
│  │                                                               │  │
│  │  RegimeRouter ── wraps any model, routes by market regime     │  │
│  │  IsotonicCalibrator ── calibrates probabilities               │  │
│  │  MetaLabeler ── secondary model filters primary predictions   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              TWO PARALLEL MODELS                              │  │
│  │                                                               │  │
│  │  SWING MODEL              TIMING MODEL                       │  │
│  │  ┌──────────────────┐    ┌──────────────────┐               │  │
│  │  │ Predicts: UP/DOWN│    │ Predicts: LOW    │               │  │
│  │  │ Target: target_up│    │ before HIGH      │               │  │
│  │  │ AUC: 0.818       │    │ AUC: 0.778       │               │  │
│  │  └──────────────────┘    └──────────────────┘               │  │
│  │                                                               │  │
│  │  COMBINED SIGNAL:                                            │  │
│  │    Buy = Swing(UP) + Timing(Low-first) + Confidence > thresh │  │
│  │    Sell = Swing(DOWN) + other conditions                     │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Robustness Validation Framework

```
┌─────────────────────────────────────────────────────────────────────┐
│                WEIGHTED MODEL EVALUATION SCORE (WMES)               │
│                                                                     │
│  Components (weights):                                              │
│                                                                     │
│  win_rate          (0.15) ▓▓▓░░░░░░░░░░░░░  Capped at 75%         │
│  robustness        (0.25) ▓▓▓▓▓░░░░░░░░░░░  CV score stability    │
│  profit_potential  (0.20) ▓▓▓▓░░░░░░░░░░░░  Sharpe + profit factor│
│  noise_tolerance   (0.15) ▓▓▓░░░░░░░░░░░░░  Noisy data perf       │
│  plateau_stability (0.15) ▓▓▓░░░░░░░░░░░░░  HP sensitivity        │
│  complexity_penalty(0.10) ▓▓░░░░░░░░░░░░░░  Feature count penalty │
│                                                                     │
│  Interpretation:                                                    │
│    > 0.65  EXCELLENT ─── Likely robust                             │
│    0.55-0.65  GOOD ───── Proceed with caution                      │
│    < 0.55  SUSPECT ───── Likely overfit                            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│            ADVANCED STABILITY SUITE (22 Methods)                    │
│                                                                     │
│  ┌──── FAST (per-model) ────────────────────────────────────────┐  │
│  │  1. hp_perturbation          8. sfi_feature_importance       │  │
│  │  2. bootstrap_stability      9. regime_stability             │  │
│  │  3. noise_injection         10. walk_forward                 │  │
│  │  4. permutation_importance  11. conformal_prediction         │  │
│  │  5. train_size_sensitivity  12. disagreement_smoothing       │  │
│  │  6. feature_subset_stab     13. adversarial_examples         │  │
│  │  7. prediction_consistency                                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──── EXPENSIVE (skip on LOW tier) ────────────────────────────┐  │
│  │  14. cpcv (Combinatorial Purged CV)                          │  │
│  │  15. stability_selection                                     │  │
│  │  16. rashomon_set                                            │  │
│  │  17. adversarial_overfitting                                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──── EXTERNAL GATES ──────────────────────────────────────────┐  │
│  │  18. shap_consistency                                        │  │
│  │  19. calibration_quality                                     │  │
│  │  20. label_noise_test                                        │  │
│  │  21. fi_stability_gate                                       │  │
│  │  22. knockoff_gate                                           │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Output: composite_score (0.0-1.0), per-method scores              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Trading Execution Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRADING BOT EXECUTION LOOP                       │
│                                                                     │
│  STARTUP                                                            │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  1. StalenessChecker ── verify model age                       │ │
│  │  2. CircuitBreaker ── check state file                         │ │
│  │  3. FeatureDriftMonitor ── self-baseline on first call         │ │
│  │  4. ThompsonSelector ── load from thompson_state.json          │ │
│  │  5. ConformalSizer ── load calibration from joblib             │ │
│  │  6. SystemResources ── print tier summary                      │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  EACH TRADING DAY                                                   │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                                                                │ │
│  │  8:00 AM ── Pre-market data collection                        │ │
│  │             Extended hours features computed                   │ │
│  │                                                                │ │
│  │  9:30 AM ── Market open                                       │ │
│  │             ┌─────────────────────────────────┐               │ │
│  │             │      SIGNAL GENERATION          │               │ │
│  │             │                                 │               │ │
│  │             │  1. Engineer features (200+)    │               │ │
│  │             │  2. Reduce dimensions (50)      │               │ │
│  │             │  3. Run swing model             │               │ │
│  │             │  4. Run timing model            │               │ │
│  │             │  5. Thompson-weighted ensemble  │               │ │
│  │             │  6. BMA ensemble weighting      │               │ │
│  │             │  7. Dynamic weight adjustment   │               │ │
│  │             │  8. Meta-labeler filter         │               │ │
│  │             │  9. Feature drift check         │               │ │
│  │             │  10. Conformal position sizing  │               │ │
│  │             │  11. CVaR tail risk scaling     │               │ │
│  │             └──────────────┬──────────────────┘               │ │
│  │                            │                                  │ │
│  │                            ▼                                  │ │
│  │             ┌─────────────────────────────────┐               │ │
│  │             │      TRADING GATES (5)          │               │ │
│  │             │                                 │               │ │
│  │             │  AAII Sentiment Gate             │               │ │
│  │             │  COT Positioning Gate            │               │ │
│  │             │  EDGAR Filing Gate               │               │ │
│  │             │  Insider Trading Gate            │               │ │
│  │             │  Staleness Gate                  │               │ │
│  │             └──────────────┬──────────────────┘               │ │
│  │                            │                                  │ │
│  │                            ▼                                  │ │
│  │             ┌─────────────────────────────────┐               │ │
│  │             │  CIRCUIT BREAKER CHECK          │               │ │
│  │             │                                 │               │ │
│  │             │  Daily loss < 2%?               │               │ │
│  │             │  Consecutive losses < 5?        │               │ │
│  │             │  Drawdown < 10%?                │               │ │
│  │             └──────────────┬──────────────────┘               │ │
│  │                            │                                  │ │
│  │                            ▼                                  │ │
│  │             ┌─────────────────────────────────┐               │ │
│  │             │  BATCH EXECUTION (Edge 5)       │               │ │
│  │             │                                 │               │ │
│  │             │  Entry: 40% → 35% → 25%        │               │ │
│  │             │  Wait 5 min between batches     │               │ │
│  │             │  Cancel if signal reverses       │               │ │
│  │             └─────────────────────────────────┘               │ │
│  │                                                                │ │
│  │  4:00 PM ── Market close                                      │ │
│  │             Record PnL → CircuitBreaker                       │ │
│  │                                                                │ │
│  │  4:15 PM ── Post-close                                        │ │
│  │             OnlineUpdater ── buffer and retrain daily          │ │
│  │             ThompsonSelector ── update Beta distributions      │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Resource Scaling System

```
┌─────────────────────────────────────────────────────────────────────┐
│                  SYSTEM RESOURCE DETECTION                           │
│                                                                     │
│  src/core/system_resources.py                                       │
│                                                                     │
│  DETECTION (once at import):                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  psutil.virtual_memory()  → total RAM, available RAM         │   │
│  │  os.cpu_count()           → logical cores                    │   │
│  │  psutil.cpu_count(False)  → physical cores                   │   │
│  │  torch.cuda.is_available()→ CUDA GPU                         │   │
│  │  xgboost.XGBClassifier()  → XGBoost GPU support              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  TIER CLASSIFICATION:                                               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                                                              │   │
│  │  < 16 GB   ─────── LOW                                      │   │
│  │                       │  Aggressive GC, skip expensive       │   │
│  │                       │  methods, single-threaded             │   │
│  │                       │                                      │   │
│  │  16-32 GB  ─────── MEDIUM                                   │   │
│  │                       │  GC between feature steps,           │   │
│  │                       │  half cores, full methods             │   │
│  │                       │                                      │   │
│  │  32-128 GB ─────── HIGH                                     │   │
│  │                       │  No GC pressure, near-full cores     │   │
│  │                       │  More synthetic universes             │   │
│  │                       │                                      │   │
│  │  128+ GB   ─────── ULTRA                                    │   │
│  │                       │  Maximum parallelism, 30 universes   │   │
│  │                       │  100 Optuna trials, all cores        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  INTEGRATION POINTS:                                                │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  ExperimentConfig.resources  ── ResourceConfig dataclass     │   │
│  │  anti_overfit_integration    ── GC calls, n_universes        │   │
│  │  experiment_runner           ── memory thresholds, cache     │   │
│  │  group_aware_processor       ── n_jobs, Nystroem threshold   │   │
│  │  advanced_stability          ── skip_expensive flag          │   │
│  │  train_robust_model          ── startup summary              │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Configuration Architecture

The `ExperimentConfig` dataclass has 13 sections totaling 98+ flags:

```
ExperimentConfig
├── data: DataConfig
│     ├── use_extended_hours: bool
│     ├── use_information_bars: bool
│     ├── information_bar_type: str
│     └── ... (data source toggles)
│
├── features: FeatureEngineeringConfig
│     ├── feature_groups: list
│     ├── feature_group_mode: str
│     └── ... (feature selection)
│
├── anti_overfit: AntiOverfitConfig
│     ├── use_synthetic_universes: bool
│     ├── use_cross_assets: bool
│     ├── use_breadth_streaks: bool
│     ├── use_economic_features: bool
│     ├── use_entropy_features: bool
│     ├── ... (43+ feature toggles)
│     └── use_tda_homology: bool
│
├── model: ModelConfig
│     ├── model_type: ModelType (44 options)
│     ├── use_regime_router: bool
│     ├── use_class_weights: bool
│     └── ... (model hyperparams)
│
├── cross_validation: CrossValidationConfig
│     ├── n_folds: int
│     ├── purge_days: int
│     ├── embargo_days: int
│     └── use_cpcv: bool
│
├── dim_reduction: DimReductionConfig
│     ├── method: str  ("ensemble_plus", "mutual_info", etc.)
│     ├── target_dims: int
│     └── ... (method-specific params)
│
├── risk: RiskConfig
│     ├── max_position_pct: float
│     ├── stop_loss_pct: float
│     └── daily_loss_limit: float
│
├── temporal_cascade: TemporalCascadeConfig
├── training_augmentation: TrainingAugmentationConfig
├── trading: TradingConfig
│     ├── use_conformal_sizer: bool
│     ├── use_cvar_sizer: bool
│     ├── use_thompson_selector: bool
│     └── ... (execution params)
│
├── resources: ResourceConfig          ◄── Auto-detected from hardware
│     ├── tier: str
│     ├── n_synthetic_universes: int
│     ├── n_jobs: int
│     ├── memory_pressure_threshold_mb: int
│     └── ... (scaling params)
│
└── metadata: Dict[str, Any]
```

---

## Storage and Persistence

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STORAGE ARCHITECTURE                              │
│                                                                     │
│  SQLite Registry (data/giga_trader.db)                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  experiments ── config hash, AUC, WMES, status, timestamps   │   │
│  │  models      ── model type, tier gates passed, file path     │   │
│  │  model_entries── individual model instances within ensemble   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Model Files (models/)                                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  production/                                                  │   │
│  │    spy_robust_models.joblib  ── trained model ensemble        │   │
│  │    conformal_sizer.joblib    ── calibrated position sizer     │   │
│  │    thompson_state.json       ── Thompson bandit state         │   │
│  │                                                               │   │
│  │  checkpoints/                                                 │   │
│  │    experiment_*.joblib       ── mid-training checkpoints      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Configuration Files (config/)                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  pipeline.yaml          ── 265-step pipeline definition       │   │
│  │  pipeline_status.yaml   ── completion tracking                │   │
│  │  edge_strategies.yaml   ── 5 core alpha strategies            │   │
│  │  risk_management.yaml   ── risk limits and circuit breakers   │   │
│  │  data_sources.yaml      ── API endpoints and fallback chains  │   │
│  │  monitoring.yaml        ── health check intervals             │   │
│  │  trading_modes.yaml     ── simulation/paper/live configs      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Data Files (data/)                                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  raw/        ── untransformed market data                     │   │
│  │  processed/  ── cleaned, merged data                          │   │
│  │  synthetic/  ── generated alternative SPY histories           │   │
│  │  cache/      ── feature computation cache                     │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Design Decisions

### Why Not PCA?

PCA assumes linear relationships — financial data has non-linear patterns. The **Ensemble+** method combines four complementary approaches:

| Method | What It Captures | Components |
|--------|-----------------|------------|
| Mutual Information | Any-type dependency with target | 20 features |
| Kernel PCA (RBF) | Non-linear manifold structure | 12 components |
| ICA | Statistically independent signals | 8 components |
| K-Medoids | Outlier-robust cluster distances | 10 clusters |
| **Total** | | **50 dimensions** |

### Why Regularization-First?

Complex models memorize noise; simple models find patterns. All models use:
- L1/L2 regularization (alpha 0.1-1.0)
- Tree depth capped at 5
- Early stopping (patience 5 rounds)
- Feature count penalty in WMES score

### Why Synthetic SPY Universes?

A model might overfit to the specific SPY price path. Synthetic universes test generalization:
- Filter extreme/middle component performers
- Volatility/momentum component filters
- Bootstrap random 70% subsets
- Bear market shifted universes

### Why Walk-Forward CV?

Standard k-fold leaks future information. Walk-forward with purging/embargo:
- **Purge**: 5-day gap between train and test to prevent leakage
- **Embargo**: 2-day buffer after test to prevent serial correlation
- **Expanding window**: Each fold adds more training data

---

*Last updated: 2026-02-28*

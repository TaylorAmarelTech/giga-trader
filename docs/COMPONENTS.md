# Giga Trader — Component Reference

> Detailed reference for all 27 phases, 210 modules, and integration patterns.

---

## Table of Contents

1. [Phase 01-02: Data Infrastructure](#phase-01-02-data-infrastructure)
2. [Phase 03-04: Synthetic Data](#phase-03-04-synthetic-data)
3. [Phase 05: Target Creation](#phase-05-target-creation)
4. [Phase 06-09: Feature Engineering](#phase-06-09-feature-engineering)
5. [Phase 10: Feature Processing](#phase-10-feature-processing)
6. [Phase 11: Cross-Validation](#phase-11-cross-validation)
7. [Phase 12: Model Training](#phase-12-model-training)
8. [Phase 13-14: Validation & Robustness](#phase-13-14-validation--robustness)
9. [Phase 15: Strategy & Signals](#phase-15-strategy--signals)
10. [Phase 16-17: Backtesting](#phase-16-17-backtesting)
11. [Phase 18: Persistence & Registry](#phase-18-persistence--registry)
12. [Phase 19: Paper Trading](#phase-19-paper-trading)
13. [Phase 20-21: Monitoring & Continuous Learning](#phase-20-21-monitoring--continuous-learning)
14. [Phase 22-27: Advanced & Live Trading](#phase-22-27-advanced--live-trading)
15. [Core Modules](#core-modules)
16. [Feature Groups Reference](#feature-groups-reference)
17. [Model Types Reference](#model-types-reference)

---

## Phase 01-02: Data Infrastructure

### Phase 01: Data Acquisition

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `alpaca_client.py` | Alpaca API wrapper for historical and real-time data | `AlpacaDataClient` |
| `data_manager.py` | Data download orchestration, caching, merge logic | `DataManager` |
| `historical_constituents.py` | Point-in-time S&P 500 constituent lists (survivorship bias prevention) | `HistoricalConstituents` |

**Data sources**: Alpaca (primary, 1-min bars with extended hours), yfinance (backup, daily), FRED (economic indicators).

### Phase 02: Preprocessing

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `ohlc_validator.py` | Validates OHLC relationships (High >= max(O,C), Low <= min(O,C)) | `OHLCValidator` |
| `information_bars.py` | Converts time bars to dollar/volume/tick bars (AFML Ch.2) | `InformationBarGenerator` |
| `bar_resampler.py` | Resamples minute bars to custom frequencies | `BarResampler` |

**OHLCValidator** is called at 4 integration points:
- `anti_overfit_integration.py` step 0 (before features)
- `signal_generator.py` (inference time)
- `experiment_runner.py` (per experiment)
- `train_robust_model.py` (startup)

---

## Phase 03-04: Synthetic Data

### Phase 03: Synthetic Data Generation

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `bear_universes.py` | Generates 8 bear-market-shifted SPY alternatives | `BearUniverseGenerator` |
| `multiscale_bootstrap.py` | 7 bootstrap scenarios across multiple timescales | `MultiscaleBootstrap` |

**Bear Universe Types**: Mean-shifted, vol-amplified, vol-dampened, drawdown-injected, trend-reversed, correlation-stressed, tail-extended, regime-mixed.

**Synthetic Weight**: Real data gets 70% weight, synthetic gets 30% (configurable). The number of universes scales with system resources (3-30).

---

## Phase 05: Target Creation

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `target_generator.py` | Binary up/down labels + soft targets | `TargetGenerator` |
| `cusum_filter.py` | CUSUM event-driven sampling (AFML Ch.5) | `CUSUMFilter` |
| `triple_barrier.py` | Triple barrier labeling (profit/stop/time) | `TripleBarrierLabeler` |

**Target types**:
- `target_up`: Binary (1 if close > threshold above open)
- `timing`: Binary (1 if daily low occurs before daily high)
- `soft_label`: Sigmoid transform of return, 0.0-1.0 (Edge 4)
- `triple_barrier`: Path-dependent profit/stop/time labels

---

## Phase 06-09: Feature Engineering

### Phase 06: Intraday Features

Extended hours signals (Edge 2): premarket return, range, direction, momentum, VWAP deviation. Afterhours lagged features with 1/2/3/5-day lookback.

### Phase 07: Daily Features

Standard technical indicators: RSI, MACD, Bollinger Bands, ATR, OBV, momentum windows (5/10/20/40 days).

### Phase 08: Feature Breadth (48 Modules)

The largest phase, containing 40+ individual feature computation modules. Each module follows the same pattern:

```python
class SomeFeatures:
    def __init__(self, lookback: int = 60, ...):
        self.lookback = lookback

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # Adds columns with a consistent prefix (e.g., "ent_", "hurst_")
        return df_with_features
```

#### Economic & Macro Features

| Module | Prefix | Features | Description |
|--------|--------|----------|-------------|
| `economic_features.py` | `econ_` | ~120 | VIX term structure, yield curve, gold, bonds, commodities, FRED |
| `fear_greed_features.py` | `fg_` | 5 | CNN Fear & Greed Index signals |

#### Sentiment & Alternative Data

| Module | Prefix | Features | Description |
|--------|--------|----------|-------------|
| `crypto_sentiment_features.py` | `csent_` | 8 | Crypto market sentiment signals |
| `reddit_sentiment_features.py` | `rsent_` | 6 | Reddit WSB/investing sentiment |
| `finnhub_social_features.py` | `fsoc_` | 6 | Finnhub social sentiment metrics |
| `congressional_features.py` | `cong_` | 8 | Congressional trading signals |
| `insider_aggregate_features.py` | `insider_agg_` | 10 | Insider buying/selling aggregate |

#### Market Microstructure

| Module | Prefix | Features | Description |
|--------|--------|----------|-------------|
| `vpin_features.py` | `vpin_` | 6 | Volume-Synchronized PIN (toxicity) |
| `dark_pool_features.py` | `dp_` | 8 | Dark pool volume ratios |
| `block_structure_features.py` | `blk_` | 54 | Block trade analysis |
| `amihud_features.py` | `liq_` | 6 | Amihud illiquidity ratio |
| `options_features.py` | `opt_` | 15 | Put/call ratio, IV, GEX |
| `gamma_exposure_features.py` | `gex_` | 8 | Net gamma exposure signals |

#### Statistical & Information Theory

| Module | Prefix | Features | Description |
|--------|--------|----------|-------------|
| `entropy_features.py` | `ent_` | 6 | Sample, spectral, approximate entropy |
| `hurst_features.py` | `hurst_` | 4 | Hurst exponent (mean reversion/trend) |
| `nmi_features.py` | `nmi_` | 4 | Normalized mutual information |
| `transfer_entropy_features.py` | `te_` | 6 | Causal information flow |
| `multiscale_entropy_features.py` | `mse_` | 3 | Multi-scale entropy complexity |
| `l_moments_features.py` | `lmom_` | 4 | L-CV, L-skewness, L-kurtosis |

#### Regime & Structural

| Module | Prefix | Features | Description |
|--------|--------|----------|-------------|
| `hmm_features.py` | `hmm_` | 8 | Hidden Markov Model regime states |
| `changepoint_features.py` | `chg_` | 6 | Structural break detection |
| `drift_features.py` | `drift_` | 4 | Distribution drift monitoring |
| `wasserstein_regime.py` | `wreg_` | 8 | EMD-based regime change |
| `market_structure_features.py` | `mstr_` | 18 | Compression, squeeze, attractors |

#### Wavelet & Signal Processing

| Module | Prefix | Features | Description |
|--------|--------|----------|-------------|
| `wavelet_features.py` | `wvt_` | 8 | Wavelet decomposition features |
| `wavelet_scattering_features.py` | `wscat_` | 12 | 2-layer wavelet scattering |
| `sax_features.py` | `sax_` | 6 | Symbolic Aggregate Approximation |
| `har_rv_features.py` | `harv_` | 4 | Heterogeneous Autoregressive RV |

#### Advanced Mathematical

| Module | Prefix | Features | Description |
|--------|--------|----------|-------------|
| `path_signature_features.py` | `psig_` | 17 | Iterated integrals of price paths |
| `mfdfa_features.py` | `mfdfa_` | 6 | Multifractal Detrended FA |
| `rqa_features.py` | `rqa_` | 6 | Recurrence Quantification Analysis |
| `copula_features.py` | `copula_` | 6 | Copula dependence structures |
| `network_features.py` | `net_` | 6 | Correlation network topology |
| `tda_features.py` | `tda_` | 5 | Topological Data Analysis (optional) |
| `rv_signature_features.py` | `rvsp_` | 3 | Realized volatility signature plot |

#### Flow & Positioning

| Module | Prefix | Features | Description |
|--------|--------|----------|-------------|
| `etf_flow_features.py` | `etf_flow_` | 8 | ETF fund flow signals |
| `futures_basis_features.py` | `fut_` | 6 | Futures basis and roll signals |
| `intraday_momentum_features.py` | `imom_` | 8 | Intraday momentum patterns |
| `absorption_ratio_features.py` | `absr_` | 4 | Market fragility indicator |

#### Event & Calendar

| Module | Prefix | Features | Description |
|--------|--------|----------|-------------|
| `event_recency_features.py` | `dts_` | 88 | Days-to/since events (FOMC, expiry, etc.) |

### Phase 09: Calendar Features

| Module | Prefix | Features | Description |
|--------|--------|----------|-------------|
| `calendar_features.py` | `cal_` | 29 | Day of week, month, quarter, FOMC proximity |
| `feature_researcher.py` | — | — | Automated feature idea generation |

---

## Phase 10: Feature Processing

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `group_aware_processor.py` | Group-based dimensionality reduction with multiple methods | `GroupAwareProcessor` |
| `feature_neutralizer.py` | Remove market/sector exposure from features | `FeatureNeutralizer` |
| `interaction_discovery.py` | Discover pairwise feature interactions | `InteractionDiscovery` |

### GroupAwareProcessor

The central feature processing engine. Handles:

- **43 feature groups** organized by prefix
- **7 dimensionality reduction methods**: Mutual Info, Kernel PCA, ICA, UMAP, K-Medoids, Agglomeration, Ensemble+
- **Adaptive parameters**: n_jobs and Nystroem threshold scale with system resources
- **Ensemble+ (default)**: Combines MI (20) + KernelPCA (12) + ICA (8) + K-Medoids (10) = 50 dimensions

---

## Phase 11: Cross-Validation

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `leak_proof_cv.py` | Time-series CV with purging and embargo | `LeakProofCV` |
| `cpcv.py` | Combinatorial Purged Cross-Validation (AFML) | `CombinatorialPurgedCV` |
| `walk_forward.py` | Expanding-window walk-forward validation | `WalkForwardValidator` |

**Walk-Forward parameters** (scaled by resource tier):
- Windows: 3-10 (LOW: 3, ULTRA: 10)
- Per-window AUC threshold: 0.53 (normal), 0.48 (crisis regime)
- Variance threshold: 0.07

---

## Phase 12: Model Training

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `model_types.py` | 44 model type definitions and factory | `create_model()` |
| `stacking_ensemble.py` | Multi-layer stacking with diverse base learners | `StackingEnsemble` |
| `quantile_forest_wrapper.py` | Quantile regression forest for uncertainty | `QuantileForestWrapper` |

### Model Types (44 total)

**Linear Models**: L1_strong, L1_moderate, L1_weak, L2_strong, L2_moderate, L2_weak, elastic_net_balanced, elastic_net_l1_heavy, elastic_net_l2_heavy

**Tree Models**: tree_shallow (depth 3), tree_moderate (depth 4), tree_deep (depth 5 — maximum allowed)

**Gradient Boosting**: GB variants with different hyperparameters, XGBoost, LightGBM, CatBoost

**Ensembles**: diverse_ensemble, stacking_ensemble, quantile_forest

**All models**: Max depth capped at 5 (Edge 1: regularization-first philosophy)

---

## Phase 13-14: Validation & Robustness

### Phase 13: Anti-Overfit Integration

`anti_overfit_integration.py` — The central orchestrator with 50 feature engineering steps. Accepts a `resource_config` parameter for memory-aware scaling:

- **GC checkpoints**: Inserted after steps 8, 18, 31, 40, 50 (only on LOW/MEDIUM tiers)
- **Synthetic universes**: Scaled by resource tier (3-30)
- Each step is gated by a `use_*` config flag (all independently toggleable)

### Phase 14: Robustness Testing

| Module | Purpose | Key Metric |
|--------|---------|------------|
| `advanced_stability.py` | 22-method stability suite | composite score 0-1 |
| `feature_importance_stability.py` | Feature importance consistency across folds | FI stability score |
| `knockoff_gate.py` | Knockoff filter for false discovery control | FDR-controlled features |
| `label_noise_test.py` | Model performance under label corruption | noise tolerance score |
| `wasserstein_regime.py` | EMD-based regime change detection | regime distance features |

### 3-Tier Validation Gate

```
Tier 1: AUC > 0.56, AUC < 0.85, gap < 0.10, WMES >= 0.45, walk_forward_passed
Tier 2: stability_score >= 0.60 (multi-radius HP perturbation)
Tier 3: fragility < 0.40, AUC >= 0.58, suite_composite >= 0.45
```

---

## Phase 15: Strategy & Signals

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `signal_generator.py` | Combines models into trading signals | `SignalGenerator` |
| `meta_labeler.py` | Secondary model filters primary predictions | `MetaLabeler` |
| `conformal_sizer.py` | Conformal prediction-based position sizing | `ConformalSizer` |
| `cvar_position_sizer.py` | CVaR (Expected Shortfall) tail-risk sizing | `CVaRPositionSizer` |
| `thompson_selector.py` | Thompson Sampling for adaptive model weighting | `ThompsonSamplingSelector` |
| `bayesian_averaging.py` | Bayesian Model Averaging for ensemble | `BayesianModelAverager` |
| `dynamic_weights.py` | Rolling-AUC dynamic ensemble weighting | `DynamicEnsembleWeighter` |
| `isotonic_calibrator.py` | Probability calibration via isotonic regression | `IsotonicCalibrator` |
| `ensemble_disagreement.py` | Model disagreement as confidence signal | `EnsembleDisagreement` |
| `regime_router.py` | Route predictions through regime-specific models | `RegimeRouter` |

### Signal Generation Flow

```
Raw model predictions
    ├── Thompson-weighted ensemble (adaptive model selection)
    ├── BMA-weighted averaging (Bayesian posterior weights)
    ├── Dynamic reweighting (rolling AUC performance)
    ├── Meta-labeler filter (secondary classifier)
    ├── Feature drift check (distribution shift alert)
    ├── Conformal position sizing (coverage-guaranteed)
    └── CVaR tail-risk scaling (Expected Shortfall at 5%)
```

**Signal thresholds**:
- STRONG_BUY: net_sentiment >= 0.7, expectation >= 0.5
- BUY: net_sentiment >= 0.4, expectation >= 0.2
- SELL: net_sentiment <= -0.4, expectation <= -0.2
- STRONG_SELL: net_sentiment <= -0.7, expectation <= -0.5

---

## Phase 16-17: Backtesting

| Module | Purpose |
|--------|---------|
| `backtest_engine.py` | Historical simulation with realistic execution modeling |
| `portfolio_tracker.py` | Track positions, PnL, drawdown, exposure |
| `monte_carlo.py` | Stochastic return simulation for risk estimation |

---

## Phase 18: Persistence & Registry

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `registry_db.py` | SQLite database for experiment/model tracking | `RegistryDB` |
| `grid_search_generator.py` | Generate configuration grids with bundles | `GridSearchGenerator` |
| `registry_configs.py` | Pre-defined configuration templates | — |
| `registry_enums.py` | Enum definitions (ModelType, FeatureGroup, etc.) | `ModelType`, `FEATURE_GROUPS` |

### Grid Bundles

| Bundle | Level | Focus |
|--------|-------|-------|
| MINIMAL | 160 configs | Quick validation |
| STANDARD | 69,120 configs | Production search |
| COMPREHENSIVE | ~50K configs | Deep exploration |
| FULL | Sampled | All flags including experimental |
| sentiment | STANDARD+ | Sentiment-focused features |
| social | STANDARD+ | Social media features |
| flow | STANDARD+ | Fund flow features |
| cvar_sizing | FULL | CVaR position sizing |
| thompson_selector | FULL | Thompson sampling |

---

## Phase 19: Paper Trading

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `trading_bot.py` | Main trading execution loop | `TradingBot` |
| `signal_generator.py` | Real-time signal generation | `LiveSignalGenerator` |
| `trading_gates.py` | 5-gate system for trade approval | `TradingGates` |
| `aaii_gate.py` | AAII Investor Sentiment Survey gate | `AAIIGate` |
| `cot_gate.py` | CFTC Commitments of Traders gate | `COTGate` |
| `edgar_gate.py` | SEC EDGAR filing activity gate | `EDGARGate` |
| `insider_gate.py` | Insider trading activity gate | `InsiderGate` |
| `circuit_breaker.py` | Daily loss limit and consecutive loss tracking | `CircuitBreaker` |
| `staleness_checker.py` | Model file age verification | `StalenessChecker` |

### Trading Gate System

Each gate independently approves or vetoes a trade. All 5 must pass for execution:

```
SIGNAL GENERATED
    │
    ├── AAII Gate: Retail sentiment not at extreme?
    ├── COT Gate: Institutional positioning aligned?
    ├── EDGAR Gate: No unusual filing activity?
    ├── Insider Gate: No abnormal insider selling?
    └── Staleness Gate: Model trained within N days?
    │
    ALL PASS → Execute trade
    ANY FAIL → Skip trade (logged with reason)
```

### Circuit Breaker

Tracks and enforces:
- **Daily loss limit**: 2% (configurable)
- **Consecutive losses**: Max 5 before pause
- **Max drawdown**: 10% from peak equity
- **State persistence**: JSON file survives restarts

---

## Phase 20-21: Monitoring & Continuous Learning

### Phase 20: Monitoring

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `health_checker.py` | System health monitoring (60-second interval) | `HealthChecker` |
| `feature_drift_monitor.py` | Detects feature distribution shifts | `FeatureDriftMonitor` |
| `dashboard_server.py` | Flask-based web dashboard with SSE updates | `DashboardServer` |
| `web_monitor.py` | Real-time web monitoring interface | `WebMonitor` |

**FeatureDriftMonitor**: Self-baselines on first multi-row call when training data is unavailable. Uses KS-test and PSI for drift detection.

### Phase 21: Continuous Learning

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `experiment_runner.py` | Automated experiment execution with campaign support | `ExperimentRunner` |
| `online_updater.py` | Daily post-close model updates (4:15 PM buffer) | `OnlineUpdater` |
| `experiment_tracking.py` | Experiment metadata logging | `ExperimentTracker` |

**ExperimentRunner** integration with resources:
- Feature cache size: 1-10 entries (scaled by tier)
- Memory pressure threshold: 1.5-32 GB
- Stability method parameters: scaled N for CPCV, stability selection, Rashomon

---

## Phase 22-27: Advanced & Live Trading

### Phase 23: Analytics

| Module | Purpose |
|--------|---------|
| `thick_weave_search.py` | Multi-configuration search across grid bundles |
| `advanced_analytics.py` | Performance analytics and attribution |

### Phase 25: Risk Management

| Module | Purpose |
|--------|---------|
| `model_selector.py` | Dynamic model selection based on regime and performance |
| `risk_manager.py` | Position-level and portfolio-level risk controls |

### Phase 26: Temporal Models

| Module | Purpose |
|--------|---------|
| `temporal_cascade_models.py` | Multi-resolution temporal models |
| `temporal_cascade_trainer.py` | Training for temporal cascades |

---

## Core Modules

### `src/core/`

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `base.py` | Base class for all pipeline phases | `PhaseRunner` |
| `state_manager.py` | Persistent state management | `StateManager` |
| `registry_db.py` | SQLite model/experiment registry | `RegistryDB`, `get_registry_db()` |
| `system_resources.py` | Hardware detection and adaptive scaling | `SystemResources`, `ResourceConfig`, `ResourceTier` |

### `src/mega_ensemble/`

5-layer ensemble pipeline (untouched, standalone):

```
Layer 1: Individual model predictions
Layer 2: Diversity selection (maximize disagreement)
Layer 3: Config interpolation (blend configs)
Layer 4: Registry-based ensemble (historical best)
Layer 5: Final meta-ensemble
```

### `src/supervision/`

Trading supervision service (8 modules):

```
TradingSupervisionService (orchestrator)
├── AlertSystem (notifications)
├── CircuitBreakerEnforcer (loss limits)
├── FeatureValidator (input validation)
├── ForceCloseManager (emergency exits)
├── ModelHealthMonitor (model degradation)
└── PositionReconciler (broker sync)
```

---

## Feature Groups Reference

All 43 feature groups with their config flags:

| # | Group | Prefix | Config Flag | Default |
|---|-------|--------|------------|---------|
| 1 | technical | various | — | Always on |
| 2 | extended_hours | `pm_`, `ah_` | — | Always on |
| 3 | breadth_streaks | `pct_`, `wtd_` | `use_breadth_streaks` | True |
| 4 | cross_assets | `cross_` | `use_cross_assets` | True |
| 5 | economic | `econ_` | `use_economic_features` | True |
| 6 | calendar | `cal_` | `use_calendar_features` | True |
| 7 | sentiment | `sent_` | `use_sentiment_features` | True |
| 8 | fear_greed | `fg_` | `use_fear_greed` | False |
| 9 | options | `opt_` | `use_options_features` | False |
| 10 | dark_pool | `dp_` | `use_dark_pool` | False |
| 11 | block_structure | `blk_` | `use_block_structure` | False |
| 12 | event_recency | `dts_` | `use_event_recency` | False |
| 13 | crypto_sentiment | `csent_` | `use_crypto_sentiment` | False |
| 14 | reddit_sentiment | `rsent_` | `use_reddit_sentiment` | False |
| 15 | finnhub_social | `fsoc_` | `use_finnhub_social` | False |
| 16 | congressional | `cong_` | `use_congressional` | False |
| 17 | insider_aggregate | `insider_agg_` | `use_insider_aggregate` | False |
| 18 | etf_flow | `etf_flow_` | `use_etf_flow` | False |
| 19 | futures_basis | `fut_` | `use_futures_basis` | False |
| 20 | gamma_exposure | `gex_` | `use_gamma_exposure` | False |
| 21 | amihud | `liq_` | `use_amihud_features` | True |
| 22 | entropy | `ent_` | `use_entropy_features` | True |
| 23 | hurst | `hurst_` | `use_hurst_features` | True |
| 24 | nmi | `nmi_` | `use_nmi_features` | True |
| 25 | absorption_ratio | `absr_` | `use_absorption_ratio` | True |
| 26 | drift | `drift_` | `use_drift_features` | True |
| 27 | changepoint | `chg_` | `use_changepoint` | True |
| 28 | hmm | `hmm_` | `use_hmm_features` | True |
| 29 | vpin | `vpin_` | `use_vpin` | True |
| 30 | intraday_momentum | `imom_` | `use_intraday_momentum` | True |
| 31 | range_vol | `rvol_` | `use_range_vol` | True |
| 32 | wavelet | `wvt_` | `use_wavelet_features` | True |
| 33 | sax | `sax_` | `use_sax_features` | True |
| 34 | transfer_entropy | `te_` | `use_transfer_entropy` | True |
| 35 | mfdfa | `mfdfa_` | `use_mfdfa` | True |
| 36 | rqa | `rqa_` | `use_rqa` | True |
| 37 | copula | `copula_` | `use_copula` | True |
| 38 | network | `net_` | `use_network_features` | True |
| 39 | path_signature | `psig_` | `use_path_signatures` | True |
| 40 | wavelet_scattering | `wscat_` | `use_wavelet_scattering` | True |
| 41 | wasserstein_regime | `wreg_` | `use_wasserstein_regime` | True |
| 42 | market_structure | `mstr_` | `use_market_structure` | True |
| 43 | time_series_model | `tsm_` | `use_time_series_models` | False |

**Conditional (require external packages)**:
- `har_rv` (`harv_`, default True)
- `l_moments` (`lmom_`, default True)
- `multiscale_entropy` (`mse_`, default True)
- `rv_signature` (`rvsp_`, default False)
- `tda_homology` (`tda_`, default False, requires giotto-tda)

---

## Model Types Reference

All 44 model types available in the `ModelType` enum:

### Linear Models (9)
| Type | Regularization | Default C |
|------|---------------|-----------|
| L1_STRONG | Lasso | 0.001 |
| L1_MODERATE | Lasso | 0.01 |
| L1_WEAK | Lasso | 0.1 |
| L2_STRONG | Ridge | 0.01 |
| L2_MODERATE | Ridge | 0.1 |
| L2_WEAK | Ridge | 1.0 |
| ELASTIC_NET_BALANCED | Elastic | l1_ratio=0.5 |
| ELASTIC_NET_L1_HEAVY | Elastic | l1_ratio=0.7 |
| ELASTIC_NET_L2_HEAVY | Elastic | l1_ratio=0.3 |

### Tree Models (3)
| Type | Max Depth |
|------|-----------|
| TREE_SHALLOW | 3 |
| TREE_MODERATE | 4 |
| TREE_DEEP | 5 (maximum) |

### Gradient Boosting (6+)
| Type | Framework | Notes |
|------|-----------|-------|
| GRADIENT_BOOSTING | sklearn | Default GB |
| XGBOOST | XGBoost | Auto GPU detection |
| LIGHTGBM | LightGBM | Fast training |
| CATBOOST | CatBoost | Categorical native |
| Various GB configs | sklearn | Different HP presets |

### Ensembles (3+)
| Type | Description |
|------|-------------|
| DIVERSE_ENSEMBLE | Mix of L1/L2/EN/tree models |
| STACKING_ENSEMBLE | Multi-layer with diverse bases |
| QUANTILE_FOREST | Uncertainty quantification |

---

## Integration Pattern

Every feature module follows the same integration pattern across 6 touchpoints:

```
1. CONFIG FLAG        → AntiOverfitConfig.use_<feature_name> = True/False
2. FEATURE MODULE     → src/phase_08_features_breadth/<name>_features.py
3. INTEGRATION STEP   → anti_overfit_integration.py step N
4. FEATURE GROUP      → registry_enums.py FEATURE_GROUPS["<name>"]
5. CALL SITES         → train_robust_model.py, signal_generator.py, experiment_runner.py
6. EXPERIMENT TRACKING → experiment_tracking.py source_flags + config_hash
```

Additional touchpoints for inference modules:
- `signal_generator.py` → source health prefix
- `trading_bot.py` → startup gate checks

---

*Last updated: 2026-02-28*

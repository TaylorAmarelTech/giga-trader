# Giga Trader - Project Instructions

> Master configuration for Claude Code and Moltbot agents working on this ML trading pipeline.

---

## Environment

| Property | Value |
|----------|-------|
| **Platform** | Windows 11 |
| **Project Home** | `C:\Users\amare\OneDrive\Documents\giga_trader` |
| **User Home** | `C:\Users\amare` |
| **Node.js** | v24.13.0 |
| **Python** | **3.11 or 3.12 required** (3.13+ not supported - package compatibility) |
| **Package Manager** | pnpm (Node), pip/uv (Python) |

### Tool Locations

| Tool | Path |
|------|------|
| Moltbot | `C:\Users\amare\moltbot` |
| Claude Code Harness | `C:\Users\amare\claude-plugins\claude-code-harness` |
| Moltbot Config | `C:\Users\amare\.moltbot\moltbot.json` |

---

## Project Overview

**Giga Trader** is a comprehensive ML trading system with two pipelines:

### 1. SPY Swing Trading Pipeline (265 steps, 27 phases)
- 200+ engineered features
- 130+ synthetic data scenarios
- 16,000+ grid search configurations
- 500+ trained models

### 2. SPY Component Sentiment Analyzer (10 stages)
- Multi-source news collection (Alpha Vantage, Finnhub, NewsAPI)
- LLM-based sentiment analysis (Mistral + fallbacks)
- Market-cap weighted aggregation
- Autonomous signal generation

### Pipeline Status: 217 steps | 148 done (68%) | 13 partial (6%) | 56 missing (26%)

---

## Development Standards (MANDATORY)

> **Every change MUST follow these rules. No exceptions.**

### Code Modification Rules

```
EVERY EDIT MUST:
1. Be verified to work (run the code after changes)
2. Not break any existing tests
3. Be fully integrated into the existing pipeline
4. Use real data from Alpaca API (not made-up data)

NEVER:
- Create side branches or shortcuts
- Use placeholder or synthetic mock data for training
- Skip verification after edits
- Leave code in a broken state
- Implement partial features that don't integrate

ALWAYS:
- Run the modified script after changes
- Test with actual Alpaca API data
- Ensure models save/load correctly
- Verify end-to-end pipeline works
- Check syntax before running (Python AST parse)
```

### Testing Protocol

```bash
# After every code edit, verify:
1. Syntax check: python -c "import ast; ast.parse(open('file.py', encoding='utf-8').read())"
2. Import check: python -c "import src.module_name"
3. Full run: python src/train_robust_model.py
4. Model verification: Check models/production/ for saved models
```

---

## Critical Rules

### 1. Data Integrity

```
NEVER:
- Use future data to predict past (look-ahead bias)
- Skip data validation steps
- Ignore missing value handling
- Mix timezone-unaware timestamps
- Use current S&P 500 list for historical analysis (survivorship bias)

ALWAYS:
- Validate OHLC relationships: High >= max(O,C), Low <= min(O,C)
- Use Eastern Time for all market data
- Purge and embargo cross-validation folds
- Store raw data before transformations
- Use point-in-time constituent lists
```

### 2. Feature Engineering

```
NEVER:
- Include target-leaking features
- Use features unavailable at prediction time
- Skip feature scaling on train-only data
- Keep highly correlated features (>0.95)
- Calculate features using future data

ALWAYS:
- Fit scalers on training data only
- Document feature calculation formulas
- Test feature temporal stability
- Use lag features for time-series
- Validate feature availability at prediction time
```

### 3. Model Training

```
NEVER:
- Train on test data
- Skip cross-validation
- Ignore class imbalance
- Use default hyperparameters without tuning
- Deploy models with train-test gap > 0.10

ALWAYS:
- Use time-series aware splitting
- Apply purging (5 days) and embargo (2 days)
- Handle class imbalance (weights/SMOTE)
- Checkpoint every 30 minutes
- Validate on 100+ synthetic SPY scenarios
```

### 4. Validation

```
NEVER:
- Trust in-sample metrics
- Skip robustness testing
- Deploy without permutation test (p < 0.05)
- Ignore train-test gap > 0.10

ALWAYS:
- Validate on truly unseen data
- Run 10+ robustness tests
- Require test AUC > 0.58
- Document all validation results
- Test on different market regimes
```

### 5. Risk Management

```
NEVER:
- Risk more than 25% in one position
- Exceed 2% daily loss limit
- Trade during non-market hours (unless extended hours strategy)
- Skip risk checks before entry
- Ignore max drawdown limits (10%)

ALWAYS:
- Use stop losses (default: -0.2%, strong signals: -0.3%)
- Exit by market close (4:00 PM) for intraday
- Track consecutive losses (limit: 5)
- Implement trailing stops
- Reduce position size in high volatility (>10%)
```

### 6. Sentiment Analysis

```
NEVER:
- Use sentiment without source weighting
- Skip article deduplication
- Ignore LLM confidence scores
- Generate signals more than 1x per hour

ALWAYS:
- Apply source weights: News (1.5x), Reddit (0.8x), Twitter (0.7x)
- Deduplicate using content hash (MD5 of headline+content)
- Use LLM fallback chain: Mistral → OpenAI → Anthropic
- Wait 4+ hours between signals (unless significant change ≥0.2)
- Apply dispersion penalty to thresholds
```

### 7. Signal Generation

```
NEVER:
- Generate signals without minimum component coverage
- Ignore breadth confirmation
- Skip confidence calculation
- Trade against momentum

ALWAYS:
- Require net_sentiment thresholds:
  - STRONG_BUY: ≥0.7 AND expectation ≥0.5
  - BUY: ≥0.4 AND expectation ≥0.2
  - SELL: ≤-0.4 AND expectation ≤-0.2
  - STRONG_SELL: ≤-0.7 AND expectation ≤-0.5
- Calculate strength: 70% net_sentiment + 30% expectation
- Verify breadth agreement (majority components aligned)
```

### 8. Autonomous Operation

```
NEVER:
- Run without health checks
- Skip error recovery
- Ignore rate limits
- Process tasks without priority queue

ALWAYS:
- Health check every 60 seconds
- Signal generation every 3600 seconds (1 hour)
- Use priority queue: CRITICAL > HIGH > NORMAL > LOW
- Retry up to 3 times with exponential backoff
- Log all decisions and errors
```

---

## Edge-Generating Strategies (CRITICAL)

> **These 5 strategies represent the core alpha-generating philosophy. Reference: `config/edge_strategies.yaml`**

### EDGE 1: Regularization-First Model Philosophy

```
PRINCIPLE: Features that survive heavy regularization are TRUE signals.
           Complex models memorize noise; simple models find patterns.

MODEL PRIORITY (in order):
  1. Logistic Regression (L1) - Feature discovery
  2. Logistic Regression (L2) - Baseline predictions
  3. Elastic Net - Production candidate
  4. Shallow Trees (max_depth ≤ 5) - Non-linear patterns

NEVER USE:
  - Deep neural networks
  - Decision trees with max_depth > 5
  - Unregularized models
  - Complex ensemble stacking

REGULARIZATION DEFAULTS:
  - L1 (Lasso): alpha = 0.1
  - L2 (Ridge): alpha = 1.0
  - Elastic Net: alpha = 0.5, l1_ratio = 0.5
  - Tree max_depth: 3-5 (NEVER > 5)
  - Early stopping: patience = 5 rounds
```

### EDGE 2: Extended Hours Signal Extraction

```
PRINCIPLE: Premarket and afterhours contain unique predictive signals
           that most algorithms miss.

PREMARKET TODAY (4AM-9:30AM ET):
  - pm_return_today: Current PM % change
  - pm_range_today: PM high-low range
  - pm_direction_today: PM trend direction
  - pm_momentum_30min: Last 30-min PM momentum
  - pm_momentum_60min: Last 60-min PM momentum
  - pm_vwap_deviation: Distance from PM VWAP

PREMARKET LAGGED (prior days):
  - Lags: 1, 2, 3, 5 days
  - Features: return, range, direction, volume_ratio

AFTERHOURS LAGGED (prior days):
  - Lags: 1, 2, 3, 5 days
  - Features: return, range, direction, volume_ratio

COMBINED TIME SERIES:
  - overnight_return: pm_close_today - ah_close_yesterday
  - pm_only_series: Pure premarket data across days
  - ah_only_series: Pure afterhours data across days
  - extended_combined_series: PM + AH merged

ALWAYS:
  - Apply higher slippage multipliers in extended hours (2-2.5x)
  - Use limit orders only (no market orders)
  - Apply min_volume filters (30% of avg for PM, 20% for AH)
```

### EDGE 3: Intraday Tradable Opportunities

```
PRINCIPLE: Specific intraday patterns offer tradable opportunities
           with configurable detection windows.

MORNING DIP PATTERN:
  Window: 10:15 - 12:30 ET
  Trigger: Price drops ≥ 0.3% from open
  Grid Search: Time windows × swing thresholds

AFTERNOON SWING PATTERN:
  Window: 12:30 - 15:30 ET
  Trigger: Swing of ≥ 0.4% from intraday low
  Grid Search: Time windows × swing thresholds

CONFIGURABLE PARAMETERS (grid search all):
  - Time Window Start: 10:00, 10:15, 10:30, 11:00
  - Time Window End: 12:00, 12:30, 13:00, 14:00
  - Minimum Swing %: 0.2%, 0.3%, 0.4%, 0.5%
  - Entry Delay Minutes: 0, 5, 10, 15
  - Pattern Confirmation: Volume spike, RSI level

DETECTION RULES:
  - Require minimum volume threshold
  - Confirm with technical indicator
  - Apply risk-adjusted position sizing
```

### EDGE 4: Confidence Soft Targets

```
PRINCIPLE: Binary labels (0/1) cause overfitting.
           Soft targets (0.0-1.0) provide better gradients.

GENERATION METHODS:

Sigmoid Transform:
  soft_label = 1 / (1 + exp(-k * (return - threshold)))
  - k: steepness (default 50)
  - threshold: where 0.5 probability occurs

Label Smoothing:
  soft_label = (1 - epsilon) * hard_label + epsilon / 2
  - epsilon: smoothing factor (default 0.1)
  - If hard_label = 1: soft_label = 0.95
  - If hard_label = 0: soft_label = 0.05

Return Percentile:
  soft_label = percentile_rank(return) / 100
  - Based on rolling window percentile

TRAINING RULES:
  - Use BCEWithLogitsLoss or cross-entropy with soft targets
  - Combine with heavy regularization
  - Apply class weights for imbalance

BENEFITS:
  - Reduced overfitting to edge cases
  - Better probability calibration
  - Smoother decision boundaries
```

### EDGE 5: Batch Position Scaling

```
PRINCIPLE: Never enter or exit positions all-at-once.
           Scale in/out in batches to reduce timing risk.

ENTRY SCALING (3 batches):
  Batch 1: 40% immediately on initial signal
  Batch 2: 35% on confirmation (price continues favorable)
  Batch 3: 25% on momentum (5-min trend aligned)

  Confirmation Rules:
    - Price moves favorably by ≥ 0.1% from entry
    - Wait minimum 5 minutes between batches
    - Max time to full position: 30 minutes
    - Cancel remaining batches if signal reverses

EXIT SCALING (3 batches):
  Batch 1: 30% at first target (+0.2%)
  Batch 2: 40% at second target (+0.4%)
  Batch 3: 30% at time-based exit or stop loss

  Trailing Stop Rules:
    - Activate after +0.2% gain
    - Trail by 0.1% (tight) or 0.15% (normal)
    - Move in 0.05% increments

BENEFITS:
  - Reduces timing risk
  - Allows price discovery
  - Better average entry/exit prices
  - Avoids slippage on large orders
```

---

## Advanced Dimensionality Reduction (CRITICAL)

> **PCA is inadequate for complex financial data interrelations. Use these methods instead.**

### Why Not PCA?

```
PCA LIMITATIONS:
- Assumes linear relationships (finance has non-linear patterns)
- Sensitive to outliers (common in financial data)
- Maximizes variance, not predictive power
- Cannot capture interaction effects
- Loses interpretability of original features
```

### Recommended Methods (in `train_robust_model.py`)

#### 1. UMAP (Uniform Manifold Approximation)

```
USE WHEN: You need to preserve both local AND global structure
STRENGTHS:
  - Handles non-linear manifolds
  - Preserves cluster structure
  - Fast computation
  - Better than t-SNE for inference

CONFIG:
  dim_reduction_method: "umap"
  umap_n_components: 20
  umap_n_neighbors: 15  # Balance local/global
  umap_min_dist: 0.1    # Cluster tightness
```

#### 2. Kernel PCA (Non-linear PCA)

```
USE WHEN: You suspect non-linear relationships between features
STRENGTHS:
  - RBF kernel captures smooth manifolds
  - Polynomial kernel finds interaction effects
  - More stable than UMAP
  - Better for regression targets

CONFIG:
  dim_reduction_method: "kernel_pca"
  kpca_n_components: 25
  kpca_kernel: "rbf"    # or "poly", "sigmoid"
  kpca_gamma: 0.01      # Kernel width (lower = smoother)
```

#### 3. ICA (Independent Component Analysis)

```
USE WHEN: You want to separate mixed signals (market, sector, stock-specific)
STRENGTHS:
  - Finds statistically INDEPENDENT components
  - Captures higher-order statistics
  - Good for separating signal sources
  - Robust to non-Gaussian distributions

CONFIG:
  dim_reduction_method: "ica"
  ica_n_components: 20
  ica_max_iter: 500
```

#### 4. Mutual Information Feature Selection

```
USE WHEN: You want to keep interpretable features with non-linear relevance
STRENGTHS:
  - Captures ANY type of dependency
  - No distribution assumptions
  - Directly measures predictive power
  - Preserves original feature meaning

CONFIG:
  dim_reduction_method: "mutual_info"
  mi_n_features: 30     # Top K by MI score
  mi_n_neighbors: 5     # For MI estimation
```

#### 5. Feature Agglomeration

```
USE WHEN: You have many similar/redundant features
STRENGTHS:
  - Groups similar features via clustering
  - Creates interpretable "super-features"
  - Reduces redundancy intelligently
  - Works with hierarchical relationships

CONFIG:
  dim_reduction_method: "agglomeration"
  agglom_n_clusters: 25
```

#### 6. K-Medoids Clustering

```
USE WHEN: You need outlier-robust clustering (better than K-Means)
STRENGTHS:
  - Uses actual data points as centers (medoids)
  - More robust to outliers than K-Means
  - Better for non-spherical clusters
  - Interpretable medoids

CONFIG:
  dim_reduction_method: "kmedoids"
  kmedoids_n_clusters: 20
  kmedoids_metric: "euclidean"  # or manhattan, cosine
```

#### 7. ENSEMBLE+ (RECOMMENDED)

```
USE WHEN: You want maximum robustness with all methods (DEFAULT)
COMBINES:
  1. Mutual Info selection (20 features) - Relevance
  2. Kernel PCA (12 components) - Non-linear structure
  3. ICA (8 components) - Independent signals
  4. K-Medoids (10 clusters) - Outlier-robust distances
  = 50 total dimensions

CONFIG:
  dim_reduction_method: "ensemble_plus"

RESULTS (verified 2026-01-29 with 5 years of data):
  - Training Data: 1,260 trading days, 1,031,015 bars
  - Swing Model AUC: 0.769
  - Timing Model AUC: 0.706
  - Buy Win Rate: 78.1%
  - Total Return on Buy Signals: 33.17%
```

---

## Intelligent Hyperparameter Optimization (Optuna)

> **Bayesian optimization replaces grid search for efficient hyperparameter tuning.**

```
CONFIGURATION:
  use_optuna: true
  optuna_n_trials: 50
  optuna_timeout: 300  # seconds
  optuna_sampler: "tpe"  # Tree-Parzen Estimator

SEARCH SPACE:
  l2_C: [0.01, 10.0]  (log scale)
  gb_n_estimators: [30, 150]
  gb_max_depth: [2, 5]  (NEVER > 5 per EDGE 1)
  gb_learning_rate: [0.01, 0.3]  (log scale)
  gb_min_samples_leaf: [20, 100]
  gb_subsample: [0.6, 1.0]

BENEFITS:
  - 10-100x more efficient than grid search
  - Early pruning of bad trials
  - Intelligent exploration via TPE
  - Handles continuous and categorical params

EXAMPLE OUTPUT:
  Best AUC: 0.7851
  Best parameters:
    l2_C: 2.5664
    gb_n_estimators: 94
    gb_max_depth: 3
    gb_learning_rate: 0.0415
```

---

## Anti-Overfitting Measures (CRITICAL)

> **Win rates > 75% are suspicious. Use these measures to validate model robustness.**

### Core Components (`src/anti_overfit.py`)

#### 1. Weighted Model Evaluation Score (WMES)

```
PROBLEM: High win rate alone doesn't mean a robust model
SOLUTION: Multi-dimensional evaluation with weighted components

COMPONENTS (weights):
  - win_rate (0.15): Traditional metric, capped at 75%
  - robustness (0.25): CV score stability (penalize high variance)
  - profit_potential (0.20): Risk-adjusted returns (Sharpe + profit factor)
  - noise_tolerance (0.15): Performance degradation on noisy data
  - plateau_stability (0.15): Sensitivity to HP changes (lower = better)
  - complexity_penalty (0.10): Fewer features preferred (optimal ~30)

INTERPRETATION:
  WMES > 0.65: Excellent, likely robust
  WMES 0.55-0.65: Good, proceed with caution
  WMES < 0.55: Suspect, likely overfit

CONFIG:
  wmes_threshold: 0.55  # Minimum acceptable score
```

#### 2. Synthetic SPY Universes ("What SPY Could Have Been")

```
PROBLEM: Model may be overfit to specific SPY price path
SOLUTION: Generate alternative histories using SPY components

METHODS:
  1. Filter Extremes (10%, 20%): Remove top/bottom daily performers
  2. Filter Middle (10%, 20%): Remove middle performers (keep extremes)
  3. Volatility Filter: Remove high or low volatility stocks
  4. Momentum Filter: Remove trailing winners or losers
  5. Bootstrap (2x): Random 70% subset of components

WEIGHTING:
  - Real SPY: 70% weight (default)
  - Synthetic: 30% weight (split across 10 universes)

CONFIG:
  use_synthetic_universes: true
  synthetic_weight: 0.3  # Conservative default
```

#### 3. Cross-Asset Features

```
ASSETS INCLUDED:
  - TLT: Treasury bonds (20+ year) - risk-off signal
  - QQQ: NASDAQ 100 - tech sentiment
  - GLD: Gold ETF - flight to safety
  - IWM: Russell 2000 - small cap risk appetite
  - EEM: Emerging markets - global risk sentiment
  - VXX: VIX futures - volatility expectations
  - HYG: High yield bonds - credit risk appetite

FEATURES PER ASSET:
  - Daily return
  - 5-day return
  - 20-day volatility
  - 20-day momentum
  - RSI (14-period)

CONFIG:
  use_cross_assets: true
```

#### 4. Component Streak Breadth Features

```
CONCEPT: Track % of SPY components with consecutive up/down days

FEATURES:
  - pct_green_Nd: % of components green N+ days (N=2,3,4,5,6,7)
  - pct_red_Nd: % of components red N+ days
  - wtd_pct_green_Nd: Market-cap weighted version
  - wtd_pct_red_Nd: Market-cap weighted version
  - net_green_Nd: pct_green - pct_red
  - breadth_divergence: SPY direction vs breadth change

INTERPRETATION:
  - High pct_green_5d: Strong breadth, trend confirmation
  - Negative breadth_divergence: SPY up, breadth declining = warning

CONFIG:
  use_breadth_streaks: true
```

#### 5. Hyperparameter Stability Analysis

```
PROBLEM: Fragile solutions change drastically with small HP changes
SOLUTION: Perturb hyperparameters +/- 5% and measure score variance

METRIC:
  - stability_score: 0-1 (1 = very stable, parameter plateau)
  - sensitivity: Average score change per parameter perturbation

INTERPRETATION:
  stability_score > 0.7: Robust solution, on parameter plateau
  stability_score 0.5-0.7: Acceptable, proceed with caution
  stability_score < 0.5: Fragile, likely overfit to specific HPs

CONFIG:
  stability_threshold: 0.5  # Minimum acceptable stability
```

#### 6. Robustness Ensemble (Dimension + Parameter Perturbation)

```
PROBLEM: Model may be overfit to specific dimension count or exact parameter values
SOLUTION: Train ensemble of "adjacent" models with slight perturbations

STRATEGY:
  1. Train base model with optimal n dimensions and parameters
  2. Train models with n-2, n-1, n+1, n+2 dimensions
  3. Train models with parameters perturbed by +/- 5% noise
  4. Ensemble all models with weighted averaging
  5. If performance drops drastically -> solution is fragile

DIMENSION VARIANTS:
  optimal_dims = 30  # From Optuna/CV
  variants = [28, 29, 30, 31, 32]  # +/- 2 dimensions

  For each variant:
    - Reduce features to N dimensions
    - Train model with same parameters
    - Record CV score

PARAMETER VARIANTS:
  base_params = {"C": 2.5, "max_depth": 3, ...}

  For each variant:
    - Add +/- 5% noise to each parameter
    - Train model with perturbed parameters
    - Record CV score

WEIGHTING:
  - Center model (optimal): 50% weight
  - Adjacent models: Split remaining 50%
  - Can adjust if absolutely necessary

FRAGILITY DETECTION:
  fragility_score = weighted_avg(
    score_variance * 0.3,
    max_score_drop * 0.3,
    dim_sensitivity * 0.2,
    param_sensitivity * 0.2
  )

  fragility < 0.15: VERY_ROBUST
  fragility 0.15-0.25: ROBUST
  fragility 0.25-0.35: MODERATE
  fragility 0.35-0.50: FRAGILE
  fragility > 0.50: VERY_FRAGILE

CONFIG:
  use_robustness_ensemble: true
  n_dimension_variants: 2      # +/- 2 dimensions
  n_param_variants: 2          # 2 parameter perturbations
  param_noise_pct: 0.05        # 5% noise
  ensemble_center_weight: 0.5  # Weight for optimal model
  fragility_threshold: 0.35    # Warn if above this
```

### Running with Anti-Overfitting

```python
# In train_robust_model.py CONFIG:
CONFIG = {
    # ... other settings ...
    "use_anti_overfit": True,
    "use_synthetic_universes": True,
    "use_cross_assets": True,
    "use_breadth_streaks": True,
    "synthetic_weight": 0.3,
    "wmes_threshold": 0.55,
    "stability_threshold": 0.5,
    # Robustness Ensemble
    "use_robustness_ensemble": True,
    "n_dimension_variants": 2,
    "n_param_variants": 2,
    "param_noise_pct": 0.05,
    "ensemble_center_weight": 0.5,
    "fragility_threshold": 0.35,
}
```

### Expected Output

```
ANTI-OVERFITTING FEATURE AUGMENTATION
======================================================================

[BREADTH] Downloading component data for streak analysis...
  Downloaded 50 components, 1260 days
[BREADTH] Computing component streak features...
  Added 42 streak-based breadth features

[CROSS-ASSETS] Downloading correlated asset data...
  Downloaded 7 assets, 1260 days
[CROSS-ASSETS] Engineering features...
  Added 35 cross-asset features

[SYNTHETIC SPY] Downloading component data...
  Downloaded 50 components, 1260 days
[SYNTHETIC SPY] Generating 10 alternative universes...
  Created: Filter extreme 10% performers
  Created: Filter extreme 20% performers
  ...
[PASS] Generated 10 synthetic universes

[AUGMENTED DATA] Creating combined dataset...
  Real samples: 1,250
  Synthetic samples: 12,500
  Total augmented: 13,750
  Effective weight ratio: 70% real, 30% synthetic

[WEIGHTED MODEL EVALUATION]
  WMES Score: 0.612
    Win Rate Component: 0.867
    Robustness Component: 0.583
    Profit Potential: 0.472
    Plateau Stability: 0.724
    Complexity Penalty: 0.833

  [GOOD] WMES above threshold - model appears robust

======================================================================
ROBUSTNESS ENSEMBLE TRAINING
======================================================================

[DIM VARIANTS] Testing dimensions: [28, 29, 30, 31, 32]
  dim=28: AUC=0.7612
  dim=29: AUC=0.7685
  dim=30: AUC=0.7721 (OPTIMAL)
  dim=31: AUC=0.7698
  dim=32: AUC=0.7654

[PARAM VARIANTS] Testing 3 parameter sets
  variant_0: AUC=0.7721 (BASE)
  variant_1: AUC=0.7689
  variant_2: AUC=0.7702

[ENSEMBLE] Trained 8 models
  Fragility Score: 0.182 (0=robust, 1=fragile)
  [GOOD] Low fragility - solution appears robust

[ENSEMBLE EVALUATION]
  Ensemble AUC: 0.7735
  Best Individual AUC: 0.7721
  Improvement: 0.0014

======================================================================
STEP 9D: ENTRY/EXIT TIMING MODEL
======================================================================

[INFO] Training ML model for entry/exit timing predictions
  This model predicts:
    - Optimal entry time (minutes from open)
    - Optimal exit time (minutes from open)
    - Position size based on conditions
    - Dynamic stop loss / take profit levels
    - Batch entry schedules
    - Guardrail triggers

[EntryExitTimingModel] Starting training...
  Creating targets from historical data...
  Engineering timing features...
  Training samples: 1,234

  Training fold 1/3...
  Training fold 2/3...
  Training fold 3/3...
  Final fit on all data...

  ═══════════════════════════════════════════════════
  ENTRY/EXIT TIMING MODEL - Training Summary
  ═══════════════════════════════════════════════════

  Entry Time Model:
    MAE: 18.3 ± 2.1 min
    Within 15 min: 52.4%
    Within 30 min: 78.6%

  Exit Time Model:
    MAE: 22.7 ± 3.4 min
    Within 15 min: 45.2%
    Within 30 min: 72.1%

  Position Size Model:
    MAE: 2.34%
    R²: 0.412

  Stop/Take Profit Model:
    Stop MAE: 12.3 bps
    TP MAE: 15.7 bps

  Batch Schedule Model:
    Should Batch Accuracy: 68.4%
    N Batches Accuracy: 62.1%
  ═══════════════════════════════════════════════════

  [PASS] Entry/Exit Timing Model trained successfully
```

#### 7. Entry/Exit Timing Model (Predicts Specific Trading Decisions)

**Problem Solved:** The swing/timing models predict DIRECTION but not SPECIFICS:

| Model Predicts | Model Does NOT Predict |
|----------------|----------------------|
| Up/Down day    | Specific entry time  |
| Low before High| Specific exit time   |
| General timing | Position size        |
|                | Stop/take profit     |
|                | Batch schedule       |
|                | Guardrails           |

**Solution:** `src/entry_exit_model.py` - ML models that LEARN optimal decisions from historical data.

**Sub-Models:**

1. **EntryTimeModel** - Predicts optimal entry time (minutes from open)
2. **ExitTimeModel** - Predicts optimal exit time (minutes from open)
3. **PositionSizeModel** - Predicts position size based on volatility & confidence
4. **StopTakeProfitModel** - Predicts dynamic stop loss and take profit levels
5. **BatchScheduleModel** - Predicts whether to batch and how many tranches
6. **GuardrailModel** - Predicts when emergency exits should trigger

**Target Labeling (TargetLabeler):**

For each historical day, computes what WOULD have been optimal:
- Optimal entry = minute with lowest price in entry window (for LONG)
- Optimal exit = minute with highest price in exit window
- Optimal stop = max adverse excursion + buffer
- Optimal position = volatility-adjusted based on achieved return

**Usage:**

```python
from src.entry_exit_model import EntryExitTimingModel

# Create and train
model = EntryExitTimingModel(
    model_type="gradient_boosting",
    entry_window=(0, 120),    # First 2 hours
    exit_window=(180, 385),   # Last 3.5 hours
)

metrics = model.fit(
    daily_data=df_daily,
    intraday_data=df_1min,
    directions=directions,  # "LONG" or "SHORT" per day
    cv_folds=3,
)

# Predict for new day
prediction = model.predict(
    features=today_features,
    swing_proba=0.72,       # From swing model
    timing_proba=0.65,      # From timing model
    current_price=450.0,
)

print(prediction)
# {
#   "entry_time_minutes": 45,
#   "exit_time_minutes": 320,
#   "position_size_pct": 0.12,
#   "stop_loss_pct": 0.012,
#   "take_profit_pct": 0.018,
#   "should_batch": True,
#   "n_batches": 3,
#   "batch_schedule": [
#     {"batch_num": 1, "entry_minute": 45, "position_pct": 0.02},
#     {"batch_num": 2, "entry_minute": 55, "position_pct": 0.04},
#     {"batch_num": 3, "entry_minute": 65, "position_pct": 0.06},
#   ],
#   "guardrails": {
#     "risk_level": "NORMAL",
#     "guardrail_probability": 0.23,
#   }
# }
```

**CONFIG Settings:**

```python
CONFIG = {
    # Entry/Exit Timing Model
    "train_entry_exit_model": True,
    "entry_exit_model_type": "gradient_boosting",
    "entry_window": (0, 120),      # Entry window (minutes from open)
    "exit_window": (180, 385),     # Exit window (minutes from open)
    "min_position_pct": 0.05,      # Min position 5%
    "max_position_pct": 0.25,      # Max position 25%
}
```

---

### When to Use Each Method

| Scenario | Recommended Method |
|----------|-------------------|
| General production | `ensemble` |
| Need interpretability | `mutual_info` |
| Suspect non-linear patterns | `kernel_pca` |
| Want cluster preservation | `umap` |
| Separating signal sources | `ica` |
| Many redundant features | `agglomeration` |

---

## Data Sources

### Market Data

| Source | Data Type | Priority | API Key |
|--------|-----------|----------|---------|
| Alpaca | Equities, minute-level | Primary | Yes |
| yfinance | Historical OHLCV | Backup | No |
| Polygon | Alternative data | Optional | Yes |
| FRED | Economic indicators | Optional | Yes |

### Sentiment Data

| Source | Data Type | Pre-scored | Priority |
|--------|-----------|------------|----------|
| Alpha Vantage | News + sentiment | Yes (-1 to +1) | 1 |
| Finnhub | Company news | No | 2 |
| NewsAPI | General news | No | 3 |

### LLM Providers (Fallback Chain)

| Provider | Model | Priority | Use Case |
|----------|-------|----------|----------|
| Mistral | mistral-large-latest | Primary | Sentiment analysis |
| OpenAI | gpt-4o-mini | Fallback 1 | If Mistral fails |
| Anthropic | claude-3-haiku | Fallback 2 | If OpenAI fails |

---

## Configuration Files

| File | Purpose |
|------|---------|
| `config/pipeline.yaml` | Master ML pipeline config (265 steps) |
| `config/sentiment_pipeline.yaml` | Sentiment analyzer config (10 stages) |
| `config/pipeline_status.yaml` | Step completion tracker |
| `config/edge_strategies.yaml` | **5 critical edge-generating strategies** |
| `config/trading_modes.yaml` | Simulation, Paper, Live mode configs |
| `config/risk_management.yaml` | Regime-aware risk limits, stops, circuit breakers |
| `config/monitoring.yaml` | Health checks, alerts, dashboards |
| `config/data_sources.yaml` | Multi-source data with fallback chains |
| `.env` | API keys and secrets |

### Key Source Files

| File | Purpose |
|------|---------|
| `src/train_robust_model.py` | **Main training script with all edge strategies** |
| `scripts/test_alpaca_data.py` | Alpaca API connection and data tests |
| `models/production/spy_robust_models.joblib` | Saved production models |

---

## Modular Architecture

### Directory Structure

```
giga_trader/
├── src/
│   ├── phase_01_data_acquisition/    # Steps 1-5
│   ├── phase_02_preprocessing/        # Steps 6-14
│   ├── ...                            # Phases 3-27
│   ├── core/                          # Base classes, interfaces
│   └── utils/                         # Shared utilities
├── config/
│   ├── pipeline.yaml                  # Master pipeline config
│   ├── sentiment_pipeline.yaml        # Sentiment config
│   └── pipeline_status.yaml           # Status tracker
├── data/
│   ├── raw/                           # Untransformed data
│   ├── processed/                     # Cleaned data
│   └── synthetic/                     # Augmented scenarios
├── models/
│   ├── checkpoints/                   # Training checkpoints
│   └── production/                    # Deployed models
├── logs/                              # Execution logs
├── reports/                           # Generated reports
└── scripts/                           # Utility scripts
```

---

## Coding Standards

### Python Style

```python
# Type hints required
def calculate_feature(
    data: pd.DataFrame,
    period: int = 14,
    column: str = "close"
) -> pd.Series:
    """Calculate feature with full docstring."""
    if data.empty:
        raise ValueError("Empty DataFrame provided")
    return data[column].rolling(period).mean()
```

### Concurrency Pattern

```python
# Use semaphore for API calls
async def analyze_batch(articles: list[Article]) -> list[Sentiment]:
    semaphore = asyncio.Semaphore(3)  # Max 3 concurrent
    async with semaphore:
        return await llm.analyze(articles)
```

### Retry Pattern

```python
# Exponential backoff for API calls
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10)
)
async def call_api(endpoint: str) -> Response:
    return await client.get(endpoint)
```

---

## Pipeline Status Summary

| Category | Total | Done | Partial | Missing |
|----------|-------|------|---------|---------|
| Data Infrastructure | 13 | 5 | 0 | 8 |
| Data Preprocessing | 9 | 4 | 1 | 4 |
| Universe Management | 8 | 5 | 1 | 2 |
| Price/Volume Features | 18 | 15 | 2 | 1 |
| Breadth/Streak Features | 15 | 15 | 0 | 0 |
| Sentiment Features | 15 | 8 | 0 | 7 |
| Extended Hours Features | 23 | 23 | 0 | 0 |
| Model Training | 15 | 11 | 1 | 3 |
| Monitoring & Alerting | 4 | 0 | 0 | 4 |
| **TOTAL** | **217** | **148** | **13** | **56** |

### Critical Missing Items

1. **Historical S&P 500 constituents** - Survivorship bias (CRITICAL)
2. **News/Sentiment APIs** - Core sentiment data (HIGH)
3. **FOMC meeting flags** - Event-driven trading (HIGH)
4. **Monitoring & Alerting** - Production readiness (HIGH)

---

## Performance Targets

| Metric | Minimum | Target | Stretch | **Current** |
|--------|---------|--------|---------|-------------|
| Swing Model AUC | 0.58 | 0.65 | 0.75 | **0.818** |
| Timing Model AUC | 0.55 | 0.62 | 0.72 | **0.778** |
| Train-Test Gap | < 0.10 | < 0.07 | < 0.05 | **~0.05** |
| Buy Win Rate | 55% | 65% | 75% | **81.8%** |
| Sell Win Rate | 55% | 65% | 75% | **77.3%** |
| Sharpe Ratio | 0.5 | 1.0 | 1.5 | TBD |
| Max Drawdown | < 15% | < 10% | < 7% | TBD |

### Current Model Performance (2026-01-29)

```
Trained on: 1,260 trading days (5 years of 1-min data with extended hours)
Total bars: 1,031,015
Features: 133 raw -> 33 after ensemble+ dim reduction -> 32 robust
Method: Ensemble+ (Mutual Info + Kernel PCA + ICA + K-Medoids)
Optimization: Optuna Bayesian (21 trials)

Swing Direction Model:
  - AUC: 0.769
  - Accuracy: 71.8%
  - Precision (Up): 68%

Timing Model (Low before High):
  - AUC: 0.706
  - Accuracy: 63.7%

Combined Signals (Test Period: 248 days):
  - Buy signals: 25.8% of days
  - Sell signals: 47.6% of days
  - Buy win rate: 78.1%
  - Sell win rate: 61.9%
  - Total return on buy signals: 33.17%
```

---

## Commands Reference

### Moltbot Continuous Building

```bash
# SETUP: Run from C:\Users\amare\moltbot
node moltbot.mjs doctor                    # Health check (fix any issues)
node moltbot.mjs onboard --install-daemon  # Install gateway daemon

# RUN CONTINUOUS BUILD: Use workspace flag
node moltbot.mjs agent --workspace "c:\Users\amare\OneDrive\Documents\giga_trader" \
  --message "Build the next component following config/moltbot_workflow.yaml"

# VALIDATE AFTER EACH BUILD:
cd "c:\Users\amare\OneDrive\Documents\giga_trader"
.venv/Scripts/python.exe -c "import ast; ast.parse(open('src/train_robust_model.py', encoding='utf-8').read())"
.venv/Scripts/python.exe src/train_robust_model.py
```

### Moltbot Build Workflow

The build workflow in `config/moltbot_workflow.yaml` defines 9 phases:

1. **Data Infrastructure** - Alpaca client, validators, storage
2. **Feature Engineering** - Technical indicators, extended hours, patterns
3. **Dimensionality Reduction** - UMAP, Kernel PCA, ICA, K-Medoids
4. **Model Training** - Swing + Timing models with Optuna
5. **Risk Management** - Position sizing, stops, drawdown limits
6. **Signal Generation** - Dual model agreement, confidence scoring
7. **Backtesting** - Historical simulation, performance metrics
8. **Paper Trading** - Alpaca paper trading integration
9. **Monitoring** - Health checks, dashboards, alerts

Each phase:
- Has explicit dependencies (must be complete before starting)
- Requires validation before moving to next phase
- Uses real Alpaca API data (no mocks)

### Quick Commands

```bash
# Check moltbot status
node moltbot.mjs doctor

# Start gateway (for channels)
node moltbot.mjs gateway --port 18789 --verbose

# Run single agent task
node moltbot.mjs agent --message "Task description" --thinking high
```

### Claude Code Harness Commands

```
/plan-with-agent     # Create structured development plans
/work                # Execute planned tasks
/harness-review      # Run 8-expert code review
/sync-status         # Check progress
```

### Pipeline Commands

```bash
# Environment setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Run pipeline stages
python -m src.run_pipeline --stage 1     # Data acquisition
python -m src.run_pipeline --stage 1-5   # Phases 1-5
python -m src.run_pipeline --stage all   # Full pipeline

# Preflight checks
node scripts/preflight-tests.mjs
```

---

## Security

```
NEVER commit:
- .env files
- API keys or credentials
- Production model weights
- Trading account credentials

ALWAYS:
- Use .env.example as template
- Store secrets in environment variables
- Rotate API keys regularly
- Use paper trading for testing
```

---

## Phase Summary

| Phase | Steps | Description | Status |
|-------|-------|-------------|--------|
| 1-2 | 1-14 | Data acquisition & cleaning | Partial |
| 3-4 | 15-40 | Synthetic data & Monte Carlo | Done |
| 5 | 41-48 | Target variable creation | Done |
| 6-9 | 49-95 | Feature engineering (200+ features) | Mostly Done |
| 10-11 | 96-112 | Feature processing & CV | Done |
| 12-14 | 113-150 | Model training & validation | Done |
| 15-17 | 151-185 | Strategy & backtesting | Done |
| 18-19 | 186-210 | Production deployment | Partial |
| 20-22 | 211-240 | Monitoring & automation | Missing |
| 23-27 | 241-265 | Advanced analytics & live | Partial |

**Total: 265 steps, 27 phases, 200+ features, 50+ code files**

---

_Last updated: 2026-01-29 16:00_
_Pipeline version: 1.1_
_Status: 68% complete (models trained and validated)_
_Latest Model: Swing AUC 0.818, Timing AUC 0.778, Buy Win Rate 81.8%_

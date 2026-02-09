"""
GIGA TRADER - Temporal Cascade Model Architecture
==================================================
Multiple models trained on different temporal slices to reduce overfitting
and enable real-time prediction updates as the trading day progresses.

CONCEPT:
--------
Instead of a single swing model, we train a CASCADE of models:
  - T0:   Historical features only (pre-market, overnight)
  - T30:  Historical + first 30 minutes of trading
  - T60:  Historical + first 60 minutes
  - T90:  Historical + first 90 minutes
  - T120: Historical + first 2 hours
  - T180: Historical + first 3 hours

Each temporal slice model predicts:
  1. Swing direction (UP/DOWN probability)
  2. Expected magnitude (SMALL/MEDIUM/LARGE)
  3. Day regime (TRENDING/RANGING/VOLATILE/QUIET)
  4. Optimal entry window
  5. Confidence score

BENEFITS:
---------
  1. ANTI-OVERFITTING: Models with different inputs can't all overfit to same noise
  2. REAL-TIME UPDATES: As day progresses, later models can correct early predictions
  3. AGREEMENT SIGNAL: When T0, T30, T60 all agree → stronger signal
  4. DISAGREEMENT DETECTION: Early warning when models start disagreeing
  5. ADAPTIVE: Can adjust position as new information arrives

Usage:
    from src.temporal_cascade_models import TemporalCascadeEnsemble

    cascade = TemporalCascadeEnsemble()
    cascade.fit(historical_data, intraday_data)

    # Pre-market prediction (T0 only)
    pred_t0 = cascade.predict_at_time(features, minutes_since_open=0)

    # Update at 10:00 AM (30 min in)
    pred_t30 = cascade.predict_at_time(features, minutes_since_open=30)

    # Get ensemble prediction with all available models
    ensemble_pred = cascade.get_ensemble_prediction(features, minutes_since_open=60)
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import warnings

import numpy as np
import pandas as pd
import joblib

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score

logger = logging.getLogger("TEMPORAL_CASCADE")


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================
class SwingDirection(Enum):
    DOWN = 0
    UP = 1


class DayMagnitude(Enum):
    SMALL = 0      # < 0.3% move
    MEDIUM = 1     # 0.3% - 0.6% move
    LARGE = 2      # > 0.6% move


class DayRegime(Enum):
    QUIET = 0      # Low volatility, small range
    RANGING = 1    # Medium vol, no clear direction
    TRENDING = 2   # Clear directional move
    VOLATILE = 3   # High volatility, large swings


class IntradayPattern(Enum):
    MORNING_DIP = 0       # Dip in first 1-2 hours, then recovery
    GAP_FADE = 1          # Gap up/down that fades
    ORB_BREAKOUT = 2      # Opening range breakout
    V_REVERSAL = 3        # Sharp reversal pattern
    AFTERNOON_RALLY = 4   # Flat morning, rally afternoon
    GRIND_UP = 5          # Steady upward grind
    GRIND_DOWN = 6        # Steady downward grind
    CHOP = 7              # No clear pattern


class OptimalWindow(Enum):
    VERY_EARLY = 0   # First 30 min
    EARLY = 1        # 30-60 min
    MID_MORNING = 2  # 60-120 min
    LATE_MORNING = 3 # 120-180 min
    AFTERNOON = 4    # After 180 min


@dataclass
class TemporalPrediction:
    """Prediction from a single temporal slice model."""
    temporal_slice: int  # Minutes of intraday data used
    timestamp: str

    # Core predictions
    swing_direction: float  # Probability of UP
    swing_confidence: float

    # Magnitude prediction
    magnitude_proba: Dict[str, float] = field(default_factory=dict)  # SMALL/MEDIUM/LARGE
    expected_magnitude: str = "MEDIUM"

    # Regime prediction
    regime_proba: Dict[str, float] = field(default_factory=dict)
    predicted_regime: str = "TRENDING"

    # Pattern prediction
    pattern_proba: Dict[str, float] = field(default_factory=dict)
    predicted_pattern: str = "GRIND_UP"

    # Optimal window
    optimal_entry_window: str = "EARLY"
    optimal_exit_window: str = "AFTERNOON"

    # Meta
    model_agreement_score: float = 0.0  # How much this agrees with other temporal models

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CascadeEnsemblePrediction:
    """Combined prediction from all temporal cascade models."""
    timestamp: str
    minutes_since_open: int
    models_available: List[int]  # Which temporal slices are available

    # Ensemble predictions (weighted by recency and confidence)
    swing_direction: float
    swing_confidence: float
    expected_magnitude: str
    predicted_regime: str
    predicted_pattern: str
    optimal_entry_window: str
    optimal_exit_window: str

    # Agreement metrics
    model_agreement: float  # 0-1, how much models agree
    prediction_stability: float  # How stable is prediction across time slices

    # Individual model predictions for transparency
    individual_predictions: List[TemporalPrediction] = field(default_factory=list)

    # Trading recommendation
    should_trade: bool = False
    recommended_position_size: float = 0.0
    recommended_entry_time: int = 0  # Minutes from open
    recommended_exit_time: int = 385  # Minutes from open

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['individual_predictions'] = [p.to_dict() for p in self.individual_predictions]
        return d


# =============================================================================
# TEMPORAL SLICE FEATURE ENGINEERING
# =============================================================================
class TemporalFeatureEngineer:
    """
    Engineers features for different temporal slices.

    Each slice includes:
      - All historical features (same for all slices)
      - Intraday features up to the time slice cutoff
    """

    # Define temporal slices (minutes from market open)
    TEMPORAL_SLICES = [0, 30, 60, 90, 120, 180, 240]

    def __init__(self):
        self.historical_feature_cols = []
        self.intraday_feature_templates = []

    def engineer_historical_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from historical data only (available pre-market).
        These are the SAME for all temporal slices.
        """
        df = df_daily.copy()

        # Previous day features
        for lag in [1, 2, 3, 5]:
            df[f'return_lag_{lag}'] = df['close'].pct_change(lag)
            df[f'volume_ratio_lag_{lag}'] = df['volume'] / df['volume'].rolling(20).mean()
            df[f'range_lag_{lag}'] = (df['high'] - df['low']) / df['close']

        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'price_vs_sma_{window}'] = df['close'] / df[f'sma_{window}'] - 1

        # Volatility
        df['volatility_5d'] = df['close'].pct_change().rolling(5).std()
        df['volatility_20d'] = df['close'].pct_change().rolling(20).std()
        df['volatility_ratio'] = df['volatility_5d'] / df['volatility_20d']

        # Momentum
        df['momentum_5d'] = df['close'].pct_change(5)
        df['momentum_10d'] = df['close'].pct_change(10)
        df['momentum_20d'] = df['close'].pct_change(20)

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # Day of week
        if 'date' in df.columns:
            df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek

        # Store historical feature columns
        self.historical_feature_cols = [c for c in df.columns
                                         if c not in ['date', 'open', 'high', 'low', 'close', 'volume']]

        return df

    def engineer_intraday_features(
        self,
        df_1min: pd.DataFrame,
        minutes_cutoff: int
    ) -> Dict[str, float]:
        """
        Engineer features from intraday data up to the specified cutoff.

        Args:
            df_1min: Minute-level data for the current day
            minutes_cutoff: How many minutes of data to use (0, 30, 60, etc.)

        Returns:
            Dictionary of intraday features
        """
        features = {}

        if minutes_cutoff == 0 or len(df_1min) == 0:
            # No intraday data available (pre-market)
            return self._get_empty_intraday_features()

        # Filter to only use data up to cutoff
        df = df_1min.head(minutes_cutoff).copy()

        if len(df) < 5:
            return self._get_empty_intraday_features()

        # Basic price features
        open_price = df['open'].iloc[0]
        current_price = df['close'].iloc[-1]
        high_so_far = df['high'].max()
        low_so_far = df['low'].min()

        features['intraday_return'] = (current_price - open_price) / open_price
        features['intraday_range'] = (high_so_far - low_so_far) / open_price
        features['price_vs_open'] = current_price / open_price - 1
        features['high_vs_open'] = high_so_far / open_price - 1
        features['low_vs_open'] = low_so_far / open_price - 1

        # Where is price relative to intraday range?
        intraday_range = high_so_far - low_so_far
        if intraday_range > 0:
            features['price_position_in_range'] = (current_price - low_so_far) / intraday_range
        else:
            features['price_position_in_range'] = 0.5

        # Timing features
        high_minute = df['high'].idxmax() if hasattr(df['high'].idxmax(), '__int__') else 0
        low_minute = df['low'].idxmin() if hasattr(df['low'].idxmin(), '__int__') else 0

        # Normalize to 0-1 range
        features['high_timing'] = high_minute / max(minutes_cutoff, 1)
        features['low_timing'] = low_minute / max(minutes_cutoff, 1)
        features['low_before_high'] = 1.0 if low_minute < high_minute else 0.0

        # Volume features
        total_volume = df['volume'].sum()
        avg_volume_per_min = total_volume / len(df)
        features['volume_intensity'] = avg_volume_per_min

        # First vs second half volume
        mid_point = len(df) // 2
        first_half_vol = df['volume'].iloc[:mid_point].sum()
        second_half_vol = df['volume'].iloc[mid_point:].sum()
        features['volume_trend'] = (second_half_vol - first_half_vol) / max(total_volume, 1)

        # Momentum in recent bars
        if len(df) >= 10:
            recent_return = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
            features['recent_momentum'] = recent_return
        else:
            features['recent_momentum'] = 0.0

        # Volatility of intraday moves
        intraday_returns = df['close'].pct_change().dropna()
        features['intraday_volatility'] = intraday_returns.std() if len(intraday_returns) > 0 else 0.0

        # Direction consistency (what % of bars are positive?)
        positive_bars = (intraday_returns > 0).sum()
        features['direction_consistency'] = positive_bars / max(len(intraday_returns), 1)

        # VWAP features
        if 'volume' in df.columns and df['volume'].sum() > 0:
            vwap = (df['close'] * df['volume']).sum() / df['volume'].sum()
            features['price_vs_vwap'] = current_price / vwap - 1
        else:
            features['price_vs_vwap'] = 0.0

        # Pattern detection features
        features['is_morning_dip'] = 1.0 if (features['low_timing'] < 0.5 and
                                              features['price_position_in_range'] > 0.6) else 0.0
        features['is_grinding_up'] = 1.0 if (features['direction_consistency'] > 0.6 and
                                              features['intraday_return'] > 0.001) else 0.0
        features['is_grinding_down'] = 1.0 if (features['direction_consistency'] < 0.4 and
                                                features['intraday_return'] < -0.001) else 0.0
        features['is_choppy'] = 1.0 if (features['intraday_volatility'] > 0.001 and
                                         abs(features['intraday_return']) < 0.002) else 0.0

        return features

    def _get_empty_intraday_features(self) -> Dict[str, float]:
        """Return empty/neutral intraday features for T0 (pre-market)."""
        return {
            'intraday_return': 0.0,
            'intraday_range': 0.0,
            'price_vs_open': 0.0,
            'high_vs_open': 0.0,
            'low_vs_open': 0.0,
            'price_position_in_range': 0.5,
            'high_timing': 0.5,
            'low_timing': 0.5,
            'low_before_high': 0.5,
            'volume_intensity': 0.0,
            'volume_trend': 0.0,
            'recent_momentum': 0.0,
            'intraday_volatility': 0.0,
            'direction_consistency': 0.5,
            'price_vs_vwap': 0.0,
            'is_morning_dip': 0.0,
            'is_grinding_up': 0.0,
            'is_grinding_down': 0.0,
            'is_choppy': 0.0,
        }

    def get_feature_names(self, temporal_slice: int) -> List[str]:
        """Get feature names for a specific temporal slice."""
        intraday_features = list(self._get_empty_intraday_features().keys())
        return self.historical_feature_cols + [f't{temporal_slice}_{f}' for f in intraday_features]


# =============================================================================
# TEMPORAL SLICE MODEL
# =============================================================================
class TemporalSliceModel:
    """
    A model trained on a specific temporal slice of data.

    Each model predicts:
      - Swing direction (UP/DOWN)
      - Magnitude (SMALL/MEDIUM/LARGE)
      - Regime (TRENDING/RANGING/VOLATILE/QUIET)
      - Pattern (various intraday patterns)
    """

    def __init__(
        self,
        temporal_slice: int,  # Minutes of intraday data (0, 30, 60, etc.)
        model_type: str = "gradient_boosting",
        regularization_strength: float = 0.1,
    ):
        self.temporal_slice = temporal_slice
        self.model_type = model_type
        self.regularization_strength = regularization_strength

        # Models for different prediction tasks
        self.swing_model = None
        self.magnitude_model = None
        self.regime_model = None
        self.pattern_model = None

        # Scalers
        self.scaler = StandardScaler()

        # Feature info
        self.feature_cols = []
        self.is_fitted = False

        # Performance metrics
        self.cv_auc = 0.0
        self.test_auc = 0.0

    def _create_model(self, n_classes: int = 2):
        """Create model based on configuration."""
        if self.model_type == "logistic":
            return LogisticRegression(
                C=self.regularization_strength,
                max_iter=1000,
                random_state=42 + self.temporal_slice,
                multi_class='multinomial' if n_classes > 2 else 'auto',
            )
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                min_samples_leaf=50,
                subsample=0.8,
                random_state=42 + self.temporal_slice,
            )
        else:
            # Ensemble
            return VotingClassifier(
                estimators=[
                    ('lr', LogisticRegression(C=self.regularization_strength, max_iter=1000)),
                    ('gb', GradientBoostingClassifier(n_estimators=30, max_depth=3)),
                ],
                voting='soft'
            )

    def fit(
        self,
        X: np.ndarray,
        y_swing: np.ndarray,
        y_magnitude: np.ndarray = None,
        y_regime: np.ndarray = None,
        y_pattern: np.ndarray = None,
        sample_weights: np.ndarray = None,
    ) -> Dict[str, float]:
        """
        Fit all prediction models for this temporal slice.

        Args:
            X: Feature matrix
            y_swing: Swing direction labels (0/1)
            y_magnitude: Magnitude labels (0/1/2 for SMALL/MEDIUM/LARGE)
            y_regime: Regime labels (0/1/2/3)
            y_pattern: Pattern labels (0-7)
            sample_weights: Optional sample weights

        Returns:
            Dictionary of training metrics
        """
        metrics = {}

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train swing model (required)
        logger.info(f"  [T{self.temporal_slice}] Training swing model...")
        self.swing_model = self._create_model(n_classes=2)
        if sample_weights is not None:
            try:
                self.swing_model.fit(X_scaled, y_swing, sample_weight=sample_weights)
            except TypeError:
                self.swing_model.fit(X_scaled, y_swing)
        else:
            self.swing_model.fit(X_scaled, y_swing)

        # CV score
        try:
            cv_scores = cross_val_score(
                self._create_model(n_classes=2),
                X_scaled, y_swing,
                cv=3, scoring='roc_auc'
            )
            self.cv_auc = float(np.mean(cv_scores))
            metrics['swing_cv_auc'] = self.cv_auc
        except Exception as e:
            logger.warning(f"  CV failed: {e}")
            self.cv_auc = 0.5

        # Train magnitude model (if labels provided)
        if y_magnitude is not None:
            logger.info(f"  [T{self.temporal_slice}] Training magnitude model...")
            self.magnitude_model = self._create_model(n_classes=3)
            try:
                self.magnitude_model.fit(X_scaled, y_magnitude)
                metrics['magnitude_trained'] = True
            except Exception as e:
                logger.warning(f"  Magnitude model failed: {e}")
                self.magnitude_model = None

        # Train regime model (if labels provided)
        if y_regime is not None:
            logger.info(f"  [T{self.temporal_slice}] Training regime model...")
            self.regime_model = self._create_model(n_classes=4)
            try:
                self.regime_model.fit(X_scaled, y_regime)
                metrics['regime_trained'] = True
            except Exception as e:
                logger.warning(f"  Regime model failed: {e}")
                self.regime_model = None

        # Train pattern model (if labels provided)
        if y_pattern is not None:
            logger.info(f"  [T{self.temporal_slice}] Training pattern model...")
            self.pattern_model = self._create_model(n_classes=8)
            try:
                self.pattern_model.fit(X_scaled, y_pattern)
                metrics['pattern_trained'] = True
            except Exception as e:
                logger.warning(f"  Pattern model failed: {e}")
                self.pattern_model = None

        self.is_fitted = True
        logger.info(f"  [T{self.temporal_slice}] Training complete. CV AUC: {self.cv_auc:.4f}")

        return metrics

    def predict(self, X: np.ndarray) -> TemporalPrediction:
        """
        Generate predictions for this temporal slice.

        Args:
            X: Feature matrix (single sample or batch)

        Returns:
            TemporalPrediction object
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self.scaler.transform(X)

        # Swing prediction
        swing_proba = self.swing_model.predict_proba(X_scaled)[0, 1]
        swing_confidence = abs(swing_proba - 0.5) * 2  # 0 to 1

        # Magnitude prediction
        magnitude_proba = {}
        expected_magnitude = "MEDIUM"
        if self.magnitude_model is not None:
            mag_probs = self.magnitude_model.predict_proba(X_scaled)[0]
            magnitude_proba = {
                "SMALL": float(mag_probs[0]) if len(mag_probs) > 0 else 0.33,
                "MEDIUM": float(mag_probs[1]) if len(mag_probs) > 1 else 0.34,
                "LARGE": float(mag_probs[2]) if len(mag_probs) > 2 else 0.33,
            }
            expected_magnitude = max(magnitude_proba, key=magnitude_proba.get)

        # Regime prediction
        regime_proba = {}
        predicted_regime = "TRENDING"
        if self.regime_model is not None:
            reg_probs = self.regime_model.predict_proba(X_scaled)[0]
            regime_names = ["QUIET", "RANGING", "TRENDING", "VOLATILE"]
            regime_proba = {name: float(reg_probs[i]) if i < len(reg_probs) else 0.25
                          for i, name in enumerate(regime_names)}
            predicted_regime = max(regime_proba, key=regime_proba.get)

        # Pattern prediction
        pattern_proba = {}
        predicted_pattern = "GRIND_UP"
        if self.pattern_model is not None:
            pat_probs = self.pattern_model.predict_proba(X_scaled)[0]
            pattern_names = ["MORNING_DIP", "GAP_FADE", "ORB_BREAKOUT", "V_REVERSAL",
                           "AFTERNOON_RALLY", "GRIND_UP", "GRIND_DOWN", "CHOP"]
            pattern_proba = {name: float(pat_probs[i]) if i < len(pat_probs) else 0.125
                           for i, name in enumerate(pattern_names)}
            predicted_pattern = max(pattern_proba, key=pattern_proba.get)

        # Determine optimal windows based on predictions
        optimal_entry = self._determine_optimal_entry(
            swing_proba, predicted_regime, predicted_pattern
        )
        optimal_exit = self._determine_optimal_exit(
            swing_proba, predicted_regime, predicted_pattern
        )

        return TemporalPrediction(
            temporal_slice=self.temporal_slice,
            timestamp=datetime.now().isoformat(),
            swing_direction=float(swing_proba),
            swing_confidence=float(swing_confidence),
            magnitude_proba=magnitude_proba,
            expected_magnitude=expected_magnitude,
            regime_proba=regime_proba,
            predicted_regime=predicted_regime,
            pattern_proba=pattern_proba,
            predicted_pattern=predicted_pattern,
            optimal_entry_window=optimal_entry,
            optimal_exit_window=optimal_exit,
        )

    def _determine_optimal_entry(
        self,
        swing_proba: float,
        regime: str,
        pattern: str
    ) -> str:
        """Determine optimal entry window based on predictions."""
        # Pattern-based rules
        if pattern == "MORNING_DIP":
            return "MID_MORNING"  # Wait for dip
        elif pattern == "GAP_FADE":
            return "VERY_EARLY"  # Enter quickly to fade
        elif pattern == "ORB_BREAKOUT":
            return "EARLY"  # After ORB forms
        elif pattern == "AFTERNOON_RALLY":
            return "LATE_MORNING"  # Wait for rally to start

        # Regime-based rules
        if regime == "TRENDING":
            return "EARLY"  # Get in early on trend days
        elif regime == "VOLATILE":
            return "MID_MORNING"  # Wait for dust to settle
        elif regime == "RANGING":
            return "MID_MORNING"  # Wait for range to establish

        return "EARLY"  # Default

    def _determine_optimal_exit(
        self,
        swing_proba: float,
        regime: str,
        pattern: str
    ) -> str:
        """Determine optimal exit window based on predictions."""
        # Pattern-based rules
        if pattern == "MORNING_DIP":
            return "AFTERNOON"  # Ride the recovery
        elif pattern == "GAP_FADE":
            return "MID_MORNING"  # Exit once gap filled
        elif pattern == "AFTERNOON_RALLY":
            return "LATE_MORNING"  # Wait for rally (exit early if wrong)

        # Regime-based rules
        if regime == "TRENDING":
            return "AFTERNOON"  # Let it run
        elif regime == "VOLATILE":
            return "MID_MORNING"  # Exit early
        elif regime == "RANGING":
            return "LATE_MORNING"  # Exit at range boundary

        return "AFTERNOON"  # Default


# =============================================================================
# TARGET LABELER
# =============================================================================
class TemporalTargetLabeler:
    """
    Creates training labels for temporal models.

    Labels:
      - Swing: 1 if day closes higher than open
      - Magnitude: SMALL/MEDIUM/LARGE based on absolute return
      - Regime: Based on volatility and trend characteristics
      - Pattern: Based on intraday price action
    """

    MAGNITUDE_THRESHOLDS = {
        'small': 0.003,   # < 0.3%
        'large': 0.006,   # > 0.6%
    }

    def label_swing(self, df_daily: pd.DataFrame) -> np.ndarray:
        """Label swing direction: 1 if close > open."""
        return (df_daily['close'] > df_daily['open']).astype(int).values

    def label_magnitude(self, df_daily: pd.DataFrame) -> np.ndarray:
        """Label magnitude: 0=SMALL, 1=MEDIUM, 2=LARGE."""
        returns = abs((df_daily['close'] - df_daily['open']) / df_daily['open'])

        labels = np.ones(len(returns), dtype=int)  # Default MEDIUM
        labels[returns < self.MAGNITUDE_THRESHOLDS['small']] = 0  # SMALL
        labels[returns > self.MAGNITUDE_THRESHOLDS['large']] = 2  # LARGE

        return labels

    def label_regime(
        self,
        df_daily: pd.DataFrame,
        df_1min_dict: Dict[str, pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Label regime based on volatility and trend.
        0=QUIET, 1=RANGING, 2=TRENDING, 3=VOLATILE
        """
        labels = []

        for idx in range(len(df_daily)):
            row = df_daily.iloc[idx]
            day_range = (row['high'] - row['low']) / row['open']
            day_return = abs((row['close'] - row['open']) / row['open'])

            # Efficiency ratio: how much of the range was captured by direction
            efficiency = day_return / max(day_range, 0.0001)

            if day_range < 0.005:  # Less than 0.5% range
                labels.append(0)  # QUIET
            elif efficiency > 0.6:  # Most of range was directional
                labels.append(2)  # TRENDING
            elif day_range > 0.015:  # More than 1.5% range
                labels.append(3)  # VOLATILE
            else:
                labels.append(1)  # RANGING

        return np.array(labels)

    def label_pattern(
        self,
        df_daily: pd.DataFrame,
        df_1min_dict: Dict[str, pd.DataFrame] = None
    ) -> np.ndarray:
        """
        Label intraday pattern.
        Requires intraday data for accurate labeling.
        """
        # If no intraday data, use simple heuristics
        labels = []

        for idx in range(len(df_daily)):
            row = df_daily.iloc[idx]
            day_return = (row['close'] - row['open']) / row['open']

            # Simple heuristics without intraday data
            if day_return > 0.004:
                labels.append(5)  # GRIND_UP
            elif day_return < -0.004:
                labels.append(6)  # GRIND_DOWN
            elif abs(day_return) < 0.002:
                labels.append(7)  # CHOP
            else:
                labels.append(5 if day_return > 0 else 6)  # Default to grind

        return np.array(labels)


# =============================================================================
# TEMPORAL CASCADE ENSEMBLE
# =============================================================================
class TemporalCascadeEnsemble:
    """
    Ensemble of models trained on different temporal slices.

    This is the main interface for training and prediction.

    Architecture:
        T0   (Historical only) ─┐
        T30  (+30 min intraday) ─┼─► Weighted Ensemble ─► Final Prediction
        T60  (+60 min intraday) ─┤
        T90  (+90 min intraday) ─┤
        T120 (+120 min intraday)─┤
        T180 (+180 min intraday)─┘
    """

    TEMPORAL_SLICES = [0, 30, 60, 90, 120, 180]

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        regularization_strength: float = 0.1,
    ):
        self.model_type = model_type
        self.regularization_strength = regularization_strength

        # Create models for each temporal slice
        self.models: Dict[int, TemporalSliceModel] = {}
        for ts in self.TEMPORAL_SLICES:
            self.models[ts] = TemporalSliceModel(
                temporal_slice=ts,
                model_type=model_type,
                regularization_strength=regularization_strength,
            )

        # Feature engineer
        self.feature_engineer = TemporalFeatureEngineer()

        # Target labeler
        self.labeler = TemporalTargetLabeler()

        # Tracking
        self.is_fitted = False
        self.training_metrics = {}

    def fit(
        self,
        df_daily: pd.DataFrame,
        df_1min_dict: Dict[str, pd.DataFrame] = None,
        sample_weights: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        Train all temporal slice models.

        Args:
            df_daily: Daily OHLCV data with features
            df_1min_dict: Dictionary mapping date strings to minute-level DataFrames
            sample_weights: Optional sample weights

        Returns:
            Dictionary of training metrics
        """
        logger.info("=" * 70)
        logger.info("TEMPORAL CASCADE ENSEMBLE - Training")
        logger.info("=" * 70)

        # Engineer historical features
        logger.info("[STEP 1] Engineering historical features...")
        df_with_features = self.feature_engineer.engineer_historical_features(df_daily)

        # Create labels
        logger.info("[STEP 2] Creating training labels...")
        y_swing = self.labeler.label_swing(df_daily)
        y_magnitude = self.labeler.label_magnitude(df_daily)
        y_regime = self.labeler.label_regime(df_daily, df_1min_dict)
        y_pattern = self.labeler.label_pattern(df_daily, df_1min_dict)

        logger.info(f"  Swing: {np.sum(y_swing)} UP / {len(y_swing) - np.sum(y_swing)} DOWN")
        logger.info(f"  Magnitude: SMALL={np.sum(y_magnitude==0)}, MED={np.sum(y_magnitude==1)}, LARGE={np.sum(y_magnitude==2)}")

        # Train each temporal slice model
        logger.info("[STEP 3] Training temporal slice models...")

        all_metrics = {}
        for ts in self.TEMPORAL_SLICES:
            logger.info(f"\n[T{ts}] Training model with {ts} minutes of intraday data...")

            # Get features for this temporal slice
            X = self._prepare_features_for_slice(
                df_with_features,
                df_1min_dict,
                ts
            )

            # Remove NaN rows
            valid_mask = ~np.isnan(X).any(axis=1)
            X_valid = X[valid_mask]
            y_swing_valid = y_swing[valid_mask]
            y_magnitude_valid = y_magnitude[valid_mask]
            y_regime_valid = y_regime[valid_mask]
            y_pattern_valid = y_pattern[valid_mask]
            weights_valid = sample_weights[valid_mask] if sample_weights is not None else None

            logger.info(f"  Samples: {len(X_valid)} (after removing NaN)")

            if len(X_valid) < 50:
                logger.warning(f"  [SKIP] Not enough samples for T{ts}")
                continue

            # Train
            metrics = self.models[ts].fit(
                X_valid,
                y_swing_valid,
                y_magnitude_valid,
                y_regime_valid,
                y_pattern_valid,
                weights_valid,
            )

            all_metrics[f'T{ts}'] = metrics

        self.is_fitted = True
        self.training_metrics = all_metrics

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 70)
        for ts in self.TEMPORAL_SLICES:
            if self.models[ts].is_fitted:
                logger.info(f"  T{ts:3d}: CV AUC = {self.models[ts].cv_auc:.4f}")

        return all_metrics

    def _prepare_features_for_slice(
        self,
        df_with_features: pd.DataFrame,
        df_1min_dict: Dict[str, pd.DataFrame],
        temporal_slice: int,
    ) -> np.ndarray:
        """Prepare feature matrix for a specific temporal slice."""
        historical_features = df_with_features[self.feature_engineer.historical_feature_cols].values

        # Add intraday features for this slice
        intraday_features_list = []

        for idx in range(len(df_with_features)):
            if df_1min_dict is not None:
                # Get date for this row
                if 'date' in df_with_features.columns:
                    date_str = str(df_with_features.iloc[idx]['date'])[:10]
                else:
                    date_str = str(df_with_features.index[idx])[:10]

                # Get intraday data for this date
                df_1min = df_1min_dict.get(date_str, pd.DataFrame())
            else:
                df_1min = pd.DataFrame()

            # Engineer intraday features
            intraday_feat = self.feature_engineer.engineer_intraday_features(
                df_1min,
                temporal_slice
            )
            intraday_features_list.append(list(intraday_feat.values()))

        intraday_features = np.array(intraday_features_list)

        # Combine
        X = np.hstack([historical_features, intraday_features])

        return X

    def predict_at_time(
        self,
        historical_features: np.ndarray,
        intraday_features: Dict[str, float],
        minutes_since_open: int,
    ) -> TemporalPrediction:
        """
        Get prediction from the appropriate temporal slice model.

        Args:
            historical_features: Historical feature vector
            intraday_features: Intraday features up to current time
            minutes_since_open: How many minutes since market open

        Returns:
            TemporalPrediction from the appropriate model
        """
        # Find the appropriate temporal slice
        available_slices = [ts for ts in self.TEMPORAL_SLICES
                          if ts <= minutes_since_open and self.models[ts].is_fitted]

        if not available_slices:
            raise ValueError("No models available for this time")

        # Use the most recent available slice
        best_slice = max(available_slices)

        # Combine features
        intraday_array = np.array(list(intraday_features.values()))
        X = np.concatenate([historical_features, intraday_array])

        return self.models[best_slice].predict(X)

    def get_ensemble_prediction(
        self,
        historical_features: np.ndarray,
        intraday_features_by_slice: Dict[int, Dict[str, float]],
        minutes_since_open: int,
        weight_by_recency: bool = True,
    ) -> CascadeEnsemblePrediction:
        """
        Get weighted ensemble prediction from all available temporal models.

        Args:
            historical_features: Historical feature vector
            intraday_features_by_slice: Dict mapping temporal slice to intraday features
            minutes_since_open: Current time in minutes from open
            weight_by_recency: Whether to weight more recent models higher

        Returns:
            CascadeEnsemblePrediction with weighted ensemble
        """
        # Get predictions from all available models
        predictions = []
        weights = []

        available_slices = [ts for ts in self.TEMPORAL_SLICES
                          if ts <= minutes_since_open and self.models[ts].is_fitted]

        for ts in available_slices:
            intraday_feat = intraday_features_by_slice.get(ts, {})
            if not intraday_feat:
                intraday_feat = self.feature_engineer._get_empty_intraday_features()

            intraday_array = np.array(list(intraday_feat.values()))
            X = np.concatenate([historical_features, intraday_array])

            pred = self.models[ts].predict(X)
            predictions.append(pred)

            # Weight by recency and model quality
            if weight_by_recency:
                recency_weight = 1.0 + (ts / 180)  # More recent = higher weight
            else:
                recency_weight = 1.0

            quality_weight = self.models[ts].cv_auc
            weights.append(recency_weight * quality_weight)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Weighted ensemble of swing predictions
        ensemble_swing = sum(p.swing_direction * w for p, w in zip(predictions, weights))
        ensemble_confidence = sum(p.swing_confidence * w for p, w in zip(predictions, weights))

        # Majority vote for categorical predictions
        magnitude_votes = [p.expected_magnitude for p in predictions]
        regime_votes = [p.predicted_regime for p in predictions]
        pattern_votes = [p.predicted_pattern for p in predictions]

        from collections import Counter
        ensemble_magnitude = Counter(magnitude_votes).most_common(1)[0][0]
        ensemble_regime = Counter(regime_votes).most_common(1)[0][0]
        ensemble_pattern = Counter(pattern_votes).most_common(1)[0][0]

        # Calculate agreement score
        swing_directions = [1 if p.swing_direction > 0.5 else 0 for p in predictions]
        agreement = sum(1 for d in swing_directions if d == swing_directions[0]) / len(swing_directions)

        # Prediction stability (how much swing prediction varies across slices)
        swing_values = [p.swing_direction for p in predictions]
        stability = 1.0 - np.std(swing_values) * 2  # Lower std = higher stability
        stability = max(0.0, min(1.0, stability))

        # Determine optimal windows (weighted by recency)
        entry_votes = [p.optimal_entry_window for p in predictions[-3:]]  # Last 3 models
        exit_votes = [p.optimal_exit_window for p in predictions[-3:]]
        optimal_entry = Counter(entry_votes).most_common(1)[0][0]
        optimal_exit = Counter(exit_votes).most_common(1)[0][0]

        # Trading recommendation
        should_trade = (
            ensemble_confidence > 0.3 and
            agreement > 0.6 and
            stability > 0.5
        )

        # Position size based on confidence and agreement
        base_size = 0.10  # 10% base
        confidence_mult = ensemble_confidence
        agreement_mult = agreement
        recommended_size = base_size * confidence_mult * agreement_mult
        recommended_size = min(0.25, max(0.05, recommended_size))  # Cap at 5-25%

        # Entry/exit times based on optimal windows
        window_to_minutes = {
            "VERY_EARLY": 15,
            "EARLY": 45,
            "MID_MORNING": 90,
            "LATE_MORNING": 150,
            "AFTERNOON": 240,
        }
        recommended_entry = window_to_minutes.get(optimal_entry, 45)
        recommended_exit = window_to_minutes.get(optimal_exit, 330)

        return CascadeEnsemblePrediction(
            timestamp=datetime.now().isoformat(),
            minutes_since_open=minutes_since_open,
            models_available=available_slices,
            swing_direction=float(ensemble_swing),
            swing_confidence=float(ensemble_confidence),
            expected_magnitude=ensemble_magnitude,
            predicted_regime=ensemble_regime,
            predicted_pattern=ensemble_pattern,
            optimal_entry_window=optimal_entry,
            optimal_exit_window=optimal_exit,
            model_agreement=float(agreement),
            prediction_stability=float(stability),
            individual_predictions=predictions,
            should_trade=should_trade,
            recommended_position_size=float(recommended_size),
            recommended_entry_time=recommended_entry,
            recommended_exit_time=recommended_exit,
        )

    def save(self, path: Path):
        """Save ensemble to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'model_type': self.model_type,
            'regularization_strength': self.regularization_strength,
            'temporal_slices': self.TEMPORAL_SLICES,
            'models': {},
            'feature_engineer': self.feature_engineer,
            'training_metrics': self.training_metrics,
        }

        for ts, model in self.models.items():
            if model.is_fitted:
                save_dict['models'][ts] = {
                    'swing_model': model.swing_model,
                    'magnitude_model': model.magnitude_model,
                    'regime_model': model.regime_model,
                    'pattern_model': model.pattern_model,
                    'scaler': model.scaler,
                    'cv_auc': model.cv_auc,
                }

        joblib.dump(save_dict, path)
        logger.info(f"Saved temporal cascade ensemble to {path}")

    @classmethod
    def load(cls, path: Path) -> "TemporalCascadeEnsemble":
        """Load ensemble from disk."""
        save_dict = joblib.load(path)

        ensemble = cls(
            model_type=save_dict['model_type'],
            regularization_strength=save_dict['regularization_strength'],
        )

        ensemble.feature_engineer = save_dict['feature_engineer']
        ensemble.training_metrics = save_dict['training_metrics']

        for ts, model_data in save_dict['models'].items():
            ensemble.models[ts].swing_model = model_data['swing_model']
            ensemble.models[ts].magnitude_model = model_data['magnitude_model']
            ensemble.models[ts].regime_model = model_data['regime_model']
            ensemble.models[ts].pattern_model = model_data['pattern_model']
            ensemble.models[ts].scaler = model_data['scaler']
            ensemble.models[ts].cv_auc = model_data['cv_auc']
            ensemble.models[ts].is_fitted = True

        ensemble.is_fitted = True
        logger.info(f"Loaded temporal cascade ensemble from {path}")

        return ensemble


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    print("=" * 70)
    print("TEMPORAL CASCADE MODEL - Demo")
    print("=" * 70)
    print("""
This module implements a Temporal Cascade Ensemble:

  - T0:   Historical features only (pre-market prediction)
  - T30:  Historical + first 30 min of trading
  - T60:  Historical + first 60 min
  - T90:  Historical + first 90 min
  - T120: Historical + first 2 hours
  - T180: Historical + first 3 hours

Each model predicts:
  1. Swing direction (UP/DOWN)
  2. Expected magnitude (SMALL/MEDIUM/LARGE)
  3. Day regime (TRENDING/RANGING/VOLATILE/QUIET)
  4. Intraday pattern (MORNING_DIP, GRIND_UP, etc.)
  5. Optimal entry/exit windows

Benefits:
  - Anti-overfitting through temporal diversity
  - Real-time prediction updates
  - Agreement signals for confidence
  - Adaptive strategy selection
""")

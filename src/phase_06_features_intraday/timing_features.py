"""
GIGA TRADER - Timing Feature Engineering
==========================================
Creates features specifically for predicting optimal entry/exit timing.

Features include:
  - Pre-market activity
  - Previous day patterns
  - Volatility regime
  - Day of week / calendar effects
  - Recent momentum
"""

from datetime import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TimingFeatureEngineer:
    """
    Creates features specifically for predicting optimal entry/exit timing.

    Features include:
      - Pre-market activity
      - Previous day patterns
      - Volatility regime
      - Day of week / calendar effects
      - Recent momentum
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []

    def create_features(
        self,
        daily_data: pd.DataFrame,
        intraday_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Create timing-specific features.

        Args:
            daily_data: Daily OHLCV data
            intraday_data: Optional intraday data for premarket features

        Returns:
            DataFrame with timing features
        """
        features = pd.DataFrame(index=daily_data.index)

        # Previous day patterns
        features['prev_return'] = daily_data['close'].pct_change()
        features['prev_range'] = (daily_data['high'] - daily_data['low']) / daily_data['open']
        features['prev_body'] = abs(daily_data['close'] - daily_data['open']) / daily_data['open']
        features['prev_upper_wick'] = (daily_data['high'] - daily_data[['open', 'close']].max(axis=1)) / daily_data['open']
        features['prev_lower_wick'] = (daily_data[['open', 'close']].min(axis=1) - daily_data['low']) / daily_data['open']

        # Where did high/low occur (approximation using body position)
        features['body_position'] = (daily_data['close'] - daily_data['low']) / (daily_data['high'] - daily_data['low'] + 1e-8)

        # Volatility regime
        features['volatility_5d'] = daily_data['close'].pct_change().rolling(5).std()
        features['volatility_20d'] = daily_data['close'].pct_change().rolling(20).std()
        features['volatility_ratio'] = features['volatility_5d'] / (features['volatility_20d'] + 1e-8)

        # Range patterns
        features['range_5d_avg'] = ((daily_data['high'] - daily_data['low']) / daily_data['open']).rolling(5).mean()
        features['range_vs_avg'] = features['prev_range'] / (features['range_5d_avg'] + 1e-8)

        # Momentum
        features['momentum_3d'] = daily_data['close'].pct_change(3)
        features['momentum_5d'] = daily_data['close'].pct_change(5)
        features['momentum_10d'] = daily_data['close'].pct_change(10)

        # Trend strength
        features['close_vs_ma5'] = daily_data['close'] / daily_data['close'].rolling(5).mean() - 1
        features['close_vs_ma20'] = daily_data['close'] / daily_data['close'].rolling(20).mean() - 1

        # Calendar features
        if hasattr(daily_data.index, 'dayofweek'):
            features['day_of_week'] = daily_data.index.dayofweek
        elif 'date' in daily_data.columns:
            features['day_of_week'] = pd.to_datetime(daily_data['date']).dt.dayofweek

        features['is_monday'] = (features['day_of_week'] == 0).astype(int)
        features['is_friday'] = (features['day_of_week'] == 4).astype(int)

        # Volume patterns
        if 'volume' in daily_data.columns:
            features['volume_ratio'] = daily_data['volume'] / daily_data['volume'].rolling(20).mean()
            features['volume_trend'] = daily_data['volume'].pct_change(5)

        # Gap
        features['gap'] = daily_data['open'] / daily_data['close'].shift(1) - 1
        features['gap_filled_prev'] = ((daily_data['low'].shift(1) <= daily_data['close'].shift(2)) |
                                        (daily_data['high'].shift(1) >= daily_data['close'].shift(2))).astype(int)

        # Premarket features (if available)
        if intraday_data is not None:
            premarket_features = self._extract_premarket_features(intraday_data, daily_data.index)
            features = features.join(premarket_features)

        # Historical timing patterns (what worked on similar days)
        features['similar_day_entry'] = self._similar_day_pattern(features, 'entry')
        features['similar_day_exit'] = self._similar_day_pattern(features, 'exit')

        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        self.feature_names = features.columns.tolist()

        return features

    def _extract_premarket_features(
        self,
        intraday_data: pd.DataFrame,
        dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Extract pre-market features (4:00 AM - 9:30 AM)."""
        premarket_features = []

        for date in dates:
            day_data = intraday_data[intraday_data.index.date == date.date()]

            # Filter to premarket (before 9:30 AM)
            premarket = day_data[day_data.index.time < time(9, 30)]

            if len(premarket) > 0:
                pm_high = premarket['high'].max()
                pm_low = premarket['low'].min()
                pm_close = premarket['close'].iloc[-1]
                pm_volume = premarket['volume'].sum()
                prev_close = day_data['close'].iloc[0] if len(day_data) > 0 else pm_close

                premarket_features.append({
                    'date': date,
                    'pm_range': (pm_high - pm_low) / pm_close,
                    'pm_direction': (pm_close - premarket['open'].iloc[0]) / premarket['open'].iloc[0],
                    'pm_volume_pct': pm_volume / day_data['volume'].sum() if day_data['volume'].sum() > 0 else 0,
                    'pm_vs_prev_close': pm_close / prev_close - 1 if prev_close > 0 else 0,
                })
            else:
                premarket_features.append({
                    'date': date,
                    'pm_range': 0,
                    'pm_direction': 0,
                    'pm_volume_pct': 0,
                    'pm_vs_prev_close': 0,
                })

        return pd.DataFrame(premarket_features).set_index('date')

    def _similar_day_pattern(
        self,
        features: pd.DataFrame,
        pattern_type: str,
    ) -> pd.Series:
        """Find similar historical days and their patterns."""
        # Simplified - returns rolling mean of volatility as proxy
        # In production, would use KNN or clustering
        if pattern_type == 'entry':
            return features['volatility_5d'].rolling(20).mean() * 60  # Minutes
        else:
            return 300 + features['volatility_5d'].rolling(20).mean() * 60

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit scaler and transform features."""
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        return self.scaler.fit_transform(X_clean)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted scaler."""
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        return self.scaler.transform(X_clean)

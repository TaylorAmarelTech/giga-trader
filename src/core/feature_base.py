"""
Feature Module Base Class
=========================
Provides shared infrastructure for all feature modules in phase_08_features_breadth/.

Eliminates per-module boilerplate for:
  - Input validation (required columns check)
  - NaN/Inf cleanup
  - Zero-fill fallback
  - Feature name registry
"""

import logging
from typing import ClassVar, List, Set

import numpy as np
import pandas as pd


class FeatureModuleBase:
    """Base class for all feature engineering modules.

    Subclasses MUST define:
        REQUIRED_COLS: set of column names that must be present in input df
        FEATURE_NAMES: list of output column names this module produces
    """

    REQUIRED_COLS: ClassVar[Set[str]] = set()
    FEATURE_NAMES: ClassVar[List[str]] = []

    def _validate_input(self, df: pd.DataFrame, min_rows: int = 2) -> bool:
        """Check that required columns exist and df has enough rows.

        Returns True if valid, False otherwise. Logs a warning on failure.
        """
        logger = logging.getLogger(self.__class__.__name__)
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            logger.warning(f"{self.__class__.__name__}: missing columns {missing}, skipping")
            return False
        if len(df) < min_rows:
            logger.warning(f"{self.__class__.__name__}: only {len(df)} rows (need {min_rows}), skipping")
            return False
        return True

    def _cleanup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace NaN and Inf with 0.0 for all FEATURE_NAMES columns."""
        for col in self.FEATURE_NAMES:
            if col in df.columns:
                df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)
        return df

    def _zero_fill_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set all FEATURE_NAMES columns to 0.0 (fallback when data unavailable)."""
        for col in self.FEATURE_NAMES:
            df[col] = 0.0
        return df

    @classmethod
    def _all_feature_names(cls) -> List[str]:
        """Return list of feature names. Backward-compatible class method."""
        return list(cls.FEATURE_NAMES)

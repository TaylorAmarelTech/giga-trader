"""
GIGA TRADER - Feature Neutralizer
===================================
Sklearn-compatible transformer that removes market beta / style exposures
from features before model training.

This helps the model learn alpha signals rather than simply tracking
market moves.  Three neutralization methods are supported:

1. ``demeaning``  -- Subtract column means (cross-sectional neutralization).
2. ``residual``   -- For each feature, regress out a market factor and keep
   the residuals.
3. ``winsorized_residual`` -- Same as residual but winsorize features at
   configurable percentiles before regression.

Usage
-----
>>> from src.phase_10_feature_processing.feature_neutralizer import FeatureNeutralizer
>>> neutralizer = FeatureNeutralizer(method="residual", market_col_idx=0)
>>> X_train_neutral = neutralizer.fit_transform(X_train)
>>> X_test_neutral  = neutralizer.transform(X_test)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class FeatureNeutralizer(BaseEstimator, TransformerMixin):
    """
    Remove market beta / style exposures from feature columns.

    Parameters
    ----------
    method : str, default ``"demeaning"``
        Neutralization strategy.  One of:

        * ``"demeaning"`` -- subtract column means learned at fit time.
        * ``"residual"`` -- regress each feature on a market factor and
          keep the residual.
        * ``"winsorized_residual"`` -- same as residual but winsorize
          features at ``winsorize_pct`` / ``1 - winsorize_pct`` before
          regression.

    market_col_idx : int, default 0
        Column index used as the market factor for ``residual`` and
        ``winsorized_residual`` methods.  Ignored for ``demeaning``.

    winsorize_pct : float, default 0.01
        Lower percentile bound for winsorization.  Upper bound is
        ``1 - winsorize_pct``.  Only used when
        ``method="winsorized_residual"``.

    min_samples : int, default 50
        Minimum number of rows required for fitting.  If ``X`` has fewer
        rows, the transformer passes through data unchanged.
    """

    # Valid method names -------------------------------------------------------
    _VALID_METHODS = ("demeaning", "residual", "winsorized_residual")

    def __init__(
        self,
        method: str = "demeaning",
        market_col_idx: int = 0,
        winsorize_pct: float = 0.01,
        min_samples: int = 50,
    ):
        self.method = method
        self.market_col_idx = market_col_idx
        self.winsorize_pct = winsorize_pct
        self.min_samples = min_samples

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        market_returns: Optional[np.ndarray] = None,
    ) -> "FeatureNeutralizer":
        """
        Learn neutralization parameters from *training* data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : ignored
            Present for sklearn pipeline compatibility.
        market_returns : ndarray of shape (n_samples,), optional
            Explicit market factor.  If ``None`` the column at
            ``market_col_idx`` is used.

        Returns
        -------
        self
        """
        if self.method not in self._VALID_METHODS:
            raise ValueError(
                f"Unknown method '{self.method}'. "
                f"Choose from {self._VALID_METHODS}."
            )

        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape

        # Store shape info
        self.n_features_in_: int = n_features
        self.n_samples_seen_: int = n_samples

        # If too few samples, record pass-through mode and return early.
        if n_samples < self.min_samples:
            self.passthrough_: bool = True
            self.is_fitted_: bool = True
            return self
        self.passthrough_ = False

        # Replace NaN with 0 for regression but note the original positions.
        X_clean = np.where(np.isfinite(X), X, 0.0)

        if self.method == "demeaning":
            self.means_: np.ndarray = np.mean(X_clean, axis=0)

        elif self.method in ("residual", "winsorized_residual"):
            # Determine market factor
            if market_returns is not None:
                mkt = np.asarray(market_returns, dtype=np.float64).ravel()
            else:
                if self.market_col_idx >= n_features:
                    raise ValueError(
                        f"market_col_idx={self.market_col_idx} "
                        f"but X only has {n_features} columns."
                    )
                mkt = X_clean[:, self.market_col_idx].copy()

            mkt = np.where(np.isfinite(mkt), mkt, 0.0)

            # If winsorized_residual, compute and store percentile bounds
            if self.method == "winsorized_residual":
                lo = self.winsorize_pct * 100.0
                hi = (1.0 - self.winsorize_pct) * 100.0
                self.clip_lo_: np.ndarray = np.percentile(X_clean, lo, axis=0)
                self.clip_hi_: np.ndarray = np.percentile(X_clean, hi, axis=0)
                # Apply winsorization before regression
                X_reg = np.clip(X_clean, self.clip_lo_, self.clip_hi_)
            else:
                X_reg = X_clean

            # Fit per-feature linear regression: feature_j = beta_j * mkt + intercept_j
            betas = np.zeros(n_features, dtype=np.float64)
            intercepts = np.zeros(n_features, dtype=np.float64)

            # Build the design matrix once: [[mkt_0, 1], [mkt_1, 1], ...]
            A = np.column_stack([mkt, np.ones(n_samples, dtype=np.float64)])

            for j in range(n_features):
                col = X_reg[:, j]
                # Skip constant columns (std == 0)
                if np.std(col) == 0.0:
                    betas[j] = 0.0
                    intercepts[j] = np.mean(col)
                    continue
                # lstsq: solve A @ [beta, intercept] = col
                result, _, _, _ = np.linalg.lstsq(A, col, rcond=None)
                betas[j] = result[0]
                intercepts[j] = result[1]

            self.betas_: np.ndarray = betas
            self.intercepts_: np.ndarray = intercepts

        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Neutralize features using parameters learned during ``fit``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix (train or test).

        Returns
        -------
        X_neutral : ndarray of shape (n_samples, n_features)
            Neutralized features.
        """
        check_is_fitted(self, "is_fitted_")

        X = np.asarray(X, dtype=np.float64)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features but FeatureNeutralizer was "
                f"fitted on {self.n_features_in_} features."
            )

        # Pass-through when training had too few samples
        if self.passthrough_:
            return X.copy()

        # Record NaN positions to restore later
        nan_mask = ~np.isfinite(X)
        X_clean = np.where(np.isfinite(X), X, 0.0)

        if self.method == "demeaning":
            X_out = X_clean - self.means_

        elif self.method in ("residual", "winsorized_residual"):
            # Get market factor from the transform-time data
            mkt = X_clean[:, self.market_col_idx].copy()

            if self.method == "winsorized_residual":
                X_work = np.clip(X_clean, self.clip_lo_, self.clip_hi_)
            else:
                X_work = X_clean

            # Subtract fitted market exposure
            # residual_j = X_j - (beta_j * mkt + intercept_j)
            X_out = X_work - (self.betas_[np.newaxis, :] * mkt[:, np.newaxis]
                              + self.intercepts_[np.newaxis, :])
        else:
            X_out = X_clean  # pragma: no cover

        # Restore NaN positions
        X_out[nan_mask] = np.nan

        return X_out

    # ------------------------------------------------------------------
    # fit_transform (explicit override for market_returns passthrough)
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        market_returns: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y=y, market_returns=market_returns).transform(X)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        params = (
            f"method='{self.method}', "
            f"market_col_idx={self.market_col_idx}, "
            f"winsorize_pct={self.winsorize_pct}, "
            f"min_samples={self.min_samples}"
        )
        return f"FeatureNeutralizer({params})"

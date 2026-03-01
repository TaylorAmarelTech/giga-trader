"""
GIGA TRADER - Meta-Labeling Module
=====================================
Secondary classifier that predicts when primary model signals are profitable.

Concept (Lopez de Prado, "Advances in Financial Machine Learning"):
  - Primary model predicts direction (swing) and timing
  - Meta-labeler predicts WHETHER those predictions will be profitable
  - Output probability feeds position sizing (Half Kelly)

Training:
  - Trained on TEST fold data only (never sees training data)
  - Uses nested StratifiedKFold(3) within test fold
  - Requires 30+ signals with balanced classes (15+ each)
  - Meta-features: original features + swing_proba + timing_proba
  - Meta-target: was this signal profitable after transaction costs?

Inference:
  - predict(X, swing_proba, timing_proba) -> meta_proba (0-1)
  - meta_proba scales position_size (low confidence = smaller position)
  - With Kelly: position_size = half_kelly(meta_proba, win_loss_ratio)
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("META_LABELER")


def half_kelly_fraction(p: float, b: float) -> float:
    """Compute half-Kelly bet fraction.

    Args:
        p: Probability of winning (0-1)
        b: Win/loss ratio (avg_win / avg_loss)

    Returns:
        Optimal fraction of bankroll to bet (0 if no edge)
    """
    if b <= 0 or p <= 0 or p >= 1:
        return 0.0
    q = 1 - p
    edge = p * b - q
    if edge <= 0:
        return 0.0
    return 0.5 * edge / b


class MetaLabeler:
    """Secondary classifier predicting when primary signals are profitable.

    Learns WHEN to trade, not WHAT direction.
    """

    def __init__(
        self,
        C: float = 1.0,
        min_signals: int = 30,
        min_per_class: int = 15,
        cv_folds: int = 3,
    ):
        self.C = C
        self.min_signals = min_signals
        self.min_per_class = min_per_class
        self.cv_folds = cv_folds
        self.model_ = None
        self.meta_auc_ = 0.0
        self.is_fitted_ = False
        self.n_meta_features_ = 0

    def fit(
        self,
        X_test: np.ndarray,
        swing_proba: np.ndarray,
        timing_proba: np.ndarray,
        signals: np.ndarray,
        returns: np.ndarray,
        slippage_bps: float = 5,
        commission_bps: float = 1,
    ) -> Dict:
        """Train meta-model on test-fold data.

        Args:
            X_test: Feature matrix from test fold (n_samples, n_features)
            swing_proba: Primary swing model probabilities (n_samples,)
            timing_proba: Primary timing model probabilities (n_samples,)
            signals: Binary trading signals (n_samples,) - 1=trade, 0=no trade
            returns: Daily returns (n_samples,)
            slippage_bps: Slippage per trade in basis points
            commission_bps: Commission per trade in basis points

        Returns:
            dict with 'fitted', 'meta_auc', 'n_signals', 'n_wins', 'n_losses', 'reason'
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        X_test = np.asarray(X_test)
        swing_proba = np.asarray(swing_proba).ravel()
        timing_proba = np.asarray(timing_proba).ravel()
        signals = np.asarray(signals).ravel()
        returns = np.asarray(returns).ravel()

        # 1. Filter to signaled days only
        signal_mask = signals.astype(bool)
        n_signals = int(signal_mask.sum())

        if n_signals < self.min_signals:
            return {
                "fitted": False,
                "reason": f"too_few_signals ({n_signals} < {self.min_signals})",
            }

        # 2. Build meta-features for signaled days
        X_meta = np.column_stack([
            X_test[signal_mask],
            swing_proba[signal_mask].reshape(-1, 1),
            timing_proba[signal_mask].reshape(-1, 1),
        ])

        # 3. Create meta-targets: was this signal profitable after costs?
        cost_per_trade = 2 * (slippage_bps + commission_bps) / 10000
        signal_returns = returns[signal_mask] - cost_per_trade
        meta_target = (signal_returns > 0).astype(int)

        # 4. Check class balance
        n_wins = int(meta_target.sum())
        n_losses = int(len(meta_target) - n_wins)

        if n_wins < self.min_per_class or n_losses < self.min_per_class:
            return {
                "fitted": False,
                "reason": f"class_imbalance (wins={n_wins}, losses={n_losses})",
            }

        # 5. Nested CV to evaluate meta-model
        base_lr = LogisticRegression(C=self.C, max_iter=500, random_state=42)
        n_splits = min(self.cv_folds, n_wins, n_losses)
        if n_splits < 2:
            return {"fitted": False, "reason": "insufficient_folds"}

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(
            base_lr, X_meta, meta_target, cv=cv, scoring="roc_auc"
        )
        self.meta_auc_ = float(np.mean(scores))

        # 6. Guard: meta-AUC must be meaningfully above random
        if self.meta_auc_ < 0.52:
            return {
                "fitted": False,
                "reason": f"meta_auc_too_low ({self.meta_auc_:.3f} < 0.52)",
                "meta_auc": round(self.meta_auc_, 4),
            }

        # 7. Fit calibrated model on all signaled data
        cal_cv = min(n_splits, n_wins, n_losses)
        if cal_cv < 2:
            cal_cv = 2

        self.model_ = CalibratedClassifierCV(
            LogisticRegression(C=self.C, max_iter=500, random_state=42),
            cv=cal_cv,
            method="sigmoid",
        )
        self.model_.fit(X_meta, meta_target)
        self.is_fitted_ = True
        self.n_meta_features_ = X_meta.shape[1]

        return {
            "fitted": True,
            "meta_auc": round(self.meta_auc_, 4),
            "n_signals": n_signals,
            "n_wins": n_wins,
            "n_losses": n_losses,
        }

    def predict(
        self,
        X: np.ndarray,
        swing_proba: np.ndarray,
        timing_proba: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Predict probability that signal is profitable.

        Args:
            X: Feature matrix (n_samples, n_features)
            swing_proba: Swing model probabilities (n_samples,)
            timing_proba: Timing model probabilities (n_samples,)

        Returns:
            Array of meta-probabilities (0-1), or None if not fitted.
        """
        if not self.is_fitted_ or self.model_ is None:
            return None

        X = np.asarray(X)
        swing_proba = np.asarray(swing_proba).ravel()
        timing_proba = np.asarray(timing_proba).ravel()

        # Handle single-sample case
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_meta = np.column_stack([
            X,
            swing_proba.reshape(-1, 1),
            timing_proba.reshape(-1, 1),
        ])

        try:
            return self.model_.predict_proba(X_meta)[:, 1]
        except Exception as e:
            logger.warning(f"Meta-label prediction failed: {e}")
            return None

    def save(self, path: Path):
        """Save meta-model to joblib file."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            "model": self.model_,
            "meta_auc": self.meta_auc_,
            "is_fitted": self.is_fitted_,
            "n_meta_features": self.n_meta_features_,
            "C": self.C,
            "min_signals": self.min_signals,
            "min_per_class": self.min_per_class,
            "cv_folds": self.cv_folds,
        }, path)

    @classmethod
    def load(cls, path: Path) -> Optional["MetaLabeler"]:
        """Load meta-model from joblib file. Returns None if not found."""
        import joblib

        path = Path(path)
        if not path.is_file():
            return None

        try:
            data = joblib.load(path)
            obj = cls(
                C=data.get("C", 1.0),
                min_signals=data.get("min_signals", 30),
                min_per_class=data.get("min_per_class", 15),
                cv_folds=data.get("cv_folds", 3),
            )
            obj.model_ = data["model"]
            obj.meta_auc_ = data.get("meta_auc", 0.0)
            obj.is_fitted_ = data.get("is_fitted", False)
            obj.n_meta_features_ = data.get("n_meta_features", 0)
            return obj
        except Exception as e:
            logger.warning(f"Failed to load meta-labeler: {e}")
            return None

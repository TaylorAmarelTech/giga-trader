"""
Kolmogorov-Arnold Network (KAN) Classifier
============================================

A pure-NumPy implementation of a Kolmogorov-Arnold Network for binary
classification, compatible with the sklearn classifier interface.

KANs differ from standard MLPs: instead of fixed activation functions on
nodes, they use **learnable B-spline activations on edges**. Each edge
transforms its input through a linear combination of B-spline basis
functions with learnable coefficients, replacing the traditional
weight * activation(x) paradigm.

Architecture (per EDGE 1 -- shallow, regularization-first):
    Input (n_features) -> Hidden (hidden_dim, B-spline edges) -> Output (1, sigmoid)

Training:
    - Mini-batch gradient descent with Adam optimizer
    - Binary cross-entropy loss
    - L1 regularization on spline coefficients (feature sparsity)
    - Early stopping with patience
    - Grid pruning: near-zero coefficients set to exactly zero

Feature importance:
    L1 norm of spline coefficients per input feature, normalized to sum to 1.

Usage:
    from src.phase_12_model_training.kan_model import KANClassifier

    clf = KANClassifier(hidden_dim=16, grid_size=5, l1_lambda=0.01)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)

References:
    Liu et al. (2024) "KAN: Kolmogorov-Arnold Networks"
"""

import logging
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)


# ======================================================================
# B-spline utilities
# ======================================================================

def _make_knot_vector(grid_size: int, order: int) -> np.ndarray:
    """Create a clamped uniform knot vector on [0, 1].

    Parameters
    ----------
    grid_size : int
        Number of interior intervals.
    order : int
        Spline order (degree + 1).

    Returns
    -------
    knots : ndarray
        Clamped knot vector with repeated boundary knots.
    """
    degree = order - 1
    interior = np.linspace(0.0, 1.0, grid_size + 1)
    return np.concatenate([np.zeros(degree), interior, np.ones(degree)])


def _bspline_basis_batch(
    X_flat: np.ndarray, knots: np.ndarray, order: int
) -> np.ndarray:
    """Evaluate B-spline basis functions for a batch of scalar inputs.

    Fully vectorized -- no Python loops over samples or columns. The caller
    is responsible for flattening/reshaping as needed.

    Parameters
    ----------
    X_flat : ndarray of shape (N,)
        All evaluation points (already normalized to [0, 1]).
    knots : ndarray of shape (n_knots,)
        Non-decreasing knot vector including repeated boundary knots.
    order : int
        Spline order (degree + 1). 4 gives cubic splines.

    Returns
    -------
    basis : ndarray of shape (N, n_basis)
        n_basis = len(knots) - order.
    """
    n_knots = len(knots)
    N = len(X_flat)
    n_intervals = n_knots - 1

    # Degree-0 basis: piecewise constants
    # B_i^0(x) = 1 if knots[i] <= x < knots[i+1], else 0
    # Shape: (N, n_intervals)
    k_lo = knots[:-1]  # (n_intervals,)
    k_hi = knots[1:]   # (n_intervals,)
    x = X_flat[:, None]  # (N, 1)

    # Non-degenerate intervals
    nondegenerate = (k_lo < k_hi)  # (n_intervals,)
    B = ((x >= k_lo) & (x < k_hi) & nondegenerate).astype(np.float64)

    # Handle right boundary: assign x == knots[-1] to the last
    # non-degenerate interval
    right_mask = (X_flat == knots[-1])  # (N,)
    if right_mask.any():
        # Find last non-degenerate interval index
        nondeg_idx = np.where(nondegenerate)[0]
        if len(nondeg_idx) > 0:
            last_idx = nondeg_idx[-1]
            B[right_mask, last_idx] = 1.0

    # Cox-de Boor recursion for degrees 1 .. order-1
    for d in range(1, order):
        n_funcs = n_knots - 1 - d
        B_new = np.zeros((N, n_funcs), dtype=np.float64)

        # Left term: (x - knots[i]) / (knots[i+d] - knots[i]) * B[..., i]
        k_left_lo = knots[:n_funcs]        # (n_funcs,)
        k_left_hi = knots[d:d + n_funcs]   # (n_funcs,)
        denom_left = k_left_hi - k_left_lo  # (n_funcs,)
        safe_left = (denom_left > 0)
        if safe_left.any():
            coeff_left = np.where(
                safe_left,
                (X_flat[:, None] - k_left_lo) / np.where(safe_left, denom_left, 1.0),
                0.0,
            )
            B_new += coeff_left * B[:, :n_funcs]

        # Right term: (knots[i+d+1] - x) / (knots[i+d+1] - knots[i+1]) * B[..., i+1]
        k_right_hi = knots[d + 1:d + 1 + n_funcs]  # (n_funcs,)
        k_right_lo = knots[1:1 + n_funcs]            # (n_funcs,)
        denom_right = k_right_hi - k_right_lo        # (n_funcs,)
        safe_right = (denom_right > 0)
        if safe_right.any():
            coeff_right = np.where(
                safe_right,
                (k_right_hi - X_flat[:, None]) / np.where(safe_right, denom_right, 1.0),
                0.0,
            )
            B_new += coeff_right * B[:, 1:1 + n_funcs]

        B = B_new

    n_basis = n_knots - order
    return B[:, :n_basis]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    x_safe = np.clip(x, -500.0, 500.0)
    return np.where(
        x_safe >= 0,
        1.0 / (1.0 + np.exp(-x_safe)),
        np.exp(x_safe) / (1.0 + np.exp(x_safe)),
    )


# ======================================================================
# KAN Classifier
# ======================================================================

class KANClassifier(ClassifierMixin, BaseEstimator):
    """Kolmogorov-Arnold Network classifier with B-spline edge activations.

    Each edge in the network applies a learnable B-spline transformation
    instead of the traditional weight * activation(node) pattern. This
    allows the network to learn complex nonlinear transformations while
    remaining shallow and interpretable.

    Follows EDGE 1 (Regularization-First):
        - Single hidden layer only (shallow architecture)
        - Heavy L1 regularization on spline coefficients
        - Small hidden_dim (8-32 recommended)
        - Grid pruning: near-zero coefficients snapped to zero

    Note: ClassifierMixin before BaseEstimator (sklearn 1.6+ tags).

    Parameters
    ----------
    hidden_dim : int, default=16
        Width of the hidden layer.
    grid_size : int, default=5
        Number of B-spline grid intervals per edge.
    spline_order : int, default=3
        B-spline order (3 = cubic). Degree = order - 1.
    l1_lambda : float, default=0.01
        L1 regularization strength on spline coefficients.
    learning_rate : float, default=0.01
        Adam optimizer learning rate.
    max_epochs : int, default=100
        Maximum training epochs.
    batch_size : int, default=64
        Mini-batch size for gradient descent.
    patience : int, default=10
        Early stopping patience (epochs without improvement).
    prune_threshold : float, default=1e-4
        Spline coefficients with absolute value below this are set to zero.
    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        Class labels [0, 1]. Set after fit().
    feature_importances_ : ndarray of shape (n_features,)
        L1 norm of input-layer spline coefficients per feature,
        normalized to sum to 1. Set after fit().
    n_features_in_ : int
        Number of features seen during fit().
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        grid_size: int = 5,
        spline_order: int = 3,
        l1_lambda: float = 0.01,
        learning_rate: float = 0.01,
        max_epochs: int = 100,
        batch_size: int = 64,
        patience: int = 10,
        prune_threshold: float = 1e-4,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.l1_lambda = l1_lambda
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.prune_threshold = prune_threshold
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _n_basis(self) -> int:
        """Number of B-spline basis functions per edge."""
        return self.grid_size + self.spline_order - 1

    def _init_params(self, n_features: int) -> None:
        """Initialize all learnable parameters.

        Layer 1 (input -> hidden):
            coeff1: shape (n_features, hidden_dim, n_basis)
            bias1:  shape (hidden_dim,)

        Layer 2 (hidden -> output):
            coeff2: shape (hidden_dim, n_basis)
            bias2:  scalar
        """
        rng = np.random.RandomState(self.random_state)
        nb = self._n_basis()

        scale1 = np.sqrt(2.0 / (n_features + self.hidden_dim))
        scale2 = np.sqrt(2.0 / (self.hidden_dim + 1))

        self._coeff1 = rng.randn(n_features, self.hidden_dim, nb) * scale1
        self._bias1 = np.zeros(self.hidden_dim)
        self._coeff2 = rng.randn(self.hidden_dim, nb) * scale2
        self._bias2 = np.zeros(1)

        self._knots = _make_knot_vector(self.grid_size, self.spline_order)

        # Adam optimizer state
        self._adam_state = {}
        for name in ("coeff1", "bias1", "coeff2", "bias2"):
            self._adam_state[name] = {"m": 0.0, "v": 0.0, "t": 0}

    def _normalize_input(self, X: np.ndarray) -> np.ndarray:
        """Normalize inputs to [0, 1] using fitted min/max."""
        return np.clip((X - self._x_min) / (self._x_range + 1e-10), 0.0, 1.0)

    def _compute_basis(self, X_norm: np.ndarray) -> np.ndarray:
        """Compute B-spline basis for all features at once (vectorized).

        Parameters
        ----------
        X_norm : ndarray of shape (n_samples, n_cols)
            Normalized values in [0, 1].

        Returns
        -------
        basis : ndarray of shape (n_samples, n_cols, n_basis)
        """
        n_samples, n_cols = X_norm.shape
        nb = self._n_basis()
        # Flatten all columns, compute basis in one call, reshape
        flat = X_norm.T.ravel()  # (n_cols * n_samples,)
        B_flat = _bspline_basis_batch(flat, self._knots, self.spline_order)
        # B_flat: (n_cols * n_samples, nb)
        # Reshape to (n_cols, n_samples, nb) then transpose to (n_samples, n_cols, nb)
        return B_flat.reshape(n_cols, n_samples, nb).transpose(1, 0, 2)

    def _forward(self, X_norm: np.ndarray) -> dict:
        """Forward pass through the KAN (fully vectorized).

        Parameters
        ----------
        X_norm : ndarray of shape (n_samples, n_features)

        Returns
        -------
        cache : dict with intermediate values for backpropagation.
        """
        n_samples, n_features = X_norm.shape

        # --- Layer 1: input -> hidden via B-spline edges ---
        # basis1: (n_samples, n_features, nb)
        basis1 = self._compute_basis(X_norm)

        # hidden_pre[s, k] = sum_j basis1[s, j, :] @ coeff1[j, k, :]  + bias1[k]
        # Using einsum: contract over j and nb dimensions
        # basis1:  (S, J, B)
        # coeff1:  (J, K, B)
        # result:  (S, K) = sum over J,B of basis1[S,J,B] * coeff1[J,K,B]
        hidden_pre = np.einsum("sjb,jkb->sk", basis1, self._coeff1) + self._bias1

        # Node activation: ReLU
        hidden_act = np.maximum(hidden_pre, 0.0)

        # --- Layer 2: hidden -> output via B-spline edges ---
        # Normalize hidden activations to [0, 1] for spline input
        h_max = hidden_act.max(axis=0, keepdims=True) + 1e-10  # (1, K)
        hidden_norm = np.clip(hidden_act / h_max, 0.0, 1.0)

        # basis2: (n_samples, hidden_dim, nb)
        basis2 = self._compute_basis(hidden_norm)

        # logit[s] = sum_k basis2[s, k, :] @ coeff2[k, :]  + bias2
        # basis2: (S, K, B), coeff2: (K, B) -> (S, K) -> sum over K -> (S,)
        logit = np.einsum("skb,kb->s", basis2, self._coeff2) + self._bias2[0]

        prob = _sigmoid(logit)

        return {
            "X_norm": X_norm,
            "basis1": basis1,
            "hidden_pre": hidden_pre,
            "hidden_act": hidden_act,
            "h_max": h_max,
            "hidden_norm": hidden_norm,
            "basis2": basis2,
            "logit": logit,
            "prob": prob,
        }

    def _backward(
        self, cache: dict, y: np.ndarray, sample_weight: Optional[np.ndarray]
    ) -> dict:
        """Backward pass computing gradients (vectorized).

        Parameters
        ----------
        cache : dict from _forward.
        y : ndarray of shape (n_batch,)
        sample_weight : ndarray of shape (n_batch,) or None

        Returns
        -------
        grads : dict of parameter gradients.
        """
        n = len(y)

        # BCE gradient: d_loss/d_logit = (prob - y) / n
        d_logit = cache["prob"] - y
        if sample_weight is not None:
            d_logit = d_logit * sample_weight
        d_logit = d_logit / n  # (S,)

        # --- Layer 2 gradients ---
        # logit = einsum("skb,kb->s", basis2, coeff2) + bias2
        # d_coeff2[k, b] = sum_s basis2[s, k, b] * d_logit[s]
        d_coeff2 = np.einsum("skb,s->kb", cache["basis2"], d_logit)
        d_bias2 = np.array([d_logit.sum()])

        # --- Gradient through layer-2 splines to hidden_norm ---
        # d_loss/d_hidden_norm[s, k] via numerical differentiation of the
        # spline function. This avoids computing analytic B-spline derivatives.
        eps = 1e-5
        h_norm = cache["hidden_norm"]  # (S, K)
        h_plus = np.clip(h_norm + eps, 0.0, 1.0)
        h_minus = np.clip(h_norm - eps, 0.0, 1.0)

        basis_plus = self._compute_basis(h_plus)    # (S, K, B)
        basis_minus = self._compute_basis(h_minus)  # (S, K, B)

        # spline output per hidden unit: basis @ coeff2  -> (S, K)
        out_plus = np.einsum("skb,kb->sk", basis_plus, self._coeff2)
        out_minus = np.einsum("skb,kb->sk", basis_minus, self._coeff2)
        dspline_dh = (out_plus - out_minus) / (2 * eps)  # (S, K)

        # d_loss/d_hidden_norm = dspline_dh * d_logit (broadcast)
        d_hidden_norm = dspline_dh * d_logit[:, None]  # (S, K)

        # hidden_norm = hidden_act / h_max
        d_hidden_act = d_hidden_norm / (cache["h_max"] + 1e-10)

        # ReLU gradient
        d_hidden_pre = d_hidden_act * (cache["hidden_pre"] > 0).astype(np.float64)

        # --- Layer 1 gradients ---
        # hidden_pre = einsum("sjb,jkb->sk", basis1, coeff1) + bias1
        # d_coeff1[j, k, b] = sum_s basis1[s, j, b] * d_hidden_pre[s, k]
        d_coeff1 = np.einsum("sjb,sk->jkb", cache["basis1"], d_hidden_pre)
        d_bias1 = d_hidden_pre.sum(axis=0)

        # L1 regularization sub-gradient
        lam = self.l1_lambda
        d_coeff1 = d_coeff1 + lam * np.sign(self._coeff1)
        d_coeff2 = d_coeff2 + lam * np.sign(self._coeff2)

        return {
            "coeff1": d_coeff1,
            "bias1": d_bias1,
            "coeff2": d_coeff2,
            "bias2": d_bias2,
        }

    def _adam_update(
        self, name: str, param: np.ndarray, grad: np.ndarray
    ) -> np.ndarray:
        """Apply one Adam optimizer step.

        Parameters
        ----------
        name : str
            Parameter name (state key).
        param : ndarray
            Current parameter values.
        grad : ndarray
            Gradient of loss w.r.t. this parameter.

        Returns
        -------
        updated : ndarray
        """
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        state = self._adam_state[name]
        state["t"] += 1
        t = state["t"]

        state["m"] = beta1 * state["m"] + (1 - beta1) * grad
        state["v"] = beta2 * state["v"] + (1 - beta2) * (grad ** 2)

        m_hat = state["m"] / (1 - beta1 ** t)
        v_hat = state["v"] / (1 - beta2 ** t)

        return param - self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    def _prune_coefficients(self) -> int:
        """Set near-zero spline coefficients to exactly zero.

        Returns
        -------
        n_pruned : int
        """
        mask1 = np.abs(self._coeff1) < self.prune_threshold
        mask2 = np.abs(self._coeff2) < self.prune_threshold
        n_pruned = int(mask1.sum() + mask2.sum())
        self._coeff1[mask1] = 0.0
        self._coeff2[mask2] = 0.0
        return n_pruned

    def _compute_loss(
        self, prob: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray]
    ) -> float:
        """Binary cross-entropy loss plus L1 penalty."""
        p = np.clip(prob, 1e-7, 1 - 1e-7)
        bce = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        if sample_weight is not None:
            bce = bce * sample_weight
        data_loss = bce.mean()
        l1_loss = self.l1_lambda * (
            np.abs(self._coeff1).sum() + np.abs(self._coeff2).sum()
        )
        return float(data_loss + l1_loss)

    def _compute_feature_importances(self) -> np.ndarray:
        """L1 norm of input-layer spline coefficients per feature.

        Returns
        -------
        importances : ndarray of shape (n_features,)
            Normalized to sum to 1.0.
        """
        # coeff1 shape: (n_features, hidden_dim, n_basis)
        importances = np.abs(self._coeff1).sum(axis=(1, 2))
        total = importances.sum()
        if total > 0:
            importances = importances / total
        else:
            importances = np.full_like(importances, 1.0 / len(importances))
        return importances

    # ------------------------------------------------------------------
    # Public sklearn API
    # ------------------------------------------------------------------

    def fit(self, X, y, sample_weight=None):
        """Fit the KAN classifier on binary classification data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Binary target labels (0 or 1).
        sample_weight : array-like of shape (n_samples,), optional
            Per-sample weights.

        Returns
        -------
        self
            Fitted classifier.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X.shape

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64).ravel()

        # Store normalization parameters (fit on training data only)
        self._x_min = X.min(axis=0)
        self._x_max = X.max(axis=0)
        self._x_range = self._x_max - self._x_min

        X_norm = self._normalize_input(X)

        self._init_params(n_features)

        rng = np.random.RandomState(self.random_state)
        best_loss = np.inf
        best_params = None
        epochs_no_improve = 0

        logger.info(
            "KANClassifier training: %d samples, %d features, "
            "hidden_dim=%d, grid_size=%d, spline_order=%d",
            n_samples, n_features, self.hidden_dim,
            self.grid_size, self.spline_order,
        )

        for epoch in range(self.max_epochs):
            indices = rng.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                idx = indices[start:end]

                X_batch = X_norm[idx]
                y_batch = y[idx]
                w_batch = sample_weight[idx] if sample_weight is not None else None

                cache = self._forward(X_batch)
                epoch_loss += self._compute_loss(cache["prob"], y_batch, w_batch)
                n_batches += 1

                grads = self._backward(cache, y_batch, w_batch)

                self._coeff1 = self._adam_update("coeff1", self._coeff1, grads["coeff1"])
                self._bias1 = self._adam_update("bias1", self._bias1, grads["bias1"])
                self._coeff2 = self._adam_update("coeff2", self._coeff2, grads["coeff2"])
                self._bias2 = self._adam_update("bias2", self._bias2, grads["bias2"])

            avg_loss = epoch_loss / max(n_batches, 1)

            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                best_params = (
                    self._coeff1.copy(),
                    self._bias1.copy(),
                    self._coeff2.copy(),
                    self._bias2.copy(),
                )
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 20 == 0 or epoch == 0:
                logger.info(
                    "  Epoch %d/%d  loss=%.6f  best=%.6f  patience=%d/%d",
                    epoch + 1, self.max_epochs, avg_loss, best_loss,
                    epochs_no_improve, self.patience,
                )

            if epochs_no_improve >= self.patience:
                logger.info(
                    "  Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch + 1, self.patience,
                )
                break

        # Restore best parameters
        if best_params is not None:
            self._coeff1, self._bias1, self._coeff2, self._bias2 = best_params

        # Grid pruning
        n_pruned = self._prune_coefficients()
        total_coeffs = self._coeff1.size + self._coeff2.size
        logger.info(
            "  Pruned %d/%d coefficients (%.1f%%) below threshold %.1e",
            n_pruned, total_coeffs, 100.0 * n_pruned / max(total_coeffs, 1),
            self.prune_threshold,
        )

        # Set sklearn-required attributes
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = n_features
        self.feature_importances_ = self._compute_feature_importances()

        logger.info(
            "  Training complete: best_loss=%.6f, top-3 features=%s",
            best_loss,
            np.argsort(self.feature_importances_)[-3:][::-1].tolist(),
        )

        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return class probability estimates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Column 0 = P(class=0), Column 1 = P(class=1).
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        X_norm = self._normalize_input(X)
        cache = self._forward(X_norm)
        p1 = np.clip(cache["prob"], 1e-7, 1 - 1e-7)
        return np.column_stack([1 - p1, p1])

    def predict(self, X) -> np.ndarray:
        """Return binary class predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def decision_function(self, X) -> np.ndarray:
        """Return raw logit scores (before sigmoid).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray of shape (n_samples,)
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        X_norm = self._normalize_input(X)
        cache = self._forward(X_norm)
        return cache["logit"]

    def _check_fitted(self) -> None:
        """Raise if the model has not been fitted."""
        if not hasattr(self, "_coeff1"):
            raise AttributeError(
                "KANClassifier is not fitted yet. Call fit() before predict."
            )

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for this estimator (sklearn-compatible).

        Parameters
        ----------
        deep : bool, default=True
            Not used; included for sklearn API compatibility.

        Returns
        -------
        params : dict
        """
        return {
            "hidden_dim": self.hidden_dim,
            "grid_size": self.grid_size,
            "spline_order": self.spline_order,
            "l1_lambda": self.l1_lambda,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "prune_threshold": self.prune_threshold,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        """Set parameters of this estimator (sklearn-compatible).

        Parameters
        ----------
        **params : dict
            Estimator parameters to set.

        Returns
        -------
        self
        """
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(
                    f"Invalid parameter '{key}' for KANClassifier."
                )
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        return (
            f"KANClassifier("
            f"hidden_dim={self.hidden_dim}, "
            f"grid_size={self.grid_size}, "
            f"spline_order={self.spline_order}, "
            f"l1_lambda={self.l1_lambda}, "
            f"lr={self.learning_rate}, "
            f"epochs={self.max_epochs})"
        )

"""
Selective State-Space Model (Mamba-style) Classifier
=====================================================

A pure-NumPy implementation of a Mamba-inspired selective state-space model
for binary classification, compatible with the sklearn classifier interface.

Mamba differs from traditional RNNs/Transformers: it uses **selective
state-space dynamics** where the discretization parameters (A, B, C, dt) are
input-dependent, allowing the model to selectively propagate or forget
information along the sequence dimension.

Architecture (per EDGE 1 -- shallow, regularization-first):
    Input projection -> N SSM layers (diagonal A, selective dt) -> Output head

How features become a sequence:
    - Input (n_features,) is reshaped into (seq_len, d_model) chunks
    - If n_features is not divisible by d_model, pad to the nearest multiple
    - Each SSM layer runs a selective scan over the sequence
    - The final hidden state is projected to a logit for classification

State-space model (per layer):
    Continuous: x'(t) = A * x(t) + B * u(t),  y(t) = C * x(t)
    Discretized (ZOH):
        A_bar = exp(A * dt)           (diagonal A -> element-wise exp)
        B_bar = (A_bar - I) * A^{-1} * B   (simplified for diagonal A)
        x_{t+1} = A_bar * x_t + B_bar * u_t
        y_t = C * x_t

Selective mechanism:
    dt = softplus(Linear(u_t))  -- step size is input-dependent
    B = Linear(u_t)             -- input matrix is input-dependent
    C = Linear(u_t)             -- output matrix is input-dependent
    This makes the SSM *selective*: it can choose what to remember/forget.

Training:
    - Mini-batch gradient descent with Adam optimizer
    - Binary cross-entropy loss
    - L2 regularization on all weight matrices
    - Early stopping with patience
    - Analytical backpropagation through the selective scan

Feature importance:
    Gradient magnitude of the loss w.r.t. each input feature, averaged over
    training data, normalized to sum to 1.

Usage:
    from src.phase_12_model_training.mamba_model import MambaClassifier

    clf = MambaClassifier(hidden_dim=16, n_layers=2, l2_lambda=0.01)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)

References:
    Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling with Selective
    State Spaces"
"""

import logging
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)


# ======================================================================
# Activation utilities
# ======================================================================

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    x_safe = np.clip(x, -500.0, 500.0)
    return np.where(
        x_safe >= 0,
        1.0 / (1.0 + np.exp(-x_safe)),
        np.exp(x_safe) / (1.0 + np.exp(x_safe)),
    )


def _softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus: log(1 + exp(x))."""
    x_safe = np.clip(x, -500.0, 500.0)
    return np.where(
        x_safe > 20.0,
        x_safe,
        np.log1p(np.exp(x_safe)),
    )


def _softplus_grad(x: np.ndarray) -> np.ndarray:
    """Derivative of softplus: sigmoid(x)."""
    return _sigmoid(x)


def _silu(x: np.ndarray) -> np.ndarray:
    """SiLU / Swish activation: x * sigmoid(x)."""
    return x * _sigmoid(x)


def _silu_grad(x: np.ndarray) -> np.ndarray:
    """Derivative of SiLU: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))."""
    s = _sigmoid(x)
    return s + x * s * (1.0 - s)


# ======================================================================
# SSM Layer parameters
# ======================================================================

class _SSMLayerParams:
    """Container for the learnable parameters of one SSM layer.

    Each layer has:
        - W_in:  (d_model, d_inner)   input projection
        - b_in:  (d_inner,)
        - A_log: (d_inner, d_state)   log of diagonal A (learned in log-space)
        - W_B:   (d_inner, d_state)   input-dependent B projection
        - W_C:   (d_inner, d_state)   input-dependent C projection
        - W_dt:  (d_inner, 1)         input-dependent dt projection
        - b_dt:  (1,)                 dt bias
        - D:     (d_inner,)           skip connection
        - W_out: (d_inner, d_model)   output projection
        - b_out: (d_model,)
    """

    def __init__(
        self, d_model: int, d_inner: int, d_state: int, rng: np.random.RandomState
    ):
        scale_in = np.sqrt(2.0 / (d_model + d_inner))
        scale_out = np.sqrt(2.0 / (d_inner + d_model))
        scale_s = np.sqrt(2.0 / (d_inner + d_state))

        self.W_in = rng.randn(d_model, d_inner) * scale_in
        self.b_in = np.zeros(d_inner)

        # A is learned in log-space for stability; initialized near -1
        # so exp(A_log) ~ exp(-1) ~ 0.37 (mild decay)
        self.A_log = -np.ones((d_inner, d_state)) + rng.randn(d_inner, d_state) * 0.1

        self.W_B = rng.randn(d_inner, d_state) * scale_s
        self.W_C = rng.randn(d_inner, d_state) * scale_s

        self.W_dt = rng.randn(d_inner, 1) * scale_s
        self.b_dt = np.zeros(1)

        self.D = np.ones(d_inner)  # skip connection initialized to 1

        self.W_out = rng.randn(d_inner, d_model) * scale_out
        self.b_out = np.zeros(d_model)

    def param_names(self):
        """Return list of parameter attribute names."""
        return [
            "W_in", "b_in", "A_log", "W_B", "W_C",
            "W_dt", "b_dt", "D", "W_out", "b_out",
        ]


# ======================================================================
# SSM Layer forward + backward (analytical gradients)
# ======================================================================

def _ssm_layer_forward(
    x_seq: np.ndarray, params: _SSMLayerParams
) -> dict:
    """Forward pass through one SSM layer, caching intermediates.

    Parameters
    ----------
    x_seq : ndarray of shape (seq_len, d_model)
        Input sequence for a single sample.
    params : _SSMLayerParams

    Returns
    -------
    cache : dict with output and all intermediates for backprop.
    """
    seq_len, d_model = x_seq.shape
    d_inner = params.W_in.shape[1]
    d_state = params.A_log.shape[1]

    # Input projection (pre-activation)
    z_pre = x_seq @ params.W_in + params.b_in  # (seq_len, d_inner)
    z = _silu(z_pre)  # (seq_len, d_inner)

    # Continuous A (diagonal, negative for stability)
    A = -np.exp(params.A_log)  # (d_inner, d_state), all negative

    # Selective scan over the sequence -- cache everything
    y_inner = np.zeros((seq_len, d_inner))
    h_states = np.zeros((seq_len + 1, d_inner, d_state))  # h[0] = zeros
    dt_raws = np.zeros((seq_len, 1))
    dt_vals = np.zeros(seq_len)
    A_bars = np.zeros((seq_len, d_inner, d_state))
    B_bars = np.zeros((seq_len, d_inner, d_state))
    B_ts = np.zeros((seq_len, d_inner, d_state))
    C_ts = np.zeros((seq_len, d_inner, d_state))

    for t in range(seq_len):
        u_t = z[t]  # (d_inner,)

        # Input-dependent parameters (selective mechanism)
        dt_raw = u_t @ params.W_dt + params.b_dt  # (1,)
        dt_raws[t] = dt_raw
        dt_val = _softplus(dt_raw)[0]
        dt_vals[t] = dt_val

        B_t = u_t[:, None] * params.W_B  # (d_inner, d_state)
        C_t = u_t[:, None] * params.W_C  # (d_inner, d_state)
        B_ts[t] = B_t
        C_ts[t] = C_t

        # Discretize: ZOH for diagonal A
        A_bar = np.exp(A * dt_val)  # (d_inner, d_state)
        A_bars[t] = A_bar

        # B_bar = (A_bar - I) / A * B_t
        A_safe = np.where(np.abs(A) > 1e-8, A, -1e-8)
        B_bar = (A_bar - 1.0) / A_safe * B_t  # (d_inner, d_state)
        B_bars[t] = B_bar

        # State update
        h_states[t + 1] = A_bar * h_states[t] + B_bar

        # Output
        y_t = (C_t * h_states[t + 1]).sum(axis=1)  # (d_inner,)

        # Skip connection
        y_inner[t] = y_t + params.D * u_t

    # Output projection
    y_seq = y_inner @ params.W_out + params.b_out  # (seq_len, d_model)

    return {
        "x_seq": x_seq,
        "z_pre": z_pre,
        "z": z,
        "A": A,
        "y_inner": y_inner,
        "y_seq": y_seq,
        "h_states": h_states,
        "dt_raws": dt_raws,
        "dt_vals": dt_vals,
        "A_bars": A_bars,
        "B_bars": B_bars,
        "B_ts": B_ts,
        "C_ts": C_ts,
    }


def _ssm_layer_backward(
    d_y_seq: np.ndarray, cache: dict, params: _SSMLayerParams
) -> tuple[np.ndarray, dict]:
    """Backward pass through one SSM layer.

    Parameters
    ----------
    d_y_seq : ndarray of shape (seq_len, d_model)
        Gradient of loss w.r.t. this layer's output.
    cache : dict from _ssm_layer_forward.
    params : _SSMLayerParams

    Returns
    -------
    d_x_seq : ndarray of shape (seq_len, d_model)
        Gradient w.r.t. input.
    grads : dict mapping parameter names to gradient arrays.
    """
    seq_len = cache["z"].shape[0]
    d_inner = params.W_in.shape[1]
    d_state = params.A_log.shape[1]

    z = cache["z"]
    z_pre = cache["z_pre"]
    x_seq = cache["x_seq"]
    A = cache["A"]
    h_states = cache["h_states"]
    dt_raws = cache["dt_raws"]
    dt_vals = cache["dt_vals"]
    A_bars = cache["A_bars"]
    B_bars = cache["B_bars"]
    B_ts = cache["B_ts"]
    C_ts = cache["C_ts"]
    y_inner = cache["y_inner"]

    # Gradient through output projection: y_seq = y_inner @ W_out + b_out
    d_y_inner = d_y_seq @ params.W_out.T  # (seq_len, d_inner)
    d_W_out = y_inner.T @ d_y_seq  # (d_inner, d_model)
    d_b_out = d_y_seq.sum(axis=0)  # (d_model,)

    # Backward through selective scan (reverse time)
    d_z = np.zeros((seq_len, d_inner))
    d_A_log = np.zeros_like(params.A_log)
    d_W_B = np.zeros_like(params.W_B)
    d_W_C = np.zeros_like(params.W_C)
    d_W_dt = np.zeros_like(params.W_dt)
    d_b_dt = np.zeros_like(params.b_dt)
    d_D = np.zeros_like(params.D)

    d_h = np.zeros((d_inner, d_state))  # gradient flowing backward through h

    A_safe = np.where(np.abs(A) > 1e-8, A, -1e-8)

    for t in reversed(range(seq_len)):
        u_t = z[t]
        dy_t = d_y_inner[t]  # (d_inner,)

        # Skip connection: y_inner[t] = y_t + D * u_t
        d_D += dy_t * u_t
        d_u_skip = dy_t * params.D  # grad through skip to u_t

        # Output: y_t = (C_t * h_{t+1}).sum(axis=1)
        # dy_t_inner shape: (d_inner,)
        d_Ch = dy_t[:, None]  # broadcast to (d_inner, d_state)
        d_h += d_Ch * C_ts[t]  # grad to h_{t+1}
        d_C_t = d_Ch * h_states[t + 1]  # grad to C_t

        # C_t = u_t[:, None] * W_C
        # d_u from C_t: sum over d_state
        d_u_from_C = (d_C_t * params.W_C).sum(axis=1)
        d_W_C += u_t[:, None] * d_C_t

        # State update: h_{t+1} = A_bar * h_t + B_bar
        # d_h is d_loss / d_h_{t+1}
        d_A_bar = d_h * h_states[t]  # (d_inner, d_state)
        d_B_bar = d_h.copy()
        d_h_prev = d_h * A_bars[t]  # propagate gradient to h_t

        # B_bar = (A_bar - 1) / A * B_t
        # d_B_t from B_bar
        coeff = (A_bars[t] - 1.0) / A_safe
        d_B_t = d_B_bar * coeff

        # B_t = u_t[:, None] * W_B
        d_u_from_B = (d_B_t * params.W_B).sum(axis=1)
        d_W_B += u_t[:, None] * d_B_t

        # Gradient through A_bar -> A_log and dt
        # A_bar = exp(A * dt)
        # d_A_bar contributions through both direct and B_bar path:
        # B_bar = (A_bar - 1) / A * B_t
        # d_A_bar_total = d_A_bar (from state update) + d_B_bar * B_t / A (from B_bar)
        d_A_bar_total = d_A_bar + d_B_bar * B_ts[t] / A_safe

        # d(exp(A*dt))/d(A) = dt * exp(A*dt) = dt * A_bar
        # But A = -exp(A_log), so dA/dA_log = -exp(A_log) = A (wait, A is neg)
        # Actually A = -exp(A_log) so dA/dA_log = -exp(A_log)
        # Chain: d_A_log += d_A_bar_total * A_bar * dt * (-exp(A_log))
        #      = d_A_bar_total * A_bar * dt * A  (since A = -exp(A_log))
        # But we need to be careful: d_A_bar_total * (dA_bar/dA) * (dA/dA_log)
        # dA_bar/dA = dt * A_bar (since A_bar = exp(A*dt))
        # dA/dA_log = -exp(A_log) = A (since A = -exp(A_log) -> actually no)
        # Let me re-derive:
        # A = -exp(A_log)
        # A_bar = exp(A * dt) = exp(-exp(A_log) * dt)
        # dA_bar/dA_log = exp(-exp(A_log)*dt) * (-exp(A_log)*dt)
        #               = A_bar * A * dt
        d_A_log += (d_A_bar_total * A_bars[t] * A * dt_vals[t])

        # dA_bar/ddt = A * A_bar
        # dt = softplus(dt_raw)
        # d_dt_raw = d_loss/d_dt * sigmoid(dt_raw)
        d_dt = (d_A_bar_total * A * A_bars[t]).sum()
        # Also B_bar depends on dt through A_bar
        # Already captured above via d_A_bar_total
        d_dt_raw_val = d_dt * _softplus_grad(dt_raws[t])[0]

        # dt_raw = u_t @ W_dt + b_dt
        d_W_dt += u_t[:, None] * d_dt_raw_val
        d_b_dt += d_dt_raw_val
        d_u_from_dt = params.W_dt.ravel() * d_dt_raw_val

        # Aggregate gradient to u_t = z[t]
        d_z[t] = d_u_skip + d_u_from_B + d_u_from_C + d_u_from_dt

        # Propagate h gradient backward
        d_h = d_h_prev

    # Gradient through SiLU: z = silu(z_pre)
    d_z_pre = d_z * _silu_grad(z_pre)

    # Gradient through input projection: z_pre = x_seq @ W_in + b_in
    d_W_in = x_seq.T @ d_z_pre  # (d_model, d_inner)
    d_b_in = d_z_pre.sum(axis=0)  # (d_inner,)
    d_x_seq = d_z_pre @ params.W_in.T  # (seq_len, d_model)

    grads = {
        "W_in": d_W_in,
        "b_in": d_b_in,
        "A_log": d_A_log,
        "W_B": d_W_B,
        "W_C": d_W_C,
        "W_dt": d_W_dt,
        "b_dt": d_b_dt,
        "D": d_D,
        "W_out": d_W_out,
        "b_out": d_b_out,
    }

    return d_x_seq, grads


# ======================================================================
# MambaClassifier
# ======================================================================

class MambaClassifier(ClassifierMixin, BaseEstimator):
    """Selective State-Space Model (Mamba-style) classifier.

    Features are reshaped into a short sequence and processed through
    stacked SSM layers with selective (input-dependent) discretization.
    The final hidden representation is projected to a binary classification
    logit.

    Follows EDGE 1 (Regularization-First):
        - Shallow architecture (1-3 SSM layers recommended)
        - Heavy L2 regularization on all weights
        - Small hidden_dim (8-32 recommended)
        - Early stopping with patience

    Note: ClassifierMixin before BaseEstimator (sklearn 1.6+ tags).

    Parameters
    ----------
    hidden_dim : int, default=16
        Dimension of the internal SSM state (d_model / d_inner).
    n_layers : int, default=2
        Number of stacked SSM layers.
    d_state : int, default=8
        State-space dimension per layer.
    learning_rate : float, default=0.01
        Adam optimizer learning rate.
    max_epochs : int, default=100
        Maximum training epochs.
    batch_size : int, default=64
        Mini-batch size for gradient descent.
    patience : int, default=5
        Early stopping patience (epochs without improvement).
    l2_lambda : float, default=0.01
        L2 regularization strength on all weight matrices.
    seed : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        Class labels [0, 1]. Set after fit().
    feature_importances_ : ndarray of shape (n_features,)
        Gradient-based feature importance, normalized to sum to 1.
        Set after fit().
    n_features_in_ : int
        Number of features seen during fit().
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        n_layers: int = 2,
        d_state: int = 8,
        learning_rate: float = 0.01,
        max_epochs: int = 100,
        batch_size: int = 64,
        patience: int = 5,
        l2_lambda: float = 0.01,
        seed: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.d_state = d_state
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.l2_lambda = l2_lambda
        self.seed = seed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_params(self, n_features: int) -> None:
        """Initialize all learnable parameters."""
        rng = np.random.RandomState(self.seed)

        d_model = self.hidden_dim
        self._d_model = d_model
        self._seq_len = max(1, int(np.ceil(n_features / d_model)))
        self._padded_features = self._seq_len * d_model

        # SSM layers
        self._layers: list[_SSMLayerParams] = []
        for _ in range(self.n_layers):
            layer = _SSMLayerParams(d_model, d_model, self.d_state, rng)
            self._layers.append(layer)

        # Output head: from d_model to 1 logit
        scale_head = np.sqrt(2.0 / (d_model + 1))
        self._W_head = rng.randn(d_model, 1) * scale_head
        self._b_head = np.zeros(1)

        # Adam optimizer state for all parameters
        self._adam_state = {}
        for layer_idx, layer in enumerate(self._layers):
            for name in layer.param_names():
                key = f"layer{layer_idx}_{name}"
                self._adam_state[key] = {"m": 0.0, "v": 0.0, "t": 0}
        self._adam_state["W_head"] = {"m": 0.0, "v": 0.0, "t": 0}
        self._adam_state["b_head"] = {"m": 0.0, "v": 0.0, "t": 0}

    def _normalize_input(self, X: np.ndarray) -> np.ndarray:
        """Standardize inputs using fitted mean/std."""
        return (X - self._x_mean) / (self._x_std + 1e-8)

    def _reshape_to_sequence(self, X: np.ndarray) -> np.ndarray:
        """Reshape (batch_size, n_features) -> (batch_size, seq_len, d_model).

        Pads with zeros if n_features is not a multiple of d_model.
        """
        batch_size = X.shape[0]
        n_features = X.shape[1]

        if n_features < self._padded_features:
            pad_width = self._padded_features - n_features
            X = np.pad(X, ((0, 0), (0, pad_width)), mode="constant")

        return X[:, :self._padded_features].reshape(
            batch_size, self._seq_len, self._d_model
        )

    def _forward_full(self, X_norm: np.ndarray) -> dict:
        """Full forward pass with caches for backprop.

        Parameters
        ----------
        X_norm : ndarray of shape (batch_size, n_features)

        Returns
        -------
        result : dict with prob, logit, h_last, and per-sample layer caches.
        """
        batch_size = X_norm.shape[0]
        x_seq_batch = self._reshape_to_sequence(X_norm)  # (B, T, D)

        # Per-sample, per-layer caches
        layer_caches = []  # list of list of dicts
        h_inputs = []  # (n_layers+1) entries, each (B, T, D)

        h = x_seq_batch  # current input
        h_inputs.append(h.copy())

        for layer_idx, layer in enumerate(self._layers):
            sample_caches = []
            h_out = np.zeros_like(h)
            for i in range(batch_size):
                cache = _ssm_layer_forward(h[i], layer)
                sample_caches.append(cache)
                h_out[i] = cache["y_seq"]
            layer_caches.append(sample_caches)

            h = h + h_out  # residual connection
            h_inputs.append(h.copy())

        # Take the last time step
        h_last = h[:, -1, :]  # (B, d_model)

        # Output head
        logit = (h_last @ self._W_head + self._b_head).ravel()  # (B,)
        prob = _sigmoid(logit)

        return {
            "prob": prob,
            "logit": logit,
            "h_last": h_last,
            "h_inputs": h_inputs,
            "layer_caches": layer_caches,
            "x_seq_batch": x_seq_batch,
        }

    def _forward(self, X_norm: np.ndarray) -> dict:
        """Forward pass for inference (no backprop caches needed)."""
        result = self._forward_full(X_norm)
        return {
            "prob": result["prob"],
            "logit": result["logit"],
            "h_last": result["h_last"],
        }

    def _compute_loss(
        self, prob: np.ndarray, y: np.ndarray,
        sample_weight: Optional[np.ndarray],
    ) -> float:
        """Binary cross-entropy loss plus L2 penalty."""
        p = np.clip(prob, 1e-7, 1 - 1e-7)
        bce = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        if sample_weight is not None:
            bce = bce * sample_weight
        data_loss = float(bce.mean())

        # L2 penalty on all weight matrices
        l2_loss = 0.0
        for layer in self._layers:
            for name in layer.param_names():
                param = getattr(layer, name)
                l2_loss += float(np.sum(param ** 2))
        l2_loss += float(np.sum(self._W_head ** 2))
        l2_loss *= self.l2_lambda * 0.5

        return data_loss + l2_loss

    def _backward(
        self, fwd: dict, y: np.ndarray,
        sample_weight: Optional[np.ndarray],
    ) -> dict:
        """Analytical backward pass computing gradients for all parameters.

        Parameters
        ----------
        fwd : dict from _forward_full.
        y : ndarray of shape (batch_size,)
        sample_weight : ndarray or None

        Returns
        -------
        grads : dict mapping parameter keys to gradient arrays.
        """
        batch_size = len(y)
        prob = fwd["prob"]
        h_last = fwd["h_last"]

        # BCE gradient: d_loss/d_logit = (prob - y) / n
        d_logit = (prob - y)
        if sample_weight is not None:
            d_logit = d_logit * sample_weight
        d_logit = d_logit / batch_size  # (B,)

        # Output head: logit = h_last @ W_head + b_head
        # d_logit is (B,), W_head is (d_model, 1)
        d_h_last = d_logit[:, None] * self._W_head.T  # (B, d_model)
        d_W_head = h_last.T @ d_logit[:, None]  # (d_model, 1)
        d_b_head = np.array([d_logit.sum()])

        # L2 regularization gradient for head
        d_W_head += self.l2_lambda * self._W_head
        d_b_head += self.l2_lambda * self._b_head

        # Gradient through "take last time step": h_last = h[:, -1, :]
        # So d_h (B, T, D) has nonzero only at t = T-1
        seq_len = self._seq_len
        d_h = np.zeros((batch_size, seq_len, self._d_model))
        d_h[:, -1, :] = d_h_last

        # Backward through layers (reverse order)
        all_grads = {}
        for layer_idx in reversed(range(self.n_layers)):
            layer = self._layers[layer_idx]
            sample_caches = fwd["layer_caches"][layer_idx]

            # Accumulate gradients across batch
            layer_grads = {name: np.zeros_like(getattr(layer, name))
                          for name in layer.param_names()}

            for i in range(batch_size):
                # Residual: h_out = h_prev + layer_out
                # d_h flows to both d_layer_out and d_h_prev
                d_layer_out = d_h[i]  # (T, D) -- gradient through residual
                d_x_seq_i, grads_i = _ssm_layer_backward(
                    d_layer_out, sample_caches[i], layer,
                )
                # Residual gradient: d_h_prev += d_h (identity path) + d_x_seq_i
                d_h[i] = d_h[i] + d_x_seq_i

                # Accumulate parameter gradients
                for name in layer.param_names():
                    layer_grads[name] += grads_i[name]

            # Add L2 regularization gradients for this layer
            for name in layer.param_names():
                layer_grads[name] += self.l2_lambda * getattr(layer, name)
                key = f"layer{layer_idx}_{name}"
                all_grads[key] = layer_grads[name]

        all_grads["W_head"] = d_W_head
        all_grads["b_head"] = d_b_head

        # Clip gradients for stability
        for key in all_grads:
            np.clip(all_grads[key], -5.0, 5.0, out=all_grads[key])

        return all_grads

    def _adam_update(
        self, key: str, param: np.ndarray, grad: np.ndarray
    ) -> np.ndarray:
        """Apply one Adam optimizer step."""
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        state = self._adam_state[key]
        state["t"] += 1
        t = state["t"]

        state["m"] = beta1 * state["m"] + (1 - beta1) * grad
        state["v"] = beta2 * state["v"] + (1 - beta2) * (grad ** 2)

        m_hat = state["m"] / (1 - beta1 ** t)
        v_hat = state["v"] / (1 - beta2 ** t)

        return param - self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    def _apply_gradients(self, grads: dict) -> None:
        """Apply Adam updates to all parameters using computed gradients."""
        for layer_idx, layer in enumerate(self._layers):
            for name in layer.param_names():
                key = f"layer{layer_idx}_{name}"
                param = getattr(layer, name)
                updated = self._adam_update(key, param, grads[key])
                setattr(layer, name, updated)

        self._W_head = self._adam_update("W_head", self._W_head, grads["W_head"])
        self._b_head = self._adam_update("b_head", self._b_head, grads["b_head"])

    def _collect_all_params(self) -> dict:
        """Snapshot all parameters for checkpointing."""
        snapshot = {}
        for layer_idx, layer in enumerate(self._layers):
            for name in layer.param_names():
                key = f"layer{layer_idx}_{name}"
                snapshot[key] = getattr(layer, name).copy()
        snapshot["W_head"] = self._W_head.copy()
        snapshot["b_head"] = self._b_head.copy()
        return snapshot

    def _restore_params(self, snapshot: dict) -> None:
        """Restore parameters from a checkpoint snapshot."""
        for layer_idx, layer in enumerate(self._layers):
            for name in layer.param_names():
                key = f"layer{layer_idx}_{name}"
                setattr(layer, name, snapshot[key].copy())
        self._W_head = snapshot["W_head"].copy()
        self._b_head = snapshot["b_head"].copy()

    def _compute_feature_importances(
        self, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Gradient-based feature importance via input perturbation.

        Computes the average absolute gradient of the loss with respect
        to each input feature, using a small subset of the training data.
        """
        n_features = X.shape[1]
        n_use = min(50, X.shape[0])
        X_sub = X[:n_use]
        y_sub = y[:n_use]

        importances = np.zeros(n_features)
        eps = 1e-4

        for j in range(n_features):
            X_plus = X_sub.copy()
            X_minus = X_sub.copy()
            X_plus[:, j] += eps
            X_minus[:, j] -= eps

            cache_plus = self._forward(X_plus)
            cache_minus = self._forward(X_minus)

            loss_plus = self._compute_loss(cache_plus["prob"], y_sub, None)
            loss_minus = self._compute_loss(cache_minus["prob"], y_sub, None)

            importances[j] = abs(loss_plus - loss_minus) / (2 * eps)

        total = importances.sum()
        if total > 0:
            importances = importances / total
        else:
            importances = np.full(n_features, 1.0 / n_features)
        return importances

    # ------------------------------------------------------------------
    # Public sklearn API
    # ------------------------------------------------------------------

    def fit(self, X, y, sample_weight=None):
        """Fit the Mamba SSM classifier on binary classification data.

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

        # Fit standardization on training data only
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0)
        X_norm = self._normalize_input(X)

        self._init_params(n_features)

        rng = np.random.RandomState(self.seed)
        best_loss = np.inf
        best_params = None
        epochs_no_improve = 0

        logger.info(
            "MambaClassifier training: %d samples, %d features, "
            "hidden_dim=%d, n_layers=%d, d_state=%d, seq_len=%d",
            n_samples, n_features, self.hidden_dim,
            self.n_layers, self.d_state, self._seq_len,
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

                # Forward (with caches for backprop)
                fwd = self._forward_full(X_batch)
                batch_loss = self._compute_loss(fwd["prob"], y_batch, w_batch)
                epoch_loss += batch_loss
                n_batches += 1

                # Analytical backward pass
                grads = self._backward(fwd, y_batch, w_batch)

                # Apply Adam updates
                self._apply_gradients(grads)

            avg_loss = epoch_loss / max(n_batches, 1)

            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                best_params = self._collect_all_params()
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
            self._restore_params(best_params)

        # Set sklearn-required attributes
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = n_features
        self.feature_importances_ = self._compute_feature_importances(X_norm, y)

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
        if not hasattr(self, "_layers"):
            raise AttributeError(
                "MambaClassifier is not fitted yet. Call fit() before predict."
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
            "n_layers": self.n_layers,
            "d_state": self.d_state,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "l2_lambda": self.l2_lambda,
            "seed": self.seed,
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
                    f"Invalid parameter '{key}' for MambaClassifier."
                )
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        return (
            f"MambaClassifier("
            f"hidden_dim={self.hidden_dim}, "
            f"n_layers={self.n_layers}, "
            f"d_state={self.d_state}, "
            f"l2_lambda={self.l2_lambda}, "
            f"lr={self.learning_rate}, "
            f"epochs={self.max_epochs})"
        )

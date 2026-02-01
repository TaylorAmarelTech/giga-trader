#!/usr/bin/env python
"""
Quick test demonstrating data leakage in CV.
Uses simple variance-based feature selection instead of MI for speed.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

print("=" * 60)
print("QUICK DATA LEAKAGE DEMONSTRATION")
print("=" * 60)

# Create data with signal in first few features
np.random.seed(42)
n_samples, n_features = 500, 50

X = np.random.randn(n_samples, n_features)

# Target correlated with first 3 features
signal = 0.4 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2]
y = (signal + np.random.randn(n_samples) * 0.4 > 0).astype(int)

print(f"\nData: {n_samples} samples, {n_features} features")
print(f"Target: {y.mean():.1%} positive")

# Simple CV parameters
n_folds = 5
fold_size = n_samples // n_folds
n_select = 10  # Select top 10 features

def variance_select(X, n):
    """Select top n features by variance."""
    variances = np.var(X, axis=0)
    return np.argsort(variances)[::-1][:n]


# =====================================================================
# LEAKY CV
# =====================================================================
print("\n" + "-" * 40)
print("LEAKY CV (scale + select BEFORE split)")
print("-" * 40)

# LEAK: Scale on all data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# LEAK: Select features using all data
top_idx = variance_select(X_scaled, n_select)
X_selected = X_scaled[:, top_idx]

leaky_test_aucs = []
for fold in range(n_folds):
    test_start = fold * fold_size
    test_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples

    train_mask = np.ones(n_samples, dtype=bool)
    train_mask[test_start:test_end] = False

    X_train, X_test = X_selected[train_mask], X_selected[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    leaky_test_aucs.append(test_auc)
    print(f"  Fold {fold + 1}: Test AUC = {test_auc:.3f}")

leaky_mean = np.mean(leaky_test_aucs)
print(f"  Mean Test AUC: {leaky_mean:.3f}")


# =====================================================================
# LEAK-PROOF CV
# =====================================================================
print("\n" + "-" * 40)
print("LEAK-PROOF CV (scale + select INSIDE each fold)")
print("-" * 40)

proof_test_aucs = []
for fold in range(n_folds):
    test_start = fold * fold_size
    test_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_samples

    train_mask = np.ones(n_samples, dtype=bool)
    train_mask[test_start:test_end] = False

    X_train_raw, X_test_raw = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    # Scale on TRAIN only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Select on TRAIN only
    top_idx = variance_select(X_train_scaled, n_select)
    X_train = X_train_scaled[:, top_idx]
    X_test = X_test_scaled[:, top_idx]

    model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    proof_test_aucs.append(test_auc)
    print(f"  Fold {fold + 1}: Test AUC = {test_auc:.3f}")

proof_mean = np.mean(proof_test_aucs)
print(f"  Mean Test AUC: {proof_mean:.3f}")


# =====================================================================
# COMPARISON
# =====================================================================
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

diff = leaky_mean - proof_mean
print(f"\n  Leaky CV Mean:      {leaky_mean:.3f}")
print(f"  Leak-Proof CV Mean: {proof_mean:.3f}")
print(f"  Difference:         {diff:+.3f}")

if diff > 0.01:
    print(f"\n  [CONFIRMED] Leaky CV inflates test AUC by {diff:.1%}")
    print(f"              The leak-proof result is more realistic!")
else:
    print(f"\n  [NOTE] Small difference - leak effect depends on data structure")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)

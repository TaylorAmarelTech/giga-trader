#!/usr/bin/env python
"""
Test script for the leak-proof CV pipeline.

Compares:
1. Leak-proof CV (transforms inside fold)
2. Leaky CV (transforms before fold) - for comparison

Expected result: Leak-proof should show lower but more realistic test AUC.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def create_test_data(n_samples=1000, n_features=100, signal_strength=0.3):
    """Create synthetic data with known signal."""
    np.random.seed(42)

    # Features (random noise)
    X = np.random.randn(n_samples, n_features)

    # Target with signal in first few features
    signal = signal_strength * (X[:, 0] + 0.5 * X[:, 1] + 0.3 * X[:, 2])
    noise = np.random.randn(n_samples) * 0.5
    y = (signal + noise > 0).astype(int)

    print(f"Data: {n_samples} samples, {n_features} features")
    print(f"Target balance: {y.mean():.2%} positive")

    return X, y


def leaky_cv(X, y, n_folds=5, n_features_select=20):
    """
    LEAKY cross-validation.

    Problem: Feature selection and scaling happen BEFORE the CV split,
    so test fold information leaks into the transformation.
    """
    print("\n" + "=" * 60)
    print("LEAKY CV (transforms BEFORE split)")
    print("=" * 60)

    # LEAK 1: Scale on ALL data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # LEAK 2: Feature selection on ALL data with ALL targets
    mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
    top_idx = np.argsort(mi_scores)[::-1][:n_features_select]
    X_selected = X_scaled[:, top_idx]

    print(f"  Selected {n_features_select} features using Mutual Info")

    # Now do CV (but damage is done - test info already leaked)
    fold_size = len(X) // n_folds
    train_aucs = []
    test_aucs = []

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(X)

        test_idx = np.arange(test_start, test_end)
        train_idx = np.concatenate([np.arange(0, test_start), np.arange(test_end, len(X))])

        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train model
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        train_proba = model.predict_proba(X_train)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]

        train_auc = roc_auc_score(y_train, train_proba)
        test_auc = roc_auc_score(y_test, test_proba)

        train_aucs.append(train_auc)
        test_aucs.append(test_auc)

        print(f"  Fold {fold + 1}: Train AUC={train_auc:.3f}, Test AUC={test_auc:.3f}")

    mean_train = np.mean(train_aucs)
    mean_test = np.mean(test_aucs)
    gap = mean_train - mean_test

    print(f"\n  Summary:")
    print(f"    Mean Train AUC: {mean_train:.3f}")
    print(f"    Mean Test AUC:  {mean_test:.3f} +/- {np.std(test_aucs):.3f}")
    print(f"    Train-Test Gap: {gap:.3f}")

    return {"mean_train": mean_train, "mean_test": mean_test, "gap": gap}


def leak_proof_cv(X, y, n_folds=5, n_features_select=20):
    """
    LEAK-PROOF cross-validation.

    All transformations happen INSIDE each fold on training data only.
    """
    print("\n" + "=" * 60)
    print("LEAK-PROOF CV (transforms INSIDE each fold)")
    print("=" * 60)

    fold_size = len(X) // n_folds
    train_aucs = []
    test_aucs = []

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(X)

        test_idx = np.arange(test_start, test_end)
        train_idx = np.concatenate([np.arange(0, test_start), np.arange(test_end, len(X))])

        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # STEP 1: Scale on TRAIN only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # STEP 2: Feature selection on TRAIN only
        mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=42)
        top_idx = np.argsort(mi_scores)[::-1][:n_features_select]

        X_train = X_train_scaled[:, top_idx]
        X_test = X_test_scaled[:, top_idx]

        # STEP 3: Train model on TRAIN only
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        train_proba = model.predict_proba(X_train)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]

        train_auc = roc_auc_score(y_train, train_proba)
        test_auc = roc_auc_score(y_test, test_proba)

        train_aucs.append(train_auc)
        test_aucs.append(test_auc)

        print(f"  Fold {fold + 1}: Train AUC={train_auc:.3f}, Test AUC={test_auc:.3f}")

    mean_train = np.mean(train_aucs)
    mean_test = np.mean(test_aucs)
    gap = mean_train - mean_test

    print(f"\n  Summary:")
    print(f"    Mean Train AUC: {mean_train:.3f}")
    print(f"    Mean Test AUC:  {mean_test:.3f} +/- {np.std(test_aucs):.3f}")
    print(f"    Train-Test Gap: {gap:.3f}")

    return {"mean_train": mean_train, "mean_test": mean_test, "gap": gap}


def test_leak_proof_pipeline():
    """Test the new LeakProofPipeline class."""
    print("\n" + "=" * 60)
    print("TESTING LeakProofPipeline CLASS")
    print("=" * 60)

    try:
        from src.leak_proof_cv import LeakProofPipeline, train_with_leak_proof_cv

        X, y = create_test_data(n_samples=500, n_features=50)

        config = {
            "n_cv_folds": 5,
            "purge_days": 2,
            "embargo_days": 1,
            "feature_selection_method": "mutual_info",
            "n_features": 15,
            "dim_reduction_method": "pca",
            "n_components": 10,
            "use_ensemble": True,
        }

        pipeline, results = train_with_leak_proof_cv(X, y, config=config, verbose=True)

        print(f"\n[PASS] LeakProofPipeline test successful!")
        return results

    except Exception as e:
        print(f"\n[FAIL] LeakProofPipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("=" * 70)
    print("LEAK-PROOF CV - COMPARISON TEST")
    print("=" * 70)

    # Create test data
    X, y = create_test_data(n_samples=1000, n_features=100, signal_strength=0.3)

    # Run both CV methods
    leaky_results = leaky_cv(X, y, n_folds=5, n_features_select=20)
    leak_proof_results = leak_proof_cv(X, y, n_folds=5, n_features_select=20)

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    print(f"\n  {'Method':<20} {'Train AUC':<12} {'Test AUC':<12} {'Gap':<10}")
    print(f"  {'-' * 54}")
    print(f"  {'Leaky CV':<20} {leaky_results['mean_train']:<12.3f} {leaky_results['mean_test']:<12.3f} {leaky_results['gap']:<10.3f}")
    print(f"  {'Leak-Proof CV':<20} {leak_proof_results['mean_train']:<12.3f} {leak_proof_results['mean_test']:<12.3f} {leak_proof_results['gap']:<10.3f}")

    # Analysis
    test_diff = leaky_results['mean_test'] - leak_proof_results['mean_test']
    gap_diff = leaky_results['gap'] - leak_proof_results['gap']

    print(f"\n  Analysis:")
    print(f"    Test AUC difference: {test_diff:+.3f} (leaky is {'higher' if test_diff > 0 else 'lower'})")
    print(f"    Gap difference: {gap_diff:+.3f} (leaky has {'larger' if gap_diff > 0 else 'smaller'} gap)")

    if test_diff > 0.02:
        print(f"\n  [WARNING] Leaky CV shows {test_diff:.1%} higher test AUC - this is the overfitting!")
        print(f"            The leak-proof AUC is more realistic for production.")

    # Test the pipeline class
    pipeline_results = test_leak_proof_pipeline()

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

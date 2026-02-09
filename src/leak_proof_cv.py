"""
GIGA TRADER - Leak-Proof Cross-Validation Pipeline (Shim)
==========================================================
Re-exports from decomposed phase modules:
  - phase_10_feature_processing.leak_proof_selector: LeakProofFeatureSelector
  - phase_10_feature_processing.leak_proof_reducer: LeakProofDimReducer
  - phase_11_cv_splitting.leak_proof_cv_core: CVFoldResult, LeakProofCV
  - phase_11_cv_splitting.ensemble_reducer: EnsembleReducer, LeakProofPipeline, train_with_leak_proof_cv
"""

# Phase 10: Feature processing
from src.phase_10_feature_processing.leak_proof_selector import LeakProofFeatureSelector
from src.phase_10_feature_processing.leak_proof_reducer import LeakProofDimReducer

# Phase 11: CV splitting
from src.phase_11_cv_splitting.leak_proof_cv_core import CVFoldResult, LeakProofCV
from src.phase_11_cv_splitting.ensemble_reducer import (
    EnsembleReducer,
    LeakProofPipeline,
    train_with_leak_proof_cv,
)

__all__ = [
    "LeakProofFeatureSelector",
    "LeakProofDimReducer",
    "CVFoldResult",
    "LeakProofCV",
    "EnsembleReducer",
    "LeakProofPipeline",
    "train_with_leak_proof_cv",
]

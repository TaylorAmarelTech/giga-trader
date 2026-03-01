"""
Ensemble Disagreement Features — computed from model prediction spread.

When multiple models in the production ensemble disagree, this is a
predictive signal: high disagreement = low confidence = tighter risk management.

Features (4, prefix ens_):
  ens_std        — Std dev of model probabilities across ensemble members
  ens_range      — Max - min probability
  ens_agreement  — Fraction of models agreeing on direction (>0.5 = up)
  ens_entropy    — Shannon entropy of probability distribution
"""

import math
from typing import Dict, List


def compute_ensemble_disagreement(probas: List[float]) -> Dict[str, float]:
    """
    Compute disagreement metrics from a list of model probabilities.

    Parameters
    ----------
    probas : List[float]
        Probability predictions from each ensemble member (0-1 scale,
        where >0.5 indicates bullish and <0.5 indicates bearish).

    Returns
    -------
    Dict[str, float]
        Dictionary with ens_std, ens_range, ens_agreement, ens_entropy.
    """
    if not probas or len(probas) < 2:
        return {
            "ens_std": 0.0,
            "ens_range": 0.0,
            "ens_agreement": 1.0,
            "ens_entropy": 0.0,
        }

    n = len(probas)

    # Standard deviation of probabilities
    mean_p = sum(probas) / n
    variance = sum((p - mean_p) ** 2 for p in probas) / n
    ens_std = math.sqrt(variance)

    # Range (max - min)
    ens_range = max(probas) - min(probas)

    # Agreement: fraction of models agreeing on direction
    n_bullish = sum(1 for p in probas if p > 0.5)
    n_bearish = n - n_bullish
    ens_agreement = max(n_bullish, n_bearish) / n

    # Shannon entropy of the probability distribution
    # Treat each model's prediction as a "vote" — bin into bullish/bearish
    if n_bullish == 0 or n_bearish == 0:
        ens_entropy = 0.0
    else:
        p_bull = n_bullish / n
        p_bear = n_bearish / n
        ens_entropy = -(p_bull * math.log2(p_bull) + p_bear * math.log2(p_bear))

    return {
        "ens_std": round(ens_std, 6),
        "ens_range": round(ens_range, 6),
        "ens_agreement": round(ens_agreement, 4),
        "ens_entropy": round(ens_entropy, 6),
    }

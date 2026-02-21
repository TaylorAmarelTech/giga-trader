"""
GIGA TRADER - Symbolic Cross-Model Learner
============================================
Cross-model knowledge transfer beyond prediction-level ensembling.

Components:
  1. FeatureImportanceExtractor  - Extract importances from any model type
  2. UniversalFeatureMap         - Aggregate importances across all models
  3. SymbolicRuleExtractor       - Extract IF-THEN rules from tree models
  4. RuleCrossAnalyzer           - Compare rules across models for consensus/gaps
  5. CrossModelAugmenter         - Generate meta-features from existing models
"""

import json
import logging
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger("GigaTrader")


# =============================================================================
# 1. FEATURE IMPORTANCE EXTRACTOR
# =============================================================================

class FeatureImportanceExtractor:
    """
    Extract feature importances from any fitted model or pipeline.

    Handles:
      - Tree models (feature_importances_)
      - Linear models (abs(coef_))
      - EnsembleReducer (per-base-model extraction + averaging)
      - LeakProofPipeline (feature_selector_ masks + final model)
      - Registry ModelMetrics (pre-stored top_features)
    """

    @staticmethod
    def from_tree_model(model, feature_names: List[str] = None) -> Dict[str, float]:
        """Extract importances from a tree-based model."""
        if not hasattr(model, "feature_importances_"):
            return {}
        importances = model.feature_importances_
        if feature_names and len(feature_names) == len(importances):
            return {name: float(imp) for name, imp in zip(feature_names, importances)}
        return {f"feature_{i}": float(imp) for i, imp in enumerate(importances)}

    @staticmethod
    def from_linear_model(model, feature_names: List[str] = None) -> Dict[str, float]:
        """Extract importances from a linear model using abs(coef_)."""
        if not hasattr(model, "coef_"):
            return {}
        coef = np.abs(model.coef_).ravel()
        # Normalize to [0, 1]
        total = coef.sum()
        if total > 0:
            coef = coef / total
        if feature_names and len(feature_names) == len(coef):
            return {name: float(c) for name, c in zip(feature_names, coef)}
        return {f"feature_{i}": float(c) for i, c in enumerate(coef)}

    @staticmethod
    def from_ensemble_reducer(ensemble, feature_names: List[str] = None) -> Dict[str, float]:
        """
        Extract importances from an EnsembleReducer by averaging
        across its base models (tree importances + linear coefs).
        """
        all_importances = defaultdict(list)

        # Access base models from VotingClassifier or manual list
        models_to_check = []
        if hasattr(ensemble, "ensemble_") and ensemble.ensemble_ is not None:
            if hasattr(ensemble.ensemble_, "estimators_"):
                models_to_check = [
                    (name, est) for (name, _), est
                    in zip(ensemble.base_models_, ensemble.ensemble_.estimators_)
                ]
        if not models_to_check and hasattr(ensemble, "base_models_"):
            models_to_check = ensemble.base_models_

        for name, model in models_to_check:
            if hasattr(model, "feature_importances_"):
                imp = FeatureImportanceExtractor.from_tree_model(model, feature_names)
            elif hasattr(model, "coef_"):
                imp = FeatureImportanceExtractor.from_linear_model(model, feature_names)
            else:
                continue
            for feat, val in imp.items():
                all_importances[feat].append(val)

        # Average across base models
        return {feat: float(np.mean(vals)) for feat, vals in all_importances.items()}

    @staticmethod
    def from_pipeline(pipeline, raw_feature_names: List[str] = None) -> Dict[str, float]:
        """
        Extract importances from a LeakProofPipeline.

        Maps importances back through the selection/reduction chain
        to raw feature space where possible.
        """
        result = {}

        # Step 1: Get selection mask from feature_selector_
        selected_raw_names = None
        if hasattr(pipeline, "feature_selector_") and pipeline.feature_selector_ is not None:
            selector = pipeline.feature_selector_
            if raw_feature_names is not None and hasattr(selector, "var_mask_") and selector.var_mask_ is not None:
                # Map through var_mask -> corr_mask -> selected_idx
                var_names = [n for n, m in zip(raw_feature_names, selector.var_mask_) if m]
                if hasattr(selector, "corr_mask_") and selector.corr_mask_ is not None:
                    corr_names = [n for n, m in zip(var_names, selector.corr_mask_) if m]
                else:
                    corr_names = var_names
                if hasattr(selector, "selected_idx_") and selector.selected_idx_ is not None:
                    selected_raw_names = [corr_names[i] for i in selector.selected_idx_
                                          if i < len(corr_names)]
                else:
                    selected_raw_names = corr_names

                # Mark selected features with 1.0, unselected with 0.0
                for name in raw_feature_names:
                    result[name] = 1.0 if (selected_raw_names and name in selected_raw_names) else 0.0

        # Step 2: Get model-level importances on reduced features
        model_importances = {}
        if hasattr(pipeline, "model_") and pipeline.model_ is not None:
            model = pipeline.model_
            # EnsembleReducer
            if hasattr(model, "base_models_"):
                model_importances = FeatureImportanceExtractor.from_ensemble_reducer(model)
            elif hasattr(model, "feature_importances_"):
                model_importances = FeatureImportanceExtractor.from_tree_model(model)
            elif hasattr(model, "coef_"):
                model_importances = FeatureImportanceExtractor.from_linear_model(model)

        # Step 3: If we have both selection mask and model importances,
        # back-project model importances to selected feature names
        if selected_raw_names and model_importances:
            # model_importances are on dim-reduced space, not directly mappable
            # to raw features. But the selection mask IS mappable.
            # Combine: selection (binary) * model-level importance (if mappable)
            n_model_feats = len(model_importances)
            n_selected = len(selected_raw_names)
            if n_model_feats == n_selected:
                # Perfect alignment: model operates on selected features
                for feat_key, imp_val in model_importances.items():
                    idx = int(feat_key.replace("feature_", "")) if feat_key.startswith("feature_") else -1
                    if 0 <= idx < n_selected:
                        result[selected_raw_names[idx]] = imp_val
            # Otherwise, keep binary selection mask (still useful)

        return result

    @staticmethod
    def from_metrics(metrics_dict: Dict) -> Dict[str, float]:
        """
        Extract importances from stored ModelMetrics.top_features.
        """
        features = metrics_dict.get("top_features", [])
        importances = metrics_dict.get("top_feature_importances", [])
        if not features:
            return {}
        if len(importances) < len(features):
            # Assign decreasing importance by rank
            importances = [1.0 / (i + 1) for i in range(len(features))]
        return {f: float(imp) for f, imp in zip(features, importances)}

    @staticmethod
    def extract(model_or_pipeline, feature_names: List[str] = None,
                metrics: Dict = None) -> Dict[str, float]:
        """
        Auto-detect model type and extract importances.
        """
        # Try pipeline first
        if hasattr(model_or_pipeline, "feature_selector_"):
            return FeatureImportanceExtractor.from_pipeline(
                model_or_pipeline, feature_names
            )
        # EnsembleReducer
        if hasattr(model_or_pipeline, "base_models_"):
            return FeatureImportanceExtractor.from_ensemble_reducer(
                model_or_pipeline, feature_names
            )
        # Tree
        if hasattr(model_or_pipeline, "feature_importances_"):
            return FeatureImportanceExtractor.from_tree_model(
                model_or_pipeline, feature_names
            )
        # Linear
        if hasattr(model_or_pipeline, "coef_"):
            return FeatureImportanceExtractor.from_linear_model(
                model_or_pipeline, feature_names
            )
        # Fall back to stored metrics
        if metrics:
            return FeatureImportanceExtractor.from_metrics(metrics)
        return {}


# =============================================================================
# 2. UNIVERSAL FEATURE MAP
# =============================================================================

@dataclass
class FeatureProfile:
    """Aggregated importance profile for a single feature."""
    name: str
    # How many models use this feature (selected or non-zero importance)
    n_models_using: int = 0
    # Total models considered
    n_models_total: int = 0
    # Quality-weighted mean importance (weighted by model AUC)
    weighted_importance: float = 0.0
    # Unweighted mean importance
    mean_importance: float = 0.0
    # Std of importances across models
    std_importance: float = 0.0
    # Max importance in any single model
    max_importance: float = 0.0
    # Which model types rely on this feature most
    top_model_types: List[str] = field(default_factory=list)

    def universality_score(self) -> float:
        """How universal is this feature? 0=niche, 1=used by all models."""
        if self.n_models_total == 0:
            return 0.0
        return self.n_models_using / self.n_models_total

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["universality_score"] = self.universality_score()
        return d


class UniversalFeatureMap:
    """
    Aggregate feature importances across all models to build a
    universal ranking of features.

    Identifies:
      - Universal features (important across many model types)
      - Niche features (important only to specific models)
      - Unused features (candidates for removal)
      - Feature set suggestions for new model training
    """

    def __init__(self, persist_path: Optional[Path] = None):
        self._persist_path = persist_path
        self.profiles: Dict[str, FeatureProfile] = {}
        self._model_contributions: List[Dict] = []
        self._load()

    def _load(self):
        """Load persisted map from disk."""
        if not self._persist_path:
            return
        try:
            p = Path(self._persist_path)
            if not p.is_file():
                return
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            for name, profile_dict in data.get("profiles", {}).items():
                self.profiles[name] = FeatureProfile(
                    name=profile_dict["name"],
                    n_models_using=profile_dict.get("n_models_using", 0),
                    n_models_total=profile_dict.get("n_models_total", 0),
                    weighted_importance=profile_dict.get("weighted_importance", 0.0),
                    mean_importance=profile_dict.get("mean_importance", 0.0),
                    std_importance=profile_dict.get("std_importance", 0.0),
                    max_importance=profile_dict.get("max_importance", 0.0),
                    top_model_types=profile_dict.get("top_model_types", []),
                )
            self._model_contributions = data.get("model_contributions", [])
            logger.info(f"Loaded universal feature map: {len(self.profiles)} features")
        except (json.JSONDecodeError, ValueError, TypeError, IOError) as e:
            logger.warning(f"Could not load feature map: {e}")

    def _save(self):
        """Persist map to disk."""
        if not self._persist_path:
            return
        try:
            from src.core.state_manager import atomic_write_json
            data = {
                "updated_at": datetime.now().isoformat(),
                "n_features": len(self.profiles),
                "n_models": len(self._model_contributions),
                "profiles": {
                    name: profile.to_dict()
                    for name, profile in self.profiles.items()
                },
                "model_contributions": self._model_contributions[-100:],  # Keep last 100
            }
            atomic_write_json(Path(self._persist_path), data)
        except Exception as e:
            logger.debug(f"Could not persist feature map: {e}")

    def add_model(
        self,
        model_id: str,
        importances: Dict[str, float],
        model_auc: float = 0.5,
        model_type: str = "unknown",
    ):
        """
        Add a model's feature importances to the universal map.

        Args:
            model_id: Unique identifier for the model
            importances: Feature name -> importance score mapping
            model_auc: Model's test AUC (used as quality weight)
            model_type: Type of model (e.g., "gradient_boosting", "logistic_l2")
        """
        if not importances:
            return

        # Record contribution
        self._model_contributions.append({
            "model_id": model_id,
            "model_type": model_type,
            "model_auc": model_auc,
            "n_features": len(importances),
            "added_at": datetime.now().isoformat(),
        })

        # Update profiles
        all_feature_names = set(self.profiles.keys()) | set(importances.keys())
        n_total = len(self._model_contributions)

        for feat_name in all_feature_names:
            if feat_name not in self.profiles:
                self.profiles[feat_name] = FeatureProfile(name=feat_name)
            profile = self.profiles[feat_name]
            profile.n_models_total = n_total

        # Recompute aggregates from all contributions
        self._recompute_aggregates()
        self._save()

    def _recompute_aggregates(self):
        """Recompute all aggregate statistics from contribution history."""
        # Collect all importances per feature
        feature_data: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)
        # (importance, auc, model_type)

        for contrib in self._model_contributions:
            model_id = contrib["model_id"]
            model_auc = contrib.get("model_auc", 0.5)
            model_type = contrib.get("model_type", "unknown")

            # We need to re-extract importances — but we only store summary
            # So we track importances at add time. This is a simplified approach:
            # profiles are updated incrementally.
            pass

        # Note: Full recomputation requires storing per-model importances.
        # For memory efficiency, we use incremental updates in add_model instead.
        # The _recompute_aggregates is called after each add_model to update n_models_total.
        n_total = len(self._model_contributions)
        for profile in self.profiles.values():
            profile.n_models_total = n_total

    def add_model_full(
        self,
        model_id: str,
        importances: Dict[str, float],
        model_auc: float = 0.5,
        model_type: str = "unknown",
    ):
        """
        Add model with full incremental aggregate update.
        This is the preferred method — updates running statistics.
        """
        if not importances:
            return

        # Record contribution
        self._model_contributions.append({
            "model_id": model_id,
            "model_type": model_type,
            "model_auc": model_auc,
            "n_features": len(importances),
            "added_at": datetime.now().isoformat(),
        })

        n_total = len(self._model_contributions)
        quality_weight = max(model_auc - 0.5, 0.01)  # AUC > 0.5 is informative

        for feat_name, imp_val in importances.items():
            if feat_name not in self.profiles:
                self.profiles[feat_name] = FeatureProfile(name=feat_name)
            p = self.profiles[feat_name]

            is_used = imp_val > 0.0
            if is_used:
                p.n_models_using += 1

            # Running weighted importance (weighted by quality)
            old_weight_sum = p.weighted_importance * max(p.n_models_using - (1 if is_used else 0), 1)
            p.weighted_importance = (old_weight_sum + imp_val * quality_weight) / max(p.n_models_using, 1)

            # Running mean importance
            old_count = p.n_models_using - (1 if is_used else 0)
            if old_count > 0:
                p.mean_importance = (p.mean_importance * old_count + imp_val) / p.n_models_using
            elif is_used:
                p.mean_importance = imp_val

            # Max
            p.max_importance = max(p.max_importance, imp_val)

            # Top model types (keep top 3)
            if is_used and model_type not in p.top_model_types:
                p.top_model_types.append(model_type)
                if len(p.top_model_types) > 3:
                    p.top_model_types = p.top_model_types[-3:]

        # Update n_models_total for all profiles
        for p in self.profiles.values():
            p.n_models_total = n_total

        self._save()

    def get_universal_ranking(self, min_universality: float = 0.5) -> List[Tuple[str, float]]:
        """
        Get features ranked by combined universality + importance.

        Returns features used by >= min_universality fraction of models,
        sorted by weighted importance.
        """
        result = []
        for name, profile in self.profiles.items():
            if profile.universality_score() >= min_universality:
                score = profile.weighted_importance * profile.universality_score()
                result.append((name, score))
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def get_niche_features(self, max_universality: float = 0.3,
                           min_importance: float = 0.05) -> List[Tuple[str, float, str]]:
        """
        Get features that are niche — important to few models but highly valued.

        Returns: List of (feature_name, max_importance, model_type)
        """
        result = []
        for name, profile in self.profiles.items():
            if (profile.universality_score() <= max_universality
                    and profile.max_importance >= min_importance):
                top_type = profile.top_model_types[0] if profile.top_model_types else "unknown"
                result.append((name, profile.max_importance, top_type))
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def get_consensus_features(self, top_n: int = 20, min_models: int = 5) -> List[Tuple[str, float, int]]:
        """Get features that consistently appear as important across models.

        Wave 26: Used for daily campaign reports.

        Returns:
            List of (feature_name, consensus_score, n_models_using) tuples,
            sorted by consensus_score descending.
        """
        n_total = len(self._model_contributions)
        if n_total < min_models:
            return []

        consensus = []
        for name, profile in self.profiles.items():
            if profile.n_models_using > 0:
                freq = profile.n_models_using / max(n_total, 1)
                consensus_score = profile.mean_importance * freq
                consensus.append((name, round(consensus_score, 4), profile.n_models_using))

        consensus.sort(key=lambda x: x[1], reverse=True)
        return consensus[:top_n]

    def get_unused_features(self) -> List[str]:
        """Get features that no model uses (candidates for removal)."""
        return [name for name, p in self.profiles.items() if p.n_models_using == 0]

    def suggest_feature_set(
        self,
        strategy: str = "balanced",
        n_features: int = 30,
    ) -> List[str]:
        """
        Suggest a feature set for new model training.

        Strategies:
          - "universal": Top-N most universally important features
          - "niche": Features only used by few models (explore gaps)
          - "balanced": 70% universal + 30% niche
          - "complementary": Features not used by best models (find new edges)
        """
        if strategy == "universal":
            ranked = self.get_universal_ranking(min_universality=0.3)
            return [name for name, _ in ranked[:n_features]]

        elif strategy == "niche":
            niche = self.get_niche_features(max_universality=0.3)
            return [name for name, _, _ in niche[:n_features]]

        elif strategy == "balanced":
            n_universal = int(n_features * 0.7)
            n_niche = n_features - n_universal
            universal = [name for name, _ in self.get_universal_ranking(0.3)[:n_universal]]
            niche = [name for name, _, _ in self.get_niche_features(0.3)[:n_niche]
                     if name not in universal]
            return universal + niche[:n_features - len(universal)]

        elif strategy == "complementary":
            # Features with high importance but low universality
            result = []
            for name, profile in sorted(
                self.profiles.items(),
                key=lambda x: x[1].max_importance * (1 - x[1].universality_score()),
                reverse=True,
            ):
                if profile.max_importance > 0 and profile.universality_score() < 0.5:
                    result.append(name)
                if len(result) >= n_features:
                    break
            return result

        else:
            return self.suggest_feature_set("balanced", n_features)

    def get_report(self) -> Dict:
        """Generate a summary report of the feature landscape."""
        if not self.profiles:
            return {"n_features": 0, "n_models": 0}

        universality_scores = [p.universality_score() for p in self.profiles.values()]
        importances = [p.weighted_importance for p in self.profiles.values()
                       if p.weighted_importance > 0]

        return {
            "n_features": len(self.profiles),
            "n_models": len(self._model_contributions),
            "n_universal_features": sum(1 for s in universality_scores if s >= 0.5),
            "n_niche_features": sum(1 for s in universality_scores if 0 < s < 0.3),
            "n_unused_features": sum(1 for s in universality_scores if s == 0),
            "mean_universality": float(np.mean(universality_scores)) if universality_scores else 0.0,
            "mean_importance": float(np.mean(importances)) if importances else 0.0,
            "top_10_universal": self.get_universal_ranking(0.3)[:10],
            "top_5_niche": self.get_niche_features(0.3)[:5],
        }


# =============================================================================
# 3. SYMBOLIC RULE EXTRACTION
# =============================================================================

@dataclass
class SymbolicRule:
    """A single IF-THEN rule extracted from a decision tree path."""
    conditions: List[Tuple[str, str, float]]  # [(feature, operator, threshold), ...]
    prediction: int  # 0 or 1
    confidence: float  # Probability at leaf
    n_samples: int  # Number of training samples reaching this leaf
    source_model: str = ""  # Which model this came from
    depth: int = 0

    def to_readable(self) -> str:
        """Convert rule to human-readable string."""
        parts = []
        for feat, op, thresh in self.conditions:
            parts.append(f"{feat} {op} {thresh:.4f}")
        cond_str = " AND ".join(parts)
        label = "UP" if self.prediction == 1 else "DOWN"
        return f"IF {cond_str} THEN {label} (conf={self.confidence:.2f}, n={self.n_samples})"

    def feature_set(self) -> set:
        """Get the set of features used in this rule."""
        return {feat for feat, _, _ in self.conditions}

    def to_dict(self) -> Dict:
        return {
            "conditions": [(f, op, float(t)) for f, op, t in self.conditions],
            "prediction": self.prediction,
            "confidence": self.confidence,
            "n_samples": self.n_samples,
            "source_model": self.source_model,
            "depth": self.depth,
            "readable": self.to_readable(),
        }


class SymbolicRuleExtractor:
    """
    Extract IF-THEN rules from tree-based models.

    Works with:
      - DecisionTreeClassifier
      - GradientBoostingClassifier (per-tree extraction)
      - RandomForestClassifier (per-tree extraction)
      - EnsembleReducer (extracts from GB sub-model)
    """

    @staticmethod
    def from_decision_tree(
        tree_model,
        feature_names: List[str] = None,
        source_model: str = "",
        min_samples: int = 10,
        min_confidence: float = 0.55,
    ) -> List[SymbolicRule]:
        """
        Extract all decision paths from a single decision tree.
        """
        try:
            from sklearn.tree import DecisionTreeClassifier
        except ImportError:
            return []

        # Get the underlying tree
        if hasattr(tree_model, "tree_"):
            tree = tree_model.tree_
        else:
            return []

        n_features = tree.n_features
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        rules = []
        _extract_paths(
            tree, 0, [], feature_names,
            source_model, min_samples, min_confidence, rules
        )
        return rules

    @staticmethod
    def from_gradient_boosting(
        gb_model,
        feature_names: List[str] = None,
        source_model: str = "",
        min_samples: int = 10,
        min_confidence: float = 0.55,
        max_trees: int = 10,
    ) -> List[SymbolicRule]:
        """
        Extract rules from a GradientBoostingClassifier's internal trees.
        """
        if not hasattr(gb_model, "estimators_"):
            return []

        rules = []
        n_trees = min(len(gb_model.estimators_), max_trees)
        for i in range(n_trees):
            tree_regressor = gb_model.estimators_[i, 0]
            if hasattr(tree_regressor, "tree_"):
                tree_rules = SymbolicRuleExtractor.from_decision_tree(
                    tree_regressor, feature_names,
                    source_model=f"{source_model}_tree{i}",
                    min_samples=min_samples,
                    min_confidence=0.0,  # GB trees don't have class confidence
                )
                rules.extend(tree_rules)
        return rules

    @staticmethod
    def from_ensemble_reducer(
        ensemble,
        feature_names: List[str] = None,
        source_model: str = "ensemble",
        min_samples: int = 10,
        min_confidence: float = 0.55,
    ) -> List[SymbolicRule]:
        """
        Extract rules from the tree sub-models within an EnsembleReducer.
        """
        rules = []
        models_to_check = []

        if hasattr(ensemble, "ensemble_") and ensemble.ensemble_ is not None:
            if hasattr(ensemble.ensemble_, "estimators_"):
                models_to_check = [
                    (name, est) for (name, _), est
                    in zip(ensemble.base_models_, ensemble.ensemble_.estimators_)
                ]
        if not models_to_check and hasattr(ensemble, "base_models_"):
            models_to_check = ensemble.base_models_

        for name, model in models_to_check:
            if hasattr(model, "estimators_"):
                # GradientBoostingClassifier
                sub_rules = SymbolicRuleExtractor.from_gradient_boosting(
                    model, feature_names,
                    source_model=f"{source_model}_{name}",
                    min_samples=min_samples,
                    min_confidence=min_confidence,
                )
                rules.extend(sub_rules)
            elif hasattr(model, "tree_"):
                sub_rules = SymbolicRuleExtractor.from_decision_tree(
                    model, feature_names,
                    source_model=f"{source_model}_{name}",
                    min_samples=min_samples,
                    min_confidence=min_confidence,
                )
                rules.extend(sub_rules)
        return rules

    @staticmethod
    def extract(model_or_pipeline, feature_names: List[str] = None,
                source_model: str = "", **kwargs) -> List[SymbolicRule]:
        """Auto-detect model type and extract rules."""
        # Pipeline with model inside
        if hasattr(model_or_pipeline, "model_"):
            return SymbolicRuleExtractor.extract(
                model_or_pipeline.model_, feature_names, source_model, **kwargs
            )
        # EnsembleReducer
        if hasattr(model_or_pipeline, "base_models_"):
            return SymbolicRuleExtractor.from_ensemble_reducer(
                model_or_pipeline, feature_names, source_model, **kwargs
            )
        # GradientBoosting
        if hasattr(model_or_pipeline, "estimators_") and hasattr(model_or_pipeline, "n_estimators"):
            return SymbolicRuleExtractor.from_gradient_boosting(
                model_or_pipeline, feature_names, source_model, **kwargs
            )
        # Single decision tree
        if hasattr(model_or_pipeline, "tree_"):
            return SymbolicRuleExtractor.from_decision_tree(
                model_or_pipeline, feature_names, source_model, **kwargs
            )
        return []


def _extract_paths(
    tree, node_id: int,
    current_conditions: List[Tuple[str, str, float]],
    feature_names: List[str],
    source_model: str,
    min_samples: int,
    min_confidence: float,
    rules: List[SymbolicRule],
):
    """Recursively extract all root-to-leaf paths from a sklearn tree."""
    left = tree.children_left[node_id]
    right = tree.children_right[node_id]

    # Leaf node
    if left == right:  # -1 == -1
        n_samples = int(tree.n_node_samples[node_id])
        if n_samples < min_samples:
            return

        # Get prediction and confidence
        values = tree.value[node_id].ravel()
        total = values.sum()
        if total == 0:
            return

        if len(values) >= 2:
            prediction = int(np.argmax(values))
            confidence = float(values[prediction] / total)
        else:
            # Regression tree (GB uses these) — use sign of value
            prediction = 1 if values[0] > 0 else 0
            confidence = min(abs(float(values[0])), 1.0)

        if confidence < min_confidence:
            return

        rules.append(SymbolicRule(
            conditions=list(current_conditions),
            prediction=prediction,
            confidence=confidence,
            n_samples=n_samples,
            source_model=source_model,
            depth=len(current_conditions),
        ))
        return

    # Internal node — branch left and right
    feature_idx = tree.feature[node_id]
    threshold = float(tree.threshold[node_id])
    feat_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"feature_{feature_idx}"

    # Left child: feature <= threshold
    left_conditions = current_conditions + [(feat_name, "<=", threshold)]
    _extract_paths(tree, left, left_conditions, feature_names, source_model,
                   min_samples, min_confidence, rules)

    # Right child: feature > threshold
    right_conditions = current_conditions + [(feat_name, ">", threshold)]
    _extract_paths(tree, right, right_conditions, feature_names, source_model,
                   min_samples, min_confidence, rules)


# =============================================================================
# 4. RULE CROSS-ANALYZER
# =============================================================================

class RuleCrossAnalyzer:
    """
    Compare symbolic rules across models to find:
      - Consensus rules (multiple models agree)
      - Complementary rules (cover different market regimes)
      - Contradictory rules (models disagree)
      - Feature blind spots (features no model uses in rules)
    """

    def __init__(self, rules: List[SymbolicRule] = None):
        self.rules = rules or []

    def add_rules(self, rules: List[SymbolicRule]):
        """Add rules from another model."""
        self.rules.extend(rules)

    def find_consensus_rules(self, min_agreement: int = 2) -> List[Dict]:
        """
        Find conditions that appear in rules from multiple models.

        Groups rules by their feature conditions (ignoring exact thresholds)
        and returns groups where >= min_agreement different models agree.
        """
        # Group by condition features + prediction
        pattern_groups: Dict[str, List[SymbolicRule]] = defaultdict(list)

        for rule in self.rules:
            # Create a pattern key: sorted feature set + prediction
            feat_key = "_".join(sorted(rule.feature_set())) + f"_pred{rule.prediction}"
            pattern_groups[feat_key].append(rule)

        consensus = []
        for pattern_key, group_rules in pattern_groups.items():
            # Count distinct source models
            source_models = set(r.source_model.split("_tree")[0] for r in group_rules)
            if len(source_models) >= min_agreement:
                # Compute average confidence and threshold ranges
                avg_conf = np.mean([r.confidence for r in group_rules])
                total_samples = sum(r.n_samples for r in group_rules)
                features_used = set()
                for r in group_rules:
                    features_used.update(r.feature_set())

                consensus.append({
                    "pattern": pattern_key,
                    "n_models_agreeing": len(source_models),
                    "n_rules": len(group_rules),
                    "avg_confidence": float(avg_conf),
                    "total_samples": total_samples,
                    "features_used": sorted(features_used),
                    "source_models": sorted(source_models),
                    "example_rule": group_rules[0].to_readable(),
                })

        consensus.sort(key=lambda x: x["n_models_agreeing"] * x["avg_confidence"], reverse=True)
        return consensus

    def find_contradictions(self) -> List[Dict]:
        """
        Find cases where models using similar features predict opposite directions.
        """
        # Group rules by feature set (ignoring prediction)
        feature_groups: Dict[str, List[SymbolicRule]] = defaultdict(list)
        for rule in self.rules:
            feat_key = "_".join(sorted(rule.feature_set()))
            feature_groups[feat_key].append(rule)

        contradictions = []
        for feat_key, group_rules in feature_groups.items():
            predictions = set(r.prediction for r in group_rules)
            if len(predictions) > 1:
                # Both UP and DOWN predictions from same features
                up_rules = [r for r in group_rules if r.prediction == 1]
                down_rules = [r for r in group_rules if r.prediction == 0]
                up_conf = np.mean([r.confidence for r in up_rules]) if up_rules else 0
                down_conf = np.mean([r.confidence for r in down_rules]) if down_rules else 0

                contradictions.append({
                    "features": feat_key.split("_"),
                    "n_up_rules": len(up_rules),
                    "n_down_rules": len(down_rules),
                    "avg_up_confidence": float(up_conf),
                    "avg_down_confidence": float(down_conf),
                    "dominant_direction": "UP" if up_conf > down_conf else "DOWN",
                    "conflict_severity": abs(float(up_conf - down_conf)),
                })

        contradictions.sort(key=lambda x: x["conflict_severity"])
        return contradictions

    def find_feature_blind_spots(self, all_feature_names: List[str]) -> List[str]:
        """
        Find features that no model uses in any extracted rule.
        These represent potential unexplored signal sources.
        """
        used_features = set()
        for rule in self.rules:
            used_features.update(rule.feature_set())
        return sorted(set(all_feature_names) - used_features)

    def get_feature_rule_frequency(self) -> Dict[str, int]:
        """Count how many rules each feature appears in."""
        freq = defaultdict(int)
        for rule in self.rules:
            for feat in rule.feature_set():
                freq[feat] += 1
        return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))

    def get_report(self, all_feature_names: List[str] = None) -> Dict:
        """Generate cross-analysis report."""
        return {
            "total_rules": len(self.rules),
            "unique_source_models": len(set(
                r.source_model.split("_tree")[0] for r in self.rules
            )),
            "consensus_rules": self.find_consensus_rules()[:10],
            "contradictions": self.find_contradictions()[:5],
            "feature_rule_frequency": self.get_feature_rule_frequency(),
            "blind_spots": self.find_feature_blind_spots(all_feature_names or []),
        }


# =============================================================================
# 5. CROSS-MODEL FEATURE AUGMENTER
# =============================================================================

class CrossModelAugmenter:
    """
    Generate meta-features from existing models for use in new model training.

    Meta-feature types:
      1. Rule activation features: Binary flags for symbolic rule matches
      2. Importance-weighted features: Scale features by universal importance
      3. Model prediction features: Cross-model probability outputs
    """

    def __init__(
        self,
        feature_map: Optional[UniversalFeatureMap] = None,
        rules: Optional[List[SymbolicRule]] = None,
    ):
        self.feature_map = feature_map
        self.rules = rules or []
        self._selected_rules: List[SymbolicRule] = []

    def select_rules(self, max_rules: int = 20, min_confidence: float = 0.6,
                     min_samples: int = 20) -> int:
        """
        Select the most informative rules for meta-feature generation.
        Returns number of rules selected.
        """
        candidates = [
            r for r in self.rules
            if r.confidence >= min_confidence and r.n_samples >= min_samples
        ]
        # Sort by confidence * samples (most reliable rules first)
        candidates.sort(key=lambda r: r.confidence * np.log1p(r.n_samples), reverse=True)
        self._selected_rules = candidates[:max_rules]
        return len(self._selected_rules)

    def generate_rule_activations(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate binary rule activation features.

        For each selected rule, creates a column that is 1 when all
        conditions in the rule are satisfied, 0 otherwise.

        Returns:
            (n_samples, n_rules) array and list of meta-feature names
        """
        if not self._selected_rules:
            self.select_rules()

        if not self._selected_rules:
            return np.empty((X.shape[0], 0)), []

        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        activations = []
        meta_names = []

        for rule_idx, rule in enumerate(self._selected_rules):
            activation = np.ones(X.shape[0], dtype=np.float32)
            valid = True

            for feat, op, thresh in rule.conditions:
                if feat not in name_to_idx:
                    valid = False
                    break
                col_idx = name_to_idx[feat]
                if op == "<=":
                    activation *= (X[:, col_idx] <= thresh).astype(np.float32)
                elif op == ">":
                    activation *= (X[:, col_idx] > thresh).astype(np.float32)

            if valid:
                activations.append(activation)
                direction = "up" if rule.prediction == 1 else "down"
                meta_names.append(f"rule_{rule_idx}_{direction}")

        if not activations:
            return np.empty((X.shape[0], 0)), []

        return np.column_stack(activations), meta_names

    def generate_importance_weighted(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Scale features by their universal importance score.

        Features with higher universality get amplified, features with
        low universality get dampened. This creates a "consensus view"
        of the data that emphasizes what multiple models agree matters.
        """
        if self.feature_map is None or not self.feature_map.profiles:
            return np.empty((X.shape[0], 0)), []

        weights = []
        valid_cols = []
        valid_names = []

        for i, name in enumerate(feature_names):
            if name in self.feature_map.profiles:
                p = self.feature_map.profiles[name]
                w = p.weighted_importance * p.universality_score()
                if w > 0:
                    weights.append(w)
                    valid_cols.append(i)
                    valid_names.append(f"iw_{name}")

        if not valid_cols:
            return np.empty((X.shape[0], 0)), []

        # Normalize weights to mean=1
        weights = np.array(weights)
        weights = weights / (weights.mean() + 1e-8)

        X_selected = X[:, valid_cols]
        X_weighted = X_selected * weights[np.newaxis, :]

        return X_weighted, valid_names

    def generate_all_meta_features(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate all meta-features and concatenate.

        Returns:
            Combined (n_samples, n_meta_features) array and names
        """
        all_features = []
        all_names = []

        # Rule activations
        rule_X, rule_names = self.generate_rule_activations(X, feature_names)
        if rule_X.shape[1] > 0:
            all_features.append(rule_X)
            all_names.extend(rule_names)

        # Importance-weighted features
        iw_X, iw_names = self.generate_importance_weighted(X, feature_names)
        if iw_X.shape[1] > 0:
            all_features.append(iw_X)
            all_names.extend(iw_names)

        if not all_features:
            return np.empty((X.shape[0], 0)), []

        return np.hstack(all_features), all_names

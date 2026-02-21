"""
GIGA TRADER - Model Registry v2: Registry Class
=================================================
ModelRegistryV2 class and get_registry() singleton.

Storage is fully backed by SQLite via RegistryDB.
"""

import json
import logging
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from src.phase_18_persistence.registry_enums import (
    TargetType,
    ModelStatus,
)
from src.phase_18_persistence.registry_configs import (
    ModelEntry,
    ModelMetrics,
    ModelArtifacts,
)

project_root = Path(__file__).parent.parent.parent
logger = logging.getLogger("MODEL_REGISTRY_V2")


# =============================================================================
# MODEL REGISTRY
# =============================================================================

class ModelRegistryV2:
    """
    Comprehensive model registry for tracking full pipeline configurations.

    Features:
    - Store complete pipeline configurations
    - Query by any configuration parameter
    - Track model lifecycle (training -> validation -> production)
    - Compare models by configuration
    - Export/import for reproducibility
    """

    def __init__(
        self,
        db,
        models_dir: Path = None,
        **_kwargs,
    ):
        self.models_dir = models_dir or (project_root / "models" / "artifacts")
        self._db = db

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, ModelEntry] = {}

        self._load_registry()

    def _load_registry(self):
        """Load registry from SQLite."""
        try:
            entries = self._db.query_model_entries()
            self.models = {
                d["model_id"]: ModelEntry.from_dict(d)
                for d in entries
            }
            logger.info(f"Loaded {len(self.models)} models from SQLite")
        except Exception as e:
            logger.warning(f"Could not load model registry: {e}")
            self.models = {}

    def register(self, entry: ModelEntry) -> str:
        """
        Register a new model entry.

        Auto-generates a unique model_id if one isn't provided.

        Returns:
            model_id
        """
        if not entry.model_id:
            entry.model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        entry.updated_at = datetime.now().isoformat()
        self.models[entry.model_id] = entry
        self._db.add_model_entry(entry.to_dict())

        logger.info(f"Registered model: {entry.model_id}")
        return entry.model_id

    def update(self, model_id: str, **kwargs):
        """Update model entry fields."""
        if model_id not in self.models:
            raise KeyError(f"Model not found: {model_id}")

        entry = self.models[model_id]

        for key, value in kwargs.items():
            if hasattr(entry, key):
                setattr(entry, key, value)
            elif hasattr(entry.metrics, key):
                setattr(entry.metrics, key, value)

        entry.updated_at = datetime.now().isoformat()
        self._db.update_model_entry(model_id, entry.to_dict())

    def update_metrics(self, model_id: str, metrics: ModelMetrics):
        """Update model metrics."""
        if model_id not in self.models:
            raise KeyError(f"Model not found: {model_id}")

        self.models[model_id].metrics = metrics
        self.models[model_id].updated_at = datetime.now().isoformat()
        self._db.update_model_entry(model_id, self.models[model_id].to_dict())

    def update_artifacts(self, model_id: str, artifacts: ModelArtifacts):
        """Update model artifacts."""
        if model_id not in self.models:
            raise KeyError(f"Model not found: {model_id}")

        self.models[model_id].artifacts = artifacts
        self.models[model_id].updated_at = datetime.now().isoformat()
        self._db.update_model_entry(model_id, self.models[model_id].to_dict())

    def set_status(self, model_id: str, status: ModelStatus):
        """Set model status."""
        if model_id not in self.models:
            raise KeyError(f"Model not found: {model_id}")

        self.models[model_id].status = status.value
        self.models[model_id].updated_at = datetime.now().isoformat()
        self._db.update_model_entry(model_id, self.models[model_id].to_dict())

    def get(self, model_id: str) -> Optional[ModelEntry]:
        """Get model by ID."""
        return self.models.get(model_id)

    def delete(self, model_id: str, delete_artifacts: bool = False):
        """Delete model from registry."""
        if model_id not in self.models:
            return

        if delete_artifacts:
            entry = self.models[model_id]
            for path_attr in ['model_path', 'scaler_path', 'dim_reducer_path']:
                path = getattr(entry.artifacts, path_attr, None)
                if path and Path(path).exists():
                    Path(path).unlink()

        del self.models[model_id]
        self._db.delete_model_entry(model_id)

        logger.info(f"Deleted model: {model_id}")

    def query(
        self,
        target_type: Optional[str] = None,
        cascade_type: Optional[str] = None,
        model_type: Optional[str] = None,
        dim_reduction: Optional[str] = None,
        status: Optional[str] = None,
        min_cv_auc: Optional[float] = None,
        min_test_auc: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ModelEntry]:
        """
        Query models by criteria.

        Returns:
            List of matching ModelEntry objects
        """
        results = []

        for entry in self.models.values():
            # Filter by target type
            if target_type and entry.target_type != target_type:
                continue

            # Filter by cascade type
            if cascade_type and entry.cascade_config.cascade_type != cascade_type:
                continue

            # Filter by model type
            if model_type and entry.model_config.model_type != model_type:
                continue

            # Filter by dim reduction
            if dim_reduction and entry.dim_reduction_config.method != dim_reduction:
                continue

            # Filter by status
            if status and entry.status != status:
                continue

            # Filter by CV AUC
            if min_cv_auc and entry.metrics.cv_auc < min_cv_auc:
                continue

            # Filter by test AUC
            if min_test_auc and entry.metrics.test_auc < min_test_auc:
                continue

            # Filter by tags
            if tags and not all(tag in entry.tags for tag in tags):
                continue

            results.append(entry)

        return results

    def get_best(
        self,
        target_type: str,
        metric: str = "cv_auc",
        cascade_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Optional[ModelEntry]:
        """Get best model by metric."""
        candidates = self.query(
            target_type=target_type,
            cascade_type=cascade_type,
            status=status,
        )

        if not candidates:
            return None

        return max(candidates, key=lambda e: getattr(e.metrics, metric, 0))

    def get_production_models(
        self,
        target_type: Optional[str] = None
    ) -> List[ModelEntry]:
        """Get all production models."""
        return self.query(
            target_type=target_type,
            status=ModelStatus.PRODUCTION.value
        )

    def promote_to_production(self, model_id: str):
        """Promote model to production status."""
        if model_id not in self.models:
            raise KeyError(f"Model not found: {model_id}")

        entry = self.models[model_id]

        # Demote current production models of same type
        for other in self.query(
            target_type=entry.target_type,
            cascade_type=entry.cascade_config.cascade_type,
            status=ModelStatus.PRODUCTION.value
        ):
            if other.model_id != model_id:
                self.set_status(other.model_id, ModelStatus.DEPRECATED)

        self.set_status(model_id, ModelStatus.PRODUCTION)
        logger.info(f"Promoted to production: {model_id}")

    def compare(
        self,
        model_ids: List[str],
        metrics: List[str] = None
    ) -> pd.DataFrame:
        """
        Compare models side by side.

        Returns:
            DataFrame with model comparisons
        """
        metrics = metrics or ['cv_auc', 'test_auc', 'win_rate', 'sharpe_ratio']

        rows = []
        for model_id in model_ids:
            entry = self.models.get(model_id)
            if not entry:
                continue

            row = {
                'model_id': model_id,
                'target': entry.target_type,
                'cascade': entry.cascade_config.cascade_type,
                'model': entry.model_config.model_type,
                'dim_red': entry.dim_reduction_config.method,
                'n_features': entry.artifacts.n_output_features,
            }

            for metric in metrics:
                row[metric] = getattr(entry.metrics, metric, None)

            rows.append(row)

        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Get registry summary."""
        lines = [
            "=" * 60,
            "MODEL REGISTRY V2 SUMMARY",
            "=" * 60,
            f"Total models: {len(self.models)}",
            "",
        ]

        # Count by status
        status_counts = {}
        for entry in self.models.values():
            status_counts[entry.status] = status_counts.get(entry.status, 0) + 1

        lines.append("By Status:")
        for status, count in sorted(status_counts.items()):
            lines.append(f"  {status}: {count}")

        # Count by target type
        target_counts = {}
        for entry in self.models.values():
            target_counts[entry.target_type] = target_counts.get(entry.target_type, 0) + 1

        lines.append("\nBy Target Type:")
        for target, count in sorted(target_counts.items()):
            lines.append(f"  {target}: {count}")

        # Count by cascade type
        cascade_counts = {}
        for entry in self.models.values():
            cascade_counts[entry.cascade_config.cascade_type] = \
                cascade_counts.get(entry.cascade_config.cascade_type, 0) + 1

        lines.append("\nBy Cascade Type:")
        for cascade, count in sorted(cascade_counts.items()):
            lines.append(f"  {cascade}: {count}")

        # Best models
        lines.append("\nBest Models (by CV AUC):")
        for target in TargetType:
            best = self.get_best(target.value)
            if best:
                lines.append(f"  {target.value}: {best.model_id} (AUC={best.metrics.cv_auc:.4f})")

        # Production models
        production = self.get_production_models()
        lines.append(f"\nProduction Models: {len(production)}")
        for entry in production:
            lines.append(f"  - {entry.model_id}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def export(self, filepath: Path):
        """Export registry to file."""
        data = {
            'version': '2.0',
            'exported_at': datetime.now().isoformat(),
            'models': {
                model_id: entry.to_dict()
                for model_id, entry in self.models.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported registry to {filepath}")

    def import_registry(self, filepath: Path, merge: bool = True):
        """Import registry from a JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        imported = {
            model_id: ModelEntry.from_dict(entry)
            for model_id, entry in data.get('models', {}).items()
        }

        if merge:
            self.models.update(imported)
        else:
            self.models = imported

        # Sync all entries to SQLite
        for model_id, entry in imported.items():
            self._db.add_model_entry(entry.to_dict())

        logger.info(f"Imported {len(imported)} models")

    def reset(self):
        """Reset registry (delete all entries from SQLite)."""
        for model_id in list(self.models.keys()):
            self._db.delete_model_entry(model_id)

        self.models = {}
        logger.info("Registry reset")


def get_registry(db=None) -> ModelRegistryV2:
    """Get the global model registry instance."""
    return ModelRegistryV2(db=db)

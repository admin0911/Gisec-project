"""Detector-facing feature result types."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _metadata_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Metadata value {type(value).__name__} is not JSON serializable")


@dataclass
class FeatureBundle:
    """Aligned representations and evaluation-only labels for a dataset."""

    features: np.ndarray
    scaled_features: np.ndarray
    reduced_features: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    modality: str
    encoder: str
    dataset_name: str = ""
    visual_features: Optional[np.ndarray] = None
    metadata: Optional[dict[str, Any]] = None
    original_labels: Optional[np.ndarray] = None
    is_poisoned: Optional[np.ndarray] = None
    poison_type: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        lengths = {
            len(self.features),
            len(self.scaled_features),
            len(self.reduced_features),
            len(self.labels),
            len(self.sample_ids),
        }
        if len(lengths) != 1:
            raise ValueError("features, labels, and sample_ids must have equal lengths")
        for name in ("original_labels", "is_poisoned", "poison_type"):
            values = getattr(self, name)
            if values is not None and len(values) != len(self.features):
                raise ValueError(f"{name} must align with feature rows")
        for name in ("features", "scaled_features", "reduced_features"):
            values = getattr(self, name)
            if values.ndim != 2 or not np.isfinite(values).all():
                raise ValueError(f"{name} must be a finite two-dimensional array")

    @property
    def original_feature_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def reduced_feature_dim(self) -> int:
        return int(self.reduced_features.shape[1])

    def save(self, path: str | Path) -> None:
        """Save all detector inputs and metadata to a compressed NumPy archive."""
        metadata = dict(self.metadata or {})
        metadata.update({
            "modality": self.modality,
            "encoder": self.encoder,
            "dataset_name": self.dataset_name,
        })
        np.savez_compressed(
            path,
            features=self.features,
            scaled_features=self.scaled_features,
            reduced_features=self.reduced_features,
            labels=self.labels,
            sample_ids=self.sample_ids,
            visual_features=self.visual_features
            if self.visual_features is not None else np.empty((0, 0)),
            original_labels=self.original_labels
            if self.original_labels is not None else np.empty((0,)),
            is_poisoned=self.is_poisoned
            if self.is_poisoned is not None else np.empty((0,), dtype=bool),
            poison_type=self.poison_type
            if self.poison_type is not None else np.empty((0,), dtype="<U1"),
            format_version=np.array("1.1"),
            metadata_json=np.array(json.dumps(metadata, sort_keys=True, default=_metadata_json)),
            metadata=np.array(metadata, dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureBundle":
        """Load a bundle produced by :meth:`save`."""
        with np.load(path, allow_pickle=True) as archive:
            metadata = archive["metadata"].item()
            visual = archive["visual_features"]
            original = archive["original_labels"]
            poisoned = archive["is_poisoned"]
            poison_type = archive["poison_type"]
            return cls(
                features=archive["features"],
                scaled_features=archive["scaled_features"],
                reduced_features=archive["reduced_features"],
                labels=archive["labels"],
                sample_ids=archive["sample_ids"],
                modality=metadata.pop("modality"),
                encoder=metadata.pop("encoder"),
                dataset_name=metadata.pop("dataset_name", ""),
                visual_features=None if visual.size == 0 else visual,
                original_labels=None if original.size == 0 else original,
                is_poisoned=None if poisoned.size == 0 else poisoned,
                poison_type=None if poison_type.size == 0 else poison_type,
                metadata=metadata,
            )

"""Detector-facing feature result types."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


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
            metadata=np.array(metadata, dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureBundle":
        """Load a bundle produced by :meth:`save`."""
        with np.load(path, allow_pickle=True) as archive:
            metadata = archive["metadata"].item()
            visual = archive["visual_features"]
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
                metadata=metadata,
            )


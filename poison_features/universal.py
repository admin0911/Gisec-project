"""Public orchestration API for supported datasets."""

from typing import Any

import numpy as np
from .bundle import FeatureBundle
from .image import ResNet18ImageEncoder
from .preprocessing import prepare_representations, prepare_visual_features


class UniversalFeatureExtractor:
    """Create detector-ready bundles while keeping labels separate."""

    def __init__(self, device: str | None = None, batch_size: int = 128):
        self.device = device
        self.batch_size = batch_size

    def extract_images(
        self,
        dataset: Any,
        *,
        labels: Any = None,
        sample_ids: Any = None,
        dataset_name: str = "",
        encoder: str = "resnet18",
        visual: bool = True,
    ) -> FeatureBundle:
        if encoder != "resnet18":
            raise ValueError(f"Unsupported image encoder: {encoder}")
        image_encoder = ResNet18ImageEncoder(device=self.device)
        features = image_encoder.extract(
            dataset,
            batch_size=self.batch_size,
            progress=lambda done, total: None,
        )
        labels_array = (
            np.asarray(labels) if labels is not None
            else np.asarray([item[1] for item in dataset])
        )
        ids = (
            np.asarray(sample_ids) if sample_ids is not None
            else np.arange(len(features), dtype=np.int64)
        )
        if len(labels_array) != len(features) or len(ids) != len(features):
            raise ValueError("labels and sample_ids must align with dataset rows")
        scaled, reduced = prepare_representations(features)
        visual_features = prepare_visual_features(features) if visual else None
        attack_metadata = getattr(dataset, "metadata", None)
        return FeatureBundle(
            features=features,
            scaled_features=scaled,
            reduced_features=reduced,
            labels=labels_array,
            sample_ids=ids,
            modality="image",
            encoder=encoder,
            dataset_name=dataset_name,
            visual_features=visual_features,
            original_labels=None if attack_metadata is None else attack_metadata.original_labels,
            is_poisoned=None if attack_metadata is None else attack_metadata.is_poisoned,
            poison_type=None if attack_metadata is None else attack_metadata.poison_type,
            metadata={"device": str(image_encoder.device)},
        )

"""Sentence-transformer text embeddings."""

from typing import Any

import numpy as np

from .preprocessing import prepare_representations, prepare_visual_features
from .bundle import FeatureBundle


class MiniLMTextEncoder:
    """Encode text with sentence-transformers/all-MiniLM-L6-v2."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Text extraction requires sentence-transformers; install requirements.txt"
            ) from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def extract(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        values = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=False,
        )
        return np.asarray(values, dtype=np.float32)


def extract_text(
    texts: list[str],
    *,
    labels: Any = None,
    sample_ids: Any = None,
    dataset_name: str = "text",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    original_labels: Any = None,
    is_poisoned: Any = None,
    poison_type: Any = None,
) -> FeatureBundle:
    features = MiniLMTextEncoder(model_name).extract(texts, batch_size)
    scaled, reduced = prepare_representations(features)
    visual = prepare_visual_features(features)
    return FeatureBundle(
        features=features,
        scaled_features=scaled,
        reduced_features=reduced,
        labels=np.full(len(texts), -1) if labels is None else np.asarray(labels),
        sample_ids=np.arange(len(texts)) if sample_ids is None else np.asarray(sample_ids),
        modality="text",
        encoder=model_name,
        dataset_name=dataset_name,
        visual_features=visual,
        original_labels=None if original_labels is None else np.asarray(original_labels),
        is_poisoned=None if is_poisoned is None else np.asarray(is_poisoned),
        poison_type=None if poison_type is None else np.asarray(poison_type),
    )

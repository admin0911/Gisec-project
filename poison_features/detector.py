"""Small connector contract for detector implementations."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .bundle import FeatureBundle


@dataclass(frozen=True)
class DetectorInput:
    """The only data a detector needs, with labels optional by design."""

    X: np.ndarray
    sample_ids: np.ndarray
    y: np.ndarray | None = None


class Detector(Protocol):
    def score(self, inputs: DetectorInput) -> np.ndarray:
        """Return one anomaly score per sample; higher means more suspicious."""


def detector_input(
    bundle: FeatureBundle,
    *,
    representation: str = "scaled",
    label_aware: bool = False,
) -> DetectorInput:
    """Adapt a FeatureBundle to an unsupervised or label-aware detector."""
    representations = {
        "raw": bundle.features,
        "scaled": bundle.scaled_features,
        "reduced": bundle.reduced_features,
    }
    if representation not in representations:
        raise ValueError("representation must be 'raw', 'scaled', or 'reduced'")
    return DetectorInput(
        X=representations[representation],
        sample_ids=bundle.sample_ids,
        y=bundle.labels if label_aware else None,
    )


# Leila: public pixel adapter produces the same truth-free DetectorInput as feature extraction.
def image_detector_input(bundle, *, label_aware=False):
    """Flatten an ImageInputBundle without rescaling or exposing attack metadata."""
    import numpy as np
    ids = np.asarray(bundle.sample_ids)
    labels = np.asarray(bundle.labels)
    if (ids.ndim != 1 or len(ids) != len(bundle.images) or not len(ids)
            or ids.dtype.kind not in 'iuUS' or len(np.unique(ids)) != len(ids)
            or labels.shape != ids.shape):
        raise ValueError('Image connector requires unique aligned sample IDs and labels.')
    return DetectorInput(X=bundle.images.reshape(len(ids),-1),sample_ids=ids.copy(),
                         y=labels.copy() if label_aware else None)

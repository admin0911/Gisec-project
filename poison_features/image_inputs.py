"""Aligned pixel inputs for image-based detectors."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ImageInputBundle:
    """Images and row metadata aligned by position.

    ``images[i]``, ``labels[i]``, and ``sample_ids[i]`` always describe the
    same row. Images are float32 channel-first arrays in the original size and
    pixel range [0, 1]. These pixels are separate from FeatureBundle vectors.
    """

    images: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray

    def __post_init__(self) -> None:
        if self.images.ndim != 4 or not np.isfinite(self.images).all():
            raise ValueError("images must be a finite NCHW array")
        if self.images.min() < 0 or self.images.max() > 1:
            raise ValueError("image pixels must be in [0, 1]")
        if not (len(self.images) == len(self.labels) == len(self.sample_ids)):
            raise ValueError("images, labels, and sample_ids must have equal lengths")

    def save(self, path: str) -> None:
        """Save aligned pixels and public row metadata."""
        np.savez_compressed(
            path,
            images=self.images,
            labels=self.labels,
            sample_ids=self.sample_ids,
        )

    @classmethod
    def load(cls, path: str) -> "ImageInputBundle":
        """Load an image bundle produced by :meth:`save`."""
        with np.load(path, allow_pickle=False) as archive:
            return cls(
                images=archive["images"],
                labels=archive["labels"],
                sample_ids=archive["sample_ids"],
            )


def _to_chw_float(image: Any) -> np.ndarray:
    from PIL import Image

    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return np.transpose(image, (2, 0, 1))
    try:
        import torch
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
    except ImportError:
        pass
    values = np.asarray(image)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3:
        raise ValueError("each image must be HWC, HW, or CHW")
    if values.shape[0] in (1, 3):
        channels_first = values
    elif values.shape[-1] in (1, 3):
        channels_first = np.transpose(values, (2, 0, 1))
    else:
        raise ValueError("could not determine image channel dimension")
    channels_first = channels_first.astype(np.float32, copy=False)
    if np.issubdtype(values.dtype, np.integer):
        channels_first /= 255.0
    return channels_first


def load_image_inputs(
    dataset: Any,
    *,
    sample_ids: Any = None,
    limit: int | None = None,
) -> ImageInputBundle:
    """Load original or post-attack pixels without creating feature vectors.

    A ``PoisonedImageDataset`` returns its attacked image from ``__getitem__``,
    so this function automatically returns the post-attack pixels.
    """
    count = len(dataset) if limit is None else min(limit, len(dataset))
    if count < 1:
        raise ValueError("dataset must contain at least one image")
    ids = np.arange(count, dtype=np.int64) if sample_ids is None else np.asarray(sample_ids)
    if len(ids) != count:
        raise ValueError("sample_ids must align with the requested rows")
    images: list[np.ndarray] = []
    labels: list[Any] = []
    for index in range(count):
        item = dataset[index]
        image, label = item[0], item[1]
        images.append(_to_chw_float(image))
        labels.append(label)
    try:
        stacked = np.stack(images).astype(np.float32)
    except ValueError as exc:
        raise ValueError("all images must have the same dimensions") from exc
    return ImageInputBundle(stacked, np.asarray(labels), ids)

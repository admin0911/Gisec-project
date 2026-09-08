"""Evaluation-only poisoning wrappers for image datasets."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PoisonMetadata:
    """Ground truth kept separate from detector-facing features."""

    original_labels: np.ndarray
    current_labels: np.ndarray
    is_poisoned: np.ndarray
    poison_type: np.ndarray


class PoisonedImageDataset:
    """Wrap a torchvision-style dataset and apply one poisoning strategy."""

    def __init__(
        self,
        dataset: Any,
        *,
        attack: str,
        poison_rate: float = 0.05,
        target_label: int = 0,
        seed: int = 0,
    ) -> None:
        if not 0 <= poison_rate <= 1:
            raise ValueError("poison_rate must be between 0 and 1")
        if attack not in {"label_flip", "backdoor"}:
            raise ValueError("attack must be 'label_flip' or 'backdoor'")
        self.dataset = dataset
        self.attack = attack
        self.target_label = target_label
        labels = np.asarray([int(dataset[i][1]) for i in range(len(dataset))])
        rng = np.random.default_rng(seed)
        count = int(round(len(labels) * poison_rate))
        poisoned_indices = np.sort(rng.choice(len(labels), size=count, replace=False))
        is_poisoned = np.zeros(len(labels), dtype=bool)
        is_poisoned[poisoned_indices] = True
        current = labels.copy()
        if attack == "label_flip":
            current[is_poisoned] = (current[is_poisoned] + 1) % 10
        else:
            current[is_poisoned] = target_label
        self.metadata = PoisonMetadata(
            original_labels=labels,
            current_labels=current,
            is_poisoned=is_poisoned,
            poison_type=np.where(is_poisoned, attack, "clean"),
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        image, _ = self.dataset[index]
        if self.attack == "backdoor" and self.metadata.is_poisoned[index]:
            image = image.clone()
            image[..., -3:, -3:] = 1.0
        return image, int(self.metadata.current_labels[index])

    def take(self, count: int) -> "PoisonedImageDataset":
        """Return a metadata-preserving prefix for smoke tests and UI previews."""
        if count < 1 or count > len(self):
            raise ValueError("count must be within the dataset length")
        clone = object.__new__(PoisonedImageDataset)
        clone.dataset = self.dataset
        clone.attack = self.attack
        clone.target_label = self.target_label
        clone.metadata = PoisonMetadata(
            original_labels=self.metadata.original_labels[:count],
            current_labels=self.metadata.current_labels[:count],
            is_poisoned=self.metadata.is_poisoned[:count],
            poison_type=self.metadata.poison_type[:count],
        )
        clone.dataset = _DatasetPrefix(self.dataset, count)
        return clone


class _DatasetPrefix:
    def __init__(self, dataset: Any, count: int) -> None:
        self.dataset, self.count = dataset, count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]


def poison_dataset(
    dataset: Any,
    attack: str,
    *,
    poison_rate: float = 0.05,
    target_label: int = 0,
    seed: int = 0,
) -> PoisonedImageDataset:
    return PoisonedImageDataset(
        dataset,
        attack=attack,
        poison_rate=poison_rate,
        target_label=target_label,
        seed=seed,
    )


def poison_texts(
    texts: list[str],
    labels: Any,
    *,
    attack: str,
    poison_rate: float = 0.05,
    target_label: int = 1,
    trigger: str = "excellent cinematic signal",
    seed: int = 0,
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    """Apply label-flip or phrase-trigger poisoning to text rows."""
    if attack not in {"label_flip", "backdoor"}:
        raise ValueError("attack must be 'label_flip' or 'backdoor'")
    if not 0 <= poison_rate <= 1:
        raise ValueError("poison_rate must be between 0 and 1")
    original = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    count = int(round(len(original) * poison_rate))
    selected = np.sort(rng.choice(len(original), size=count, replace=False))
    poisoned = np.zeros(len(original), dtype=bool)
    poisoned[selected] = True
    current = original.copy()
    result_texts = list(texts)
    if attack == "label_flip":
        current[poisoned] = 1 - current[poisoned]
    else:
        current[poisoned] = target_label
        result_texts = [
            f"{text} {trigger}" if poisoned[index] else text
            for index, text in enumerate(result_texts)
        ]
    return result_texts, current, {
        "original_labels": original,
        "is_poisoned": poisoned,
        "poison_type": np.where(poisoned, attack, "clean"),
    }

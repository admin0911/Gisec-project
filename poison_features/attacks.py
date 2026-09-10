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
        source_label: int | None = None,
        target_label: int = 0,
        blend_alpha: float = 0.10,
        poison_count: int | None = None,
        seed: int = 0,
    ) -> None:
        if not 0 <= poison_rate <= 1:
            raise ValueError("poison_rate must be between 0 and 1")
        if attack not in {"label_flip", "targeted_label_flip", "backdoor", "blended_injection"}:
            raise ValueError("unsupported image attack")
        if attack == "targeted_label_flip":
            if source_label is None:
                raise ValueError("source_label is required for targeted_label_flip")
            if source_label == target_label:
                raise ValueError("source_label and target_label must differ")
        if not 0 < blend_alpha <= 1:
            raise ValueError("blend_alpha must be greater than 0 and at most 1")
        if poison_count is not None and not 0 <= poison_count <= len(dataset):
            raise ValueError("poison_count must fit within the dataset")
        self.dataset = dataset
        self.attack = attack
        self.source_label = source_label
        self.target_label = target_label
        self.blend_alpha = blend_alpha
        self._noise_seed = seed + 104729
        labels = np.asarray([int(dataset[i][1]) for i in range(len(dataset))])
        rng = np.random.default_rng(seed)
        count = int(round(len(labels) * poison_rate)) if poison_count is None else poison_count
        candidates = np.arange(len(labels))
        if attack in {"targeted_label_flip", "blended_injection"}:
            if attack == "targeted_label_flip":
                candidates = candidates[labels == source_label]
            else:
                candidates = candidates[labels != target_label]
            if count > len(candidates):
                raise ValueError("poison_count exceeds eligible source samples")
        poisoned_indices = np.sort(rng.choice(candidates, size=count, replace=False))
        is_poisoned = np.zeros(len(labels), dtype=bool)
        is_poisoned[poisoned_indices] = True
        current = labels.copy()
        if attack == "label_flip":
            current[is_poisoned] = (current[is_poisoned] + 1) % 10
        elif attack == "targeted_label_flip":
            current[is_poisoned] = target_label
        elif attack in {"backdoor", "blended_injection"}:
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
        elif self.attack == "blended_injection" and self.metadata.is_poisoned[index]:
            image = image.clone().float()
            generator = np.random.default_rng(self._noise_seed)
            noise = np.clip(
                generator.normal(0.5, 0.2, size=tuple(image.shape)),
                0.0,
                1.0,
            ).astype(np.float32)
            import torch
            trigger = torch.from_numpy(noise).to(image.device)
            image = ((1 - self.blend_alpha) * image + self.blend_alpha * trigger).clamp(0.0, 1.0)
        return image, int(self.metadata.current_labels[index])

    def take(self, count: int) -> "PoisonedImageDataset":
        """Return a metadata-preserving prefix for smoke tests and UI previews."""
        if count < 1 or count > len(self):
            raise ValueError("count must be within the dataset length")
        clone = object.__new__(PoisonedImageDataset)
        clone.dataset = self.dataset
        clone.attack = self.attack
        clone.source_label = self.source_label
        clone.target_label = self.target_label
        clone.blend_alpha = self.blend_alpha
        clone._noise_seed = self._noise_seed
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
    source_label: int | None = None,
    target_label: int = 0,
    blend_alpha: float = 0.10,
    poison_count: int | None = None,
    seed: int = 0,
) -> PoisonedImageDataset:
    return PoisonedImageDataset(
        dataset,
        attack=attack,
        poison_rate=poison_rate,
        source_label=source_label,
        target_label=target_label,
        blend_alpha=blend_alpha,
        poison_count=poison_count,
        seed=seed,
    )


def poison_texts(
    texts: list[str],
    labels: Any,
    *,
    attack: str,
    poison_rate: float = 0.05,
    source_label: int | None = None,
    target_label: int = 1,
    trigger: str = "excellent cinematic signal",
    # Leila: allow a visible prefix before text encoders truncate long reviews.
    trigger_position: str = "end",
    # Leila: opt-in non-target sampling preserves existing shared callers.
    selection_policy: str = "all",
    seed: int = 0,
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    """Apply label-flip or phrase-trigger poisoning to text rows."""
    if attack not in {"label_flip", "targeted_label_flip", "backdoor"}:
        raise ValueError("unsupported text attack")
    if attack == "targeted_label_flip":
        if source_label is None:
            raise ValueError("source_label is required for targeted_label_flip")
        if source_label == target_label:
            raise ValueError("source_label and target_label must differ")
    if not 0 <= poison_rate <= 1:
        raise ValueError("poison_rate must be between 0 and 1")
    # Leila: validate text alignment and phrase settings before applying an attack.
    if len(texts) != len(labels) or not all(isinstance(t, str) for t in texts):
        raise ValueError("Expected aligned text rows and labels")
    if trigger_position not in {"start", "end"} or not isinstance(trigger, str) or not trigger.strip():
        raise ValueError("Expected a nonempty trigger and start/end position")
    original = np.asarray(labels, dtype=np.int64)
    if original.ndim != 1 or not np.isin(original, [0, 1]).all() or target_label not in (0, 1):
        raise ValueError("Text attack requires binary sentiment labels")
    rng = np.random.default_rng(seed)
    count = int(round(len(original) * poison_rate))
       # Leila: preserve targeted flips and non-target backdoor sampling.
    if selection_policy not in {"all", "non_target"}:
        raise ValueError("selection_policy must be all or non_target")
    if selection_policy == "non_target" and attack != "backdoor":
        raise ValueError("non_target selection is only supported for backdoor attacks")

    if attack == "targeted_label_flip":
        eligible = np.flatnonzero(original == source_label)
    elif selection_policy == "non_target":
        eligible = np.flatnonzero(original != target_label)
    else:
        eligible = np.arange(len(original))

    if count > len(eligible):
        raise ValueError("Not enough eligible reviews for the requested poison rate")

    selected = np.sort(rng.choice(eligible, size=count, replace=False))
    poisoned = np.zeros(len(original), dtype=bool)
    poisoned[selected] = True
    current = original.copy()
    result_texts = list(texts)
    if attack == "label_flip":
        current[poisoned] = 1 - current[poisoned]
    elif attack == "targeted_label_flip":
        current[poisoned] = target_label
    else:
        current[poisoned] = target_label
        result_texts = [
            (f"{trigger} {text}" if trigger_position == "start" else f"{text} {trigger}")
            if poisoned[index] else text
            for index, text in enumerate(result_texts)
        ]
    return result_texts, current, {
        "original_labels": original,
        "is_poisoned": poisoned,
        "poison_type": np.where(poisoned, attack, "clean"),
    }

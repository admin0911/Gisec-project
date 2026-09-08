"""Downloadable dataset adapters for the standalone feature module."""

from pathlib import Path
from typing import Any


def load_image_dataset(
    name: str,
    *,
    root: str | Path = "data",
    train: bool = True,
    download: bool = True,
) -> Any:
    """Load CIFAR-10 or MNIST with stable row ordering and labels."""
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError("Install the project requirements before loading datasets") from exc

    normalized = name.lower().replace("-", "")
    root_path = str(Path(root))
    if normalized == "cifar10":
        return datasets.CIFAR10(
            root=root_path,
            train=train,
            download=download,
            transform=transforms.ToTensor(),
        )
    if normalized == "mnist":
        return datasets.MNIST(
            root=root_path,
            train=train,
            download=download,
            transform=transforms.ToTensor(),
        )
    raise ValueError("Unsupported image dataset; choose 'cifar10' or 'mnist'")


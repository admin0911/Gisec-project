"""Downloadable dataset adapters for the standalone feature module."""

from pathlib import Path
from typing import Any
import csv
import json

from PIL import Image


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


def load_imdb_dataset(
    *,
    split: str = "train",
    cache_dir: str | Path = "data",
) -> Any:
    """Load IMDB reviews with deterministic row ordering."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("IMDB extraction requires the datasets package") from exc
    return load_dataset("stanfordnlp/imdb", split=split, cache_dir=str(cache_dir))


def load_image_folder(root: str | Path) -> Any:
    """Load an image folder where each class is a subdirectory.

    Example: root/class_a/image1.jpg and root/class_b/image2.png.
    """
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError("Install torchvision before loading image folders") from exc
    path = Path(root)
    if not path.is_dir():
        raise FileNotFoundError(f"Image dataset folder does not exist: {path}")
    return datasets.ImageFolder(str(path), transform=transforms.ToTensor())


def load_text_table(
    path: str | Path,
    *,
    text_column: str = "text",
    label_column: str | None = "label",
) -> tuple[list[str], list[Any], list[str]]:
    """Load custom text rows from CSV or JSON/JSONL.

    Returns texts, labels, and stable file-based row IDs. Labels default to -1
    when the source has no label column.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Text dataset file does not exist: {file_path}")
    if file_path.suffix.lower() == ".csv":
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif file_path.suffix.lower() in {".json", ".jsonl"}:
        with file_path.open("r", encoding="utf-8") as handle:
            rows = (
                [json.loads(line) for line in handle if line.strip()]
                if file_path.suffix.lower() == ".jsonl"
                else json.load(handle)
            )
    else:
        raise ValueError("Custom text datasets must be .csv, .json, or .jsonl")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Custom text dataset must contain a non-empty list of rows")
    if any(text_column not in row for row in rows):
        raise ValueError(f"Every row must contain the text column '{text_column}'")
    texts = [str(row[text_column]) for row in rows]
    labels = [-1 if label_column is None else row.get(label_column, -1) for row in rows]
    ids = [f"{file_path.name}:{index}" for index in range(len(rows))]
    return texts, labels, ids

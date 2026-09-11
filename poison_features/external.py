"""Dataset-neutral adapters for feature matrices produced outside the extractor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .bundle import FeatureBundle
from .preprocessing import prepare_representations


FEATURE_KEYS = ("features", "X", "embeddings")
LABEL_KEYS = ("labels", "y", "targets")
SAMPLE_ID_KEYS = ("sample_ids", "ids", "row_ids")


def _one_dimensional(values: Any, name: str, length: int) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (length,):
        raise ValueError(f"{name} must contain one value per feature row")
    return array


def _validate_ids(values: Any, length: int) -> np.ndarray:
    ids = _one_dimensional(values, "sample_ids", length)
    if ids.dtype.kind not in "iuUS" or len(np.unique(ids)) != length:
        raise ValueError("sample_ids must be unique integer or string values")
    if ids.dtype.kind == "U" and np.any(ids == ""):
        raise ValueError("sample_ids must not contain empty strings")
    if ids.dtype.kind == "S" and np.any(ids == b""):
        raise ValueError("sample_ids must not contain empty strings")
    return ids


def _validate_labels(values: Any, length: int) -> np.ndarray:
    labels = _one_dimensional(values, "labels", length)
    if labels.dtype.kind not in "biufUS":
        raise ValueError("labels must be numeric or string class values")
    if labels.dtype.kind in "biuf" and not np.isfinite(labels).all():
        raise ValueError("labels must be finite")
    if labels.dtype.kind == "f":
        if not np.equal(labels, np.floor(labels)).all():
            raise ValueError("numeric class labels must be integer-valued")
        labels = labels.astype(np.int64)
    elif labels.dtype.kind == "b":
        labels = labels.astype(np.int64)
    return labels


def _feature_matrix(values: Any, name: str, *, rows: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if (array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1
            or array.dtype.kind not in "iuf" or not np.isfinite(array).all()):
        raise ValueError(f"{name} must be a finite numeric matrix with at least two rows")
    if rows is not None and len(array) != rows:
        raise ValueError(f"{name} must align with the feature rows")
    return array.astype(np.float32, copy=False)


def feature_bundle_from_arrays(
    features: Any,
    *,
    labels: Any | None = None,
    sample_ids: Any | None = None,
    modality: str = "unknown",
    encoder: str = "external",
    dataset_name: str = "external",
    scaled_features: Any | None = None,
    reduced_features: Any | None = None,
    visual_features: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> FeatureBundle:
    """Create the common detector bundle from any aligned feature matrix.

    The adapter does not inspect dataset names or feature dimensions. Labels stay
    separate from the feature matrix. Unknown labels are represented by ``-1``;
    label-aware detectors will reject them with a clear error.
    """
    raw = _feature_matrix(features, "features")
    count = len(raw)
    ids = _validate_ids(np.arange(count) if sample_ids is None else sample_ids, count)
    current_labels = _validate_labels(
        np.full(count, -1, dtype=np.int64) if labels is None else labels, count,
    )
    generated_scaled = generated_reduced = None
    if scaled_features is None or reduced_features is None:
        generated_scaled, generated_reduced = prepare_representations(raw)
    scaled = (generated_scaled if scaled_features is None
              else _feature_matrix(scaled_features, "scaled_features", rows=count))
    reduced = (generated_reduced if reduced_features is None
               else _feature_matrix(reduced_features, "reduced_features", rows=count))
    visual = (None if visual_features is None
              else _feature_matrix(visual_features, "visual_features", rows=count))
    if not isinstance(modality, str) or not modality.strip():
        raise ValueError("modality must be a non-empty string")
    if not isinstance(encoder, str) or not encoder.strip():
        raise ValueError("encoder must be a non-empty string")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("dataset_name must be a non-empty string")
    return FeatureBundle(
        features=raw,
        scaled_features=scaled,
        reduced_features=reduced,
        labels=current_labels,
        sample_ids=ids,
        modality=modality.strip(),
        encoder=encoder.strip(),
        dataset_name=dataset_name.strip(),
        visual_features=visual,
        metadata=dict(metadata or {}),
    )


def _choose_key(files: Iterable[str], explicit: str | None, aliases: tuple[str, ...], name: str,
                *, required: bool) -> str | None:
    available = set(files)
    if explicit is not None:
        if explicit not in available:
            raise ValueError(f"Requested {name} key '{explicit}' is missing")
        return explicit
    key = next((candidate for candidate in aliases if candidate in available), None)
    if key is None and required:
        raise ValueError(f"No {name} array found; expected one of {', '.join(aliases)}")
    return key


def load_external_feature_bundle(
    path: str | Path,
    *,
    features_key: str | None = None,
    labels_key: str | None = None,
    sample_ids_key: str | None = None,
    modality: str | None = None,
    encoder: str | None = None,
    dataset_name: str | None = None,
) -> FeatureBundle:
    """Safely load aligned arrays from an NPZ file without enabling pickle.

    Accepted feature aliases are ``features``, ``X`` and ``embeddings``;
    labels use ``labels``, ``y`` or ``targets``; IDs use ``sample_ids``, ``ids``
    or ``row_ids``. Files produced by current :class:`FeatureBundle` versions
    also carry JSON metadata. Evaluation-only poison truth is intentionally not
    imported by this adapter.
    """
    source = Path(path)
    if source.suffix.lower() != ".npz":
        raise ValueError("External feature input must be a .npz file")
    with np.load(source, allow_pickle=False) as archive:
        feature_name = _choose_key(archive.files, features_key, FEATURE_KEYS, "features", required=True)
        label_name = _choose_key(archive.files, labels_key, LABEL_KEYS, "labels", required=False)
        id_name = _choose_key(archive.files, sample_ids_key, SAMPLE_ID_KEYS, "sample IDs", required=False)
        stored: dict[str, Any] = {}
        if "metadata_json" in archive.files:
            try:
                parsed = json.loads(str(archive["metadata_json"].item()))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError("metadata_json must contain one valid JSON object") from exc
            if not isinstance(parsed, dict):
                raise ValueError("metadata_json must contain one valid JSON object")
            stored = parsed
        raw = archive[feature_name]
        labels = None if label_name is None else archive[label_name]
        ids = None if id_name is None else archive[id_name]
        scaled = archive["scaled_features"] if "scaled_features" in archive.files else None
        reduced = archive["reduced_features"] if "reduced_features" in archive.files else None
        visual = archive["visual_features"] if "visual_features" in archive.files else None
        if visual is not None and visual.size == 0:
            visual = None
    stored_modality = str(stored.pop("modality", "unknown"))
    stored_encoder = str(stored.pop("encoder", "external"))
    stored_dataset = str(stored.pop("dataset_name", source.stem))
    stored.update(source_format="external_npz", source_file=source.name)
    return feature_bundle_from_arrays(
        raw,
        labels=labels,
        sample_ids=ids,
        modality=modality or stored_modality,
        encoder=encoder or stored_encoder,
        dataset_name=dataset_name or stored_dataset,
        scaled_features=scaled,
        reduced_features=reduced,
        visual_features=visual,
        metadata=stored,
    )

"""Shared, validated output format for feature or image detectors."""
from copy import deepcopy

import numpy as np


def detector_result(name, version, result, settings, *, expected_sample_ids):
    """Wrap an analyze() result without changing its scores or flag policy.

    Extra detector-specific outputs are retained under evidence. Poison ground
    truth must never be supplied here. Scores need not share a common scale.
    """
    ids = np.asarray(result["sample_ids"])
    expected = np.asarray(expected_sample_ids)
    scores = np.asarray(result["scores"])
    flags = np.asarray(result["flags"])
    if ids.ndim != 1 or not np.array_equal(ids, expected):
        raise ValueError("Output sample IDs must preserve the input order")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Sample IDs must be unique")
    if scores.shape != ids.shape or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite, with one value per sample")
    if flags.shape != ids.shape or flags.dtype.kind != "b":
        raise ValueError("Flags must be boolean, with one value per sample")
    if not name or not version:
        raise ValueError("Detector name and version are required")
    return {
        "detector_name": name,
        "version": version,
        "sample_ids": ids.copy(),
        "scores": scores.copy(),
        "flags": flags.copy(),
        "settings": deepcopy(settings),
        "evidence": deepcopy({k: v for k, v in result.items()
                              if k not in {"sample_ids", "scores", "flags"}}),
    }


def to_jsonable(value):
    """Convert the connector output to values accepted by json.dumps()."""
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(v) for v in value]
    return value

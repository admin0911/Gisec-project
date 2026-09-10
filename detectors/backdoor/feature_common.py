"""Validation for truth-free, label-aware backdoor feature detectors."""
import numpy as np


def validated_inputs(inputs):
    X = np.asarray(inputs.X)
    ids = np.asarray(inputs.sample_ids)
    y = np.asarray(inputs.y)
    if (X.ndim != 2 or not X.size or X.dtype.kind not in "iuf"
            or not np.isfinite(X).all()):
        raise ValueError("X must be a nonempty finite numeric feature matrix")
    n = len(X)
    if (ids.shape != (n,) or ids.dtype.kind not in "iuUS"
            or len(np.unique(ids)) != n):
        raise ValueError("Supply one unique integer or string sample ID per row")
    if (inputs.y is None or y.shape != (n,) or y.dtype.kind not in "iuUS"
            or (y.dtype.kind in "iu" and np.any(y < 0))
            or (y.dtype.kind == "U" and np.any(y == ""))
            or (y.dtype.kind == "S" and np.any(y == b""))):
        raise ValueError("Supply known integer or string labels with label_aware=True")
    return X.astype(np.float64), y, ids


def centered_class(X):
    """One scalar rescale avoids overflow without changing class geometry."""
    scale = np.max(np.abs(X))
    values = X / scale if scale else X.copy()
    return values - values.mean(axis=0)


def integer_setting(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def positive_setting(value, name):
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")

"""Label-aware cosine distance to class centres on frozen embeddings.

Use detector_input(bundle, representation="raw", label_aware=True).
Each centre is the mean of unit-normalized features with the supplied label.
A sample is excluded from its own class centre (leave-one-out).
score = max(0, distance_to_own - distance_to_closest_other) / 2, in [0, 1].
Higher means a larger preference for another class, not a poison probability.
The default 0.1 threshold is provisional; no poisoning metadata is used.
Class means can be contaminated and may poorly represent multimodal classes.
Time O(n*d*c), memory O(n*d+n*c); no n-by-n matrix.
"""
import numpy as np


class ClassDistance:
    def __init__(self, threshold=0.1):
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = float(threshold)

    def analyze(self, inputs):
        X = np.asarray(inputs.X, dtype=np.float64)
        y = np.asarray(inputs.y)
        ids = np.asarray(inputs.sample_ids)
        if X.ndim != 2 or not len(X) or X.shape[1] == 0 or not np.isfinite(X).all():
            raise ValueError("X must be a non-empty finite 2D feature matrix")
        n = len(X)
        if inputs.y is None or y.shape != (n,) or y.dtype.kind not in "biufUS":
            raise ValueError("Supply one known label per sample with label_aware=True")
        if y.dtype.kind in "biuf" and (not np.isfinite(y).all() or np.any(y == -1)):
            raise ValueError("Labels must be finite and known (-1 is unsupported)")
        if ids.shape != (n,) or len(np.unique(ids)) != n:
            raise ValueError("Supply one unique sample ID per row")
        classes, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
        if len(classes) < 2 or np.any(counts < 2):
            raise ValueError("Need at least two classes, each with at least two samples")
        scale = np.max(np.abs(X), axis=1, keepdims=True)
        if np.any(scale == 0):
            raise ValueError("Cosine distance needs non-zero feature vectors")
        unit = X / scale
        unit /= np.linalg.norm(unit, axis=1, keepdims=True)
        sums = np.zeros((len(classes), X.shape[1]), dtype=np.float64)
        np.add.at(sums, inverse, unit)
        centre_norm = np.linalg.norm(sums, axis=1, keepdims=True)
        own_centres = sums[inverse] - unit
        own_norm = np.linalg.norm(own_centres, axis=1, keepdims=True)
        if np.any(centre_norm < 1e-12) or np.any(own_norm < 1e-12):
            raise ValueError("A class centre has no usable direction")
        centres = sums / centre_norm
        own_centres /= own_norm
        own_distance = 1 - np.clip(np.sum(unit * own_centres, axis=1), -1, 1)
        distances = 1 - np.clip(unit @ centres.T, -1, 1)
        distances[np.arange(n), inverse] = np.inf
        other_index = np.argmin(distances, axis=1)
        other_distance = distances[np.arange(n), other_index]
        scores = np.clip((own_distance - other_distance) / 2, 0, 1)
        return dict(sample_ids=ids.copy(), scores=scores,
                    flags=(scores > 0) & (scores >= self.threshold),
                    own_distance=own_distance, alternative_distance=other_distance,
                    alternative_labels=classes[other_index])

    def score(self, inputs):
        return self.analyze(inputs)["scores"]


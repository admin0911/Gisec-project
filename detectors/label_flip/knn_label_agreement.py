"""Cosine kNN label disagreement for the shared feature connector.

Use detector_input(bundle, representation="raw", label_aware=True).
"raw" means embeddings, not image pixels. Scores are disagreement fractions,
not poisoning probabilities. The default 19/20 flag threshold is provisional.
Exact search costs O(n*n*d); batches bound the similarity-matrix memory.
"""
import numpy as np


class KNNLabelAgreement:
    def __init__(self, k=20, threshold=0.95, batch_size=256):
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.k, self.threshold, self.batch_size = k, threshold, batch_size

    def analyze(self, inputs):
        """Return aligned scores, flags and neighbour row indices.

        Require at least k+1 samples rather than silently changing k.
        Equal-similarity neighbours at the cutoff may be selected arbitrarily.
        """
        X = np.asarray(inputs.X, dtype=np.float64)
        y = np.asarray(inputs.y)
        ids = np.asarray(inputs.sample_ids)
        if X.ndim != 2 or X.shape[1] == 0 or not np.isfinite(X).all():
            raise ValueError("X must be a finite 2D feature matrix")
        n = len(X)
        if n <= self.k:
            raise ValueError(f"Need at least {self.k + 1} samples for k={self.k}")
        if inputs.y is None or y.shape != (n,):
            raise ValueError("Provide one current label per row with label_aware=True")
        if y.dtype.kind not in "biufUS":
            raise ValueError("Labels must be known numeric or string class values")
        if y.dtype.kind in "biuf" and (not np.isfinite(y).all() or np.any(y == -1)):
            raise ValueError("Labels must be finite")
        if ids.shape != (n,) or len(np.unique(ids)) != n:
            raise ValueError("Provide one unique sample ID per row")
        # Scale first to avoid overflow; then normalize for cosine similarity.
        scales = np.max(np.abs(X), axis=1, keepdims=True)
        if np.any(scales == 0):
            raise ValueError("Cosine similarity is undefined for zero feature vectors")
        X = X / scales
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        neighbours = np.empty((n, self.k), dtype=np.int64)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            similarity = X[start:end] @ X.T
            # Exclude each image itself, including when duplicate images exist.
            similarity[np.arange(end - start), np.arange(start, end)] = -np.inf
            nearest = np.argpartition(-similarity, self.k - 1, axis=1)[:, :self.k]
            order = np.argsort(-np.take_along_axis(similarity, nearest, axis=1), axis=1)
            neighbours[start:end] = np.take_along_axis(nearest, order, axis=1)
        counts = np.sum(y[neighbours] != y[:, None], axis=1)
        scores = counts / self.k
        return {
            "sample_ids": ids.copy(),
            "scores": scores,
            "flags": scores >= self.threshold,
            "disagreement_counts": counts,
            "neighbour_indices": neighbours,
            "neighbour_sample_ids": ids[neighbours],
        }

    def score(self, inputs):
        """Connector contract: one score per input row, higher is suspicious."""
        return self.analyze(inputs)["scores"]


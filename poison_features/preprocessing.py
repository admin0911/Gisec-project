"""Scaling and safe dimensionality reduction."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def prepare_representations(
    features: np.ndarray, reduced_dim: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Return standardized and PCA representations without leaking labels."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("features must be a two-dimensional array with at least 2 rows")
    if not np.isfinite(values).all():
        raise ValueError("features contain NaN or infinite values")
    scaled = StandardScaler().fit_transform(values).astype(np.float32)
    components = min(reduced_dim, scaled.shape[0] - 1, scaled.shape[1])
    if components < 1:
        raise ValueError("features must have at least one usable dimension")
    reduced = PCA(n_components=components, random_state=0).fit_transform(scaled)
    return scaled, reduced.astype(np.float32)


def prepare_visual_features(features: np.ndarray) -> np.ndarray:
    """Create optional two-dimensional PCA points for visualization only."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("at least two feature rows are required for visualization")
    scaled = StandardScaler().fit_transform(values)
    components = min(2, scaled.shape[0] - 1, scaled.shape[1])
    return PCA(n_components=components, random_state=0).fit_transform(scaled).astype(np.float32)

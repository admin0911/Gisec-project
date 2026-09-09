"""Blended injection detector using shared high-frequency residual analysis.

Blended injection overlays a FIXED random-noise pattern on selected images at
low opacity and relabels them to a target class:

    poisoned = (1 - alpha) * clean_image + alpha * shared_noise_pattern

Detection strategy
------------------
The trigger is random noise and therefore high-frequency: adjacent pixels are
uncorrelated. Natural images are the opposite - smooth and low-frequency.
High-pass filtering removes most image content and leaves a residual in which
the trigger occupies far more of the energy than it does in raw pixels.

Every poisoned residual contains the same trigger, so the poisoned rows share
a direction. Individual pairs overlap only weakly - measured on CIFAR-10, two
poisoned rows correlate at about 0.14 against a clean spread of 0.06, and some
poisoned pairs barely correlate at all - so no pairwise rule is reliable.
Aggregating across rows is what recovers the signal, and the leading singular
vectors of the residual matrix isolate the trigger direction.

That first estimate is contaminated by natural image variation, and on a large
training split the trigger can even be smeared across two components with near
equal singular values. It is sharpened by averaging: take the rows the current
direction selects and average their residuals. The poisoned rows all contain
the identical trigger and reinforce, while the clean rows carry unrelated noise
that cancels, so the average is a far cleaner estimate than any singular vector.
Repeating this converges within a few passes. On the full CIFAR-10 split at a
1% poison rate this lifts detection from an F1 of 72.9% to 99.8%.

Deciding whether an attack exists at all is a separate question, and the harder
one. A clean class still has a leading singular vector, and a threshold alone
will cut its tail and report poison that is not there. The distinguishing
property is contrast: on a poisoned class the suspicious group sits many times
above the bulk, and refinement drives that contrast higher still, while on a
clean class there is nothing coherent to sharpen and refinement leaves it flat.
Measured across sample sizes and seeds, clean data never exceeded 6.3 while
every genuine attack reached at least 22.9, so the detector declines below 7.5
and flags nothing.
"""

import numpy as np

# How many times further the suspicious group must project than the bulk before
# an attack is reported. Clean data never exceeded 6.3 in testing and genuine
# attacks reached 22.9 or more, so this sits in a wide, empirically verified gap.
_MIN_CONTRAST = 7.5

# At a 10% dataset-wide poison rate the target class ends up about half
# poisoned, so a component implicating much more than half is image variety.
_MAX_FLAGGED_FRACTION = 0.6

# A genuine attack marks many rows. Fewer than this is treated as no finding.
_MIN_GROUP = 3

# Components searched for the trigger. At low poison rates the trigger explains
# less variance than natural image variation and is not necessarily first.
_N_COMPONENTS = 8

# Averaging passes used to sharpen the trigger estimate. Converges by the third.
_N_REFINEMENTS = 3


def _high_pass(images: np.ndarray) -> np.ndarray:
    """Return images minus their 3x3 box-blurred version (NCHW in, NCHW out)."""
    height, width = images.shape[2], images.shape[3]
    padded = np.pad(images, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="reflect")
    blurred = np.zeros_like(images, dtype=np.float64)
    for dy in range(3):
        for dx in range(3):
            blurred += padded[:, :, dy:dy + height, dx:dx + width]
    blurred /= 9.0
    return images.astype(np.float64) - blurred


def _top_right_vectors(matrix: np.ndarray, count: int) -> np.ndarray:
    """Return the leading `count` right singular vectors, one per row.

    Only a handful of directions are ever examined, so a full decomposition
    computes thousands that are immediately discarded. Going through whichever
    Gram matrix is smaller keeps this practical on a full training split, where
    the target class holds thousands of rows of a few thousand values each.
    """
    n_rows, n_cols = matrix.shape

    if n_rows <= n_cols:
        # Few rows: decompose the small row-wise Gram matrix, then map its
        # vectors back into the original coordinate space.
        values, vectors = np.linalg.eigh(matrix @ matrix.T)
        order = np.argsort(values)[::-1][:count]
        scale = np.sqrt(np.maximum(values[order], 0.0))
        keep = scale > 1e-12
        if not keep.any():
            return np.zeros((0, n_cols))
        return ((matrix.T @ vectors[:, order][:, keep]) / scale[keep]).T

    # Many rows: the column-wise Gram matrix yields the right vectors directly.
    values, vectors = np.linalg.eigh(matrix.T @ matrix)
    order = np.argsort(values)[::-1][:count]
    return vectors[:, order].T


def _otsu_threshold(values: np.ndarray) -> float:
    """Find the cut point maximising between-group variance (Otsu's method).

    Unlike a fixed percentile this adapts to the poison rate, which the
    detector does not know at scoring time.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return high

    hist, edges = np.histogram(finite, bins=256, range=(low, high))
    hist = hist.astype(np.float64)
    total = hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2.0

    weight_low = np.cumsum(hist)
    weight_high = total - weight_low
    valid = (weight_low > 0) & (weight_high > 0)
    if not valid.any():
        return high

    cumulative = np.cumsum(hist * centers)
    grand_total = cumulative[-1]
    mean_low = cumulative / np.maximum(weight_low, 1.0)
    mean_high = (grand_total - cumulative) / np.maximum(weight_high, 1.0)

    between = weight_low * weight_high * (mean_low - mean_high) ** 2
    between[~valid] = -1.0
    return float(centers[int(np.argmax(between))])


def _evaluate_component(projection: np.ndarray):
    """Split one set of projections and measure how far the groups sit apart.

    Returns (contrast, threshold, group_size). Contrast is the suspicious
    group's mean divided by the bulk's mean, which is scale-free and so
    comparable across components, refinement passes and datasets.
    """
    threshold = _otsu_threshold(projection)
    high = projection > threshold
    group_size = int(high.sum())

    if group_size == 0 or group_size == len(projection):
        return 0.0, threshold, 0
    if high.mean() > _MAX_FLAGGED_FRACTION:
        return 0.0, threshold, 0

    bulk_mean = float(projection[~high].mean())
    if bulk_mean <= 0:
        return 0.0, threshold, 0

    return float(projection[high].mean() / bulk_mean), threshold, group_size


class BlendedInjectionDetector:
    """Detects blended-injection poisoned samples via shared residual analysis."""

    def __init__(self, target_class: int = 0, min_contrast: float = _MIN_CONTRAST,
                 n_components: int = _N_COMPONENTS,
                 n_refinements: int = _N_REFINEMENTS):
        self.target_class = target_class
        self.min_contrast = min_contrast
        self.n_components = n_components
        self.n_refinements = n_refinements

    def score_bundle(self, bundle, pixels=None):
        """Score samples for blended injection.

        Args:
            bundle: ImageInputBundle, or a FeatureBundle when pixels is given.
            pixels: ImageInputBundle with NCHW float32 images in [0, 1].
                    Optional, so a single-argument call matching the
                    documented detector contract also works.

        Returns:
            dict with keys:
                "sample_ids": original sample IDs, order preserved
                "scores":     float64 array in [0, 1]; higher = more suspicious.
                              Non-target-class samples always score 0.
                "contrast":   how far the suspicious group sits above the bulk.
                              Below the acceptance level means no attack found.
                "group_size": how many rows fall in that group.
        """
        source = bundle if pixels is None else pixels
        if source is None or not hasattr(source, "images"):
            raise TypeError(
                "BlendedInjectionDetector scores pixels, so it needs an "
                "ImageInputBundle - pass it alone or as the second argument."
            )

        images = source.images       # (N, C, H, W), float32, [0, 1]
        labels = source.labels
        sample_ids = source.sample_ids

        n = len(images)
        all_scores = np.zeros(n, dtype=np.float64)
        empty = {"sample_ids": sample_ids, "scores": all_scores,
                 "contrast": 0.0, "group_size": 0}

        target_mask = labels == self.target_class
        target_images = images[target_mask]
        target_count = len(target_images)
        if target_count < 8:
            return empty

        # 1. High-pass residual, unit-normalised so pattern matters, not contrast.
        residual = _high_pass(target_images).reshape(target_count, -1)
        norms = np.linalg.norm(residual, axis=1, keepdims=True)
        residual = residual / np.maximum(norms, 1e-12)

        # 2. Search the leading components for one carrying a trigger.
        best_contrast = 0.0
        best_projection = None
        best_group = 0

        for vector in _top_right_vectors(residual, self.n_components):
            projection = np.abs(residual @ vector)
            contrast, _, group_size = _evaluate_component(projection)
            if contrast > best_contrast and group_size >= _MIN_GROUP:
                best_contrast = contrast
                best_projection = projection
                best_group = group_size

        if best_projection is None:
            return empty

        # 3. Sharpen the estimate. Averaging the selected rows reinforces the
        #    shared trigger and cancels their unrelated noise, so each pass
        #    describes the trigger better than the singular vector did.
        projection = best_projection
        for _ in range(self.n_refinements):
            group = projection > _otsu_threshold(projection)
            if int(group.sum()) < _MIN_GROUP:
                break
            direction = residual[group].mean(axis=0)
            magnitude = np.linalg.norm(direction)
            if magnitude <= 0:
                break
            candidate = np.abs(residual @ (direction / magnitude))
            contrast, _, group_size = _evaluate_component(candidate)
            if contrast <= 0 or group_size < _MIN_GROUP:
                break
            projection, best_contrast, best_group = candidate, contrast, group_size

        # 4. Reject when nothing separates convincingly - a clean class.
        if best_contrast < self.min_contrast:
            return {"sample_ids": sample_ids, "scores": all_scores,
                    "contrast": best_contrast, "group_size": 0}

        peak = projection.max()
        if peak > 0:
            all_scores[target_mask] = projection / peak

        return {
            "sample_ids": sample_ids,
            "scores": all_scores,
            "contrast": best_contrast,
            "group_size": best_group,
        }


def flag_samples(result, labels, target_class: int = 0) -> np.ndarray:
    """Turn suspicion scores into boolean flags.

    Accepts either the dict returned by score_bundle or a bare score array.
    Nothing is flagged when no direction separated convincingly, which is the
    correct answer for a clean dataset.
    """
    if isinstance(result, dict):
        scores = np.asarray(result["scores"])
        if result.get("group_size", 0) < _MIN_GROUP:
            return np.zeros(len(scores), dtype=bool)
    else:
        scores = np.asarray(result)

    flags = np.zeros(len(scores), dtype=bool)
    target_mask = labels == target_class
    target_scores = scores[target_mask]
    if len(target_scores) == 0 or target_scores.max() <= 0:
        return flags

    flags[target_mask] = target_scores > _otsu_threshold(target_scores)
    return flags

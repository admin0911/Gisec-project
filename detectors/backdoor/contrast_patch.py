"""Repeated exact patches with colour-independent local boundary evidence.

Development heuristic for fixed 2x2/3x3 patches, not general backdoor proof.
No clean counterpart, trigger location, attack target or poison truth is read.
"""
import math
import numpy as np

from poison_features import ImageInputBundle
from detectors.output_connector import detector_result


def boundary_contrast(images, row, column, size):
    """Minimum mean RGB jump across every visible edge, in [0,1].

    Requiring every visible edge rejects fragments of a larger flat region.
    At least two edges must be visible; an image-sized patch is not assessed.
    """
    n, _, h, w = images.shape
    sides = []
    def jump(inside, outside):
        # Compare recovered source bytes before averaging. Subtracting normalized
        # float32 values can put an exact one-level edge just below 1/255.
        a = np.rint(inside * 255).astype(np.int16)
        b = np.rint(outside * 255).astype(np.int16)
        return np.abs(a - b).mean(axis=(1, 2)) / 255
    if row > 0:
        sides.append(jump(images[:, :, row, column:column+size], images[:, :, row-1, column:column+size]))
    if row + size < h:
        sides.append(jump(images[:, :, row+size-1, column:column+size], images[:, :, row+size, column:column+size]))
    if column > 0:
        sides.append(jump(images[:, :, row:row+size, column], images[:, :, row:row+size, column-1]))
    if column + size < w:
        sides.append(jump(images[:, :, row:row+size, column+size-1], images[:, :, row:row+size, column+size]))
    return np.minimum.reduce(sides) if len(sides) >= 2 else np.zeros(n)


class ContrastPatchDetector:
    """Colour-independent exact patch recurrence after per-image edge filtering."""
    def __init__(self, patch_sizes=(2, 3), min_count=6, min_fraction=.005,
                 max_fraction=.20, min_purity=.70, min_lift=3., min_spatial_lift=3.,
                 min_contrast=1/255, diagnostic_limit=500):
        if not patch_sizes or any(type(s) is not int or s not in (2, 3) for s in patch_sizes):
            raise ValueError("patch_sizes must contain 2 and/or 3")
        if len(set(patch_sizes)) != len(patch_sizes):
            raise ValueError("patch_sizes must not contain duplicates")
        if type(min_count) is not int or min_count < 2:
            raise ValueError("min_count must be an integer >= 2")
        if not (0 < min_fraction <= max_fraction < 1 and .5 < min_purity <= 1):
            raise ValueError("Invalid fraction or purity settings")
        if any(not math.isfinite(v) or v <= 1 for v in (min_lift, min_spatial_lift)):
            raise ValueError("Lift settings must be finite and > 1")
        if not math.isfinite(min_contrast) or not 0 < min_contrast <= 1:
            raise ValueError("min_contrast must be finite and in (0,1]")
        if type(diagnostic_limit) is not int or diagnostic_limit < 0:
            raise ValueError("diagnostic_limit must be a nonnegative integer")
        self.settings = dict(patch_sizes=list(patch_sizes), min_count=min_count,
                             min_fraction=min_fraction, max_fraction=max_fraction,
                             min_purity=min_purity, min_lift=min_lift,
                             min_spatial_lift=min_spatial_lift, min_contrast=min_contrast,
                             diagnostic_limit=diagnostic_limit)

    def analyze(self, inputs, progress=None):
        if not isinstance(inputs, ImageInputBundle):
            raise TypeError("Expected ImageInputBundle")
        images, labels, ids = map(np.asarray, (inputs.images, inputs.labels, inputs.sample_ids))
        if (images.ndim != 4 or not images.size or images.shape[1] not in (1, 3)
                or not np.isfinite(images).all() or images.min() < 0 or images.max() > 1):
            raise ValueError("Expected finite nonempty NCHW pixels in [0,1]")
        n, channels, height, width = images.shape
        if ids.shape != (n,) or ids.dtype.kind not in "iuUS" or len(np.unique(ids)) != n:
            raise ValueError("Expected unique aligned integer or string IDs")
        if labels.shape != (n,) or labels.dtype.kind not in "iu" or np.any(labels < 0):
            raise ValueError("Expected known aligned integer labels")
        cfg = self.settings
        if any(s > min(height, width) for s in cfg["patch_sizes"]):
            raise ValueError("Patch sizes must fit the image dimensions")
        values = np.rint(images * 255).astype(np.uint8)
        normalized = values.astype(np.float32) / 255
        classes, encoded, class_counts = np.unique(labels, return_inverse=True, return_counts=True)
        baseline = class_counts / n
        lower = max(cfg["min_count"], math.ceil(cfg["min_fraction"] * n))
        scores = np.zeros(n); pattern_ids = np.full(n, -1, dtype=np.int64)
        patterns, diagnostics = [], []
        reasons = {}; candidate_groups = 0
        total = sum((height-s+1)*(width-s+1) for s in cfg["patch_sizes"])
        done = 0
        for size in cfg["patch_sizes"]:
            def codes(rr, cc):
                rows = np.ascontiguousarray(values[:, :, rr:rr+size, cc:cc+size]).reshape(n, -1)
                return rows.view(np.dtype((np.void, rows.shape[1]))).ravel()
            for row in range(height-size+1):
                for column in range(width-size+1):
                    keys = codes(row, column)
                    unique, counts = np.unique(keys, return_counts=True)
                    candidates = unique[counts >= lower]
                    contrast = boundary_contrast(normalized, row, column, size) if len(candidates) else None
                    for code in candidates:
                        candidate_groups += 1
                        raw = np.flatnonzero(keys == code)
                        matches = raw[contrast[raw] >= cfg["min_contrast"]]
                        entry = dict(row=row, column=column, size=size, raw_support=len(raw),
                                     boundary_support=len(matches))
                        reason = None
                        if len(matches) < lower:
                            reason = "insufficient_boundary_support"
                        elif len(matches) > cfg["max_fraction"] * n:
                            reason = "too_common_after_boundary_filter"
                        else:
                            histogram = np.bincount(encoded[matches], minlength=len(classes))
                            dominant = int(histogram.argmax())
                            purity = float(histogram[dominant] / len(matches))
                            lift = float(purity / baseline[dominant])
                            entry.update(dominant_label=int(classes[dominant]), purity=purity, lift=lift)
                            if purity < cfg["min_purity"]:
                                reason = "low_label_purity"
                            elif lift < cfg["min_lift"]:
                                reason = "low_label_lift"
                            else:
                                nearby = []
                                for dr in (-size, 0, size):
                                    for dc in (-size, 0, size):
                                        rr, cc = row+dr, column+dc
                                        if (dr == dc == 0 or rr < 0 or cc < 0 or rr+size > height or cc+size > width):
                                            continue
                                        near = codes(rr, cc) == code
                                        if near.any():
                                            near &= boundary_contrast(normalized, rr, cc, size) >= cfg["min_contrast"]
                                        nearby.append(int(near.sum()))
                                spatial = len(matches)/(1 + float(np.mean(nearby))) if nearby else 0.
                                entry["spatial_lift"] = spatial
                                if spatial < cfg["min_spatial_lift"]:
                                    reason = "low_spatial_lift"
                                else:
                                    strength = purity * (1 - 1/lift)
                                    pattern_id = len(patterns)
                                    pattern = np.frombuffer(code.tobytes(), dtype=np.uint8).reshape(channels, size, size)
                                    patterns.append(dict(entry, support=len(matches), score=strength,
                                                         quantized_pattern=pattern.transpose(1, 2, 0).tolist()))
                                    better = matches[strength > scores[matches]]
                                    scores[better] = strength; pattern_ids[better] = pattern_id
                        reason = reason or "accepted"
                        reasons[reason] = reasons.get(reason, 0) + 1
                        if len(diagnostics) < cfg["diagnostic_limit"]:
                            diagnostics.append(dict(entry, decision=reason))
                    done += 1
                    if progress and (done % 50 == 0 or done == total): progress(done, total)
        return detector_result("contrast_patch", "0.1.0",
                               dict(sample_ids=ids, scores=scores, flags=pattern_ids >= 0,
                                    pattern_id=pattern_ids, patterns=patterns, diagnostics=diagnostics,
                                    diagnostic_counts=reasons, candidate_groups=candidate_groups,
                                    diagnostics_truncated=candidate_groups > len(diagnostics)),
                               dict(cfg, profile="contrast-patch-v1", intensity_bins=256,
                                    calibration="Development heuristic; assess on independent clean and attacked data",
                                    flag_rule="Exact repeated patch with every visible boundary, label and spatial support",
                                    limitation="Fixed 2x2/3x3 patches only; triggers matching their surroundings can be invisible"),
                               expected_sample_ids=ids)

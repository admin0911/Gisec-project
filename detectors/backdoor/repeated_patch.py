"""MNIST repeated-patch heuristic; no reference model or poison metadata.

Searches every location for recurring quantized 2x2 and 3x3 patches.
Scores are evidence strengths, not calibrated probabilities of poisoning.
Natural class-specific patterns can cause false positives. Variable-position,
dark, large, or non-patch triggers may be missed. RGB supported experimentally; no CIFAR-10 calibration.
"""
import math
import numpy as np
from poison_features.image_inputs import ImageInputBundle
from detectors.output_connector import detector_result


def scan_repeated_patches(inputs, progress=None, patch_sizes=(2, 3),
                          min_fraction=.005, max_fraction=.20,
                          min_count=30, min_purity=.90, min_lift=3.0,
                          min_spatial_lift=5.0, intensity_bins=4):
    """Accept an ImageInputBundle and observed integer labels only.

    Group patches into configurable intensity bins (256 means exact uint8 matching). A candidate must recur, have
    mean quantized brightness >= 1 (out of 3), and concentrate in a class
    more than its overall frequency predicts. Flag all candidate matches,
    including matches whose labels differ from the candidate's dominant class.
    """
    # Leila: consume post-attack pixels through the shared image connector only.
    if not isinstance(inputs, ImageInputBundle):
        raise TypeError('Expected ImageInputBundle with pixels, current labels and sample IDs')
    images = np.asarray(inputs.images)
    y = np.asarray(inputs.labels)
    ids = np.asarray(inputs.sample_ids)
    if (images.ndim != 4 or images.shape[1] not in (1, 3) or images.size == 0
            or not np.isfinite(images).all() or images.min() < 0 or images.max() > 1):
        raise ValueError('Expected finite nonempty NCHW grayscale or RGB pixels in [0,1]')
    if ids.shape != (len(images),) or ids.dtype.kind not in 'iuUS' or len(np.unique(ids)) != len(ids):
        raise ValueError('Expected aligned unique integer or string sample IDs')
    # Recover source uint8 values before quantization to preserve bin boundaries.
    x = np.rint(images.transpose(0,2,3,1) * 255).astype(np.uint8)
    if x.shape[-1] == 1: x = x[...,0]
    if x.ndim not in (3,4) or (x.ndim == 4 and x.shape[-1] != 3) or x.dtype != np.uint8 or len(x) == 0:
        raise ValueError('Expected nonempty uint8 data [N,H,W] or [N,H,W,3]')
    if y.shape != (len(x),) or not np.issubdtype(y.dtype, np.integer) or (y < 0).any():
        raise ValueError('Expected aligned nonnegative integer labels')
    if (not patch_sizes or any(type(s) is not int or s not in (2, 3) or
            s > min(x.shape[1:3]) for s in patch_sizes)):
        raise ValueError('Patch sizes must be 2 or 3 and fit the images')
    if not (0 < min_fraction <= max_fraction < 1 and .5 < min_purity <= 1
            and math.isfinite(min_lift) and min_lift > 1
            and math.isfinite(min_spatial_lift) and min_spatial_lift > 1
            and type(min_count) is int and min_count > 0):
        raise ValueError('Invalid support, purity, or lift settings')
    classes, encoded, class_counts = np.unique(y, return_inverse=True, return_counts=True)
    baseline = class_counts / len(x)
    channels = 1 if x.ndim == 3 else 3
    # Leila: finer matching is opt-in; retain four-bin defaults for the web scan.
    if type(intensity_bins) is not int or intensity_bins not in (4, 8, 16, 32, 64, 128, 256):
        raise ValueError('intensity_bins must be a power of two from 4 through 256')
    quantized = x // (256 // intensity_bins)
    if x.ndim == 3: quantized = quantized[..., None]
    risk = np.zeros(len(x), dtype=np.float32)
    evidence = np.full(len(x), -1, dtype=np.int64)
    patterns = []
    total = sum((x.shape[1]-s+1)*(x.shape[2]-s+1) for s in patch_sizes)
    done = 0
    lower = max(min_count, math.ceil(min_fraction*len(x)))
    for size in patch_sizes:
        for row in range(x.shape[1]-size+1):
            for col in range(x.shape[2]-size+1):
                # Leila: byte keys preserve all RGB values; fine 3x3 RGB cannot fit in uint64.
                def patch_codes(rr, cc):
                    values = np.ascontiguousarray(quantized[:,rr:rr+size,cc:cc+size,:]).reshape(len(x),-1)
                    return values.view(np.dtype((np.void, values.shape[1]))).ravel()
                codes = patch_codes(row, col)
                unique, counts = np.unique(codes, return_counts=True)
                candidates = unique[(counts >= lower) & (counts <= max_fraction*len(x))]
                for code in candidates:
                    bins = np.frombuffer(code.tobytes(), dtype=np.uint8).tolist()
                    if np.mean(bins) / (intensity_bins - 1) < 1/3:
                        continue
                    matches = np.flatnonzero(codes == code)
                    histogram = np.bincount(encoded[matches], minlength=len(classes))
                    dominant = int(histogram.argmax())
                    purity = float(histogram[dominant]/len(matches))
                    lift = float(purity/baseline[dominant])
                    if purity < min_purity or lift < min_lift:
                        continue
                    # Ordinary strokes recur at neighbouring positions too.
                    # Compare identical codes at offsets of one patch width;
                    # this uses only the submitted images, never clean truth.
                    nearby_counts = []
                    for dr in (-size, 0, size):
                        for dc in (-size, 0, size):
                            rr, cc = row+dr, col+dc
                            if (dr == dc == 0 or rr < 0 or cc < 0 or
                                    rr+size > x.shape[1] or cc+size > x.shape[2]):
                                continue
                            other = patch_codes(rr, cc)
                            nearby_counts.append(int((other == code).sum()))
                    if not nearby_counts:
                        continue
                    spatial_lift = len(matches)/(1 + float(np.mean(nearby_counts)))
                    if spatial_lift < min_spatial_lift:
                        continue
                    score = purity * (1 - 1/lift)
                    pattern_id = len(patterns)
                    patterns.append(dict(row=row, column=col, size=size,
                        quantized_pattern=np.array(bins).reshape((size,size) if channels == 1 else (size,size,channels)).tolist(),
                        support=len(matches), dominant_label=int(classes[dominant]),
                        purity=purity, lift=lift, spatial_lift=spatial_lift, score=score))
                    better = matches[score > risk[matches]]
                    risk[better] = score
                    evidence[better] = pattern_id
                done += 1
                if progress and (done % 50 == 0 or done == total):
                    progress(done, total)
    # Leila: all rows retain their input order; pattern evidence explains each flag.
    result = dict(sample_ids=ids, scores=risk, flags=evidence >= 0,
                  pattern_id=evidence, patterns=patterns)
    settings = dict(patch_sizes=list(patch_sizes), min_fraction=min_fraction,
                    max_fraction=max_fraction, min_count=min_count,
                    min_purity=min_purity, min_lift=min_lift,
                    min_spatial_lift=min_spatial_lift, intensity_bins=intensity_bins,
                    calibration='provisional; not calibrated for CIFAR-10',
                    flag_rule='matches an accepted repeated patch')
    return detector_result('repeated_patch', '1.1.0', result, settings,
                           expected_sample_ids=inputs.sample_ids)


class RepeatedPatchDetector:
    """Shared ImageInputBundle -> standard detector dictionary.

    Only bright quantized 2x2/3x3 patterns are searched. No attack target,
    trigger position, poison rate, or poison identities are supplied.
    """
    def __init__(self, **settings):
        self.settings = dict(settings)

    def analyze(self, inputs, progress=None):
        return scan_repeated_patches(inputs, progress=progress, **self.settings)

    def score_bundle(self, bundle, pixels=None):
        """Use the team's pixel connector, optionally alongside a feature bundle."""
        source = bundle if pixels is None else pixels
        if pixels is not None and (not np.array_equal(bundle.sample_ids, pixels.sample_ids)
                                  or not np.array_equal(bundle.labels, pixels.labels)):
            raise ValueError('Feature and pixel rows or labels do not match')
        return self.analyze(source)

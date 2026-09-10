"""Consensus-pixel deviation scan for blended triggers on constant-background images.

Uses only the submitted pixels: no clean reference, trigger, target, rate, or
poison identities. A pixel is a consensus pixel when at least ``consensus`` of
rows share its median value within ``tolerance``. A global blended trigger may disturb many of them at once. Benign brightness
changes can produce the same evidence; this is not proof of a backdoor.

Scope: datasets with a near-constant background such as MNIST. Inputs with too few
consensus pixels are inconclusive. Natural photos with common borders may still
have consensus pixels; this detector is intended for MNIST-like backgrounds.
Blends applied only inside the foreground, or too faint to exceed ``tolerance``,
are missed. Scores are fractions of consensus pixels broken, not probabilities.
"""
import numpy as np
from poison_features.image_inputs import ImageInputBundle
from detectors.output_connector import detector_result


def scan_consensus_pixels(inputs, consensus=0.90, tolerance=1/255,
                          flag_fraction=0.50, min_consensus_pixels=50):
    if not isinstance(inputs, ImageInputBundle):
        raise TypeError('Expected ImageInputBundle with pixels, labels and sample IDs')
    images = np.asarray(inputs.images, dtype=np.float32)
    ids = np.asarray(inputs.sample_ids)
    if images.ndim != 4 or images.size == 0 or not np.isfinite(images).all():
        raise ValueError('Expected finite nonempty NCHW pixels in [0,1]')
    # Leila: validate the pixel range even if a caller mutated an existing bundle.
    if np.any(images < 0) or np.any(images > 1):
        raise ValueError('Expected pixels in [0,1]')
    if ids.shape != (len(images),) or len(np.unique(ids)) != len(ids):
        raise ValueError('Expected aligned unique sample IDs')
    if not (0.5 < consensus <= 1 and 0 <= tolerance < 1 and 0 < flag_fraction < 1
            and type(min_consensus_pixels) is int and min_consensus_pixels > 0):
        raise ValueError('Invalid consensus, tolerance, flag fraction, or minimum pixel settings')
    flat = images.reshape(len(images), -1)
    # Leila: the per-pixel median over the submitted rows is the only reference used.
    median = np.median(flat, axis=0)
    agreement = (np.abs(flat - median) <= tolerance).mean(axis=0)
    fixed = agreement >= consensus
    n_fixed = int(fixed.sum())
    if n_fixed >= min_consensus_pixels:
        broken = np.abs(flat[:, fixed] - median[fixed]) > tolerance
        scores = broken.mean(axis=1).astype(np.float32)
        flags = scores > flag_fraction
        status = 'scored'
    else:
        scores = np.zeros(len(flat), dtype=np.float32)
        flags = np.zeros(len(flat), dtype=bool)
        status = 'inconclusive: too few consensus pixels for this dataset'
    result = dict(sample_ids=ids, scores=scores, flags=flags,
                  consensus_pixel_count=n_fixed, status=status,
                  applicable=n_fixed >= min_consensus_pixels,
                  consensus_pixel_indices=np.flatnonzero(fixed),
                  image_shape=np.asarray(images.shape[1:]))
    settings = dict(consensus=consensus, tolerance=tolerance, flag_fraction=flag_fraction,
                    min_consensus_pixels=min_consensus_pixels,
                    calibration='provisional; MNIST development checks only',
                    flag_rule='breaks more than flag_fraction of consensus pixels',
                    scope='MNIST-like constant-background datasets; not validated for natural photos')
    return detector_result('consensus_pixels', '0.1.0', result, settings,
                           expected_sample_ids=inputs.sample_ids)


class ConsensusPixelDetector:
    """Shared ImageInputBundle -> standard detector dictionary."""
    def __init__(self, **settings):
        self.settings = dict(settings)

    def analyze(self, inputs, progress=None):
        result = scan_consensus_pixels(inputs, **self.settings)
        if progress:
            progress(1, 1)
        return result

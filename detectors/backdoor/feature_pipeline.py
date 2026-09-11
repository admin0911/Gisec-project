"""Experimental feature backdoor results, separate from cleaning policies."""
from importlib.metadata import version
import numpy as np

from .activation_clustering import ActivationClusteringDetector
from .spectral_signature import SpectralSignatureDetector
from .feature_common import validated_inputs


def scan_backdoor_features(inputs, *, progress=None):
    """No truth fields, no detector selection, no automatic sample removal."""
    _, _, ids = validated_inputs(inputs)
    outputs = {}
    for detector in (SpectralSignatureDetector(), ActivationClusteringDetector()):
        if progress:
            progress(f"Running {type(detector).__name__}")
        result = detector.analyze(inputs)
        outputs[result["detector_name"]] = result
    votes = np.sum([item["flags"] for item in outputs.values()], axis=0).astype(np.int64)
    return dict(
        sample_ids=ids.copy(), detectors=outputs, flag_count=votes,
        candidate_flags=votes >= 1, agreement_flags=votes == 2,
        settings=dict(
            status="experimental; not connected to automatic cleaning or web scans",
            candidate_rule="at least one detector flags; review only",
            agreement_rule="both detectors flag; not independent confirmation",
            packages={name: version(name) for name in ("numpy", "scikit-learn")},
        ),
    )

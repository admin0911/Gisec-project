"""Truth-free image backdoor scan: patch evidence plus optional features."""
import numpy as np

from poison_features import ImageInputBundle
from .feature_common import validated_inputs
from .feature_pipeline import scan_backdoor_features
from .repeated_patch import RepeatedPatchDetector
from .web_patch import CIFAR_PATCH_SETTINGS
from .contrast_patch import ContrastPatchDetector


def scan_backdoor_images(pixels, *, dataset, features=None, progress=None, patch_profile="legacy"):
    """Union review candidates without averaging scores or requiring agreement.

    `features` must be a DetectorInput, never a FeatureBundle carrying truth.
    The clean reference, trigger settings and poison truth are not accepted.
    The Python API defaults to legacy for existing callers; the saved-file CLI
    explicitly defaults to contrast. In contrast mode, legacy is comparison-only.
    """
    if not isinstance(pixels, ImageInputBundle):
        raise TypeError("Use ImageInputBundle for image inputs")
    if dataset not in ("cifar10", "mnist"):
        raise ValueError("Supported fixed profiles: cifar10 and mnist")
    if patch_profile not in ("legacy", "contrast"):
        raise ValueError("patch_profile must be legacy or contrast")
    if features is not None:
        _, labels, ids = validated_inputs(features)
        if not np.array_equal(ids, pixels.sample_ids) or not np.array_equal(labels, pixels.labels):
            raise ValueError("Feature and image sample IDs, row order and current labels must match")
    if progress:
        progress("Scanning pixels for repeated bright patches")
    patch = RepeatedPatchDetector(
        **(CIFAR_PATCH_SETTINGS if dataset == "cifar10" else {})
    ).analyze(pixels)
    patch["settings"]["profile"] = "cifar-bright-patch-v1" if dataset == "cifar10" else "mnist-default"
    if dataset == "cifar10":
        patch["settings"]["calibration"] = (
            "Existing CIFAR profile; limited prior clean-only calibration. "
            "No settings changed by this pipeline."
        )
    outputs = {"repeated_patch": patch}
    active = ["repeated_patch"]
    if patch_profile == "contrast":
        if progress:
            progress("Checking repeated patches against surrounding pixels")
        outputs["contrast_patch"] = ContrastPatchDetector().analyze(pixels)
        active = ["contrast_patch"]
    if features is not None:
        outputs.update(scan_backdoor_features(features, progress=progress)["detectors"])
        active.extend(["spectral_signature", "activation_clustering"])
    votes = np.sum([outputs[name]["flags"] for name in active], axis=0).astype(np.int64)
    return dict(sample_ids=pixels.sample_ids.copy(), detectors=outputs,
                candidate_flags=votes > 0, flag_count=votes,
                settings=dict(
                    candidate_rule="Any active detector flag requests review; no automatic removal",
                    active_detectors=active, patch_profile=patch_profile,
                    comparison_only=[name for name in outputs if name not in active],
                    features_used=features is not None,
                    limitation="Fixed exact 2x2/3x3 patches plus experimental feature anomalies; "
                               "zero flags is not proof of clean data. No model ASR measurement.",
                ))

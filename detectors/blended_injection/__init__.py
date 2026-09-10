"""Blended-injection detection.

The attack blends one shared noise pattern into selected images at low opacity
and relabels them to a target class. Two signatures cover different kinds of
image data:

`residual_signature` looks for the pattern shared across poisoned rows. This
needs images to be individually distinctive, which holds for photographs.

`background_lift` checks whether pixels that are dark throughout the dataset
have been lifted off zero. This suits data whose images resemble one another
closely, such as handwritten digits, where no shared pattern stands out.

`scan_all_classes` picks whichever applies and identifies the targeted class.
"""

from detectors.blended_injection.residual_signature import (
    BlendedInjectionDetector,
    flag_samples,
)
from detectors.blended_injection.background_lift import (
    background_mask,
    lift_scores,
    scan_background_lift,
)
from detectors.blended_injection.pipeline import (
    DETECTOR_NAME,
    DETECTOR_VERSION,
    scan_all_classes,
    scan_blended_injection,
    scan_settings,
)

# Leila: retain the MNIST consensus-pixel detector alongside the CIFAR detector.
from .consensus_pixels import ConsensusPixelDetector, scan_consensus_pixels

__all__ = [
    "BlendedInjectionDetector",
    "flag_samples",
    "background_mask",
    "lift_scores",
    "scan_background_lift",
    "scan_all_classes",
    "scan_blended_injection",
    "scan_settings",
    "ConsensusPixelDetector",
    "scan_consensus_pixels",
]
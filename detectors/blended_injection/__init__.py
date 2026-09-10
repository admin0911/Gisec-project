"""Blended-injection detectors for CIFAR-10 and MNIST."""

from detectors.blended_injection.residual_signature import (
    BlendedInjectionDetector,
    flag_samples,
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
    "DETECTOR_NAME",
    "DETECTOR_VERSION",
    "scan_all_classes",
    "scan_blended_injection",
    "scan_settings",
    "ConsensusPixelDetector",
    "scan_consensus_pixels",
]
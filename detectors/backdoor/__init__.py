# Leila: expose the repeated-patch detector through the backdoor package.
"""Pixel and feature backdoor detectors using shared connectors."""
from .repeated_patch import RepeatedPatchDetector
from .contrast_patch import ContrastPatchDetector
from .spectral_signature import SpectralSignatureDetector
from .activation_clustering import ActivationClusteringDetector

__all__ = [
    "RepeatedPatchDetector",
    "ContrastPatchDetector",
    "SpectralSignatureDetector",
    "ActivationClusteringDetector",
]

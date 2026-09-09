"""Blended-injection detection.

The attack blends one shared noise pattern into selected images at low opacity
and relabels them to a target class. `residual_signature` recovers the shared
pattern; `pipeline` runs it and returns connector-ready output.
"""

from detectors.blended_injection.residual_signature import (
    BlendedInjectionDetector,
    flag_samples,
)
from detectors.blended_injection.pipeline import (
    DETECTOR_NAME,
    DETECTOR_VERSION,
    scan_blended_injection,
    scan_settings,
)

__all__ = [
    "BlendedInjectionDetector",
    "flag_samples",
    "scan_blended_injection",
    "scan_settings",
    "DETECTOR_NAME",
    "DETECTOR_VERSION",
]

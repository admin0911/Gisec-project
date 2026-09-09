"""Run the blended-injection detector and return shared-connector output.

The result carries `sample_ids`, `scores` and `flags` as the shared output
connector requires. Detector-specific values are returned alongside them and
are retained under `evidence` once the connector wraps the result.
"""

from detectors.blended_injection.residual_signature import (
    BlendedInjectionDetector,
    flag_samples,
    _MIN_CONTRAST,
    _N_COMPONENTS,
    _N_REFINEMENTS,
)

DETECTOR_NAME = "blended-injection-residual"
DETECTOR_VERSION = "1.0.0"


def scan_settings(target_class: int = 0, min_contrast: float = _MIN_CONTRAST,
                  n_components: int = _N_COMPONENTS,
                  n_refinements: int = _N_REFINEMENTS) -> dict:
    """Return the settings recorded alongside a scan."""
    return {
        "target_class": target_class,
        "min_contrast": min_contrast,
        "n_components": n_components,
        "n_refinements": n_refinements,
    }


def scan_blended_injection(pixels, target_class: int = 0,
                           min_contrast: float = _MIN_CONTRAST,
                           n_components: int = _N_COMPONENTS,
                           n_refinements: int = _N_REFINEMENTS) -> dict:
    """Score an ImageInputBundle for blended injection.

    Args:
        pixels: ImageInputBundle with NCHW float32 images in [0, 1].
        target_class: label the attack assigns to poisoned rows.

    Returns:
        dict with `sample_ids`, `scores` and `flags` in input order, plus
        `contrast` and `group_size` describing how convincing the finding was.
        A contrast below the acceptance level means no attack was found, and
        no rows are flagged.
    """
    detector = BlendedInjectionDetector(
        target_class=target_class,
        min_contrast=min_contrast,
        n_components=n_components,
        n_refinements=n_refinements,
    )
    result = detector.score_bundle(pixels)
    flags = flag_samples(result, pixels.labels, target_class)

    return {
        "sample_ids": result["sample_ids"],
        "scores": result["scores"],
        "flags": flags,
        "contrast": result["contrast"],
        "group_size": result["group_size"],
    }


def scan_as_connector_result(pixels, target_class: int = 0, **kwargs) -> dict:
    """Run a scan and wrap it with the shared team output connector.

    Imported lazily so this module stays usable on a branch where the shared
    connector is not yet present.
    """
    from detectors.output_connector import detector_result

    result = scan_blended_injection(pixels, target_class=target_class, **kwargs)
    return detector_result(
        DETECTOR_NAME,
        DETECTOR_VERSION,
        result,
        scan_settings(target_class=target_class, **kwargs),
        expected_sample_ids=pixels.sample_ids,
    )

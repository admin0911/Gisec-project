"""Run the blended-injection detector and return shared-connector output.

The result carries `sample_ids`, `scores` and `flags` as the shared output
connector requires. Detector-specific values are returned alongside them and
are retained under `evidence` once the connector wraps the result.
"""

import numpy as np

from detectors.blended_injection.background_lift import scan_background_lift
from detectors.blended_injection.residual_signature import (
    BlendedInjectionDetector,
    flag_samples,
    _MIN_CONTRAST,
    _N_COMPONENTS,
    _N_REFINEMENTS,
)

DETECTOR_NAME = "blended-injection-residual"
DETECTOR_VERSION = "1.1.0"

# How far the suspect class must stand above its peers. Blended injection
# targets a single class, so the remaining classes are a control group drawn
# from the same dataset, and the comparison needs no absolute scale.
#
# Measured on CIFAR-10, a clean winner sits about 1.15x its peer median while
# a poisoned class reaches 3.9x or more. Small classes are the demanding case:
# with only 60 rows, chance coherence lifted a clean class to an absolute
# contrast of 9.6, well past the absolute bar, and the peer ratio of 2.0 was
# the only thing that caught it. Hence 3.0, which clears every clean result
# observed while leaving headroom below the weakest genuine detection.
_MIN_CLASS_RATIO = 3.0


def scan_settings(target_class=0, min_contrast: float = _MIN_CONTRAST,
                  n_components: int = _N_COMPONENTS,
                  n_refinements: int = _N_REFINEMENTS,
                  min_class_ratio: float = _MIN_CLASS_RATIO) -> dict:
    """Return the settings recorded alongside a scan."""
    return {
        "target_class": target_class,
        "min_contrast": min_contrast,
        "n_components": n_components,
        "n_refinements": n_refinements,
        "min_class_ratio": min_class_ratio,
    }


def scan_blended_injection(pixels, target_class: int = 0,
                           min_contrast: float = _MIN_CONTRAST,
                           n_components: int = _N_COMPONENTS,
                           n_refinements: int = _N_REFINEMENTS) -> dict:
    """Score an ImageInputBundle for blended injection in one known class.

    Use `scan_all_classes` when the targeted class is unknown, which is the
    realistic case.

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
        "target_class": target_class,
    }


def scan_all_classes(pixels, min_contrast: float = _MIN_CONTRAST,
                     n_components: int = _N_COMPONENTS,
                     n_refinements: int = _N_REFINEMENTS,
                     min_class_ratio: float = _MIN_CLASS_RATIO) -> dict:
    """Scan every class and report the targeted one, if any.

    On a real dataset the attacker's target is unknown, so scoring a single
    assumed class would miss an attack aimed anywhere else and wrongly report
    the data clean. Every class is scanned instead and the strongest finding
    is reported, which also identifies the target rather than assuming it.

    Scanning many classes means many chances of a false alarm, so a finding
    must clear two independent bars. It must reach the absolute contrast level
    a genuine attack produces, and it must stand well above the other classes,
    which serve as a control group from the same dataset - blended injection
    poisons one class, so the rest show what untouched classes look like here.

    Returns:
        dict with `sample_ids`, `scores` and `flags` in input order, plus
        `target_class` naming the class found, `contrast`, `group_size`,
        `class_ratio` against the peer median, and `class_contrasts` giving
        every class's score. `target_class` is None when nothing was found.
    """
    labels = np.asarray(pixels.labels)
    classes = [int(c) for c in np.unique(labels)]

    contrasts = {}
    results = {}
    for label in classes:
        result = BlendedInjectionDetector(
            target_class=label,
            min_contrast=min_contrast,
            n_components=n_components,
            n_refinements=n_refinements,
        ).score_bundle(pixels)
        contrasts[label] = float(result["contrast"])
        results[label] = result

    nothing_found = {
        "sample_ids": np.asarray(pixels.sample_ids),
        "scores": np.zeros(len(labels), dtype=np.float64),
        "flags": np.zeros(len(labels), dtype=bool),
        "target_class": None,
        "contrast": 0.0,
        "group_size": 0,
        "class_ratio": 0.0,
        "class_contrasts": contrasts,
    }
    if not contrasts:
        return nothing_found

    suspect = max(contrasts, key=contrasts.get)
    best = results[suspect]
    peers = [v for k, v in contrasts.items() if k != suspect]
    peer_median = float(np.median(peers)) if peers else 0.0
    ratio = (contrasts[suspect] / peer_median) if peer_median > 0 else float("inf")

    if contrasts[suspect] < min_contrast or ratio < min_class_ratio:
        # No shared pattern stood out. That signature needs images to be
        # individually distinctive, which holds for photographs but not for
        # datasets whose classes look alike, so fall back to the background
        # check - it asks a different question and covers exactly that case.
        fallback = scan_background_lift(pixels)
        if fallback["target_class"] is not None:
            fallback["class_contrasts"] = contrasts
            return fallback
        nothing_found["method"] = "residual-signature"
        return nothing_found

    return {
        "method": "residual-signature",
        "sample_ids": best["sample_ids"],
        "scores": best["scores"],
        "flags": flag_samples(best, labels, suspect),
        "target_class": suspect,
        "contrast": contrasts[suspect],
        "group_size": best["group_size"],
        "class_ratio": ratio,
        "class_contrasts": contrasts,
    }


def scan_as_connector_result(pixels, target_class=None, **kwargs) -> dict:
    """Run a scan and wrap it with the shared team output connector.

    Scans every class when `target_class` is None. Imported lazily so this
    module stays usable on a branch without the shared connector.
    """
    from detectors.output_connector import detector_result

    if target_class is None:
        result = scan_all_classes(pixels, **kwargs)
        settings = scan_settings(target_class="auto", **kwargs)
    else:
        result = scan_blended_injection(pixels, target_class=target_class, **kwargs)
        settings = scan_settings(target_class=target_class, **kwargs)

    return detector_result(
        DETECTOR_NAME,
        DETECTOR_VERSION,
        result,
        settings,
        expected_sample_ids=pixels.sample_ids,
    )

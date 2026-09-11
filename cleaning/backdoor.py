"""Convert a complete backdoor scan into reversible training decisions."""
from copy import deepcopy

import numpy as np

from .label_flip import ACTIONS


DEFAULT_QUARANTINE = ("contrast_patch",)
DEFAULT_REVIEW = ("spectral_signature", "activation_clustering")


def decide_backdoor_actions(
    scan,
    *,
    quarantine_detectors=DEFAULT_QUARANTINE,
    review_detectors=DEFAULT_REVIEW,
):
    """Quarantine strong pixel findings and hold feature-only findings for review.

    The function consumes detector flags and recorded settings only. Evaluation
    fields, poison identities, attack parameters, and clean references are not
    read. Comparison-only detector outputs never affect a decision.
    """
    ids = np.asarray(scan["sample_ids"])
    settings = scan.get("settings")
    detectors = scan.get("detectors")
    if (
        ids.ndim != 1
        or not len(ids)
        or ids.dtype.kind not in "iuUS"
        or len(np.unique(ids)) != len(ids)
        or not isinstance(settings, dict)
        or not isinstance(detectors, dict)
    ):
        raise ValueError("A complete backdoor scan with unique sample IDs is required")

    active = list(settings.get("active_detectors", []))
    comparison_only = list(settings.get("comparison_only", []))
    configured_quarantine = tuple(quarantine_detectors)
    configured_review = tuple(review_detectors)
    if set(configured_quarantine) & set(configured_review):
        raise ValueError("A detector cannot have two defence actions")
    unknown = set(active) - set(configured_quarantine) - set(configured_review)
    quarantine = tuple(name for name in configured_quarantine if name in active)
    review = tuple(name for name in configured_review if name in active)
    if (
        not active
        or len(active) != len(set(active))
        or set(active) & set(comparison_only)
        or unknown
        or set(active) != set(quarantine) | set(review)
        or set(active) - set(detectors)
    ):
        raise ValueError("Every active detector must have one explicit defence action")

    aligned = {}
    for name in active:
        result = detectors[name]
        other_ids = np.asarray(result.get("sample_ids"))
        flags = np.asarray(result.get("flags"))
        if (
            other_ids.shape != ids.shape
            or not np.array_equal(other_ids, ids)
            or flags.shape != ids.shape
            or flags.dtype.kind != "b"
            or not isinstance(result.get("settings"), dict)
        ):
            raise ValueError(f"{name}: aligned flags and recorded settings are required")
        aligned[name] = flags.copy()

    expected_candidates = np.any(np.column_stack([aligned[name] for name in active]), axis=1)
    candidates = np.asarray(scan.get("candidate_flags"))
    if candidates.shape != ids.shape or candidates.dtype.kind != "b" or not np.array_equal(candidates, expected_candidates):
        raise ValueError("Candidate flags must equal the union of active detector flags")

    quarantine_flags = np.any(
        np.column_stack([aligned[name] for name in quarantine]), axis=1
    ) if quarantine else np.zeros(len(ids), dtype=bool)
    review_flags = np.any(
        np.column_stack([aligned[name] for name in review]), axis=1
    ) if review else np.zeros(len(ids), dtype=bool)
    review_only = review_flags & ~quarantine_flags

    actions = np.full(len(ids), "keep", dtype="<U16")
    reasons = np.full(len(ids), "not_flagged_by_active_backdoor_detectors", dtype="<U64")
    actions[review_only] = "human_review"
    reasons[review_only] = "experimental_feature_finding"
    actions[quarantine_flags] = "quarantine"
    reasons[quarantine_flags] = "contrast_patch_finding"

    return {
        "schema_version": "1.0",
        "policy_name": "backdoor_quarantine",
        "policy_version": "1.0",
        "status": "experimental",
        "sample_ids": ids.copy(),
        "actions": actions,
        "reasons": reasons,
        "training_keep_mask": actions == "keep",
        "quarantine_flags": actions == "quarantine",
        "human_review_flags": actions == "human_review",
        "summary": {action: int(np.sum(actions == action)) for action in ACTIONS},
        "settings": {
            "quarantine_detectors": list(quarantine),
            "review_detectors": list(review),
            "comparison_only_ignored": comparison_only,
            "quarantine_rule": "any configured quarantine-detector flag",
            "review_rule": "feature finding without a quarantine-detector flag",
            "exclude_all_non_keep_from_training": True,
            "source_scan_settings": deepcopy(settings),
            "source_detector_versions": {
                name: detectors[name].get("version") for name in active
            },
        },
    }

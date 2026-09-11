"""Dataset-neutral feature scanning with explicitly separated detector tracks."""

from __future__ import annotations

from importlib.metadata import version

import numpy as np

from poison_features import detector_input
from .backdoor.feature_pipeline import scan_backdoor_features
from .label_flip.class_distance import ClassDistance
from .label_flip.confident_learning import ConfidentLearning
from .label_flip.knn_label_agreement import KNNLabelAgreement
from .output_connector import detector_result


SUPPORTED_TRACKS = ("backdoor", "label_inconsistency")


def _review_summary(detectors: dict[str, dict], sample_ids: np.ndarray) -> dict:
    flags = np.column_stack([np.asarray(result["flags"], dtype=bool)
                             for result in detectors.values()])
    counts = flags.sum(axis=1).astype(np.int64)
    return dict(
        sample_ids=sample_ids.copy(),
        detectors=detectors,
        flag_count=counts,
        candidate_flags=counts >= 1,
        agreement_flags=counts >= 2,
        settings=dict(
            status="review_only_uncalibrated",
            candidate_rule="at least one detector flag",
            agreement_rule="at least two detector flags",
            limitation="Feature anomalies or label disagreement are not proof of poisoning.",
        ),
    )


def _scan_label_inconsistency(inputs, *, k: int, batch_size: int,
                              include_cleanlab: bool, progress=None) -> dict:
    stages = [
        ("knn_label_agreement", KNNLabelAgreement(k=k, threshold=.95, batch_size=batch_size),
         dict(k=k, threshold=.95, metric="cosine", threshold_status="provisional")),
        ("class_distance", ClassDistance(threshold=.1),
         dict(threshold=.1, metric="cosine", threshold_status="provisional")),
    ]
    if include_cleanlab:
        stages.append(("confident_learning", ConfidentLearning(folds=5, seed=2026),
                       dict(folds=5, seed=2026, threshold_status="cleanlab_estimated_noise_rate")))
    results = {}
    for index, (name, detector, settings) in enumerate(stages, 1):
        if progress:
            progress(f"label_inconsistency {index} of {len(stages)}: {name}")
        raw = (detector.analyze(inputs, progress=progress)
               if name == "confident_learning" else detector.analyze(inputs))
        settings.update(
            representation="feature_bundle_selected_representation",
            score_semantics="higher means more suspicious, not a poison probability",
        )
        result = detector_result(name, "1.0", raw, settings,
                                 expected_sample_ids=inputs.sample_ids)
        # Exact neighbour indices remain available in the Python result but the
        # duplicate string-ID matrix is unnecessary in normal JSON reports.
        result["evidence"].pop("neighbour_sample_ids", None)
        results[name] = result
    return _review_summary(results, np.asarray(inputs.sample_ids))


def scan_feature_bundle(
    bundle,
    *,
    tracks=("backdoor", "label_inconsistency"),
    representation: str = "raw",
    include_cleanlab: bool = False,
    k: int = 20,
    batch_size: int = 256,
    progress=None,
) -> dict:
    """Run feature-compatible detectors without dataset-name routing.

    Class labels are required by the current feature detectors. Results remain
    separated by track and are review candidates only; this function does not
    remove, relabel, train on, or evaluate any row and never reads poison truth.
    """
    requested = tuple(dict.fromkeys(tracks))
    if not requested or any(track not in SUPPORTED_TRACKS for track in requested):
        raise ValueError(f"tracks must contain only: {', '.join(SUPPORTED_TRACKS)}")
    labels = np.asarray(bundle.labels)
    if (labels.shape != (len(bundle.features),)
            or (labels.dtype.kind in "biuf" and np.any(labels == -1))
            or (labels.dtype.kind == "U" and np.any(labels == ""))
            or (labels.dtype.kind == "S" and np.any(labels == b""))):
        raise ValueError("Dataset-neutral feature scans require one known class label per row")
    inputs = detector_input(bundle, representation=representation, label_aware=True)
    outputs = {}
    if "backdoor" in requested:
        if progress:
            progress("backdoor: spectral signatures and activation clustering")
        outputs["backdoor"] = scan_backdoor_features(inputs, progress=progress)
    if "label_inconsistency" in requested:
        outputs["label_inconsistency"] = _scan_label_inconsistency(
            inputs, k=k, batch_size=batch_size,
            include_cleanlab=include_cleanlab, progress=progress,
        )
    dimensions = {
        "raw": bundle.features.shape[1],
        "scaled": bundle.scaled_features.shape[1],
        "reduced": bundle.reduced_features.shape[1],
    }
    dependencies = {name: version(name) for name in ("numpy", "scikit-learn")}
    if include_cleanlab:
        dependencies["cleanlab"] = version("cleanlab")
    return dict(
        schema_version="1.0",
        sample_ids=np.asarray(bundle.sample_ids).copy(),
        input=dict(
            dataset_name=bundle.dataset_name,
            modality=bundle.modality,
            encoder=bundle.encoder,
            samples=len(inputs.X),
            dimensions=int(dimensions[representation]),
            representation=representation,
        ),
        tracks=outputs,
        settings=dict(
            tracks=list(requested),
            include_cleanlab=bool(include_cleanlab),
            automatic_cleaning=False,
            truth_used_for_scoring=False,
            calibration="provisional; validate thresholds on an independent clean set",
            dependencies=dependencies,
        ),
    )

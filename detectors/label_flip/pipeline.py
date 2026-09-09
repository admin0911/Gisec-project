"""Label-flip connector on raw frozen embeddings and supplied labels.

The combined rule is a CIFAR-10 development preset, not a universal cutoff.
It recommends review, not automatic removal or relabeling.
"""
from importlib.metadata import version

import numpy as np

from .knn_label_agreement import KNNLabelAgreement
from .class_distance import ClassDistance
from .confident_learning import ConfidentLearning
from ..output_connector import detector_result


CONNECTOR_VERSION = "1.0"
CLASS_THRESHOLD = 0.024119124718243068
STRICT_CLASS_THRESHOLD = 0.029466908865236896


def combine_label_results(results):
    """Apply the fixed development rule; individual flags stay unchanged."""
    knn = results["knn"]
    distance = results["class_distance"]
    cl = results["confident_learning"]
    ids = np.asarray(knn["sample_ids"])
    for item in (knn, distance, cl):
        detector_result(item["detector_name"], item["version"], item,
                        item["settings"], expected_sample_ids=ids)
    if knn["settings"]["k"] != 20:
        raise ValueError("The fixed combination requires k=20")
    votes = {
        "knn": np.asarray(knn["scores"]) >= 19 / 20,
        "class_distance": ((np.asarray(distance["scores"]) > 0)
                           & (np.asarray(distance["scores"]) >= STRICT_CLASS_THRESHOLD)),
        "confident_learning": (np.asarray(cl["flags"])
                               & (np.asarray(cl["scores"]) >= 0.90)),
    }
    counts = np.sum(np.column_stack(list(votes.values())), axis=1)
    return {
        "schema_version": "1.0",
        "pipeline_version": CONNECTOR_VERSION,
        "sample_ids": ids.copy(),
        "detectors": results,
        "combination_votes": votes,
        "flag_count": counts,
        "review_flags": counts >= 2,
        "combination_settings": {
            "preset": "cifar10_development_strict",
            "minimum_votes": 2,
            "knn_minimum_disagreement": 19,
            "knn_k": 20,
            "class_distance_threshold": STRICT_CLASS_THRESHOLD,
            "confident_learning_minimum_score": 0.90,
            "confident_learning_requires_individual_flag": True,
            "status": "development_only_not_independently_validated",
        },
    }


def scan_label_flips(inputs, *, batch_size=256, progress=None):
    """Run all three detectors. Input is DetectorInput, never FeatureBundle.

    Use detector_input(bundle, representation='raw', label_aware=True).
    Raw means embeddings, not pixels. Full exact kNN scans can be expensive.
    Optional progress receives status strings. cleanlab must be installed.
    """
    # Check optional dependencies before starting the expensive kNN stage.
    dependencies = {name: version(name) for name in
                    ("numpy", "scikit-learn", "cleanlab")}
    results = {}
    stages = [
        ("knn", "knn_label_agreement", KNNLabelAgreement(batch_size=batch_size),
         {"k": 20, "threshold": 0.95, "minimum_disagreement": 19,
          "batch_size": batch_size, "metric": "cosine", "exclude_self": True}),
        ("class_distance", "class_distance", ClassDistance(CLASS_THRESHOLD),
         {"threshold": CLASS_THRESHOLD, "metric": "cosine",
          "own_centre": "leave_one_out"}),
        ("confident_learning", "confident_learning", ConfidentLearning(),
         {"folds": 5, "seed": 2026, "max_iter": 1000,
          "classifier": "LogisticRegression", "C": 1.0, "solver": "lbfgs",
          "tol": 1e-4, "quality_method": "self_confidence",
          "filter_by": "prune_by_noise_rate"}),
    ]
    for index, (key, name, detector, settings) in enumerate(stages, 1):
        if progress:
            progress(f"{index} of 3: {name}")
        raw = (detector.analyze(inputs, progress=progress) if key == "confident_learning"
               else detector.analyze(inputs))
        settings.update(representation="raw_embeddings", normalization="row_l2",
                        dependencies=dependencies.copy())
        results[key] = detector_result(name, "1.0", raw, settings,
                                       expected_sample_ids=inputs.sample_ids)
    return combine_label_results(results)

"""Minority feature-cluster heuristic inspired by Chen et al. (2018).

https://arxiv.org/abs/1811.03728. Uses per-class PCA and two-means; does not
implement the paper's full activation analysis or exclusionary reclassification.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd

from detectors.output_connector import detector_result
from .feature_common import (
    centered_class, integer_setting, positive_setting, validated_inputs,
)


class ActivationClusteringDetector:
    def __init__(self, n_components=10, min_class_size=10, min_cluster_size=3,
                 max_cluster_fraction=0.35, min_separation=3.0, random_state=0):
        integer_setting(n_components, "n_components", 1)
        integer_setting(min_class_size, "min_class_size", 4)
        integer_setting(min_cluster_size, "min_cluster_size", 2)
        integer_setting(random_state, "random_state", 0)
        if random_state > 2**32 - 1:
            raise ValueError("random_state must fit uint32")
        if not np.isfinite(max_cluster_fraction) or not 0 < max_cluster_fraction < 0.5:
            raise ValueError("max_cluster_fraction must be between 0 and 0.5")
        positive_setting(min_separation, "min_separation")
        self.settings = dict(n_components=n_components, min_class_size=min_class_size,
                             min_cluster_size=min_cluster_size,
                             max_cluster_fraction=float(max_cluster_fraction),
                             min_separation=float(min_separation), random_state=random_state)

    def analyze(self, inputs):
        X, y, ids = validated_inputs(inputs)
        settings = self.settings
        scores = np.zeros(len(X))
        flags = np.zeros(len(X), dtype=bool)
        evaluated = np.zeros(len(X), dtype=bool)
        cluster_ids = np.full(len(X), -1, dtype=np.int64)
        summaries = []
        for label in np.unique(y):
            rows = np.flatnonzero(y == label)
            summary = dict(label=label, count=len(rows), status="too_small")
            summaries.append(summary)
            if len(rows) < settings["min_class_size"]:
                continue
            centered = centered_class(X[rows])
            if not np.any(centered):
                summary["status"] = "constant_features"
                continue
            components = min(settings["n_components"], len(rows) - 1, X.shape[1])
            _, _, vectors = randomized_svd(
                centered, n_components=components, n_iter=7,
                random_state=settings["random_state"],
            )
            reduced = centered @ vectors.T
            if len(np.unique(reduced, axis=0)) < 2:
                summary["status"] = "constant_projection"
                continue
            model = KMeans(n_clusters=2, n_init=10, max_iter=300,
                           random_state=settings["random_state"]).fit(reduced)
            counts = np.bincount(model.labels_, minlength=2)
            cluster_ids[rows] = model.labels_
            if not counts.min():
                summary["status"] = "degenerate_clusters"
                continue
            minority = int(np.argmin(counts))
            fraction = float(counts[minority] / len(rows))
            radii = [np.mean(np.sum((reduced[model.labels_ == k]
                                    - model.cluster_centers_[k])**2, axis=1))
                     for k in range(2)]
            pooled_radius = np.sqrt(np.mean(radii))
            distance = np.linalg.norm(model.cluster_centers_[0] - model.cluster_centers_[1])
            separation = float(distance / max(float(pooled_radius), 1e-12))
            minority_rows = rows[model.labels_ == minority]
            scores[minority_rows] = separation
            evaluated[rows] = True
            accepted = (counts[minority] >= settings["min_cluster_size"]
                        and fraction <= settings["max_cluster_fraction"]
                        and separation >= settings["min_separation"])
            flags[minority_rows] = accepted
            summary.update(status="evaluated", cluster_counts=counts,
                           minority_cluster=minority, minority_fraction=fraction,
                           separation=separation, accepted=bool(accepted),
                           components=components)
        return detector_result(
            "activation_clustering", "0.1.0",
            dict(sample_ids=ids, scores=scores, flags=flags, evaluated=evaluated,
                 cluster_ids=cluster_ids, classes=summaries),
            dict(settings, label_aware=True,
                 score="minority-cluster centroid separation / pooled RMS radius",
                 flag_rule="minority count, fraction, and separation gates all pass",
                 calibration="provisional; requires independent clean validation",
                 limitation="Legitimate minority subclasses can also be flagged"),
            expected_sample_ids=ids,
        )

    def score(self, inputs):
        return self.analyze(inputs)["scores"]

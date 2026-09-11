"""Class-conditional spectral ranking with a provisional robust threshold.

Adapted from Tran, Li and Madry (2018), https://arxiv.org/abs/1811.00636.
Frozen embeddings need not encode a trigger. This is not a reproduction of
their trained-model defence or their poison-budget removal policy.
"""
import numpy as np
from sklearn.utils.extmath import randomized_svd

from detectors.output_connector import detector_result
from .feature_common import (
    centered_class, integer_setting, positive_setting, validated_inputs,
)


class SpectralSignatureDetector:
    def __init__(self, threshold=6.0, min_class_size=10, random_state=0):
        positive_setting(threshold, "threshold")
        integer_setting(min_class_size, "min_class_size", 3)
        integer_setting(random_state, "random_state", 0)
        if random_state > 2**32 - 1:
            raise ValueError("random_state must fit uint32")
        self.threshold = float(threshold)
        self.min_class_size = min_class_size
        self.random_state = random_state

    def analyze(self, inputs):
        X, y, ids = validated_inputs(inputs)
        scores = np.zeros(len(X))
        projection_energy = np.zeros(len(X))
        evaluated = np.zeros(len(X), dtype=bool)
        summaries = []
        for label in np.unique(y):
            rows = np.flatnonzero(y == label)
            summary = dict(label=label, count=len(rows), status="too_small")
            summaries.append(summary)
            if len(rows) < self.min_class_size:
                continue
            centered = centered_class(X[rows])
            if not np.any(centered):
                summary["status"] = "constant_features"
                continue
            _, singular, vectors = randomized_svd(
                centered, n_components=1, n_iter=7, random_state=self.random_state,
            )
            magnitude = np.abs(centered @ vectors[0])
            median = float(np.median(magnitude))
            mad = float(np.median(np.abs(magnitude - median)))
            # Fixed numerical floor, not a fitted poisoning fraction.
            scale = max(1.4826 * mad, 1e-12)
            scores[rows] = np.maximum(0, (magnitude - median) / scale)
            projection_energy[rows] = magnitude**2
            evaluated[rows] = True
            summary.update(status="evaluated", median_projection=median,
                           robust_scale=scale, leading_singular_value=float(singular[0]))
        return detector_result(
            "spectral_signature", "0.1.0",
            dict(sample_ids=ids, scores=scores,
                 flags=evaluated & (scores >= self.threshold),
                 evaluated=evaluated, projection_energy=projection_energy,
                 classes=summaries),
            dict(threshold=self.threshold, min_class_size=self.min_class_size,
                 random_state=self.random_state, label_aware=True,
                 score="positive robust deviation of absolute leading projection",
                 flag_rule="evaluated and score >= threshold",
                 calibration="provisional; requires independent clean validation",
                 limitation="Feature outliers are not proof of a backdoor"),
            expected_sample_ids=ids,
        )

    def score(self, inputs):
        return self.analyze(inputs)["scores"]

import unittest
import numpy as np
from poison_features import DetectorInput, FeatureBundle, detector_input
from detectors.label_flip.class_distance import ClassDistance


class ClassDistanceTests(unittest.TestCase):
    def test_clean_and_mislabeled(self):
        X = np.vstack([np.tile([1., 0.], (20, 1)), np.tile([0., 1.], (20, 1))])
        y = np.repeat([0, 1], 20)
        clean = DetectorInput(X, np.arange(40), y)
        self.assertFalse(ClassDistance().analyze(clean)["flags"].any())
        y = y.copy()
        y[0] = 1
        result = ClassDistance().analyze(DetectorInput(X, np.arange(40), y))
        np.testing.assert_array_equal(np.flatnonzero(result["flags"]), [0])
        self.assertEqual(result["alternative_labels"][0], 0)

    def test_leave_one_out_against_manual_distances(self):
        X = np.array([[1., 0.], [0., 1.], [0., 1.], [.2, 1.]])
        result = ClassDistance().analyze(DetectorInput(X, np.arange(4), np.array([0, 0, 1, 1])))
        self.assertAlmostEqual(result["own_distance"][0], 1.)
        unit = X / np.linalg.norm(X, axis=1, keepdims=True)
        other = unit[2:].mean(axis=0)
        other /= np.linalg.norm(other)
        self.assertAlmostEqual(result["alternative_distance"][0], 1 - other[0])

    def test_connector_alignment_and_metadata_exclusion(self):
        X = np.array([[1., .1], [1., .2], [.1, 1.], [.2, 1.]])
        bundle = FeatureBundle(X, X, X, np.array(["a", "a", "b", "b"]),
                               np.array(["id3", "id2", "id1", "id0"]), "image", "test")
        first = ClassDistance().analyze(detector_input(bundle, representation="raw", label_aware=True))
        bundle.is_poisoned = np.ones(4, dtype=bool)
        bundle.original_labels = np.array(["b", "b", "a", "a"])
        second = ClassDistance().score(detector_input(bundle, representation="raw", label_aware=True))
        np.testing.assert_array_equal(first["scores"], second)
        np.testing.assert_array_equal(first["sample_ids"], bundle.sample_ids)

    def test_validation(self):
        X = np.array([[1., 0.], [1., .1], [0., 1.], [.1, 1.]])
        ids = np.arange(4)
        y = np.array([0, 0, 1, 1])
        for inputs in [DetectorInput(X, ids), DetectorInput(X, ids, np.zeros(4)),
                       DetectorInput(X, ids, np.array([0, 0, 0, 1])),
                       DetectorInput(X * np.nan, ids, y),
                       DetectorInput(np.zeros_like(X), ids, y),
                       DetectorInput(X, np.zeros(4), y),
                       DetectorInput(X, ids, np.full(4, -1))]:
            with self.assertRaises(ValueError):
                ClassDistance().score(inputs)
        with self.assertRaises(ValueError):
            ClassDistance(threshold=-.1)


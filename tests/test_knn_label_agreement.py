import unittest
import numpy as np
from poison_features import DetectorInput, FeatureBundle, detector_input
from detectors.knn_label_agreement import KNNLabelAgreement


class KNNTests(unittest.TestCase):
    def make_inputs(self):
        X = np.vstack([np.tile([1., 0.], (25, 1)), np.tile([0., 1.], (25, 1))])
        y = np.repeat([0, 1], 25)
        y[0] = 1
        bundle = FeatureBundle(X, X, X, y, np.arange(100, 150), 'image', 'test')
        return bundle, detector_input(bundle, representation='raw', label_aware=True)

    def test_wrong_label_and_self_exclusion_with_duplicates(self):
        _, inputs = self.make_inputs()
        result = KNNLabelAgreement().analyze(inputs)
        self.assertEqual(result['scores'][0], 1)
        np.testing.assert_array_equal(np.flatnonzero(result['flags']), [0])
        self.assertFalse(np.any(result['neighbour_indices'] == np.arange(50)[:, None]))
        np.testing.assert_array_equal(result['sample_ids'], inputs.sample_ids)
        np.testing.assert_array_equal(result['neighbour_sample_ids'], inputs.sample_ids[result['neighbour_indices']])

    def test_evaluation_metadata_does_not_change_scores(self):
        bundle, inputs = self.make_inputs()
        detector = KNNLabelAgreement()
        before = detector.score(inputs)
        bundle.is_poisoned = np.ones(50, dtype=bool)
        bundle.original_labels = 1 - bundle.labels
        after = detector.score(detector_input(bundle, representation='raw', label_aware=True))
        np.testing.assert_array_equal(before, after)

    def test_small_input_and_exact_threshold(self):
        inputs = DetectorInput(np.ones((21, 2)), np.arange(21), np.array([0, 0] + [1] * 19))
        result = KNNLabelAgreement(batch_size=3).analyze(inputs)
        self.assertEqual(result['scores'][0], 19 / 20)
        self.assertTrue(result['flags'][0])
        with self.assertRaises(ValueError):
            KNNLabelAgreement().score(DetectorInput(inputs.X[:20], inputs.sample_ids[:20], inputs.y[:20]))

    def test_invalid_inputs(self):
        _, inputs = self.make_inputs()
        for bad in [DetectorInput(inputs.X, inputs.sample_ids),
                    DetectorInput(np.zeros_like(inputs.X), inputs.sample_ids, inputs.y),
                    DetectorInput(inputs.X, np.zeros(50), inputs.y),
                    DetectorInput(inputs.X, inputs.sample_ids, np.full(50, -1)),
                    DetectorInput(inputs.X * np.nan, inputs.sample_ids, inputs.y)]:
            with self.subTest(bad=type(bad)), self.assertRaises(ValueError):
                KNNLabelAgreement().score(bad)

    def test_agrees_with_brute_force_cosine(self):
        X = np.random.default_rng(12).normal(size=(80, 8))
        y = np.arange(80) % 3
        unit = X / np.linalg.norm(X, axis=1, keepdims=True)
        sim = unit @ unit.T
        np.fill_diagonal(sim, -np.inf)
        nearest = np.argsort(-sim, axis=1)[:, :20]
        expected = np.mean(y[nearest] != y[:, None], axis=1)
        actual = KNNLabelAgreement(batch_size=7).score(DetectorInput(X, np.arange(80), y))
        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()

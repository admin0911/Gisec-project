from pathlib import Path
import tempfile
import unittest

import numpy as np

from poison_features import FeatureBundle
from detectors.label_flip.feature_inputs import paired_feature_inputs, load_paired_feature_inputs


def bundle(name, dimension):
    X = np.random.default_rng(2).normal(size=(6, dimension)).astype('float32')
    return FeatureBundle(X, X.copy(), X[:, :2].copy(), np.array([0, 0, 1, 1, 2, 2]),
                         np.array(['b','a','d','c','f','e']), 'image', name, 'cifar10')


class PairedFeatureTests(unittest.TestCase):
    def setUp(self):
        self.resnet = bundle('resnet18', 512)
        self.dino = bundle('dinov2_vits14', 384)

    def test_roundtrip_and_truth_excluded(self):
        self.resnet.is_poisoned = np.ones(6, dtype=bool)
        self.dino.is_poisoned = np.zeros(6, dtype=bool)
        self.resnet.original_labels = np.full(6, 9)
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a.npz', Path(directory) / 'b.npz'
            self.resnet.save(a); self.dino.save(b)
            inputs = load_paired_feature_inputs(a, b)
        for name, source in [('resnet18', self.resnet), ('dinov2_vits14', self.dino)]:
            np.testing.assert_array_equal(inputs[name].X, source.features)
            np.testing.assert_array_equal(inputs[name].sample_ids, source.sample_ids)
            np.testing.assert_array_equal(inputs[name].y, source.labels)
            self.assertEqual(set(vars(inputs[name])), {'X','sample_ids','y'})

    def test_reordered_ids_are_rejected(self):
        self.dino.sample_ids = self.dino.sample_ids[::-1]
        with self.assertRaisesRegex(ValueError, 'same order'):
            paired_feature_inputs(self.resnet, self.dino)

    def test_changed_labels_are_rejected(self):
        self.dino.labels[0] = 8
        with self.assertRaisesRegex(ValueError, 'identical supplied labels'):
            paired_feature_inputs(self.resnet, self.dino)

    def test_teammate_dinov2_encoder_name_is_accepted(self):
        self.dino.encoder = 'dinov2'
        result = paired_feature_inputs(self.resnet, self.dino)
        np.testing.assert_array_equal(result['dinov2_vits14'].X, self.dino.features)

    def test_wrong_encoder_or_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            paired_feature_inputs(self.dino, self.resnet)
        self.dino.features = self.dino.features[:, :32]
        with self.assertRaises(ValueError):
            paired_feature_inputs(self.resnet, self.dino)

    def test_duplicate_ids_and_invalid_features_are_rejected(self):
        self.dino.sample_ids[0] = self.dino.sample_ids[1]
        with self.assertRaises(ValueError):
            paired_feature_inputs(self.resnet, self.dino)
        self.dino = bundle('dinov2_vits14', 384)
        self.dino.features[0] = 0
        with self.assertRaises(ValueError):
            paired_feature_inputs(self.resnet, self.dino)


if __name__ == '__main__':
    unittest.main()

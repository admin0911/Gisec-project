import tempfile
import unittest
from pathlib import Path

import numpy as np

from poison_features import FeatureBundle
from poison_features.preprocessing import prepare_representations


class FeatureBundleTests(unittest.TestCase):
    def test_small_pca_is_safe_and_finite(self):
        features = np.arange(20 * 10, dtype=np.float32).reshape(20, 10)
        scaled, reduced = prepare_representations(features)
        self.assertEqual(scaled.shape, (20, 10))
        self.assertEqual(reduced.shape, (20, 10))
        self.assertTrue(np.isfinite(reduced).all())

    def test_round_trip_preserves_alignment(self):
        features = np.random.default_rng(0).normal(size=(8, 5)).astype(np.float32)
        scaled, reduced = prepare_representations(features)
        bundle = FeatureBundle(
            features, scaled, reduced, np.arange(8), np.arange(100, 108),
            "image", "test", dataset_name="small",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.npz"
            bundle.save(path)
            loaded = FeatureBundle.load(path)
        np.testing.assert_array_equal(loaded.sample_ids, np.arange(100, 108))
        np.testing.assert_array_equal(loaded.labels, np.arange(8))
        self.assertEqual(loaded.reduced_feature_dim, 5)


if __name__ == "__main__":
    unittest.main()


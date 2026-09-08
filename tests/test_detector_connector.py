import unittest

import numpy as np

from poison_features import FeatureBundle, detector_input


class DetectorConnectorTests(unittest.TestCase):
    def test_unsupervised_input_excludes_labels(self):
        values = np.ones((3, 4), dtype=np.float32)
        bundle = FeatureBundle(values, values, values[:, :2], np.array([1, 2, 3]), np.array(["a", "b", "c"]), "test", "test")
        inputs = detector_input(bundle, representation="reduced")
        self.assertIsNone(inputs.y)
        self.assertEqual(inputs.X.shape, (3, 2))

    def test_label_aware_input_keeps_labels_separate(self):
        values = np.ones((3, 4), dtype=np.float32)
        bundle = FeatureBundle(values, values, values[:, :2], np.array([1, 2, 3]), np.array(["a", "b", "c"]), "test", "test")
        inputs = detector_input(bundle, label_aware=True)
        np.testing.assert_array_equal(inputs.y, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()

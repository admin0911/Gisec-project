import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.utils.data import TensorDataset

from poison_features import load_image_inputs
from poison_features.attacks import poison_dataset
from experiments.scan_mnist_pixels import pixel_input, save_results
from detectors.output_connector import detector_result


class MNISTPixelTests(unittest.TestCase):
    def test_post_attack_labels_and_pixel_scale_are_preserved(self):
        images = torch.zeros(100, 1, 28, 28)
        images[:, :, 10:15, 10:15] = 1
        images[:, :, 8, 8] = .5
        clean = TensorDataset(images, torch.arange(100) % 10)
        attacked = poison_dataset(clean, 'label_flip', poison_rate=.05, seed=0)
        ids = np.array([f'mnist-train:{i}' for i in range(100)])
        inputs = pixel_input(load_image_inputs(attacked, sample_ids=ids))
        self.assertEqual(inputs.X.shape, (100, 784))
        np.testing.assert_array_equal(inputs.X, images.numpy().reshape(100, 784))
        np.testing.assert_array_equal(inputs.sample_ids, ids)
        np.testing.assert_array_equal(inputs.y, attacked.metadata.current_labels)
        self.assertEqual(int(np.sum(inputs.y != np.arange(100) % 10)), 5)
        self.assertFalse(hasattr(inputs, 'is_poisoned'))
        self.assertFalse(hasattr(inputs, 'original_labels'))

    def test_rejects_non_mnist_dimensions(self):
        bundle = load_image_inputs(TensorDataset(torch.ones(3, 3, 32, 32), torch.arange(3)))
        with self.assertRaisesRegex(ValueError, 'MNIST'):
            pixel_input(bundle)

    def test_saved_connector_arrays_round_trip_without_truth(self):
        ids = np.array(['a', 'b'])
        raw = dict(sample_ids=ids, scores=np.array([.2, .98]), flags=np.array([False, True]))
        result = detector_result('knn', '1.0', raw, {'threshold': .95}, expected_sample_ids=ids)
        with tempfile.TemporaryDirectory() as directory:
            save_results(Path(directory), {'knn': result})
            with np.load(Path(directory) / 'knn.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['sample_ids'], ids)
                np.testing.assert_array_equal(saved['scores'], raw['scores'])
                np.testing.assert_array_equal(saved['flags'], raw['flags'])
                self.assertNotIn('is_poisoned', saved.files)


if __name__ == '__main__':
    unittest.main()

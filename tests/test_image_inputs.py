import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from poison_features import load_image_inputs, load_image_dataset
from poison_features.attacks import poison_dataset


class ImageInputTests(unittest.TestCase):
    def test_clean_pixels_have_stable_ids(self):
        dataset = load_image_dataset("mnist", train=True, download=False)
        inputs = load_image_inputs(dataset, sample_ids=["mnist:0", "mnist:1"], limit=2)
        self.assertEqual(inputs.images.shape, (2, 1, 28, 28))
        np.testing.assert_array_equal(inputs.sample_ids, ["mnist:0", "mnist:1"])

    def test_backdoor_returns_modified_pixels(self):
        dataset = [(torch.zeros(1, 8, 8), 0) for _ in range(4)]
        poisoned = poison_dataset(dataset, "backdoor", poison_rate=0.5, seed=0)
        inputs = load_image_inputs(poisoned, limit=4)
        for index, is_poisoned in enumerate(poisoned.metadata.is_poisoned):
            expected = 1.0 if is_poisoned else 0.0
            self.assertEqual(float(inputs.images[index, :, -1, -1].max()), expected)
        self.assertEqual(inputs.images.shape, (4, 1, 8, 8))

    def test_image_bundle_round_trip(self):
        dataset = [(torch.zeros(1, 8, 8), 1)]
        inputs = load_image_inputs(dataset, sample_ids=["row-0"])
        with TemporaryDirectory() as directory:
            path = Path(directory) / "images.npz"
            inputs.save(path)
            loaded = type(inputs).load(path)
        np.testing.assert_array_equal(loaded.images, inputs.images)
        np.testing.assert_array_equal(loaded.sample_ids, inputs.sample_ids)


if __name__ == "__main__":
    unittest.main()

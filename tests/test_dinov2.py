import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from poison_features.image import DINOv2ImageEncoder
from poison_features.universal import UniversalFeatureExtractor


class FakeBackbone(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=(1, 2, 3))[:, None].repeat(1, 384)


class DinoTests(unittest.TestCase):
    @patch('torch.hub.load', return_value=FakeBackbone())
    def test_order_range_and_batching(self, load):
        encoder = DINOv2ImageEncoder(device='cpu')
        data = [(torch.zeros(3, 32, 32), 0), (torch.ones(3, 32, 32), 1)]
        progress = []
        features = encoder.extract(data, batch_size=1, progress=lambda n, t: progress.append((n, t)))
        self.assertEqual(features.shape, (2, 384))
        self.assertLess(features[0, 0], features[1, 0])
        self.assertEqual(progress, [(1, 2), (2, 2)])
        np.testing.assert_allclose(encoder.extract(data, batch_size=2), features)
        np.testing.assert_allclose(encoder._prepare(Image.new('RGB', (32, 32))), encoder._prepare(data[0][0]))
        with self.assertRaises(ValueError):
            encoder.extract([torch.full((3,32,32), 2.0)])
        with self.assertRaises(ValueError):
            encoder.extract([])
        self.assertIn(DINOv2ImageEncoder.revision, load.call_args.args[0])

    @patch('poison_features.image.DINOv2ImageEncoder')
    def test_universal_connector_preserves_labels_and_ids(self, cls):
        cls.return_value.extract.return_value = np.random.default_rng(4).normal(size=(10,384)).astype('float32')
        cls.return_value.device = 'cpu'
        cls.return_value.revision = DINOv2ImageEncoder.revision
        ids = np.array([f'id-{i}' for i in range(10)])
        bundle = UniversalFeatureExtractor().extract_images(
            [None] * 10, labels=np.arange(10), sample_ids=ids,
            encoder='dinov2', visual=False)
        self.assertEqual(bundle.encoder, 'dinov2')
        self.assertEqual(bundle.features.shape, (10,384))
        np.testing.assert_array_equal(bundle.sample_ids, ids)
        np.testing.assert_array_equal(bundle.labels, np.arange(10))


if __name__ == '__main__':
    unittest.main()

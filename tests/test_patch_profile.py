import unittest
import numpy as np
from detectors.backdoor.web_patch import scan_patch
from poison_features.image_inputs import ImageInputBundle


class PatchProfileTests(unittest.TestCase):
    def test_dataset_profiles_keep_connector_alignment(self):
        rng = np.random.default_rng(12)
        pixels = rng.integers(0, 180, (200, 3, 8, 8), dtype=np.uint8).astype(np.float32)/255
        labels = np.arange(200)%10
        pixels[:15, :, :3, :3] = 1
        labels[:15] = 7
        images = ImageInputBundle(pixels, labels, np.arange(200))
        assessment = dict(sample_ids=images.sample_ids, flags=np.zeros(200, dtype=bool))
        cifar, ui = scan_patch(images, assessment, lambda *args: None, dataset='cifar10')
        mnist, _ = scan_patch(images, assessment, lambda *args: None)
        self.assertTrue(cifar['flags'][:15].all())
        self.assertFalse(mnist['flags'].any())
        self.assertEqual(cifar['settings']['profile'], 'cifar-bright-patch-v1')
        self.assertEqual(cifar['settings']['intensity_bins'], 256)
        self.assertEqual(mnist['settings']['min_count'], 30)
        self.assertEqual(mnist['settings']['intensity_bins'], 4)
        np.testing.assert_array_equal(cifar['sample_ids'], images.sample_ids)
        self.assertEqual(ui['flagged'], int(cifar['flags'].sum()))

"""Leila: verify ASR denominator, trigger application and test immutability."""
import unittest
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset
from training.connector import training_input
from training.image_backdoor import evaluate_patch_model


class TriggerModel(nn.Module):
    def forward(self, x):
        logits = torch.zeros((len(x), 10))
        logits[:, 0] = 2 * (x[:, :, -3:, -3:].mean((1, 2, 3)) == 1)
        logits[:, 1] = 1
        return logits


class PatchASRTests(unittest.TestCase):
    def test_uses_only_non_target_test_rows_and_leaves_pixels_unchanged(self):
        x = torch.zeros((4, 3, 32, 32))
        test = training_input(TensorDataset(x, torch.tensor([0, 1, 2, 1])),
                              ['test:0', 'test:1', 'test:2', 'test:3'], dataset_version='test', split='test')
        metrics, saved = evaluate_patch_model(TriggerModel(), test,
            dict(target_label=0, size=3, value=1.), np.array([0, 1, 0, 1]), batch_size=2)
        self.assertEqual(metrics['non_target_samples'], 3)
        self.assertEqual(metrics['asr_non_target'], 1.)
        self.assertAlmostEqual(metrics['untriggered_target_rate'], 1/3)
        self.assertEqual(metrics['conditional_asr'], 1.)
        self.assertEqual(saved['sample_ids'].tolist(), ['test:1', 'test:2', 'test:3'])
        self.assertEqual(int(torch.count_nonzero(x)), 0)


if __name__ == '__main__':
    unittest.main()

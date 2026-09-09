import tempfile
import unittest
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset
from training import training_input, TrainConfig
from experiments.compare_training import compare_training, SuppliedLabels


class ComparisonTests(unittest.TestCase):
    def test_three_fresh_models_and_current_labels(self):
        clean = TensorDataset(torch.tensor([[-1.], [1.], [-2.], [2.]]), torch.tensor([0, 1, 0, 1]))
        poisoned = SuppliedLabels(clean, np.array([1, 1, 0, 1]))
        ids = np.array(['a', 'b', 'c', 'd'])
        decisions = {'sample_ids': ids, 'actions': ['quarantine', 'keep', 'human_review', 'keep']}
        test = training_input(clean, ['t1', 't2', 't3', 't4'], dataset_version='test', split='test')
        initial = []
        def factory():
            model = nn.Linear(1, 2)
            initial.append(model.weight.detach().clone())
            return model
        with tempfile.TemporaryDirectory() as root:
            result = compare_training(clean, poisoned, ids, decisions, test,
                model_factory=factory, model_name='test_linear', num_classes=2,
                output_root=root, dataset_version='testcase',
                config=TrainConfig(epochs=1, batch_size=2), progress=None)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual([r['training_samples'] for r in result['runs'].values()], [4, 4, 2])
            self.assertEqual(len(initial), 3)
            self.assertTrue(all(torch.equal(initial[0], weights) for weights in initial))
            self.assertEqual(poisoned[0][1], 1)
            self.assertEqual(clean[0][1].item(), 0)
            kept = np.load(result['runs']['after_cleaning']['artifacts']['training_ids'])
            self.assertEqual(kept.tolist(), ['b', 'd'])

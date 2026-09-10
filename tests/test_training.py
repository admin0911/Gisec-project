import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset
from cleaning.label_flip import partition_dataset
from training import training_input, TrainConfig, train_classifier


class TrainingTests(unittest.TestCase):
    def test_partition_to_real_training_and_saved_predictions(self):
        data = TensorDataset(torch.tensor([[-1.], [1.], [-2.], [2.]]),
                             torch.tensor([0, 1, 0, 1]))
        views = partition_dataset(data, ['a', 'b', 'c', 'd'], {
            'sample_ids': ['a', 'b', 'c', 'd'],
            'actions': ['keep', 'keep', 'human_review', 'quarantine']})
        train = training_input(views['keep'], views['keep'].sample_ids,
                               dataset_version='v1', split='train')
        test = training_input(data, ['t1', 't2', 't3', 't4'],
                              dataset_version='test1', split='test')
        with tempfile.TemporaryDirectory() as root:
            report = train_classifier(train, test, model_factory=lambda: nn.Linear(1, 2),
                model_name='linear_test', num_classes=2, output_root=root,
                config=TrainConfig(epochs=2, batch_size=2))
            self.assertEqual(report['training_samples'], 2)
            ids = np.load(report['artifacts']['training_ids'])
            self.assertEqual(ids.tolist(), ['a', 'b'])
            with np.load(report['artifacts']['predictions']) as saved:
                self.assertEqual(saved['sample_ids'].tolist(), test.sample_ids.tolist())
                self.assertEqual(report['metrics']['accuracy'],
                    float(np.mean(saved['predictions'] == saved['labels'])))
            self.assertEqual(sum(report['metrics']['class_support']), 4)
            json.loads(Path(report['artifacts']['report']).read_text())
            self.assertTrue(Path(report['artifacts']['checkpoint']).is_file())

    def test_overlap_rejected_before_model_creation(self):
        data = TensorDataset(torch.zeros(2, 1), torch.tensor([0, 1]))
        train = training_input(data, ['a', 'b'], dataset_version='v', split='train')
        test = training_input(data, ['a', 'c'], dataset_version='v', split='test')
        with self.assertRaisesRegex(ValueError, 'overlap'):
            train_classifier(train, test, model_factory=lambda: self.fail('factory called'),
                             model_name='test', num_classes=2, output_root='unused')

    def test_empty_duplicate_and_view_order_rejected(self):
        with self.assertRaises(ValueError):
            training_input([], [], dataset_version='v', split='train')
        with self.assertRaises(ValueError):
            training_input([1, 2], ['a', 'a'], dataset_version='v', split='train')
        view = partition_dataset([1, 2], ['a', 'b'],
            {'sample_ids': ['a', 'b'], 'actions': ['keep', 'keep']})['keep']
        with self.assertRaises(ValueError):
            training_input(view, ['b', 'a'], dataset_version='v', split='train')


if __name__ == '__main__':
    unittest.main()

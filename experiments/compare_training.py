"""Compare clean, unfiltered poisoned, and filtered training with one trainer."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from cleaning.label_flip import partition_dataset
from poison_features import load_image_dataset
from training import training_input, train_classifier, TrainConfig
from training.models import small_cnn


class SuppliedLabels:
    """Label-flip benchmark view: same pixels, saved current labels."""
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = np.asarray(labels)
        if self.labels.shape != (len(dataset),) or self.labels.dtype.kind not in 'iu':
            raise ValueError('One integer supplied label per image required')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index][0], int(self.labels[index])


def compare_training(clean_dataset, poisoned_dataset, sample_ids, decisions, test,
                     *, model_factory, model_name, num_classes, output_root,
                     dataset_version, config=None, progress=print):
    """Clean reference is evaluation-only; it never enters cleaning decisions.

    Both source datasets must describe the same rows in sample_ids order.
    The caller owns pixel/label provenance. Only keep enters filtered training.
    Each arm constructs a fresh model with the same training seed/settings.
    """
    ids = np.asarray(sample_ids)
    if len(clean_dataset) != len(poisoned_dataset):
        raise ValueError('Clean and poisoned baselines must cover the same rows')
    config = config or TrainConfig()
    views = partition_dataset(poisoned_dataset, ids, decisions)
    datasets = {'clean_baseline': clean_dataset, 'poisoned_baseline': poisoned_dataset,
                'after_cleaning': views['keep']}
    inputs = {name: training_input(data, views['keep'].sample_ids if name == 'after_cleaning' else ids,
               dataset_version=f'{dataset_version}/{name}', split='train')
              for name, data in datasets.items()}
    if any(set(value.sample_ids.tolist()) & set(test.sample_ids.tolist()) for value in inputs.values()):
        raise ValueError('Training and test IDs overlap')
    out = Path(output_root) / datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': '1.0', 'status': 'running',
              'policy': 'exclude_all_non_keep', 'dataset_version': dataset_version,
              'excluded_samples': len(ids) - len(views['keep']), 'runs': {}}
    path = out / 'comparison.json'
    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    save()
    for index, (name, value) in enumerate(inputs.items(), 1):
        if progress:
            progress(f'Model {index}/3: {name} ({len(value.dataset):,} training images)')
        report['runs'][name] = train_classifier(value, test,
            model_factory=model_factory, model_name=model_name, num_classes=num_classes,
            output_root=out / name, config=config, progress=progress)
        save()
    report['status'] = 'complete'
    report['accuracy_change_cleaning_vs_poisoned'] = (
        report['runs']['after_cleaning']['metrics']['accuracy'] -
        report['runs']['poisoned_baseline']['metrics']['accuracy'])
    save()
    lines = ['# Training comparison', '', '| Training data | Images | Test accuracy |',
             '|---|---:|---:|']
    for name, run in report['runs'].items():
        lines.append(f'| {name} | {run["training_samples"]:,} | {run["metrics"]["accuracy"]:.2%} |')
    lines += ['', 'Same fixed epochs/settings/seed and clean test split; fresh model per arm.',
              'Filtering changes data size and therefore optimizer steps per epoch.',
              'These results are specific to this model and configuration.']
    (out / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    if progress:
        progress(f'Saved comparison: {path.resolve()}')
    return report


def load_saved_case(benchmark_dir, rate, seed, data_root):
    """Load the existing CIFAR-10 label-flip benchmark, without rerunning scans."""
    root = Path(benchmark_dir)
    prefix = f'rate{rate:.2f}-seed{seed}'
    with np.load(root / f'{prefix}-resnet18-scores.npz', allow_pickle=False) as data:
        ids, labels = data['sample_ids'].copy(), data['labels'].copy()
    with np.load(root / f'{prefix}-actions.npz', allow_pickle=False) as data:
        if not np.array_equal(data['sample_ids'], ids):
            raise ValueError('Saved decisions and labels have mismatched IDs')
        actions = data['actions'].copy()
    original = load_image_dataset('cifar10', root=data_root, train=True, download=False)
    indices = []
    for value in ids.tolist():
        if not isinstance(value, str) or not value.startswith('cifar10-train:'):
            raise ValueError('Expected CIFAR-10 training sample IDs')
        index = int(value.split(':')[1])
        if not 0 <= index < len(original):
            raise ValueError('Invalid saved sample index')
        indices.append(index)
    clean = Subset(original, indices)
    clean_labels = np.asarray(original.targets)[indices]
    mask = np.zeros(len(ids), dtype=bool)
    mask[np.random.default_rng(seed).choice(len(ids), size=int(round(len(ids)*rate)), replace=False)] = True
    expected = clean_labels.copy()
    expected[mask] = (expected[mask] + 1) % 10
    if not np.array_equal(labels, expected):
        raise ValueError('Saved labels do not match this benchmark configuration')
    # This check verifies benchmark provenance only. Actions above remain untouched.
    return clean, SuppliedLabels(clean, labels), ids, {'sample_ids': ids, 'actions': actions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-dir', required=True)
    parser.add_argument('--rate', type=float, choices=[.01, .03, .05, .10], default=.05)
    parser.add_argument('--attack-seed', type=int, default=20260911)
    parser.add_argument('--train-seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--output-root', default='artifacts/training_comparison')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    clean, poisoned, ids, decisions = load_saved_case(args.benchmark_dir, args.rate,
                                                     args.attack_seed, args.data_root)
    test_data = load_image_dataset('cifar10', root=args.data_root, train=False, download=False)
    test = training_input(test_data, [f'cifar10-test:{i}' for i in range(len(test_data))],
                          dataset_version='cifar10-official-test', split='test')
    compare_training(clean, poisoned, ids, decisions, test, model_factory=small_cnn,
        model_name='small_cnn_v1', num_classes=10, output_root=args.output_root,
        dataset_version=f'cifar10-rate{args.rate:.2f}-seed{args.attack_seed}',
        config=TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                           seed=args.train_seed, device=args.device))


if __name__ == '__main__':
    main()

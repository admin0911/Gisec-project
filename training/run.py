"""Train from a saved post-attack image bundle and complete cleaning manifest."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset
from poison_features import ImageInputBundle, load_image_dataset
from cleaning.label_flip import partition_dataset
from .connector import training_input
from .models import small_cnn
from .trainer import TrainConfig, train_classifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', required=True)
    parser.add_argument('--decisions', required=True, help='decisions.json from save_decisions')
    parser.add_argument('--dataset', required=True, choices=['cifar10', 'mnist'])
    parser.add_argument('--dataset-version', required=True)
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--output-root', default='artifacts/training')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    images = ImageInputBundle.load(args.images)
    expected_channels = 3 if args.dataset == 'cifar10' else 1
    if images.images.shape[1] != expected_channels:
        raise ValueError('Image channels do not match selected dataset')
    if images.labels.ndim != 1 or images.labels.dtype.kind not in 'iu':
        raise ValueError('Saved labels must be integer class indices')
    source = TensorDataset(torch.from_numpy(images.images).float(),
                           torch.from_numpy(images.labels.astype(np.int64)))
    decisions = json.loads(Path(args.decisions).read_text(encoding='utf-8'))
    views = partition_dataset(source, images.sample_ids, decisions)
    kept = views['keep']
    train = training_input(kept, kept.sample_ids,
                           dataset_version=args.dataset_version, split='train')
    test_data = load_image_dataset(args.dataset, root=args.data_root,
                                   train=False, download=False)
    test = training_input(test_data, [f'{args.dataset}-test:{i}' for i in range(len(test_data))],
                          dataset_version=f'{args.dataset}-official-test', split='test')
    report = train_classifier(train, test,
        model_factory=lambda: small_cnn(expected_channels), model_name='small_cnn_v1',
        num_classes=10, output_root=args.output_root,
        config=TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                           seed=args.seed, device=args.device), progress=print)
    print(json.dumps(report['metrics'], indent=2))
    print('Saved:', report['artifacts']['report'])


if __name__ == '__main__':
    main()

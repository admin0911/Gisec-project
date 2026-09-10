"""Scan MNIST pixels with the three existing label-flip detectors.

python -m experiments.scan_mnist_pixels --poison-rate 0.05 --clean-baseline
No pretrained encoder, CIFAR thresholds, test images, or truth-based selection.
"""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits
from torch.utils.data import Subset

from poison_features import load_image_dataset, load_image_inputs
from poison_features.attacks import poison_dataset
from poison_features.detector import image_detector_input
from detectors.label_flip.knn_label_agreement import KNNLabelAgreement
from detectors.label_flip.class_distance import ClassDistance
from detectors.label_flip.confident_learning import ConfidentLearning
from detectors.output_connector import detector_result, to_jsonable
from experiments.run_knn import evaluation

ROOT = Path(__file__).resolve().parents[1]


def pixel_input(bundle):
    """Use the image connector, then flatten; do not package pixels as embeddings."""
    if bundle.images.shape[1:] != (1, 28, 28):
        raise ValueError('Expected MNIST images with shape N x 1 x 28 x 28.')
    # The image connector already returns float pixels in [0, 1]. Do not divide twice.
    return image_detector_input(bundle, label_aware=True)


def scan_pixels(inputs, *, knn_threshold=0.95, class_threshold=0.1, folds=5,
                seed=2026, progress=print):
    """The scoring boundary accepts only pixels, supplied labels and sample IDs."""
    stages = [
        ('knn', KNNLabelAgreement(k=20, threshold=knn_threshold),
         dict(k=20, threshold=knn_threshold, metric='cosine', exclude_self=True)),
        ('class_distance', ClassDistance(threshold=class_threshold),
         dict(threshold=class_threshold, metric='cosine', own_centre='leave_one_out')),
        ('confident_learning', ConfidentLearning(folds=folds, seed=seed),
         dict(folds=folds, seed=seed, classifier='LogisticRegression', C=1.0,
              max_iter=1000, filter_by='prune_by_noise_rate', probabilities='out_of_fold')),
    ]
    results, seconds = {}, {}
    for index, (name, detector, settings) in enumerate(stages, 1):
        progress(f'{index} of 3: {name}')
        start = time.perf_counter()
        raw = detector.analyze(inputs, progress=progress) if name == 'confident_learning' else detector.analyze(inputs)
        settings.update(representation='mnist_pixels_784', normalization='pixel_0_1_then_row_l2',
                        threshold_status='provisional_not_mnist_calibrated')
        results[name] = detector_result(name, '1.0', raw, settings, expected_sample_ids=inputs.sample_ids)
        seconds[name] = time.perf_counter() - start
        progress(f'  {int(raw["flags"].sum()):,} flagged; {seconds[name]:.1f} seconds')
    return results, seconds


def save_results(folder, results):
    """Save connector arrays separately from descriptive metadata and evaluation truth."""
    for name, result in results.items():
        arrays = {key: result[key] for key in ('sample_ids', 'scores', 'flags')}
        arrays.update({key: value for key, value in result['evidence'].items()
                       if key != 'neighbour_sample_ids'})
        np.savez_compressed(folder / f'{name}.npz', **arrays)
        metadata = {key: result[key] for key in ('detector_name', 'version', 'settings')}
        (folder / f'{name}.json').write_text(json.dumps(to_jsonable(metadata), indent=2), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poison-rate', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=0, help='Attack seed; also chooses the quick-test subset.')
    parser.add_argument('--limit', type=int, default=None, help='Optional random subset for quick tests; default all 60,000.')
    parser.add_argument('--clean-baseline', action='store_true')
    parser.add_argument('--knn-threshold', type=float, default=0.95)
    parser.add_argument('--class-threshold', type=float, default=0.1)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'artifacts/mnist_pixel_scans')
    args = parser.parse_args(argv)
    if not 0 <= args.poison_rate <= 1 or args.seed < 0 or args.threads < 1:
        parser.error('Use poison rate in [0,1], nonnegative seed and positive thread count.')
    dependencies = {name: version(name) for name in ('numpy', 'scikit-learn', 'cleanlab', 'torch', 'torchvision')}
    dataset = load_image_dataset('mnist', root=args.data_root, train=True, download=False)
    indices = np.arange(len(dataset))
    if args.limit is not None:
        if not 21 <= args.limit <= len(dataset):
            parser.error('limit must be between 21 and 60,000.')
        indices = np.sort(np.random.default_rng(args.seed).choice(indices, args.limit, replace=False))
        dataset = Subset(dataset, indices.tolist())
    ids = np.array([f'mnist-train:{i}' for i in indices])
    output = args.output_root / datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(dataset='mnist', split='train', samples=len(dataset), feature_dimensions=784,
                  representation='normalized_pixels_no_pretrained_model', attack_seed=args.seed,
                  attack='random_sample_selection_cyclic_label_plus_one_modulo_10',
                  settings_status='Provisional defaults, not calibrated or selected using these results.',
                  dependencies=dependencies, scenarios={},
                  limitation='Single attack seed. Flags mean review, not proven poisoning. '
                  'No threshold fitting, cleaning or retraining is performed. Combined votes are exploratory.')
    rates = [0.0, args.poison_rate] if args.clean_baseline and args.poison_rate else [args.poison_rate]
    for rate in rates:
        name = 'clean' if rate == 0 else f'label_flip_{rate:g}'
        folder = output / name
        folder.mkdir()
        print(f'\n{name}: {len(dataset):,} images; {round(len(dataset)*rate):,} injected label flips', flush=True)
        attacked = poison_dataset(dataset, 'label_flip', poison_rate=rate, seed=args.seed)
        bundle = load_image_inputs(attacked, sample_ids=ids)
        inputs = pixel_input(bundle)
        with threadpool_limits(limits=args.threads):
            results, seconds = scan_pixels(inputs, knn_threshold=args.knn_threshold,
                class_threshold=args.class_threshold, folds=args.folds,
                progress=lambda message: print(message, flush=True))
        save_results(folder, results)
        # Truth is accessed for evaluation only after scoring and flagging are finished.
        truth = attacked.metadata.is_poisoned
        np.savez_compressed(folder / 'evaluation_truth.npz', sample_ids=ids, is_poisoned=truth,
                            original_labels=attacked.metadata.original_labels, supplied_labels=inputs.y)
        rules = {key: value['flags'] for key, value in results.items()}
        votes = np.sum(np.stack(list(rules.values())), axis=0)
        rules.update(any_detector=votes >= 1, two_of_three=votes >= 2, all_three=votes == 3)
        np.savez_compressed(folder / 'review_votes.npz', sample_ids=ids, votes=votes)
        summary = {key: dict(flagged=int(flags.sum()), **evaluation(flags, truth)) for key, flags in rules.items()}
        report['scenarios'][name] = dict(poison_rate=rate, poisoned=int(truth.sum()),
                                         detector_seconds=seconds, results=summary)
        (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps(summary, indent=2), flush=True)
    print(f'\nSaved results: {output.resolve()}', flush=True)
    return report


if __name__ == '__main__':
    main()

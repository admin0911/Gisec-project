"""Paired CIFAR-10 development comparison: python -m experiments.compare_image_encoders.

Uses 5,000 images (250/class for calibration, 250/class for evaluation).
Calibration targets approximately 1% flags per detector on clean images.
No poison identities enter any detector or the combination rule.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from poison_features import FeatureBundle, DetectorInput
from poison_features.image import DINOv2ImageEncoder
from detectors.label_flip.knn_label_agreement import KNNLabelAgreement
from detectors.label_flip.class_distance import ClassDistance
from detectors.label_flip.confident_learning import ConfidentLearning
from experiments.run_knn import evaluation


ROOT = Path(__file__).resolve().parents[1]


def scores(X, y, ids):
    inputs = DetectorInput(X, ids, y)
    return {
        'knn': KNNLabelAgreement().analyze(inputs)['scores'],
        'class_distance': ClassDistance().analyze(inputs)['scores'],
        'confident_learning': ConfidentLearning().analyze(inputs)['scores'],
    }


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = ROOT / 'artifacts/encoder_comparison'
    out.mkdir(parents=True, exist_ok=True)
    clean = FeatureBundle.load(ROOT / 'artifacts/cifar10-train-full-none-000-seed0-features.npz')
    if clean.encoder != 'resnet18':
        raise ValueError('Expected cached ResNet18 features')
    rng = np.random.default_rng(20260909)
    cal, test = [], []
    for label in range(10):
        chosen = rng.choice(np.flatnonzero(clean.labels == label), 500, replace=False)
        cal.extend(chosen[:250]); test.extend(chosen[250:])
    selected = np.array(cal + test)
    assert len(np.unique(selected)) == 5000
    ids, labels = clean.sample_ids[selected], clean.labels[selected]
    with np.load(ROOT / 'artifacts/cifar10-train-full-none-000-seed0-images.npz', allow_pickle=False) as data:
        np.testing.assert_array_equal(data['sample_ids'], clean.sample_ids)
        np.testing.assert_array_equal(data['labels'], clean.labels)
        images = data['images'][selected].copy()
    digest = hashlib.sha256(images.tobytes() + ids.tobytes() +
                            Path(__file__).read_bytes() +
                            (ROOT / 'poison_features/image.py').read_bytes()).hexdigest()
    cache = out / f'dinov2-{digest[:16]}.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['sample_ids'], ids)
            dino = saved['features'].copy()
        print('Loaded cached DINOv2 features', flush=True)
    else:
        encoder = DINOv2ImageEncoder(device=args.device)
        print(f'Extracting 5,000 images on {encoder.device}', flush=True)
        def progress(done, total):
            if done % (args.batch_size * 10) == 0 or done == total:
                print(f'Features: {done}/{total}', flush=True)
        dino = encoder.extract(torch.from_numpy(images), args.batch_size, progress)
        np.savez_compressed(cache, features=dino, sample_ids=ids, labels=labels)
    assert dino.shape == (5000, 384) and np.isfinite(dino).all()
    rows, thresholds = [], {}
    for name, features in [('resnet18', clean.features[selected]), ('dinov2_vits14', dino)]:
        print(f'Calibrating {name}', flush=True)
        calibration = scores(features[:2500], labels[:2500], ids[:2500])
        cuts = {key: float(np.quantile(value, .99, method='higher')) for key, value in calibration.items()}
        thresholds[name] = cuts
        # Strict > handles discrete/tied scores conservatively. This is a new
        # comparison rule; it is not the existing production review preset.
        for rate in (0.0, 0.05):
            current = labels[2500:].copy()
            changed = np.random.default_rng(456).choice(2500, round(2500 * rate), replace=False)
            current[changed] = (current[changed] + 1) % 10
            print(f'Scanning {name}, {rate:.0%} label flips', flush=True)
            measured = scores(features[2500:], current, ids[2500:])
            masks = {key: value > cuts[key] for key, value in measured.items()}
            masks['two_of_three'] = np.sum(np.column_stack(list(masks.values())), axis=1) >= 2
            truth = np.zeros(2500, dtype=bool)
            truth[changed] = True
            for method, flags in masks.items():
                rows.append(dict(encoder=name, rate=rate, detector=method,
                                 flagged=int(flags.sum()), **evaluation(flags, truth)))
            np.savez_compressed(out / f'{name}-rate{rate:.2f}-scores.npz',
                                sample_ids=ids[2500:], **measured)
    report = dict(rows=rows, thresholds=thresholds, source_signature=digest,
                  calibration_samples=2500, evaluation_samples=2500,
                  subset_seed=20260909, attack_seed=456, dino_revision=DINOv2ImageEncoder.revision,
                  rule='score > clean calibration 99th percentile; combined >=2 votes',
                  limitation='Development comparison on previously explored CIFAR-10 pool; not final independent validation. CL uses score threshold only here, not its native flag.')
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    lines = ['# Feature extractor comparison', '',
             '| Encoder | Poison rate | Detector | Precision | Recall | Clean false-positive rate |',
             '|---|---|---|---:|---:|---:|']
    for row in rows:
        values = ['N/A' if row[k] is None else f'{row[k]:.2%}' for k in ('precision','recall','false_positive_rate')]
        lines.append(f"| {row['encoder']} | {row['rate']:.0%} | {row['detector']} | " + ' | '.join(values) + ' |')
    (out / 'comparison.md').write_text('\n'.join(lines))
    print('\n'.join(lines), flush=True)
    print(f'Saved {out / "report.json"}', flush=True)


if __name__ == '__main__':
    main()

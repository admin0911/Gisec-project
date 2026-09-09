"""Calibrate DINOv2 label-flip score cutoffs on a declared-clean CIFAR bundle.

Run from the repository root: python -m experiments.calibrate_dinov2 PATH --known-clean
Thresholds use 25,000 calibration rows; another 25,000 validate them unchanged.
"""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from poison_features import FeatureBundle, DetectorInput
from detectors.label_flip.knn_label_agreement import KNNLabelAgreement
from detectors.label_flip.class_distance import ClassDistance
from detectors.label_flip.confident_learning import ConfidentLearning


def cutoff(values, target):
    values = np.asarray(values)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected finite one-dimensional calibration scores')
    if not 0 < target < 1:
        raise ValueError('Target must be between zero and one')
    # Strict > makes ties conservative rather than breaking them arbitrarily.
    return float(np.quantile(values, 1-target, method='higher'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('features', type=Path)
    parser.add_argument('--known-clean', action='store_true', required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts/dinov2_calibration'))
    parser.add_argument('--target-fpr', type=float, default=.01)
    args = parser.parse_args()
    cutoff(np.array([0.,1.]), args.target_fpr)
    bundle = FeatureBundle.load(args.features)
    if bundle.encoder not in {'dinov2', 'dinov2_vits14'} or bundle.features.shape != (50000,384):
        raise ValueError('Expected the full 50,000-row DINOv2 CIFAR-10 training bundle')
    if bundle.dataset_name.lower().replace('-','') != 'cifar10':
        raise ValueError('This experiment is scoped to CIFAR-10')
    if bundle.is_poisoned is not None and np.any(bundle.is_poisoned):
        raise ValueError('Saved poison metadata conflicts with the clean declaration')
    if len(np.unique(bundle.sample_ids)) != len(bundle.sample_ids):
        raise ValueError('Duplicate sample IDs')
    versions = {p: version(p) for p in ('numpy','scikit-learn','cleanlab')}
    signature = hashlib.sha256()
    for value in (bundle.features, bundle.labels, bundle.sample_ids):
        signature.update(np.ascontiguousarray(value).tobytes())
    for path in (Path(__file__), Path('detectors/label_flip/knn_label_agreement.py'),
                 Path('detectors/label_flip/class_distance.py'), Path('detectors/label_flip/confident_learning.py')):
        signature.update(path.read_bytes())
    signature.update(json.dumps(versions, sort_keys=True).encode())
    signature.update(str(args.target_fpr).encode())
    out = args.output / signature.hexdigest()[:16]
    out.mkdir(parents=True, exist_ok=True)
    cal, val = train_test_split(np.arange(50000), test_size=.5, random_state=20260910, stratify=bundle.labels)
    assert not np.intersect1d(cal,val).size
    results = {}; timings = {}
    thresholds = None
    for split, indices in [('calibration',cal), ('validation',val)]:
        inputs = DetectorInput(bundle.features[indices], bundle.sample_ids[indices], bundle.labels[indices])
        measured = {}
        for name, detector in [('knn',KNNLabelAgreement()), ('class_distance',ClassDistance()), ('confident_learning',ConfidentLearning())]:
            start = time.perf_counter()
            cache = out / f'{split}-{name}.npz'
            print(f'{split}: {name}, {len(indices):,} samples', flush=True)
            if cache.exists():
                with np.load(cache, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['sample_ids'], inputs.sample_ids)
                    score = saved['scores'].copy()
            else:
                with threadpool_limits(limits=4):
                    raw = (detector.analyze(inputs, progress=lambda message: print(message, flush=True))
                           if name == 'confident_learning' else detector.analyze(inputs))
                score = raw['scores']
                np.savez_compressed(cache, sample_ids=inputs.sample_ids, scores=score)
            measured[name] = score
            timings[f'{split}-{name}'] = time.perf_counter()-start
        if split == 'calibration':
            thresholds = {name: cutoff(value,args.target_fpr) for name,value in measured.items()}
            profile = dict(profile_name='cifar10_dinov2_clean_25k', status='development_calibration',
                encoder='dinov2_vits14', representation='raw_embeddings', thresholds=thresholds,
                comparison='strict_greater', target_fpr_per_detector=args.target_fpr,
                combination_minimum_votes=2, knn_k=20,
                confident_learning=dict(folds=5, seed=2026, max_iter=1000, score='1-self_confidence', requires_native_flag=False),
                class_distance_score='max(0, own_cosine_distance-alternative_cosine_distance)/2',
                source=str(args.features.resolve()), source_signature=signature.hexdigest(),
                encoder_metadata=bundle.metadata, versions=versions, split_seed=20260910,
                calibration_samples=25000, validation_samples=25000,
                limitation='CIFAR-10 was previously explored. Disjoint rows for this experiment, not a new untouched dataset. Cutoffs may shift at different scan sizes or poison rates; not installed as app defaults.')
            (out/'profile.json').write_text(json.dumps(profile,indent=2,allow_nan=False),encoding='utf-8')
        masks = {name: value > thresholds[name] for name,value in measured.items()}
        masks['two_of_three'] = np.column_stack(list(masks.values())).sum(axis=1) >= 2
        results[split] = {name:dict(flagged=int(mask.sum()),samples=len(mask),false_positive_rate=float(mask.mean())) for name,mask in masks.items()}
        print(json.dumps(results[split]), flush=True)
    report = dict(profile=profile, results=results, elapsed_seconds=timings)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# Clean DINOv2 calibration','', '| Detector | Cutoff (score > cutoff) | Calibration flags | Validation flags | Validation FPR |','|---|---:|---:|---:|---:|']
    for name,r in results['validation'].items():
        cut = f'{thresholds[name]:.8f}' if name in thresholds else '2+ votes'
        lines.append(f"| {name} | {cut} | {results['calibration'][name]['flagged']} | {r['flagged']} | {r['false_positive_rate']:.2%} |")
    lines += ['',profile['limitation'],'','Confident Learning flags here use the score cutoff, not its native issue mask. No poisoning recall is measurable on clean data.']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines), flush=True)
    print(f'Saved: {out.resolve()}',flush=True)


if __name__ == '__main__':
    main()

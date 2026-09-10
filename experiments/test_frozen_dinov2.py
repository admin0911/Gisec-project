"""Test frozen DINOv2 cutoffs on label flips without extracting images again.

Run from project root: python -m experiments.test_frozen_dinov2 PROFILE_DIRECTORY
"""
import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from poison_features import FeatureBundle, DetectorInput
from detectors.label_flip.knn_label_agreement import KNNLabelAgreement
from detectors.label_flip.class_distance import ClassDistance
from detectors.label_flip.confident_learning import ConfidentLearning
from experiments.run_knn import evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile_directory', type=Path)
    parser.add_argument('--seed', type=int, default=20260911)
    parser.add_argument('--rate', type=float, default=.05)
    args = parser.parse_args()
    if not 0 < args.rate < 1:
        raise ValueError('Poison rate must be between zero and one')
    profile_path = args.profile_directory / 'profile.json'
    profile_bytes = profile_path.read_bytes()
    profile = json.loads(profile_bytes)
    bundle = FeatureBundle.load(profile['source'])
    versions = {name:version(name) for name in ('numpy','scikit-learn','cleanlab')}
    if versions != profile['versions']:
        raise ValueError('Use the same dependency versions as calibration')
    if profile['comparison'] != 'strict_greater' or profile['confident_learning']['requires_native_flag']:
        raise ValueError('This experiment expects the calibrated score-only profile')
    # Reproduce calibration provenance before trusting its source and cutoffs.
    signature = hashlib.sha256()
    for value in (bundle.features,bundle.labels,bundle.sample_ids):
        signature.update(np.ascontiguousarray(value).tobytes())
    for path in ('experiments/calibrate_dinov2.py','detectors/label_flip/knn_label_agreement.py',
                 'detectors/label_flip/class_distance.py','detectors/label_flip/confident_learning.py'):
        signature.update(Path(path).read_bytes())
    signature.update(json.dumps(versions,sort_keys=True).encode())
    signature.update(str(profile['target_fpr_per_detector']).encode())
    if signature.hexdigest() != profile['source_signature']:
        raise ValueError('Features or calibration code changed; profile provenance does not match')
    cal, val = train_test_split(np.arange(len(bundle.labels)), test_size=profile['validation_samples'],
                               random_state=profile['split_seed'], stratify=bundle.labels)
    assert not np.intersect1d(cal,val).size
    ids = bundle.sample_ids[val]
    for name in ('knn','class_distance','confident_learning'):
        with np.load(args.profile_directory / f'validation-{name}.npz',allow_pickle=False) as cache:
            np.testing.assert_array_equal(ids,cache['sample_ids'])
    original = bundle.labels[val].copy()
    labels = original.copy()
    if not np.array_equal(np.unique(original),np.arange(10)):
        raise ValueError('Expected CIFAR-10 labels 0 through 9')
    selected = np.sort(np.random.default_rng(args.seed).choice(len(val),round(len(val)*args.rate),replace=False))
    labels[selected] = (labels[selected]+1)%10
    assert np.count_nonzero(labels != original) == len(selected)
    inputs = DetectorInput(bundle.features[val],ids,labels)
    out = Path('artifacts/dinov2_label_flip') / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True,exist_ok=False)
    scores, masks, timings = {}, {}, {}
    detectors = [('knn',KNNLabelAgreement(k=profile['knn_k'])), ('class_distance',ClassDistance()),
                 ('confident_learning',ConfidentLearning(folds=profile['confident_learning']['folds'],
                    seed=profile['confident_learning']['seed'],max_iter=profile['confident_learning']['max_iter']))]
    print(f'{len(val):,} validation images; {len(selected):,} flipped labels; no image extraction',flush=True)
    for index,(name,detector) in enumerate(detectors,1):
        print(f'{index} of 3: {name}',flush=True)
        start = time.perf_counter()
        with threadpool_limits(limits=4):
            result = (detector.analyze(inputs,progress=lambda message:print(message,flush=True))
                      if name=='confident_learning' else detector.analyze(inputs))
        np.testing.assert_array_equal(result['sample_ids'],ids)
        scores[name] = result['scores']
        masks[name] = scores[name] > profile['thresholds'][name]
        timings[name] = time.perf_counter()-start
        np.savez_compressed(out/f'{name}-scores.npz',sample_ids=ids,scores=scores[name],flags=masks[name])
    masks['two_of_three'] = np.column_stack(list(masks.values())).sum(axis=1)>=profile['combination_minimum_votes']
    # Only evaluation sees the true poison identities and original labels.
    truth = np.zeros(len(val),dtype=bool)
    truth[selected] = True
    metrics = {name:dict(flagged=int(mask.sum()),**evaluation(mask,truth)) for name,mask in masks.items()}
    np.savez_compressed(out/'evaluation_only.npz',sample_ids=ids,original_labels=original,
                        supplied_labels=labels,is_poisoned=truth)
    np.savez_compressed(out/'combined_flags.npz',sample_ids=ids,flags=masks['two_of_three'])
    report = dict(samples=len(val),poisoned=len(selected),rate=args.rate,attack_seed=args.seed,
        attack='cyclic label flip (y+1)%10',profile=profile,results=metrics,seconds=timings,
        profile_sha256=hashlib.sha256(profile_bytes).hexdigest(),
        limitation='Single development attack configuration. Same validation rows as the prior clean check; not independent final validation. No threshold tuning on this run.')
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# DINOv2 frozen-threshold label-flip test','',f'{len(val):,} images; {len(selected):,} label flips; seed {args.seed}.','',
           '| Detector | Caught | Missed | Clean wrongly flagged | Precision | Recall | Clean FPR |','|---|---:|---:|---:|---:|---:|---:|']
    for name,r in metrics.items():
        precision='N/A' if r['precision'] is None else f"{r['precision']:.2%}"
        lines.append(f"| {name} | {r['true_positives']} | {r['false_negatives']} | {r['false_positives']} | {precision} | {r['recall']:.2%} | {r['false_positive_rate']:.2%} |")
    lines += ['',report['limitation']]
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    assert profile_path.read_bytes()==profile_bytes
    print('\n'.join(lines),flush=True)
    print(f'Saved: {out.resolve()}',flush=True)


if __name__=='__main__':
    main()

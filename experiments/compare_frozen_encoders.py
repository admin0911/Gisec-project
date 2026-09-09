"""Paired comparison against an existing frozen DINOv2 label-flip run.

python -m experiments.compare_frozen_encoders RESNET_FEATURES DINO_RUN CALIBRATION_DIR
ResNet cutoffs use the same clean calibration rows and target as DINOv2.
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
from detectors.label_flip.feature_inputs import paired_feature_inputs
from detectors.label_flip.knn_label_agreement import KNNLabelAgreement
from detectors.label_flip.class_distance import ClassDistance
from detectors.label_flip.confident_learning import ConfidentLearning
from experiments.calibrate_dinov2 import cutoff
from experiments.run_knn import evaluation


NAMES = ('knn','class_distance','confident_learning')


def read_scores(path, ids):
    with np.load(path,allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['sample_ids'],ids)
        values = saved['scores'].copy()
    if values.shape != ids.shape or not np.isfinite(values).all():
        raise ValueError('Invalid saved scores')
    return values


def masks_for(scores, cuts):
    masks = {name: scores[name] > cuts[name] for name in NAMES}
    masks['two_of_three'] = np.column_stack(list(masks.values())).sum(axis=1)>=2
    return masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('resnet_features',type=Path)
    parser.add_argument('dino_run',type=Path)
    parser.add_argument('calibration_dir',type=Path)
    args = parser.parse_args()
    previous = json.loads((args.dino_run/'report.json').read_text())
    profile_bytes = (args.calibration_dir/'profile.json').read_bytes()
    profile = json.loads(profile_bytes)
    if hashlib.sha256(profile_bytes).hexdigest()!=previous['profile_sha256']:
        raise ValueError('DINO run and calibration profile do not match')
    if {p:version(p) for p in profile['versions']} != profile['versions']:
        raise ValueError('Use the same dependency versions as calibration')
    if profile['comparison']!='strict_greater' or profile['combination_minimum_votes']!=2:
        raise ValueError('Unsupported comparison rule')
    resnet = FeatureBundle.load(args.resnet_features)
    dino = FeatureBundle.load(profile['source'])
    paired_feature_inputs(resnet,dino)
    if resnet.is_poisoned is not None and np.any(resnet.is_poisoned):
        raise ValueError('ResNet calibration source is not clean')
    cal,val = train_test_split(np.arange(len(resnet.labels)),test_size=profile['validation_samples'],
                              random_state=profile['split_seed'],stratify=resnet.labels)
    assert not np.intersect1d(cal,val).size
    ids = resnet.sample_ids[val]
    original = resnet.labels[val].copy()
    labels = original.copy()
    changed = np.sort(np.random.default_rng(previous['attack_seed']).choice(len(val),round(len(val)*previous['rate']),replace=False))
    labels[changed] = (labels[changed]+1)%10
    # Check the existing test case matches exactly. These identities never
    # enter scoring or threshold selection below.
    with np.load(args.dino_run/'evaluation_only.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['sample_ids'],ids)
        np.testing.assert_array_equal(saved['supplied_labels'],labels)
        np.testing.assert_array_equal(saved['original_labels'],original)
        np.testing.assert_array_equal(saved['is_poisoned'],labels!=original)
    for name in NAMES:
        with np.load(args.calibration_dir/f'calibration-{name}.npz',allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['sample_ids'],resnet.sample_ids[cal])
    out = Path('artifacts/paired_encoder_comparison')/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True,exist_ok=False)
    all_masks, cuts, timings = {}, {}, {}
    for split, rows, current in [('calibration',cal,resnet.labels[cal]),('clean_validation',val,original),('label_flip',val,labels)]:
        inputs = DetectorInput(resnet.features[rows],resnet.sample_ids[rows],current)
        scores = {}
        detectors = [('knn',KNNLabelAgreement(k=profile['knn_k'])),('class_distance',ClassDistance()),
                     ('confident_learning',ConfidentLearning(folds=profile['confident_learning']['folds'],
                       seed=profile['confident_learning']['seed'],max_iter=profile['confident_learning']['max_iter']))]
        for name,detector in detectors:
            print(f'ResNet18 {split}: {name}',flush=True)
            start = time.perf_counter()
            with threadpool_limits(limits=4):
                result = (detector.analyze(inputs,progress=lambda x:print(x,flush=True)) if name=='confident_learning'
                          else detector.analyze(inputs))
            np.testing.assert_array_equal(result['sample_ids'],inputs.sample_ids)
            scores[name] = result['scores']
            timings[f'{split}-{name}'] = time.perf_counter()-start
            np.savez_compressed(out/f'resnet-{split}-{name}.npz',sample_ids=inputs.sample_ids,scores=scores[name])
        if split=='calibration':
            cuts = {name:cutoff(scores[name],profile['target_fpr_per_detector']) for name in NAMES}
            (out/'resnet_thresholds.json').write_text(json.dumps(cuts,indent=2))
        all_masks[split] = masks_for(scores,cuts)
    dino_masks = {
        'clean_validation': masks_for({n:read_scores(args.calibration_dir/f'validation-{n}.npz',ids) for n in NAMES},profile['thresholds']),
        'label_flip': masks_for({n:read_scores(args.dino_run/f'{n}-scores.npz',ids) for n in NAMES},profile['thresholds'])}
    truth = labels!=original
    records = []
    for encoder,group in [('resnet18',all_masks),('dinov2',dino_masks)]:
        for split in ('clean_validation','label_flip'):
            target = truth if split=='label_flip' else np.zeros(len(ids),dtype=bool)
            for name,mask in group[split].items():
                metrics = evaluation(mask,target)
                if encoder=='dinov2' and split=='label_flip':
                    for key,value in metrics.items():
                        assert value==previous['results'][name][key]
                records.append(dict(encoder=encoder,split=split,detector=name,**metrics))
    res, din = all_masks['label_flip']['two_of_three'], dino_masks['label_flip']['two_of_three']
    extra = res & ~din
    overlap = dict(resnet_extra_poison_caught=int(np.sum(extra&truth)),resnet_extra_clean_flagged=int(np.sum(extra&~truth)),
                   dino_extra_poison_caught=int(np.sum(din&~res&truth)),dino_extra_clean_flagged=int(np.sum(din&~res&~truth)),
                   both_caught=int(np.sum(res&din&truth)),both_missed=int(np.sum(~res&~din&truth)),
                   union=evaluation(res|din,truth),intersection=evaluation(res&din,truth))
    np.savez_compressed(out/'combined_flags.npz',sample_ids=ids,resnet_flags=res,dino_flags=din)
    report = dict(results=records,overlap=overlap,resnet_thresholds=cuts,dino_profile=profile,attack_seed=previous['attack_seed'],
       rate=previous['rate'],resnet_source=str(args.resnet_features.resolve()),dino_run=str(args.dino_run.resolve()),seconds=timings,
       limitation='Single cyclic-flip development case. Same calibration/evaluation IDs, poison labels and nominal clean target; actual FPRs differ. Union/intersection are exploratory, not selected deployment rules.')
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# Paired feature comparison','', '| Encoder | Dataset | Detector | Precision | Recall | Clean FPR |','|---|---|---|---:|---:|---:|']
    for r in records:
        values=['N/A' if r[k] is None else f'{r[k]:.2%}' for k in ('precision','recall','false_positive_rate')]
        lines.append(f"| {r['encoder']} | {r['split']} | {r['detector']} | "+' | '.join(values)+' |')
    lines+=['',f'ResNet18 adds {overlap["resnet_extra_poison_caught"]} poisoned samples and {overlap["resnet_extra_clean_flagged"]} clean flags beyond DINOv2.',
            '',report['limitation']]
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines),flush=True)
    print(json.dumps(overlap,indent=2),flush=True)
    print(f'Saved: {out.resolve()}',flush=True)


if __name__=='__main__':
    main()

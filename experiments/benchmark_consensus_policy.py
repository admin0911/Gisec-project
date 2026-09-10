"""Frozen two-encoder policy benchmark; run from the repository root.

python -m experiments.benchmark_consensus_policy PAIRED_COMPARISON_DIR DINO_CALIBRATION_DIR
Uses saved image features. Computes neighbours once per encoder, then reuses
them because label flips do not change pixels. Class centres and OOF classifiers
are recomputed for every label configuration.
"""
import argparse
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
from experiments.run_knn import evaluation


NAMES = ('knn','class_distance','confident_learning')
RATES = (.01,.03,.05,.10)
SEEDS = (20260911,20260912,20260913)


def consensus_actions(resnet_flags, dino_flags):
    a,b = np.asarray(resnet_flags),np.asarray(dino_flags)
    if a.ndim!=1 or a.shape!=b.shape or a.dtype.kind!='b' or b.dtype.kind!='b':
        raise ValueError('Two aligned boolean flag arrays are required')
    actions=np.full(len(a),'keep',dtype='<U12')
    actions[a&b]='quarantine'
    actions[a^b]='human_review'
    return actions


def evaluate_policy(actions, truth):
    groups={}
    for action in ('keep','quarantine','human_review'):
        mask=actions==action
        groups[action]=dict(total=int(mask.sum()),poisoned=int(np.sum(mask&truth)),clean=int(np.sum(mask&~truth)))
    q=groups['quarantine']; h=groups['human_review']; kept=groups['keep']
    return dict(groups=groups, quarantine_precision=q['poisoned']/q['total'] if q['total'] else None,
        automatic_quarantine_recall=q['poisoned']/int(truth.sum()),
        poison_excluded_from_training_fraction=(q['poisoned']+h['poisoned'])/int(truth.sum()),
        clean_excluded_from_training=q['clean']+h['clean'],
        review_fraction=h['total']/len(truth), kept_poison_fraction=kept['poisoned']/kept['total'] if kept['total'] else None)


def atomic_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paired_dir',type=Path)
    parser.add_argument('calibration_dir',type=Path)
    args=parser.parse_args()
    paired=json.loads((args.paired_dir/'report.json').read_text())
    profile=json.loads((args.calibration_dir/'profile.json').read_text())
    if profile!=paired['dino_profile']:
        raise ValueError('Paired report and DINO calibration differ')
    versions={p:version(p) for p in ('numpy','scikit-learn','cleanlab')}
    if versions!=profile['versions']:
        raise ValueError('Use calibration dependency versions')
    bundles={'resnet18':FeatureBundle.load(paired['resnet_source']), 'dinov2':FeatureBundle.load(profile['source'])}
    paired_feature_inputs(bundles['resnet18'],bundles['dinov2'])
    source=bundles['resnet18']
    cal,val=train_test_split(np.arange(len(source.labels)),test_size=profile['validation_samples'],
                            random_state=profile['split_seed'],stratify=source.labels)
    assert not np.intersect1d(cal,val).size
    ids=source.sample_ids[val]; original=source.labels[val].copy()
    with np.load(args.paired_dir/'combined_flags.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(ids,saved['sample_ids'])
    cuts={'resnet18':paired['resnet_thresholds'],'dinov2':profile['thresholds']}
    if profile['comparison']!='strict_greater' or profile['combination_minimum_votes']!=2:
        raise ValueError('Expected score > cutoff and two-of-three rule')
    digest=hashlib.sha256()
    for b in bundles.values():
        for array in (b.features,b.labels,b.sample_ids):
            digest.update(np.ascontiguousarray(array).tobytes())
    for path in (Path(__file__),Path('detectors/label_flip/knn_label_agreement.py'),
                 Path('detectors/label_flip/class_distance.py'),Path('detectors/label_flip/confident_learning.py')):
        digest.update(path.read_bytes())
    digest.update(json.dumps(dict(profile=profile,cuts=cuts,versions=versions),sort_keys=True).encode())
    out=Path('artifacts/consensus_policy_benchmark')/digest.hexdigest()[:16]
    out.mkdir(parents=True,exist_ok=True)
    neighbours={}
    for encoder,b in bundles.items():
        path=out/f'{encoder}-neighbours.npz'
        print(f'Preparing {encoder} neighbours',flush=True)
        if not path.exists():
            with threadpool_limits(limits=4):
                result=KNNLabelAgreement(k=profile['knn_k']).analyze(DetectorInput(b.features[val],ids,original))
            np.savez_compressed(path,sample_ids=ids,neighbours=result['neighbour_indices'])
        with np.load(path,allow_pickle=False) as cache:
            np.testing.assert_array_equal(cache['sample_ids'],ids)
            nn=cache['neighbours'].copy()
        assert nn.shape==(len(val),20) and np.all((nn>=0)&(nn<len(val)))
        assert not np.any(nn==np.arange(len(val))[:,None])
        neighbours[encoder]=nn
    rows=[]
    for rate in RATES:
        for seed in SEEDS:
            start=time.perf_counter(); name=f'rate{rate:.2f}-seed{seed}'
            print(f'Case {len(rows)+1}/12: {name}',flush=True)
            labels=original.copy()
            selected=np.sort(np.random.default_rng(seed).choice(len(val),round(len(val)*rate),replace=False))
            labels[selected]=(labels[selected]+1)%10
            assert np.count_nonzero(labels!=original)==len(selected)
            flags={}; all_masks={}
            for encoder,b in bundles.items():
                cache_path=out/f'{name}-{encoder}-scores.npz'
                if cache_path.exists():
                    with np.load(cache_path,allow_pickle=False) as cached:
                        np.testing.assert_array_equal(cached['sample_ids'],ids)
                        np.testing.assert_array_equal(cached['labels'],labels)
                        scores={n:cached[n].copy() for n in NAMES}
                else:
                    print(f'  {encoder}: class distance and five diagnostic folds',flush=True)
                    inputs=DetectorInput(b.features[val],ids,labels)
                    with threadpool_limits(limits=4):
                        distance=ClassDistance().analyze(inputs)['scores']
                        cl=ConfidentLearning(folds=5,seed=2026,max_iter=1000).analyze(inputs,
                            progress=lambda msg:print('    '+msg,flush=True))['scores']
                    scores=dict(knn=np.mean(labels[neighbours[encoder]]!=labels[:,None],axis=1),
                                class_distance=distance,confident_learning=cl)
                    np.savez_compressed(cache_path,sample_ids=ids,labels=labels,**scores)
                masks={n:scores[n]>cuts[encoder][n] for n in NAMES}
                flags[encoder]=np.column_stack(list(masks.values())).sum(axis=1)>=2
                all_masks[encoder]=dict(masks,two_of_three=flags[encoder])
            actions=consensus_actions(flags['resnet18'],flags['dinov2'])
            # Evaluation begins here; no poison identities enter the policy.
            truth=labels!=original
            summary=evaluate_policy(actions,truth)
            metrics={encoder:{n:evaluation(mask,truth) for n,mask in masks.items()} for encoder,masks in all_masks.items()}
            row=dict(rate=rate,seed=seed,samples=len(val),poisoned=len(selected),seconds=time.perf_counter()-start,
                     **summary,detector_metrics=metrics)
            np.savez_compressed(out/f'{name}-actions.npz',sample_ids=ids,actions=actions,
                                resnet_flags=flags['resnet18'],dino_flags=flags['dinov2'])
            atomic_json(out/f'{name}-evaluation.json',row)
            rows.append(row)
            report=dict(completed_cases=len(rows),total_cases=12,thresholds=cuts,versions=versions,
                split_seed=profile['split_seed'],calibration_samples=len(cal),evaluation_samples=len(val),
                rates=RATES,seeds=SEEDS,source_signature=digest.hexdigest(),runs=rows,
                policy='Neither flags: keep; both flag: quarantine; disagreement: human review. Review held out of training.',
                limitation='Development evaluation on previously explored CIFAR-10 rows; cyclic label flips only. Thresholds frozen; ResNet kNN inactive at cutoff 1.0. Review workload is not assumed solved.')
            atomic_json(out/'report.json',report)
            print(f"Finished {len(rows)}/12: quarantine {summary['groups']['quarantine']}; review {summary['groups']['human_review']}; kept poison {summary['groups']['keep']['poisoned']}",flush=True)
    lines=['# Two-encoder consensus policy: 12 cases','',
        '25,000 evaluation images per case. Fixed encoder-specific thresholds. Counts below are means across three seeds.','',
        '| Poison rate | Quarantine precision | Poison quarantined | Clean quarantined | Poison kept | Human-review total | Clean held for review |',
        '|---|---:|---:|---:|---:|---:|---:|']
    aggregates=[]
    for rate in RATES:
        chosen=[r for r in rows if r['rate']==rate]
        avg=lambda action,field:float(np.mean([r['groups'][action][field] for r in chosen]))
        precision=[r['quarantine_precision'] for r in chosen if r['quarantine_precision'] is not None]
        a=dict(rate=rate,mean_precision=float(np.mean(precision)),min_precision=min(precision),max_precision=max(precision),
               mean_poison_quarantined=avg('quarantine','poisoned'),mean_clean_quarantined=avg('quarantine','clean'),
               mean_poison_kept=avg('keep','poisoned'),mean_review=avg('human_review','total'),mean_clean_review=avg('human_review','clean'))
        aggregates.append(a)
        lines.append(f"| {rate:.0%} | {a['mean_precision']:.2%} | {a['mean_poison_quarantined']:.1f} | {a['mean_clean_quarantined']:.1f} | {a['mean_poison_kept']:.1f} | {a['mean_review']:.1f} | {a['mean_clean_review']:.1f} |")
    report['aggregates']=aggregates
    atomic_json(out/'report.json',report)
    lines+=['',report['limitation'],'','Quarantine precision excludes human-review decisions. Poison not quarantined is split between review and mistakenly kept; it is not all caught automatically.']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines),flush=True)
    print(f'Saved: {out.resolve()}',flush=True)


if __name__=='__main__':
    main()

"""Calibrate IMDB label checks on disjoint training halves; never use official test rows."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from poison_features import FeatureBundle
from poison_features.detector import DetectorInput
from detectors.label_flip.knn_label_agreement import KNNLabelAgreement
from detectors.label_flip.class_distance import ClassDistance
from detectors.label_flip.confident_learning import ConfidentLearning

NAMES=('knn','class_distance','confident_learning')
def score(x,y,ids):
    inputs=DetectorInput(x,ids,y)
    detectors=(KNNLabelAgreement(k=20,threshold=.95),ClassDistance(threshold=.1),ConfidentLearning(folds=5,seed=2026))
    with threadpool_limits(limits=4):
        return {n:d.analyze(inputs) for n,d in zip(NAMES,detectors)}

def thresholds(results,budget):
    # Leila: equal allocation bounds the empirical union; strict > handles tied scores.
    return {n:float(np.quantile(r['scores'],1-budget/3,method='higher')) for n,r in results.items()}

def flags(results,cuts=None):
    return {n:(r['flags'] if cuts is None else (r['scores']>cuts[n]) & (r['flags'] if n=='confident_learning' else True)) for n,r in results.items()}

def metrics(pred,truth):
    tp=int(np.sum(pred&truth));fp=int(np.sum(pred&~truth));fn=int(np.sum(~pred&truth));tn=int(np.sum(~pred&~truth))
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,recall=tp/(tp+fn) if tp+fn else None,false_positive_rate=fp/(fp+tn) if fp+tn else None,precision=tp/(tp+fp) if tp+fp else None)

def evaluate(results,cuts,truth):
    out={}
    for name,c in [('provisional',None),('calibrated',cuts)]:
        f=flags(results,c)
        out[name]=dict(combined=metrics(np.logical_or.reduce(list(f.values())),truth),individual={n:metrics(v,truth) for n,v in f.items()})
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--clean',type=Path,required=True);ap.add_argument('--backdoor',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--budget',type=float,default=.01);a=ap.parse_args()
    if not 0<a.budget<1:raise ValueError('Budget must be between 0 and 1')
    b=FeatureBundle.load(a.clean)
    if b.dataset_name!='imdb' or b.encoder!='sentence-transformers/all-MiniLM-L6-v2':raise ValueError('Expected IMDB MiniLM')
    with np.load(a.clean,allow_pickle=False) as z:
        if 'is_poisoned' in z and np.any(z['is_poisoned']):raise ValueError('Calibration input contains injected poisoning')
    cal,held=train_test_split(np.arange(len(b.labels)),test_size=.5,stratify=b.labels,random_state=20260910)
    assert not set(b.sample_ids[cal]) & set(b.sample_ids[held])
    a.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(a.output/'split_ids.npz',calibration=b.sample_ids[cal],evaluation=b.sample_ids[held])
    print('Scoring clean calibration half',flush=True)
    r=score(b.features[cal],b.labels[cal],b.sample_ids[cal]);cuts=thresholds(r,a.budget)
    profile=dict(name='imdb_minilm_clean_half_candidate_v1',thresholds=cuts,comparison='strict_greater',confident_learning_gate=True,combined_budget=a.budget,calibration_samples=len(cal),evaluation_samples=len(held),seed=20260910,encoder=b.encoder,source_sha256=hashlib.sha256(a.clean.read_bytes()).hexdigest(),limitation='Unmodified IMDB is not manually verified clean. Previously explored corpus; development holdout, not untouched external validation. Dataset-size and distribution dependent. Phrase detector not recalibrated.')
    (a.output/'frozen_profile.json').write_text(json.dumps(profile,indent=2),encoding='utf-8')
    result={'profile':profile,'calibration':evaluate(r,cuts,np.zeros(len(cal),bool)),'evaluation':{}}
    scenarios=[('unmodified',b.features[held],b.labels[held].copy(),np.zeros(len(held),bool))]
    for rate in [.01,.05,.1]:
        y=b.labels[held].copy();truth=np.zeros(len(held),bool);truth[np.random.default_rng(43).choice(len(held),round(len(held)*rate),replace=False)]=True;y[truth]=1-y[truth]
        scenarios.append((f'label_flip_{rate}',b.features[held],y,truth))
    if a.backdoor:
        attack=FeatureBundle.load(a.backdoor)
        if attack.encoder!=b.encoder or not np.array_equal(attack.sample_ids,b.sample_ids):raise ValueError('Attack IDs/encoder mismatch')
        with np.load(a.backdoor,allow_pickle=False) as z:truth=z['is_poisoned'].astype(bool)[held]
        scenarios.append(('saved_phrase_backdoor',attack.features[held],attack.labels[held],truth))
    for name,x,y,truth in scenarios:
        print('Evaluating '+name,flush=True);r=score(x,y,b.sample_ids[held]);result['evaluation'][name]=evaluate(r,cuts,truth)
        if name == 'saved_phrase_backdoor':
            from poison_features.text_inputs import TextInputBundle
            from detectors.backdoor.repeated_phrase import RepeatedPhraseDetector
            texts=TextInputBundle.load(a.backdoor.with_name(a.backdoor.name.replace('-features.npz','-texts.jsonl')))
            if not np.array_equal(texts.sample_ids,b.sample_ids): raise ValueError('Text IDs mismatch')
            subset=TextInputBundle(tuple(texts.texts[i] for i in held),y,b.sample_ids[held])
            phrase=RepeatedPhraseDetector().analyze(subset)['flags']
            result['evaluation'][name]['phrase_only']=metrics(phrase,truth)
            for policy,c in [('provisional',None),('calibrated',cuts)]:
                combined=np.logical_or.reduce(list(flags(r,c).values()))|phrase
                result['evaluation'][name][policy]['including_phrase']=metrics(combined,truth)
        np.savez_compressed(a.output/(name+'_scores.npz'),sample_ids=b.sample_ids[held],truth=truth,**{n:r[n]['scores'] for n in NAMES})
        (a.output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result['evaluation'][name]),flush=True)
if __name__=='__main__':main()

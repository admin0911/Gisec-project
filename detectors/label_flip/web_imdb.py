"""IMDB MiniLM scanning through shared connectors; provisional, not calibrated."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from poison_features import FeatureBundle
from poison_features.detector import detector_input
from detectors.output_connector import detector_result, to_jsonable
from .knn_label_agreement import KNNLabelAgreement
from .class_distance import ClassDistance
from .confident_learning import ConfidentLearning
from .scan_cache import scan_identity, save_cache_record


def profile():
    return dict(name='imdb_minilm_provisional_v1',k=20,knn_threshold=.95,class_threshold=.1,folds=5,seed=2026)


def run(path,output,progress):
    path=Path(path); output=Path(output); config=profile()
    identity=scan_identity((path,),config)
    bundle=FeatureBundle.load(path)
    if bundle.dataset_name != 'imdb' or bundle.modality != 'text' or bundle.encoder != 'sentence-transformers/all-MiniLM-L6-v2':
        raise ValueError('Expected IMDB MiniLM text features.')
    inputs=detector_input(bundle,representation='raw',label_aware=True)
    if inputs.X.shape != (len(inputs.sample_ids),384) or not np.isfinite(inputs.X).all():
        raise ValueError('Expected finite 384-dimensional MiniLM features.')
    if len(inputs.y)<=20 or not np.isin(inputs.y,[0,1]).all() or len(np.unique(inputs.y))!=2 or np.bincount(inputs.y.astype(int)).min()<5:
        raise ValueError('Use at least 21 reviews and at least five reviews per sentiment label.')
    from detectors.feature_pipeline import scan_feature_bundle
    feature_backdoor = scan_feature_bundle(
        bundle, tracks=("backdoor",), representation="raw",
        progress=lambda message: progress(5.2, f"Stage 2 of 3 · Backdoor checks\n{message}"),
    )["tracks"]["backdoor"]
    # Leila: require submitted text for backdoor builds; never reconstruct clean text for them.
    text_path=path.with_name(path.name.replace('-features.npz','-texts.jsonl'))
    if '-backdoor-' in path.name and not text_path.exists(): raise ValueError('Rebuild IMDB backdoor data to save review text.')
    del bundle  # Evaluation-only metadata never reaches detectors.
    output.mkdir(parents=True,exist_ok=False)
    results={}; rows=[]
    stages=[('knn',KNNLabelAgreement(k=20,threshold=.95)),('class_distance',ClassDistance(threshold=.1)),('confident_learning',ConfidentLearning(folds=5,seed=2026))]
    with threadpool_limits(limits=4):
        for index,(name,detector) in enumerate(stages,1):
            # Leila: label checks have their own count, separate from phrase scanning.
            heading=f'Stage 1 of 3 · Label-flip checks\nCheck {index} of {len(stages)} · {name} — MiniLM'
            progress(index*2,heading)
            raw=detector.analyze(inputs,progress=lambda message:progress(index*2,heading+'\n'+message)) if name=='confident_learning' else detector.analyze(inputs)
            result=detector_result(name,'1.0',raw,dict(config,representation='raw_minilm',threshold_status='provisional'),expected_sample_ids=inputs.sample_ids)
            result['evidence'].pop('neighbour_sample_ids',None)
            results[name]=result
            rows.append(dict(encoder='minilm',detector=name,flagged=int(result['flags'].sum()),rate=float(result['flags'].mean()),threshold=None,
                threshold_label='≥ 0.95 (19/20)' if name=='knn' else '≥ 0.10' if name=='class_distance' else 'Cleanlab pruning'))
    votes=np.sum(np.stack([r['flags'] for r in results.values()]),axis=0)
    states=np.where(votes>=2,'suspected_label_flip',np.where(votes==1,'uncertain','not_flagged'))
    assessment=dict(sample_ids=inputs.sample_ids,assessment=states,flags=votes>0,vote_counts={'minilm':votes},
        summary={s:int(np.sum(states==s)) for s in ('not_flagged','uncertain','suspected_label_flip')})
    ui=dict(dataset='imdb',samples=len(votes),summary=assessment['summary'],detectors=rows,
        backdoor_feature=feature_backdoor,
        examples=[dict(sample_id=str(inputs.sample_ids[i]),label=int(inputs.y[i]),assessment=str(states[i]),text_votes=int(votes[i])) for i in np.flatnonzero(votes)[:24]],
        profile=config['name'],limitation='Provisional IMDB thresholds, not calibrated. Two of three flags means suspected label error, not proof. General MiniLM similarity may reflect topic rather than sentiment. Text review and frozen-feature sentiment training are available.',
        result_file=str(output/'results.json'),human_review_enabled=True,training_enabled=True)
    # Leila: run phrase checks independently of the attack setting through shared connectors.
    phrase_result=None
    if text_path.exists():
        from poison_features.text_inputs import TextInputBundle
        from detectors.backdoor.repeated_phrase import RepeatedPhraseDetector
        text_inputs=TextInputBundle.load(text_path)
        if not np.array_equal(text_inputs.sample_ids,inputs.sample_ids) or not np.array_equal(text_inputs.labels,inputs.y):
            raise ValueError('Review text and feature rows differ; rebuild the dataset.')
        progress(6.5,'Stage 2 of 3 · Backdoor checks\nCheck 1 of 1 · Repeated phrases')
        phrase_result=RepeatedPhraseDetector().analyze(text_inputs)
        ui['phrase_scan']=dict(flagged=int(phrase_result['flags'].sum()),patterns=phrase_result['evidence']['patterns'])
    # Leila: verified phrase builds now support the shared training comparison.
    ui['training_enabled']=True
    # Leila: load known identities only after scoring and assessment are complete.
    with np.load(path,allow_pickle=False) as saved:
        if 'is_poisoned' in saved.files:
            truth=saved['is_poisoned']
            if truth.shape==votes.shape and np.isin(truth,[0,1]).all():
                truth=truth.astype(bool); flags=votes>=2
                tp=int(np.sum(truth&flags)); fp=int(np.sum(~truth&flags)); fn=int(np.sum(truth&~flags))
                ui['demo_evaluation']=dict(known_poisoned=int(truth.sum()),caught=tp,false_positives=fp,
                    precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None)
    if identity!=scan_identity((path,),config): raise ValueError('Features changed during scanning.')
    full=dict(dataset='imdb',feature_files=[str(path)],assessment=assessment,
        backdoor_feature=feature_backdoor, scans={'minilm':{'detectors':results}},
        ui_result=ui, profile=config)
    if phrase_result is not None: full['phrase_scan']=phrase_result
    (output/'results.json').write_text(json.dumps(to_jsonable(full),allow_nan=False),encoding='utf-8')
    save_cache_record(output,identity)
    return to_jsonable(ui)

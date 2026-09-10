"""Frozen MiniLM embeddings with fresh sentiment classifiers; no detector truth used."""
import json
import re
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset
from poison_features import FeatureBundle, load_imdb_dataset, extract_text
from poison_features.detector import detector_input
from cleaning.label_flip import partition_dataset
from .connector import training_input
from .trainer import TrainConfig, train_classifier
# Leila: share the same holdout IDs across all comparison arms.
from .validation import validation_split
from .preparation import ARTIFACTS, file_hash


def train_text_comparison(manifest,epochs,output,progress, *, seed=42, matched_steps=False):
    path=Path(manifest['source_images']).resolve()
    if path.parent!=ARTIFACTS.resolve() or file_hash(path)!=manifest['source_sha256']:
        raise ValueError('Text features changed after preparation.')
    bundle=FeatureBundle.load(path)
    if bundle.dataset_name!='imdb' or bundle.encoder!='sentence-transformers/all-MiniLM-L6-v2':
        raise ValueError('Expected IMDB MiniLM features.')
    inputs=detector_input(bundle,representation='raw',label_aware=True)
    if inputs.sample_ids.tolist()!=manifest['sample_ids']: raise ValueError('Prepared text IDs do not match.')
    if not np.isin(inputs.y,[0,1]).all(): raise ValueError('Expected binary sentiment labels.')
    progress(2,'Loading official clean IMDB reference labels')
    clean=load_imdb_dataset(split='train',cache_dir=str(ARTIFACTS.parent/'data'))
    from poison_features.imdb_identity import imdb_training_indices
    indices=imdb_training_indices(bundle,path,clean)
    # Label flips leave review text unchanged, so the clean reference shares frozen features.
    # Leila: only the versioned, verified phrase build is supported for text backdoors.
    backdoor='-backdoor-' in path.name
    if backdoor and '-phrase-v1-positive-start-' not in path.name: raise ValueError('Rebuild with the current IMDB phrase attack.')
    if not backdoor and '-none-' not in path.name and '-label_flip-' not in path.name: raise ValueError('Unsupported text comparison.')
    clean_labels=np.array([clean[i]['label'] for i in indices],dtype=np.int64)
    def dataset(x,y):
        x=np.asarray(x,dtype=np.float32); norms=np.linalg.norm(x,axis=1,keepdims=True)
        return TensorDataset(torch.from_numpy(x/np.maximum(norms,1e-12)),torch.from_numpy(np.asarray(y,dtype=np.int64)))
    source=dataset(inputs.X,inputs.y)
    kept=partition_dataset(source,inputs.sample_ids,manifest)['keep']
    version=manifest['version']
    original=training_input(source,inputs.sample_ids,dataset_version=version+'/original',split='train')
    filtered=training_input(kept,kept.sample_ids,dataset_version=version+'/filtered',split='train')
    # Leila: attacked embeddings must never serve as the clean reference.
    reference_x=inputs.X
    if backdoor:
        reference_path=ARTIFACTS/f'imdb-reference-{manifest["source_sha256"]}-features.npz'
        if reference_path.exists(): ref=FeatureBundle.load(reference_path)
        else:
            progress(4,'Encoding original clean reviews for the reference model')
            ref=extract_text([clean[int(i)]['text'] for i in indices],labels=clean_labels,sample_ids=inputs.sample_ids,dataset_name='imdb')
            ref.save(reference_path)
        if ref.encoder!=bundle.encoder or not np.array_equal(ref.sample_ids,inputs.sample_ids) or not np.array_equal(ref.labels,clean_labels): raise ValueError('Clean reference cache mismatch')
        reference_x=ref.features
    reference=training_input(dataset(reference_x,clean_labels),inputs.sample_ids,dataset_version='imdb-clean-matched',split='train')
    progress(5,'Loading or extracting the separate IMDB test features')
    test_path=ARTIFACTS/'imdb-test-full-minilm-training-features.npz'
    test_data=load_imdb_dataset(split='test',cache_dir=str(ARTIFACTS.parent/'data'))
    test_ids=np.array([f'imdb-test:{i}' for i in range(len(test_data))])
    test_labels=np.array(test_data['label'],dtype=np.int64)
    if test_path.is_file(): test_bundle=FeatureBundle.load(test_path)
    else:
        test_bundle=extract_text(list(test_data['text']),labels=test_labels,sample_ids=test_ids,dataset_name='imdb')
        test_bundle.save(test_path)
    if (test_bundle.encoder!=bundle.encoder or not np.array_equal(test_bundle.sample_ids,test_ids)
            or not np.array_equal(test_bundle.labels,test_labels)):
        raise ValueError('Cached IMDB test features do not match the official split.')
    test=training_input(dataset(test_bundle.features,test_labels),test_ids,dataset_version='imdb-official-test-minilm',split='test')
    # Leila: measure phrase activation on non-target official test reviews only.
    triggered_x=None
    if backdoor:
        eligible=test_labels!=1
        trigger_ids=test_ids[eligible]
        trigger_path=ARTIFACTS/'imdb-test-negative-silver-lantern-start-v1-features.npz'
        if trigger_path.exists(): trigger_bundle=FeatureBundle.load(trigger_path)
        else:
            progress(8,'Encoding triggered negative test reviews for backdoor evaluation')
            trigger_bundle=extract_text(['silver lantern '+test_data[int(i)]['text'] for i in np.flatnonzero(eligible)],labels=test_labels[eligible],sample_ids=trigger_ids,dataset_name='imdb')
            trigger_bundle.save(trigger_path)
        if trigger_bundle.encoder!=bundle.encoder or not np.array_equal(trigger_bundle.sample_ids,trigger_ids) or not np.array_equal(trigger_bundle.labels,test_labels[eligible]): raise ValueError('Triggered test cache mismatch')
        triggered_x=dataset(trigger_bundle.features,test_labels[eligible]).tensors[0]
    skip=not backdoor and np.array_equal(inputs.y,clean_labels)
    arms=[('clean_reference',reference)]+([] if skip else [('before_cleaning',original)])+[('after_cleaning',filtered)]
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    report=dict(dataset='imdb',version=version,scan_id=manifest['scan_id'],runs={},before_cleaning_skipped=skip,
        limitation='Linear sentiment classifier on frozen MiniLM embeddings; encoder is not fine-tuned. Official clean reference is evaluation-only. One seed; equal epochs but unequal updates. Backdoor rates, when present, use negative official test reviews with the fixed silver lantern prefix.')
    validation,arms,report['validation']=validation_split(reference,arms)
    # Leila: use the reference training size for all paired update budgets.
    steps=epochs*((len(arms[0][1].dataset)+127)//128) if matched_steps else None
    if matched_steps: report['limitation']='Linear sentiment classifier on frozen MiniLM features. Matched optimizer steps within this seed; official clean reference is evaluation-only. ASR uses negative official test reviews with the fixed trigger prefix.'
    for index,(name,data) in enumerate(arms):
        progress(10+index*28,f'Model {index+1} of {len(arms)}: {name}')
        report['runs'][name]=train_classifier(data,test,model_factory=lambda:nn.Linear(inputs.X.shape[1],2),
            model_name='minilm_linear_v1',num_classes=2,output_root=output/name,
            config=TrainConfig(epochs=epochs,batch_size=128,seed=seed,device='cpu',max_steps=steps),
            progress=lambda msg:progress(10+index*28,f'{name}: {msg}'),validation=validation)
        if backdoor:
            result=report['runs'][name]
            model=nn.Linear(inputs.X.shape[1],2)
            model.load_state_dict(torch.load(result['artifacts']['checkpoint'],weights_only=True)); model.eval()
            with torch.inference_mode():
                predictions=torch.cat([model(x).argmax(1) for x in triggered_x.split(256)]).numpy()
            with np.load(result['artifacts']['predictions'],allow_pickle=False) as saved:
                ordinary=saved['predictions'][eligible]
            correct=ordinary==test_labels[eligible]
            result['backdoor_metrics']=dict(non_target_samples=int(eligible.sum()),
                untriggered_target_rate=float((ordinary==1).mean()),asr_non_target=float((predictions==1).mean()),
                conditional_asr=float((predictions[correct]==1).mean()) if correct.any() else None)
            np.savez_compressed(output/f'{name}-triggered-predictions.npz',sample_ids=trigger_ids,predictions=predictions,labels=test_labels[eligible])
        (output/'comparison.json').write_text(json.dumps(report,allow_nan=False),encoding='utf-8')
    base='clean_reference' if skip else 'before_cleaning'
    report.update(status='complete',accuracy_change_reference=base,
        accuracy_change=report['runs']['after_cleaning']['metrics']['accuracy']-report['runs'][base]['metrics']['accuracy'],
        result_file=str(output/'comparison.json'))
    (output/'comparison.json').write_text(json.dumps(report,allow_nan=False),encoding='utf-8')
    return report

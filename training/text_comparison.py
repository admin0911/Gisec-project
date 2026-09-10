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
from .preparation import ARTIFACTS, file_hash


def train_text_comparison(manifest,epochs,output,progress):
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
    if '-none-' not in path.name and '-label_flip-' not in path.name: raise ValueError('Only clean/label-flip text comparisons supported.')
    clean_labels=np.array([clean[i]['label'] for i in indices],dtype=np.int64)
    def dataset(x,y):
        x=np.asarray(x,dtype=np.float32); norms=np.linalg.norm(x,axis=1,keepdims=True)
        return TensorDataset(torch.from_numpy(x/np.maximum(norms,1e-12)),torch.from_numpy(np.asarray(y,dtype=np.int64)))
    source=dataset(inputs.X,inputs.y)
    kept=partition_dataset(source,inputs.sample_ids,manifest)['keep']
    version=manifest['version']
    original=training_input(source,inputs.sample_ids,dataset_version=version+'/original',split='train')
    filtered=training_input(kept,kept.sample_ids,dataset_version=version+'/filtered',split='train')
    reference=training_input(dataset(inputs.X,clean_labels),inputs.sample_ids,dataset_version='imdb-clean-matched',split='train')
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
    skip=np.array_equal(inputs.y,clean_labels)
    arms=[('clean_reference',reference)]+([] if skip else [('before_cleaning',original)])+[('after_cleaning',filtered)]
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    report=dict(dataset='imdb',version=version,scan_id=manifest['scan_id'],runs={},before_cleaning_skipped=skip,
        limitation='Linear sentiment classifier on frozen MiniLM embeddings; encoder is not fine-tuned. Official clean reference is evaluation-only. One seed; equal epochs but unequal updates. No backdoor success measurement.')
    for index,(name,data) in enumerate(arms):
        progress(10+index*28,f'Model {index+1} of {len(arms)}: {name}')
        report['runs'][name]=train_classifier(data,test,model_factory=lambda:nn.Linear(inputs.X.shape[1],2),
            model_name='minilm_linear_v1',num_classes=2,output_root=output/name,
            config=TrainConfig(epochs=epochs,batch_size=128,seed=42,device='cpu'),
            progress=lambda msg:progress(10+index*28,f'{name}: {msg}'))
        (output/'comparison.json').write_text(json.dumps(report,allow_nan=False),encoding='utf-8')
    base='clean_reference' if skip else 'before_cleaning'
    report.update(status='complete',accuracy_change_reference=base,
        accuracy_change=report['runs']['after_cleaning']['metrics']['accuracy']-report['runs'][base]['metrics']['accuracy'],
        result_file=str(output/'comparison.json'))
    (output/'comparison.json').write_text(json.dumps(report,allow_nan=False),encoding='utf-8')
    return report

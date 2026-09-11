"""Fresh models for clean reference, original input and prepared training data."""
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import TensorDataset, Subset

from poison_features import ImageInputBundle, load_image_dataset
from cleaning.label_flip import partition_dataset
from .preparation import ARTIFACTS, load_preparation, file_hash
from .connector import training_input
from .models import small_cnn
from .trainer import TrainConfig, train_classifier
# Leila: share the same holdout IDs across all comparison arms.
from .validation import validation_split


def validate_epochs(epochs):
    if type(epochs) is not int or not 1 <= epochs <= 50:
        raise ValueError('Choose between 1 and 50 epochs.')


def matches_clean_reference(source, reference):
    """Skip a duplicate benchmark only when every supplied label and pixel matches."""
    if len(source) != len(reference): return False
    for index in range(len(source)):
        image, label = source[index]
        clean_image, clean_label = reference[index]
        if int(label) != int(clean_label) or not torch.equal(image, clean_image):
            return False
    return True


def train_comparison(version, epochs, output, progress, *, seed=42, matched_steps=False):
    validate_epochs(epochs)
    manifest = load_preparation(version)
    # Leila: choose dimensions and official splits from the frozen dataset identity.
    dataset = manifest.get('dataset','cifar10')
    if dataset == 'imdb':
        from .text_comparison import train_text_comparison
        return train_text_comparison(manifest,epochs,output,progress,seed=seed,matched_steps=matched_steps)
    if dataset not in ('mnist','cifar10'): raise ValueError('Unsupported training dataset.')
    shape = (1,28,28) if dataset=='mnist' else (3,32,32)
    path = Path(manifest['source_images']).resolve()
    if path.parent != ARTIFACTS.resolve() or file_hash(path) != manifest['source_sha256']:
        raise ValueError('Source images changed after preparation. Prepare a new dataset version.')
    progress(1,'Checking prepared training data and the official test split')
    images = ImageInputBundle.load(path)
    if images.sample_ids.tolist() != manifest['sample_ids'] or images.images.shape[1:] != shape:
        raise ValueError('Prepared rows or image dimensions do not match the selected dataset.')
    source = TensorDataset(torch.from_numpy(images.images).float(),torch.from_numpy(images.labels.astype(np.int64)))
    kept = partition_dataset(source,images.sample_ids,manifest)['keep']
    original = training_input(source,images.sample_ids,dataset_version=f'{version}/original',split='train')
    filtered = training_input(kept,kept.sample_ids,dataset_version=f'{version}/filtered',split='train')
    # Leila: the benchmark reference uses official clean rows with matching IDs.
    # It is evaluation-only and never supplies labels to scanning or cleaning.
    clean_data = load_image_dataset(dataset,root=str(ARTIFACTS.parent/'data'),train=True,download=False)
    clean_indices = []
    for sid in images.sample_ids:
        match = re.fullmatch(rf'{dataset}-train:(0|[1-9][0-9]*)',str(sid))
        if match is None or int(match[1]) >= len(clean_data):
            raise ValueError('Sample ID cannot be mapped to the official clean training split.')
        clean_indices.append(int(match[1]))
    reference = training_input(Subset(clean_data,clean_indices),images.sample_ids,
        dataset_version=f'{dataset}-official-clean-train/matched-input-ids',split='train')
    # Leila: this changes benchmark workload only, never the kept/removed selection.
    progress(3,'Checking whether the input matches the clean reference')
    skip_before = matches_clean_reference(source,reference.dataset)
    test_data = load_image_dataset(dataset,root=str(ARTIFACTS.parent/'data'),train=False,download=False)
    test = training_input(test_data,[f'{dataset}-test:{i}' for i in range(len(test_data))],
                          dataset_version=f'{dataset}-official-test',split='test')
    # Leila: verify the actual CIFAR trigger before evaluating any trained arm.
    from .image_backdoor import patch_specification, add_patch_metrics
    patch_spec = patch_specification(manifest, images, reference.dataset)
    config = TrainConfig(epochs=epochs,batch_size=128,seed=seed,device='cpu')
    report = dict(dataset=dataset,version=version,scan_id=manifest['scan_id'],review_revision=manifest['review_revision'],
        preparation=manifest['summary'],runs={},status='running',before_cleaning_skipped=skip_before,
        accuracy_change_reference='clean_reference' if skip_before else 'before_cleaning',
        limitation='Clean reference uses the official clean training rows matching the input IDs, for benchmark evaluation only. Label-flip comparison, one training seed. Same epochs but filtering changes optimizer steps. Image patch/blended-noise ASR is evaluated when a matching attack is verified.')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    # Leila: identical clean input needs only the reference and filtered models.
    arms = [('clean_reference',reference)]
    if not skip_before: arms.append(('before_cleaning',original))
    arms.append(('after_cleaning',filtered))
    validation,arms,report['validation']=validation_split(reference,arms)
    # Leila: reference-arm size fixes the same update budget for every arm.
    if matched_steps:
        from dataclasses import replace
        config=replace(config,max_steps=epochs*((len(arms[0][1].dataset)+127)//128))
        report['limitation']='Matched optimizer steps within this seed. Official clean reference is evaluation-only. Image patch/blended-noise ASR uses non-target official test images when applicable.'
    step = 90 // len(arms)
    for index,(name,inputs) in enumerate(arms):
        progress(5+index*step,f'Model {index+1} of {len(arms)}: {name.replace("_"," ")}')
        def update(message):
            match = re.search(r'Epoch (\d+)/',message)
            epoch = int(match[1]) if match else 0
            progress(5+index*step+min(step-2,int((step-2)*epoch/epochs)),f'Model {index+1} of {len(arms)} \u00b7 {message}')
        run = train_classifier(inputs,test,model_factory=(lambda: small_cnn(channels=1)) if dataset=='mnist' else small_cnn,model_name='small_cnn_mnist_v1' if dataset=='mnist' else 'small_cnn_v1',num_classes=10,
                               output_root=output/name,config=config,progress=update,validation=validation)
        if patch_spec is not None:
            progress(5+index*step+step-1, f'Evaluating image-trigger ASR: {name}')
            add_patch_metrics(run, test, patch_spec)
        report['runs'][name] = run
        (output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    report['status'] = 'complete'
    report['accuracy_change'] = report['runs']['after_cleaning']['metrics']['accuracy']-report['runs'][report['accuracy_change_reference']]['metrics']['accuracy']
    report['result_file'] = str(output/'comparison.json')
    (output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report

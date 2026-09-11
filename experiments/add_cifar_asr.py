"""Leila: evaluate saved image checkpoints without retraining; save a new result job."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import uuid
import torch
from torch.utils.data import Subset
from poison_features import ImageInputBundle, load_image_dataset
from training.preparation import ARTIFACTS, load_preparation, file_hash
from training.connector import training_input
from training.image_backdoor import patch_specification, add_patch_metrics
from training.benchmark import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comparison', type=Path)
    args = parser.parse_args()
    report = json.loads(args.comparison.read_text(encoding='utf-8'))
    if report.get('status') != 'complete' or report.get('dataset') not in ('cifar10', 'mnist'):
        raise ValueError('Choose a completed image comparison.')
    dataset = report['dataset']
    manifest = load_preparation(report['version'])
    source = Path(manifest['source_images'])
    if file_hash(source) != manifest['source_sha256']:
        raise ValueError('Prepared source images changed.')
    images = ImageInputBundle.load(source)
    if images.sample_ids.tolist() != manifest['sample_ids']:
        raise ValueError('Prepared source IDs changed.')
    clean = load_image_dataset(dataset, root=str(ARTIFACTS.parent/'data'), train=True, download=False)
    indices = []
    for sid in images.sample_ids:
        if not str(sid).startswith(f'{dataset}-train:'):
            raise ValueError('Expected official image training IDs.')
        indices.append(int(str(sid).split(':')[1]))
    spec = patch_specification(manifest, images, Subset(clean, indices))
    if spec is None:
        raise ValueError('This comparison is not a supported image image-trigger attack.')
    data = load_image_dataset(dataset, root=str(ARTIFACTS.parent/'data'), train=False, download=False)
    test = training_input(data, [f'{dataset}-test:{i}' for i in range(len(data))], dataset_version=f'{dataset}-official-test', split='test')
    job = uuid.uuid4().hex
    out = ARTIFACTS/'web_training'/job
    out.mkdir()
    torch.set_num_threads(4)
    reports = deepcopy(report.get('benchmark', {}).get('reports', [report]))
    for index, item in enumerate(reports):
        for name, run in item['runs'].items():
            print(f'Evaluating seed {run["settings"]["seed"]}: {name}', flush=True)
            folder = out/f'seed-{run["settings"]["seed"]}'/name
            folder.mkdir(parents=True)
            run['artifacts']['report'] = str(folder/'report.json')
            add_patch_metrics(run, test, spec)
            print(run['backdoor_metrics'], flush=True)
        item['limitation'] = 'Saved final checkpoints evaluated without retraining. image image-trigger ASR uses non-target official test images with the verified demo trigger.'
        item['result_file'] = str(out/f'seed-{item["runs"]["clean_reference"]["settings"]["seed"]}'/'comparison.json')
        Path(item['result_file']).write_text(json.dumps(item, allow_nan=False), encoding='utf-8')
    result = deepcopy(reports[0])
    if 'benchmark' in report:
        result['benchmark'] = dict(report['benchmark'], reports=reports, summary=summarize(reports))
    result.update(result_file=str(out/'comparison.json'), asr_source_comparison=str(args.comparison.resolve()))
    (out/'comparison.json').write_text(json.dumps(result, allow_nan=False), encoding='utf-8')
    (out/'job.json').write_text(json.dumps(dict(job_id=job,version=result['version'],status='complete',progress=100,message='Saved models evaluated with image image-trigger ASR',result=result),allow_nan=False),encoding='utf-8')
    print('ASR_JOB', job, flush=True)


if __name__ == '__main__':
    main()

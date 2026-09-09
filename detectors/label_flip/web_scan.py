"""Local UI scan service. Detectors receive features, IDs and supplied labels only."""
from hashlib import sha256
from importlib.metadata import version
import json
import re
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from poison_features import FeatureBundle, ImageInputBundle
from detectors.output_connector import to_jsonable
from .assessment import assess_label_flips
from .feature_inputs import paired_feature_inputs
from .calibrated_pipeline import scan_calibrated_label_flips
from .thresholds import calibrated_profile

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'artifacts'


def web_profile():
    return dict(name=calibrated_profile('resnet18')['profile_name'],
        versions=calibrated_profile('resnet18')['dependencies'],
        encoders={name: calibrated_profile(name) for name in ('resnet18', 'dinov2_vits14')},
        dinov2_revision='7764ea0f912e53c92e82eb78a2a1631e92725fc8',
        limitation='Experimental CIFAR-10 thresholds calibrated on 25,000 clean samples. '
        'The full 50,000-row benchmark includes calibration rows and is not independent validation. '
        'Other scan sizes are not validated. ResNet18 kNN has no flags under its strict >1.0 cutoff. '
        'Flags suggest review, not proven poisoning; not flagged does not mean clean.')


def feature_pair(feature_file, artifacts=ARTIFACTS):
    """Resolve only locally generated artifacts; never accept uploads or other paths."""
    artifacts = Path(artifacts).resolve()
    source = Path(feature_file)
    if not source.is_absolute():
        source = artifacts.parent / source
    source = source.resolve()
    if source.parent != artifacts or not source.name.endswith('-features.npz'):
        raise ValueError('Choose a saved feature extraction from this project.')
    if not source.is_file():
        raise ValueError('The saved feature file is missing. Extract features first.')
    if not source.name.startswith('cifar10-train-'):
        raise ValueError('This paired label-flip preset currently supports CIFAR-10 training data only.')
    name = source.name
    if '-dinov2-' in name:
        dino = source
        candidates = [name.replace('-dinov2-', '-resnet18-', 1), name.replace('-dinov2-', '-', 1)]
        # Earlier clean/label-flip extractions omitted the encoder and patch parameters.
        candidates.append(re.sub(r'-(none|label_flip)-a[\d.]+-t\d+-nrate-', r'-\1-', candidates[1]))
        resnet = next((artifacts / n for n in candidates if (artifacts / n).is_file()), artifacts / candidates[0])
    else:
        resnet = source
        if '-resnet18-' in name:
            other = name.replace('-resnet18-', '-dinov2-', 1)
        else:
            parts = name.split('-', 3)
            other = '-'.join(parts[:3]) + '-dinov2-' + parts[3]
        dino = artifacts / other
    for encoder, path in [('ResNet18', resnet), ('DINOv2', dino)]:
        if path.resolve().parent != artifacts:
            raise ValueError('Feature files must remain inside the artifacts folder.')
        if not path.is_file():
            raise ValueError(f'Extract {encoder} with the same dataset, scope and attack settings first.')
        pixels = path.with_name(path.name.replace('-features.npz', '-images.npz'))
        if not pixels.is_file() or pixels.resolve().parent != artifacts:
            raise ValueError(f'Matching {encoder} image file is missing. Rebuild that extraction.')
    return resnet, dino


def saved_feature_pairs(artifacts=ARTIFACTS):
    """List available pairs; pixel/label identity is still verified before scoring."""
    pairs = []
    for path in sorted(Path(artifacts).glob('cifar10-train-*-dinov2-*-features.npz')):
        try:
            resnet, dino = feature_pair(str(path), artifacts)
        except ValueError:
            continue
        pairs.append(dict(feature_file=str(dino), label=dino.name.removesuffix('-features.npz'),
                          files=[resnet.name, dino.name]))
    return sorted(pairs, key=lambda item: ('-none-' not in item['label'], item['label']))


def availability(feature_file):
    try:
        paths = feature_pair(feature_file)
        return dict(ready=True, message='Both feature files found. Scan will verify matching images and labels.',
                    files=[p.name for p in paths])
    except (ValueError, TypeError) as exc:
        return dict(ready=False, message=str(exc))


def _image_fingerprint(feature_path, bundle):
    path = feature_path.with_name(feature_path.name.replace('-features.npz', '-images.npz'))
    images = ImageInputBundle.load(path)
    if (not np.array_equal(images.sample_ids, bundle.sample_ids)
            or not np.array_equal(images.labels, bundle.labels)):
        raise ValueError('Saved images do not match the feature sample IDs and labels.')
    values = np.ascontiguousarray(images.images)
    return (values.shape, sha256(memoryview(values).cast('B')).hexdigest())


def scan_inputs(inputs, progress):
    """Use the shared calibrated pipeline; only translate its progress for the UI."""
    scans, rows = {}, []
    for encoder_index, encoder in enumerate(('resnet18', 'dinov2_vits14')):
        step = encoder_index * 3

        def report(message):
            nonlocal step
            if ' of 3: ' in message:
                local_step, detail = message.split(' of 3: ', 1)
                step = encoder_index * 3 + int(local_step)
            else:
                detail = f'{encoder}: {message}'
            progress(step, f'{step} of 6: {detail}')

        with threadpool_limits(limits=4):
            scans[encoder] = scan_calibrated_label_flips(inputs[encoder], encoder=encoder, progress=report)
        for name, result in scans[encoder]['detectors'].items():
            flags = result['flags']
            rows.append(dict(encoder=encoder, detector=name, flagged=int(flags.sum()),
                             rate=float(flags.mean()), threshold=result['settings']['detector_threshold']))
    assessment = assess_label_flips(scans['resnet18'], scans['dinov2_vits14'])
    return scans, assessment, rows


def run_scan(feature_file, output_dir, progress):
    progress(0, 'Checking the saved features, images and supplied labels')
    resnet_path, dino_path = feature_pair(feature_file)
    profile = web_profile()
    for package, expected in profile['versions'].items():
        if version(package) != expected:
            raise ValueError(f'This scan profile requires {package} {expected}; install the project requirements.')
    resnet, dino = FeatureBundle.load(resnet_path), FeatureBundle.load(dino_path)
    if resnet.dataset_name != 'cifar10' or dino.dataset_name != 'cifar10':
        raise ValueError('This scan profile is for CIFAR-10 only.')
    if (dino.metadata or {}).get('encoder_revision') != profile['dinov2_revision']:
        raise ValueError('DINOv2 revision differs from the calibrated extractor. Re-extract features.')
    inputs = paired_feature_inputs(resnet, dino)
    if _image_fingerprint(resnet_path, resnet) != _image_fingerprint(dino_path, dino):
        raise ValueError('The two extractors used different images. Rebuild both with identical settings.')
    labels = inputs['resnet18'].y
    _, counts = np.unique(labels, return_counts=True)
    if len(labels) <= 20 or len(counts) < 2 or counts.min() < 5:
        raise ValueError('Scan needs at least 21 samples and at least five samples in each of two or more classes.')
    # Discard bundles containing evaluation-only metadata before calling detectors.
    del resnet, dino
    scans, assessment, rows = scan_inputs(inputs, progress)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    full = dict(profile=profile, feature_files=[str(resnet_path), str(dino_path)],
                scans=scans, assessment=assessment)
    (output_dir / 'results.json').write_text(json.dumps(to_jsonable(full), allow_nan=False), encoding='utf-8')
    selected = np.flatnonzero(assessment['flags'])[:24]
    examples = [dict(sample_id=str(assessment['sample_ids'][i]), label=to_jsonable(labels[i]),
        assessment=str(assessment['assessment'][i]),
        resnet_votes=int(assessment['vote_counts']['resnet18'][i]),
        dino_votes=int(assessment['vote_counts']['dinov2'][i])) for i in selected]
    return dict(samples=len(labels), summary=assessment['summary'], detectors=rows,
        examples=examples, profile=profile['name'], limitation=profile['limitation'],
        result_file=str(output_dir / 'results.json'), human_review_enabled=False)

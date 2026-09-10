"""Create immutable training selections from scan evidence and saved human choices."""
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import uuid

import numpy as np

from cleaning import human_review

ARTIFACTS = Path(__file__).resolve().parents[1] / 'artifacts'
PREPARATIONS = ARTIFACTS / 'training_preparations'


def file_hash(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


# Leila: retain this public import while preview and training share one selection contract.
from cleaning.selection import merge_choices


def prepare_dataset(scan_id):
    scan_path, scan, digest = human_review.record(scan_id)
    with human_review.LOCK:
        reviews = deepcopy(human_review._reviews(scan_path,digest))
    dataset = scan.get('dataset','cifar10')
    if dataset not in ('mnist','cifar10','imdb'): raise ValueError('Unsupported training dataset.')
    # Leila: use the same combined review connector as the review page.
    selection = merge_choices(human_review.review_assessment(scan),reviews,keep_uncertain=False)
    source = human_review.text_feature_path(scan) if dataset=='imdb' else human_review.image_path(scan)
    original_hash = file_hash(source)
    with np.load(source,allow_pickle=False) as images:
        ids, labels = images['sample_ids'], images['labels']
    if ids.tolist() != selection['sample_ids']:
        raise ValueError('Image sample IDs differ from the completed scan.')
    feature = Path(scan['feature_files'][0])
    if not feature.is_absolute(): feature = ARTIFACTS.parent/feature
    if feature.resolve().parent != ARTIFACTS.resolve():
        raise ValueError('Feature input must belong to the local artifacts directory.')
    with np.load(feature,allow_pickle=False) as bundle:
        if not np.array_equal(bundle['sample_ids'],ids) or not np.array_equal(bundle['labels'],labels):
            raise ValueError('Saved image labels and feature labels no longer match.')
    # Leila: verify legacy text row identities before freezing a selection.
    if dataset=='imdb':
        from poison_features import FeatureBundle, load_imdb_dataset
        from poison_features.imdb_identity import imdb_training_indices
        imdb_training_indices(FeatureBundle.load(feature),feature,load_imdb_dataset(split='train',cache_dir=str(ARTIFACTS.parent/'data')))
    if (labels.shape != ids.shape or labels.dtype.kind not in 'iu'
            or np.any((labels<0)|(labels>=10))
            or (dataset!='imdb' and any(not str(sid).startswith(f'{dataset}-train:') for sid in ids))):
        raise ValueError('Expected matching training sample IDs and supplied class labels.')
    kept = np.asarray(selection['actions']) == 'keep'
    if not kept.any(): raise ValueError('No samples remain for training. Review the flagged samples first.')
    version = uuid.uuid4().hex
    manifest = dict(schema_version='1.0',version=version,scan_id=scan_id,scan_sha256=digest,
        created_at=datetime.now(timezone.utc).isoformat(),dataset=dataset,source_images=str(source),
        source_sha256=original_hash,review_revision=reviews['revision'],review_snapshot=reviews['decisions'],
        policy_version='4.0-quarantine-unreviewed',
        policy='human_choices_override; keep_unflagged; quarantine_suspected; hold_uncertain_or_unsure',
        class_counts=np.bincount(labels[kept].astype(int),minlength=2 if dataset=='imdb' else 10).tolist(),**selection)
    out = PREPARATIONS/version
    out.mkdir(parents=True,exist_ok=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    for action in ('keep','quarantine','human_review'):
        np.save(out/f'{action}_ids.npy',ids[np.asarray(selection['actions'])==action],allow_pickle=False)
    return preparation_summary(manifest)


def load_preparation(version):
    if not isinstance(version,str) or not re.fullmatch('[0-9a-f]{32}',version):
        raise ValueError('Invalid training dataset version.')
    path = (PREPARATIONS/version/'manifest.json').resolve()
    if path.parent.parent != PREPARATIONS.resolve() or not path.is_file():
        raise ValueError('Prepared training dataset was not found.')
    data = json.loads(path.read_text(encoding='utf-8'))
    if data['version'] != version: raise ValueError('Training dataset version mismatch.')
    return data


def preparation_summary(data):
    # Leila: identify the saved build for chart filenames; unknown legacy builds stay unknown.
    source_name = Path(data.get('source_images','')).name
    attack = next((kind for kind in ('label_flip','backdoor','blended','none') if f'-{kind}-' in source_name), 'unknown')
    # Leila: expose the frozen policy so older training runs keep accurate captions.
    return dict({key:data[key] for key in ('version','scan_id','created_at','dataset','summary','review_revision','class_counts')},
                policy_version=data.get('policy_version','1.0'), attack=attack)

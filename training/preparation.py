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


def merge_choices(assessment, saved):
    """Saved reviews override scanner defaults; suspected rows are quarantined."""
    ids = assessment['sample_ids']
    states = assessment['assessment']
    if (len(ids) != len(states) or len(set(ids)) != len(ids) or not ids
            or any(s not in ('not_flagged','uncertain','suspected_label_flip') for s in states)):
        raise ValueError('Invalid or incomplete scanner assessment.')
    allowed = {sid for sid,state in zip(ids,states) if state != 'not_flagged'}
    decisions = saved['decisions']
    if any(sid not in allowed or value.get('decision') not in human_review.CHOICES for sid,value in decisions.items()):
        raise ValueError('Saved reviews contain an invalid choice or unknown sample ID.')
    actions, reasons = [], []
    for sid, state in zip(ids,states):
        choice = decisions.get(sid, {}).get('decision')
        if choice in ('keep','quarantine'):
            actions.append(choice); reasons.append('human_'+choice)
        elif state == 'not_flagged':
            actions.append('keep'); reasons.append('scanner_not_flagged')
        # Leila: quarantine scanner suspects unless a saved Unsure requests review.
        elif state == 'suspected_label_flip' and choice != 'unsure':
            actions.append('quarantine'); reasons.append('scanner_suspected')
        else:
            actions.append('human_review'); reasons.append('human_unsure' if choice == 'unsure' else 'flagged_unreviewed')
    return dict(sample_ids=ids, actions=actions, reasons=reasons,
                summary=dict(kept=actions.count('keep'),quarantined=actions.count('quarantine'),
                             unresolved=actions.count('human_review'),total=len(ids)))


def prepare_dataset(scan_id):
    scan_path, scan, digest = human_review.record(scan_id)
    with human_review.LOCK:
        reviews = deepcopy(human_review._reviews(scan_path,digest))
    selection = merge_choices(scan['assessment'],reviews)
    source = human_review.image_path(scan)
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
    if (labels.shape != ids.shape or labels.dtype.kind not in 'iu'
            or np.any((labels<0)|(labels>=10))
            or any(not str(sid).startswith('cifar10-train:') for sid in ids)):
        raise ValueError('Expected CIFAR-10 training samples and supplied class labels.')
    kept = np.asarray(selection['actions']) == 'keep'
    if not kept.any(): raise ValueError('No samples remain for training. Review the flagged samples first.')
    version = uuid.uuid4().hex
    manifest = dict(schema_version='1.0',version=version,scan_id=scan_id,scan_sha256=digest,
        created_at=datetime.now(timezone.utc).isoformat(),dataset='cifar10',source_images=str(source),
        source_sha256=original_hash,review_revision=reviews['revision'],review_snapshot=reviews['decisions'],
        policy_version='2.0',
        policy='human_choices_override; keep_unflagged; quarantine_suspected; hold_uncertain_or_unsure',
        class_counts=np.bincount(labels[kept].astype(int),minlength=10).tolist(),**selection)
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
    # Leila: expose the frozen policy so older training runs keep accurate captions.
    return dict({key:data[key] for key in ('version','scan_id','created_at','dataset','summary','review_revision','class_counts')},
                policy_version=data.get('policy_version','1.0'))

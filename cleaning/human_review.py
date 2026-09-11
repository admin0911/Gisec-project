"""Persist human choices separately from immutable scan evidence and image data."""
import base64
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import threading
import uuid

import numpy as np
from PIL import Image

ARTIFACTS = Path(__file__).resolve().parents[1] / 'artifacts'
LOCK = threading.RLock()
# Leila: one shared review view includes both label and patch findings.
from .review_assessment import review_assessment

GROUPS = ('uncertain', 'suspected_label_flip')
CHOICES = ('keep', 'quarantine', 'unsure')
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class ReviewConflict(ValueError):
    pass


def result_path(job_id):
    if not isinstance(job_id, str) or not re.fullmatch('[0-9a-f]{32}', job_id):
        raise ValueError('Invalid scan ID.')
    root = (ARTIFACTS / 'label_flip_scans').resolve()
    path = (root / job_id / 'results.json').resolve()
    if path.parent.parent != root or not path.is_file():
        raise ValueError('Completed scan results were not found.')
    return path


@lru_cache(maxsize=2)
def _record(path, modified, size):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    ids = data['assessment']['sample_ids']
    if len(set(ids)) != len(ids):
        raise ValueError('Scan sample IDs are not unique.')
    return data, sha256(raw).hexdigest()


def record(job_id):
    path = result_path(job_id)
    stat = path.stat()
    data, digest = _record(str(path), stat.st_mtime_ns, stat.st_size)
    return path, data, digest


# Leila: text stages share the saved FeatureBundle path, never fabricate image inputs.
def text_feature_path(data):
    path = Path(data['feature_files'][0])
    if not path.is_absolute(): path = ARTIFACTS.parent/path
    path = path.resolve()
    if path.parent != ARTIFACTS.resolve() or not path.is_file():
        raise ValueError('Matching text features are unavailable.')
    return path


def image_path(data):
    path = Path(data['feature_files'][0])
    if not path.is_absolute():
        path = ARTIFACTS.parent / path
    path = path.with_name(path.name.replace('-features.npz', '-images.npz')).resolve()
    if path.parent != ARTIFACTS.resolve() or not path.is_file():
        raise ValueError('The matching local image file is unavailable.')
    return path


@lru_cache(maxsize=1)
def _thumbnails(path, modified, size, ids, states):
    """Keep only review thumbnails in memory, not the full training image array."""
    with np.load(path, allow_pickle=False) as archive:
        if archive['sample_ids'].tolist() != list(ids):
            raise ValueError('Saved images and scan sample IDs do not match.')
        labels = archive['labels'].tolist()
        images = archive['images']
        thumbs = {}
        for row, state in enumerate(states):
            if state not in GROUPS:
                continue
            pixels = np.moveaxis(np.rint(images[row] * 255).clip(0,255).astype('uint8'), 0, -1)
            if pixels.shape[-1] == 1:
                pixels = pixels[..., 0]
            buffer = BytesIO()
            Image.fromarray(pixels).save(buffer, format='PNG')
            thumbs[ids[row]] = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
    return labels, thumbs


def _reviews(path, digest):
    saved = path.with_name('human_review.json')
    if not saved.exists():
        return dict(schema_version='1.0', scan_sha256=digest, revision=0, decisions={}, history=[])
    result = json.loads(saved.read_text(encoding='utf-8'))
    if result['scan_sha256'] != digest:
        raise ReviewConflict('Scan evidence changed. Existing reviews belong to an earlier scan.')
    return result


def review_summary(job_id):
    """Count current saved choices only, without changing detector results or loading images."""
    path, data, digest = record(job_id)
    a = review_assessment(data)
    with LOCK:
        saved = _reviews(path, digest)
    choices = [saved['decisions'].get(str(sid), {}).get('decision')
               for sid,state in zip(a['sample_ids'], a['assessment']) if state in GROUPS]
    counts = {name: choices.count(name) for name in CHOICES}
    return dict(**counts, unreviewed=choices.count(None), total=len(choices),
                saved=sum(counts.values()), revision=saved['revision'])


def review_page(job_id, group='uncertain', page=0, page_size=20):
    if group not in GROUPS or type(page) is not int or page < 0 or type(page_size) is not int or page_size not in (20,50,100):
        raise ValueError('Choose a valid review group, page and page size (20, 50 or 100).')
    path, data, digest = record(job_id)
    a = review_assessment(data)
    rows = [i for i, state in enumerate(a['assessment']) if state == group]
    pages = max(1, (len(rows) + page_size - 1) // page_size)
    if page >= pages:
        raise ValueError('This review page does not exist.')
    with LOCK:
        saved = _reviews(path, digest)
    imdb = data.get('dataset') == 'imdb'
    if imdb:
        from poison_features import FeatureBundle, load_imdb_dataset
        import re
        bundle = FeatureBundle.load(text_feature_path(data))
        if bundle.sample_ids.tolist() != a['sample_ids']: raise ValueError('Text IDs differ from scan.')
        labels = bundle.labels
        # Leila: show submitted post-attack text, retaining legacy clean/label-flip fallback.
        text_path=text_feature_path(data).with_name(text_feature_path(data).name.replace('-features.npz','-texts.jsonl'))
        if text_path.exists():
            from poison_features.text_inputs import TextInputBundle
            text_inputs=TextInputBundle.load(text_path)
            if not np.array_equal(text_inputs.sample_ids,bundle.sample_ids) or not np.array_equal(text_inputs.labels,labels): raise ValueError('Review text differs from features.')
            texts=text_inputs.texts
        else:
            if '-backdoor-' in text_path.name: raise ValueError('Backdoor review text is missing; rebuild the dataset.')
            reviews_data=load_imdb_dataset(split='train',cache_dir=str(ARTIFACTS.parent/'data'))
            from poison_features.imdb_identity import imdb_training_indices
            indices=imdb_training_indices(bundle,text_feature_path(data),reviews_data)
            texts=[reviews_data[int(i)]['text'] for i in indices]
        thumbs={a['sample_ids'][i]:texts[i] for i in rows[page*page_size:(page+1)*page_size]}

    else:
        pixels = image_path(data)
        stat = pixels.stat()
        labels, thumbs = _thumbnails(str(pixels), stat.st_mtime_ns, stat.st_size,
                                     tuple(a['sample_ids']), tuple(a['assessment']))
    items = []
    for i in rows[page * page_size:(page + 1) * page_size]:
        sid = a['sample_ids'][i]
        label = int(labels[i])
        # Leila: MNIST reviews show digits and pixel votes, never CIFAR class names or invented encoder votes.
        mnist = data.get('dataset') == 'mnist'
        items.append(dict(sample_id=sid, label=label, class_name=('positive' if label else 'negative') if imdb else str(label) if mnist else CLASSES[label] if 0 <= label < 10 else str(label),
            image=None if imdb else thumbs[sid],text=thumbs[sid] if imdb else None,text_votes=a['vote_counts']['minilm'][i] if imdb else None, resnet_votes=None if (mnist or imdb) else a['vote_counts']['resnet18'][i],
            dino_votes=None if (mnist or imdb) else a['vote_counts']['dinov2'][i],
            pixel_votes=a['vote_counts']['pixels'][i] if mnist else None, decision=saved['decisions'].get(str(sid), {}).get('decision'),
            phrase_flagged=bool(data.get('phrase_scan',{}).get('flags',[False]*len(a['sample_ids']))[i]),
            # Leila: show why a sample entered the combined review queue.
            blended_flagged=bool(data.get('blended_scan',{}).get('flags',[False]*len(a['sample_ids']))[i]),
            patch_flagged=bool(data.get('patch_scan',{}).get('flags',[False]*len(a['sample_ids']))[i]),
            feature_backdoor_flagged=bool(data.get('backdoor_feature',{}).get('candidate_flags',[False]*len(a['sample_ids']))[i]),
            assessment=a['assessment'][i]))
    decisions = [saved['decisions'].get(str(a['sample_ids'][i]), {}).get('decision') for i in rows]
    return dict(items=items, page=page, page_size=page_size, pages=pages, total=len(rows),
        revision=saved['revision'], group=group,
        resolved=sum(d in ('keep','quarantine') for d in decisions), unsure=decisions.count('unsure'),
        unreviewed=decisions.count(None), counts={g:a['assessment'].count(g) for g in GROUPS})


def save_review(job_id, changes, revision):
    if not isinstance(changes, dict) or not 0 < len(changes) <= 100 or type(revision) is not int:
        raise ValueError('Save 1–100 explicit sample choices with a review revision.')
    path, data, digest = record(job_id)
    a = review_assessment(data)
    # Leila: JSON decision keys are strings even for legacy numeric IMDB IDs.
    changes = {str(sid): decision for sid, decision in changes.items()}
    allowed = {str(sid) for sid,state in zip(a['sample_ids'], a['assessment']) if state in GROUPS}
    if any(sid not in allowed or decision not in CHOICES for sid,decision in changes.items()):
        raise ValueError('Only reviewable sample IDs and Keep, Quarantine or Unsure choices are accepted.')
    with LOCK:
        saved = _reviews(path, digest)
        if saved['revision'] != revision:
            raise ReviewConflict('Another review was saved. Reload this page before saving your choices again.')
        now = datetime.now(timezone.utc).isoformat()
        for sid, decision in changes.items():
            saved['history'].append(dict(sample_id=sid, previous=saved['decisions'].get(sid), decision=decision, saved_at=now))
            saved['decisions'][sid] = dict(decision=decision, saved_at=now)
        saved['revision'] += 1
        saved['job_id'] = job_id
        saved['scope'] = 'label_flip_and_patch_review; applied only through a new preparation'
        target = path.with_name('human_review.json')
        temporary = target.with_name(f'.human-review-{uuid.uuid4().hex}.tmp')
        try:
            temporary.write_text(json.dumps(saved, indent=2, allow_nan=False), encoding='utf-8')
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return dict(saved=len(changes), revision=saved['revision'])


def restored_scan_job(job_id):
    """Recover a completed results page after server restart without rescanning."""
    path, data, _ = record(job_id)
    # Leila: restore the persisted MNIST presentation without assuming two image encoders.
    # Leila: new CIFAR scans also preserve patch presentation in ui_result.
    if 'ui_result' in data:
        return dict(job_id=job_id,status='complete',progress=100,message='Saved scan loaded',result=data['ui_result'])
    a = data['assessment']
    rows = []
    for encoder, scan in data['scans'].items():
        for name, result in scan['detectors'].items():
            flags = result['flags']
            rows.append(dict(encoder=encoder,detector=name,flagged=sum(flags),rate=sum(flags)/len(flags),
                threshold=result['settings']['detector_threshold']))
    with np.load(image_path(data), allow_pickle=False) as images:
        if images['sample_ids'].tolist() != a['sample_ids']:
            raise ValueError('Saved image IDs do not match this scan.')
        labels = images['labels'].tolist()
    flagged = [i for i, state in enumerate(a['assessment']) if state in GROUPS][:24]
    examples = [dict(sample_id=a['sample_ids'][i], label=labels[i], assessment=a['assessment'][i],
        resnet_votes=a['vote_counts']['resnet18'][i], dino_votes=a['vote_counts']['dinov2'][i]) for i in flagged]
    result = dict(samples=len(a['sample_ids']),summary=a['summary'],detectors=rows,examples=examples,
        profile=data['profile']['name'],limitation=data['profile']['limitation'],result_file=str(path),human_review_enabled=True)
    return dict(job_id=job_id,status='complete',progress=100,message='Saved label-flip scan loaded',result=result)

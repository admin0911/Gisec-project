"""Read-only demo evaluation, separate from detector decisions and training."""
from pathlib import Path
import numpy as np
from .human_review import record, image_path, _reviews
from .selection import merge_choices


def evaluate_scan(job_id):
    result_path, data, digest = record(job_id)
    if data.get('dataset') != 'mnist':
        return dict(available=False, reason='Demo scan evaluation is currently available for MNIST.')
    image = image_path(data)
    truth_path = image.with_name(image.name.replace('-images.npz','-evaluation.npz'))
    if not truth_path.is_file():
        truth_path = image.with_name(image.name.replace('-images.npz','-features.npz'))
    if not truth_path.is_file():
        return dict(available=False, reason='Known poison metadata is unavailable; precision and recall cannot be calculated.')
    ids = np.asarray(data['assessment']['sample_ids'])
    with np.load(truth_path, allow_pickle=False) as saved:
        if 'is_poisoned' not in saved.files:
            return dict(available=False, reason='Known poison metadata is unavailable.')
        if not np.array_equal(saved['sample_ids'],ids):
            raise ValueError('Evaluation metadata and scan sample IDs do not match.')
        raw = saved['is_poisoned']
        if raw.shape != ids.shape or not np.isin(raw,[0,1]).all():
            raise ValueError('Invalid poison metadata.')
        truth = raw.astype(bool)
    def metrics(name, flags):
        flags = np.asarray(flags,dtype=bool)
        if flags.shape != truth.shape:
            raise ValueError('Detector flags and evaluation sample IDs do not match.')
        tp = int(np.sum(flags & truth)); fp = int(np.sum(flags & ~truth))
        fn = int(np.sum(~flags & truth)); tn = int(np.sum(~flags & ~truth))
        return dict(name=name,true_positives=tp,false_positives=fp,false_negatives=fn,
            precision=tp/(tp+fp) if tp+fp else None,
            recall=tp/(tp+fn) if tp+fn else None,
            false_positive_rate=fp/(fp+tn) if fp+tn else None)
    states = np.asarray(data['assessment']['assessment'])
    rows = [metrics('Combined: at least 2 of 3',states == 'suspected_label_flip'),
            metrics('Any detector: includes needs review',states != 'not_flagged')]
    for name, detector in data['scans']['pixels']['detectors'].items():
        if not np.array_equal(detector['sample_ids'],ids):
            raise ValueError('Detector sample IDs do not match.')
        rows.append(metrics(name,detector['flags']))
    # Leila: use exactly the same selection connector as frozen training preparation.
    saved_reviews = _reviews(result_path,digest) if result_path is not None else {'decisions':{}}
    selection = merge_choices(data['assessment'],saved_reviews,keep_uncertain=False)
    removed = np.asarray(selection['actions']) != 'keep'
    def counts(mask):
        return dict(clean=int(np.sum(mask & ~truth)),poisoned=int(np.sum(mask & truth)),total=int(np.sum(mask)))
    return dict(original=counts(np.ones(len(ids),dtype=bool)),kept=counts(~removed),removed=counts(removed),
        available=True,samples=len(ids),known_poisoned=int(truth.sum()),
        known_clean=int((~truth).sum()),rows=rows)

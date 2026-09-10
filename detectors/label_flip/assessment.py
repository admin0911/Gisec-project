"""Label-flip evidence from both complete scans, separate from cleaning."""
from copy import deepcopy

import numpy as np


DETECTORS = ('knn', 'class_distance', 'confident_learning')


def _read_scan(scan):
    ids = np.asarray(scan['sample_ids'])
    if ids.ndim != 1 or not len(ids) or ids.dtype.kind not in 'iuUS' or len(np.unique(ids)) != len(ids):
        raise ValueError('A completed scan must contain unique integer or string sample IDs')
    votes = scan['combination_votes']
    if set(votes) != set(DETECTORS):
        raise ValueError('All three label-flip detector votes are required')
    arrays = {name: np.asarray(votes[name]) for name in DETECTORS}
    if any(a.shape != ids.shape or a.dtype.kind != 'b' for a in arrays.values()):
        raise ValueError('Detector votes must be boolean arrays aligned with sample IDs')
    if not isinstance(scan.get('combination_settings'), dict) or not scan['combination_settings']:
        raise ValueError('Each scan must record its encoder-specific threshold settings')
    return ids, arrays


def assess_label_flips(resnet_scan, dino_scan):
    """Apply 2-of-3 within each encoder, then compare their flags by sample ID.

    Inputs contain separately calibrated combination_votes and settings.
    A partial or failed scan is an error, never an all-clear result.
    Output order matches ResNet18. No data is removed or relabeled here.
    """
    ids, resnet_votes = _read_scan(resnet_scan)
    dino_ids, dino_votes = _read_scan(dino_scan)
    positions = {key: i for i, key in enumerate(dino_ids.tolist())}
    if set(ids.tolist()) != set(positions):
        raise ValueError('Both complete scans must cover exactly the same sample IDs')
    order = np.array([positions[key] for key in ids.tolist()])
    dino_votes = {name: values[order].copy() for name, values in dino_votes.items()}
    counts = {
        'resnet18': np.column_stack(list(resnet_votes.values())).sum(axis=1),
        'dinov2': np.column_stack(list(dino_votes.values())).sum(axis=1),
    }
    encoder_flags = {name: count >= 2 for name, count in counts.items()}
    res, din = encoder_flags['resnet18'], encoder_flags['dinov2']
    statuses = np.full(len(ids), 'not_flagged', dtype='<U24')
    statuses[res & din] = 'suspected_label_flip'
    statuses[res ^ din] = 'uncertain'
    reasons = np.full(len(ids), 'neither_encoder_flagged', dtype='<U32')
    reasons[res & din] = 'both_encoders_flagged'
    reasons[res & ~din] = 'resnet_only_flagged'
    reasons[~res & din] = 'dinov2_only_flagged'
    return {
        'assessment_name': 'label_flip', 'version': '1.0', 'status': 'experimental',
        'sample_ids': ids.copy(), 'assessment': statuses, 'reasons': reasons,
        'flags': res | din,  # Review signal includes uncertainty; not a removal instruction.
        'encoder_flags': encoder_flags, 'vote_counts': counts,
        'detector_votes': {'resnet18': {k: v.copy() for k, v in resnet_votes.items()},
                           'dinov2': dino_votes},
        'summary': {name: int(np.sum(statuses == name)) for name in
                    ('not_flagged', 'suspected_label_flip', 'uncertain')},
        'settings': {'minimum_votes_per_encoder': 2,
                     'resnet18': deepcopy(resnet_scan['combination_settings']),
                     'dinov2': deepcopy(dino_scan['combination_settings'])},
    }

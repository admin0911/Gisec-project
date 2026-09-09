"""Experimental two-stage label-flip routing; no poison truth is accepted.

0 primary votes -> keep; 1/2 -> second check; 3 -> quarantine.
For routed rows: 0 secondary votes -> keep; 1/2 -> human review;
3 -> quarantine. Pending/review rows stay out of the training view.
"""
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from detectors.output_connector import to_jsonable


NAMES = ('knn', 'class_distance', 'confident_learning')
ACTIONS = ('keep', 'second_check', 'quarantine', 'human_review')


def _votes(scan):
    ids = np.asarray(scan['sample_ids'])
    if ids.ndim != 1 or ids.dtype.kind not in 'iuUS' or len(np.unique(ids)) != len(ids):
        raise ValueError('Provide unique integer or string sample IDs')
    votes = scan['combination_votes']
    if set(votes) != set(NAMES):
        raise ValueError('Expected the three named label-flip detector votes')
    for name in NAMES:
        values = np.asarray(votes[name])
        if values.shape != ids.shape or values.dtype.kind != 'b':
            raise ValueError('Each detector vote must be an aligned boolean array')
    if not isinstance(scan.get('combination_settings'), dict) or not scan['combination_settings']:
        raise ValueError('Record the thresholds/rule used to create the votes')
    return ids, np.sum(np.column_stack([votes[name] for name in NAMES]), axis=1)


def route_label_flips(primary_scan, secondary_scan=None):
    """Accept existing pipeline outputs or equivalent calibrated vote records.

    The second scan must use its own encoder-specific calibrated thresholds.
    It may cover the full input or just routed IDs, but must include every
    routed ID. Results are joined by identity, not by row position.
    This function does not extract features or invoke the second model.
    """
    ids, counts = _votes(primary_scan)
    requested = (counts == 1) | (counts == 2)
    actions = np.full(len(ids), 'keep', dtype='<U16')
    reasons = np.full(len(ids), 'no_primary_votes', dtype='<U40')
    actions[requested], reasons[requested] = 'second_check', 'primary_disagreement'
    actions[counts == 3], reasons[counts == 3] = 'quarantine', 'all_primary_detectors_flagged'
    second_counts = np.full(len(ids), -1, dtype=int)
    checked = np.zeros(len(ids), dtype=bool)
    if secondary_scan is not None:
        second_ids, second_votes = _votes(secondary_scan)
        lookup = dict(zip(second_ids.tolist(), second_votes.tolist()))
        if not set(lookup).issubset(set(ids.tolist())):
            raise ValueError('Second scan contains unknown sample IDs')
        if not set(ids[requested].tolist()).issubset(lookup):
            raise ValueError('Second scan is missing requested sample IDs')
        for index in np.flatnonzero(requested):
            count = lookup[ids[index].item()]
            second_counts[index], checked[index] = count, True
            if count == 0:
                actions[index], reasons[index] = 'keep', 'no_secondary_votes'
            elif count == 3:
                actions[index], reasons[index] = 'quarantine', 'all_secondary_detectors_flagged'
            else:
                actions[index], reasons[index] = 'human_review', 'secondary_disagreement'
    return {
        'schema_version': '1.0', 'policy_name': 'label_flip_two_stage',
        'policy_version': '1.0', 'status': 'experimental',
        'sample_ids': ids.copy(), 'actions': actions, 'reasons': reasons,
        'primary_vote_count': counts, 'secondary_vote_count': second_counts,
        'second_check_requested': requested, 'second_check_completed': checked,
        'needs_second_opinion': actions == 'second_check',
        'human_review_flags': actions == 'human_review',
        'quarantine_flags': actions == 'quarantine',
        'training_keep_mask': actions == 'keep',
        'summary': {action: int(np.sum(actions == action)) for action in ACTIONS},
        'settings': {
            'primary': deepcopy(primary_scan['combination_settings']),
            'secondary': None if secondary_scan is None else deepcopy(secondary_scan['combination_settings']),
            'primary_rule': '0 keep; 1/2 second_check; 3 quarantine',
            'secondary_rule': '0 keep; 1/2 human_review; 3 quarantine',
            'pending_and_human_review_excluded_from_training': True,
        },
    }


class DatasetView:
    """Index-only view. Leaves source images and supplied labels untouched."""
    def __init__(self, dataset, indices, sample_ids):
        self.dataset = dataset
        self.indices = np.asarray(indices).copy()
        self.sample_ids = np.asarray(sample_ids).copy()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[int(self.indices[index])]


def partition_dataset(dataset, sample_ids, decisions):
    """Build keep/quarantine/review/pending views, matching the full dataset IDs.

    Pass the actual post-attack dataset and its current labels. Never substitute
    original clean labels. A partial scan cannot silently drop unscanned rows.
    """
    ids = np.asarray(sample_ids)
    decision_ids = np.asarray(decisions['sample_ids'])
    actions = np.asarray(decisions['actions'])
    if (ids.shape != (len(dataset),) or len(np.unique(ids)) != len(ids)
            or decision_ids.ndim != 1 or len(np.unique(decision_ids)) != len(decision_ids)
            or actions.shape != decision_ids.shape
            or set(ids.tolist()) != set(decision_ids.tolist())
            or not np.isin(actions, ACTIONS).all()):
        raise ValueError('Dataset and decisions must cover exactly the same unique IDs and valid actions')
    lookup = dict(zip(decision_ids.tolist(), actions.tolist()))
    aligned = np.array([lookup[key] for key in ids.tolist()])
    return {action: DatasetView(dataset, np.flatnonzero(aligned == action), ids[aligned == action])
            for action in ACTIONS}


def save_decisions(decisions, output_root, *, source_description):
    """Save a new manifest and ID lists; never delete or modify source data."""
    out = Path(output_root) / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True, exist_ok=False)
    report = dict(decisions, source_description=source_description)
    (out / 'decisions.json').write_text(json.dumps(to_jsonable(report), indent=2, allow_nan=False), encoding='utf-8')
    for action in ACTIONS:
        np.save(out / f'{action}_ids.npy', decisions['sample_ids'][decisions['actions'] == action], allow_pickle=False)
    return out

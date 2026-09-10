"""Combine public detector evidence for review without changing original findings."""
from copy import deepcopy
import numpy as np


def review_assessment(scan):
    assessment = deepcopy(scan['assessment'])
    # Leila: text phrase and image patch flags use the same unresolved-review policy.
    patch = scan.get('phrase_scan') if scan.get('dataset') == 'imdb' else scan.get('patch_scan')
    if patch is None:
        return assessment
    ids = np.asarray(assessment['sample_ids'])
    flags = np.asarray(patch['flags'])
    if (not np.array_equal(ids, patch['sample_ids']) or flags.shape != ids.shape
            or flags.dtype.kind != 'b'):
        raise ValueError('Patch review flags must match the label-flip sample IDs')
    states = np.asarray(assessment['assessment'], dtype='<U24').copy()
    # Leila: patch-only flags request review; preserve existing suspected/quarantine routing.
    states[flags & (states == 'not_flagged')] = 'uncertain'
    assessment['assessment'] = states.tolist()
    assessment['flags'] = (states != 'not_flagged').tolist()
    assessment['summary'] = {name:int(np.sum(states == name)) for name in
                             ('not_flagged','uncertain','suspected_label_flip')}
    return assessment

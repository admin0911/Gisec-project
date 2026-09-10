"""Combine public detector evidence for review without changing original findings."""
from copy import deepcopy
import numpy as np


def review_assessment(scan):
    assessment = deepcopy(scan['assessment'])
    # Leila: text phrase and image patch flags use the same unresolved-review policy.
    patch = scan.get('phrase_scan') if scan.get('dataset') == 'imdb' else scan.get('patch_scan')
    ids = np.asarray(assessment['sample_ids'])
    states = None
    # Leila: both image datasets use noise flags without adding label votes.
    blended = scan.get('blended_scan')
    for detector in (patch, blended):
        if detector is None: continue
        flags = np.asarray(detector['flags'])
        if (not np.array_equal(ids, detector['sample_ids']) or flags.shape != ids.shape
                or flags.dtype.kind != 'b'):
            raise ValueError('Patch review flags must match the label-flip sample IDs')
        if states is None: states = np.asarray(assessment['assessment'], dtype='<U24').copy()
        if detector is blended and not detector.get('evidence',{}).get('applicable',True):
            continue  # Inconclusive is not evidence to keep or quarantine samples.
        states[flags & (states == 'not_flagged')] = 'uncertain'
    if states is None: return assessment
    assessment['assessment'] = states.tolist()
    assessment['flags'] = (states != 'not_flagged').tolist()
    assessment['summary'] = {name:int(np.sum(states == name)) for name in
                             ('not_flagged','uncertain','suspected_label_flip')}
    return assessment

"""Convert label-flip assessments into reversible training decisions."""
from copy import deepcopy

import numpy as np

from .label_flip import ACTIONS


def decide_label_flip_actions(assessment, *, human_review_enabled=False):
    """Default: keep unflagged; quarantine suspected and uncertain samples.

    This is a label-flip policy only, not clearance from other attack checks.
    Human review is opt-in; pending review is excluded from training too.
    """
    if type(human_review_enabled) is not bool:
        raise ValueError('human_review_enabled must be a boolean')
    if assessment.get('assessment_name') != 'label_flip':
        raise ValueError('Expected a label-flip assessment')
    ids = np.asarray(assessment['sample_ids'])
    states = np.asarray(assessment['assessment'])
    allowed = ('not_flagged', 'suspected_label_flip', 'uncertain')
    if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in 'iuUS'
            or len(np.unique(ids)) != len(ids) or states.shape != ids.shape
            or not np.isin(states, allowed).all()):
        raise ValueError('Assessment must contain valid states and one unique ID per sample')
    if not isinstance(assessment.get('settings'), dict) or not assessment['settings']:
        raise ValueError('Assessment settings must be recorded')
    actions = np.full(len(ids), 'keep', dtype='<U16')
    suspected, uncertain = states == 'suspected_label_flip', states == 'uncertain'
    actions[suspected] = 'quarantine'
    actions[uncertain] = 'human_review' if human_review_enabled else 'quarantine'
    reasons = np.full(len(ids), 'not_flagged_by_either_encoder', dtype='<U48')
    reasons[suspected] = 'both_encoders_flagged'
    reasons[uncertain] = ('encoder_disagreement_awaiting_review' if human_review_enabled
                          else 'encoder_disagreement_review_disabled')
    return {
        'schema_version': '1.0', 'policy_name': 'label_flip_assessment_cleaning',
        'policy_version': '1.0', 'status': 'experimental',
        'sample_ids': ids.copy(), 'assessment': states.copy(),
        'actions': actions, 'reasons': reasons,
        'quarantine_flags': actions == 'quarantine',
        'human_review_flags': actions == 'human_review',
        'training_keep_mask': actions == 'keep',
        'uncertain_flags': uncertain,
        'summary': {name: int(np.sum(actions == name)) for name in ACTIONS},
        'settings': {
            'human_review_enabled': human_review_enabled,
            'uncertain_action': 'human_review' if human_review_enabled else 'quarantine',
            'assessment_version': assessment.get('version'),
            'assessment_settings': deepcopy(assessment['settings']),
            'scope': 'label_flip_only',
        },
    }

"""Join complete attack-specific decisions into reversible dataset partitions."""
from copy import deepcopy
import numpy as np

from .label_flip import ACTIONS


def combine_decisions(sample_ids, decisions, *, required_checks):
    """Each named check supplies sample_ids, actions and nonempty settings.

    Explicit policy: quarantine > second_check > human_review > keep.
    All non-keep actions are excluded from training. No scores are averaged.
    The caller specifies required checks; missing/partial checks fail closed.
    """
    ids = np.asarray(sample_ids)
    required = list(required_checks)
    if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in 'iuUS'
            or len(np.unique(ids)) != len(ids)):
        raise ValueError('Unique nonempty integer or string sample IDs required')
    if not required or len(set(required)) != len(required) or set(required) != set(decisions):
        raise ValueError('Provide exactly the explicitly required checks')
    rank = {'keep': 0, 'human_review': 1, 'second_check': 2, 'quarantine': 3}
    severity = np.zeros(len(ids), dtype=int)
    aligned_checks = {}
    settings = {}
    for name in required:
        result = decisions[name]
        other = np.asarray(result['sample_ids'])
        actions = np.asarray(result['actions'])
        if (other.ndim != 1 or len(np.unique(other)) != len(other)
                or set(other.tolist()) != set(ids.tolist())
                or actions.shape != other.shape or not np.isin(actions, ACTIONS).all()):
            raise ValueError(f'{name}: complete IDs and valid actions required')
        if not isinstance(result.get('settings'), dict) or not result['settings']:
            raise ValueError(f'{name}: policy settings required')
        lookup = dict(zip(other.tolist(), actions.tolist()))
        aligned = np.array([lookup[key] for key in ids.tolist()])
        severity = np.maximum(severity, [rank[a] for a in aligned])
        aligned_checks[name] = aligned
        settings[name] = deepcopy(result['settings'])
    actions = np.array(['keep', 'human_review', 'second_check', 'quarantine'])[severity]
    return {
        'schema_version': '1.0', 'policy_name': 'all_required_checks',
        'policy_version': '1.0', 'status': 'experimental',
        'sample_ids': ids.copy(), 'actions': actions,
        'training_keep_mask': actions == 'keep',
        'quarantine_flags': actions == 'quarantine',
        'human_review_flags': actions == 'human_review',
        'check_actions': aligned_checks,
        'summary': {a: int(np.sum(actions == a)) for a in ACTIONS},
        'settings': {'required_checks': required, 'checks': settings,
                     'precedence': list(rank), 'exclude_all_non_keep': True},
    }

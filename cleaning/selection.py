"""Shared assessment-to-selection connector for preview and frozen training inputs."""
from . import human_review


def merge_choices(assessment, saved, *, keep_uncertain=False):
    """Saved reviews override scanner defaults; suspected rows are quarantined."""
    ids = assessment['sample_ids']
    states = assessment['assessment']
    if (len(ids) != len(states) or len(set(ids)) != len(ids) or not ids
            or any(s not in ('not_flagged','uncertain','suspected_label_flip') for s in states)):
        raise ValueError('Invalid or incomplete scanner assessment.')
    # Leila: match JSON review keys without changing the connector's original IDs.
    allowed = {str(sid) for sid,state in zip(ids,states) if state != 'not_flagged'}
    decisions = saved['decisions']
    if any(sid not in allowed or value.get('decision') not in human_review.CHOICES for sid,value in decisions.items()):
        raise ValueError('Saved reviews contain an invalid choice or unknown sample ID.')
    actions, reasons = [], []
    for sid, state in zip(ids,states):
        choice = decisions.get(str(sid), {}).get('decision')
        if choice in ('keep','quarantine'):
            actions.append(choice); reasons.append('human_'+choice)
        # Leila: MNIST keeps single-detector flags unless a human explicitly marks Unsure.
        elif keep_uncertain and state == 'uncertain' and choice is None:
            actions.append('keep'); reasons.append('single_detector_pending_review')
        elif state == 'not_flagged':
            actions.append('keep'); reasons.append('scanner_not_flagged')
        # Leila: quarantine scanner suspects unless a saved Unsure requests review.
        elif state == 'suspected_label_flip' and choice != 'unsure':
            actions.append('quarantine'); reasons.append('scanner_suspected')
        # Leila: unreviewed flags from any detector are quarantined by default.
        elif choice is None:
            actions.append('quarantine'); reasons.append('flagged_unreviewed_quarantine')
        else:
            actions.append('human_review'); reasons.append('human_unsure' if choice == 'unsure' else 'flagged_unreviewed')
    return dict(sample_ids=ids, actions=actions, reasons=reasons,
                summary=dict(kept=actions.count('keep'),quarantined=actions.count('quarantine'),
                             unresolved=actions.count('human_review'),total=len(ids)))

"""Encoder-specific scoring with the benchmark's exact threshold rules."""
from importlib.metadata import version
import numpy as np
from .thresholds import calibrated_profile
from .knn_label_agreement import KNNLabelAgreement
from .class_distance import ClassDistance
from .confident_learning import ConfidentLearning
from ..output_connector import detector_result


def combine_calibrated_scores(sample_ids, scores, *, encoder):
    """Accept aligned score arrays, not native flags or poison identities."""
    profile = calibrated_profile(encoder)
    if set(scores) != set(profile['thresholds']):
        raise ValueError('All three named detector scores are required')
    results, votes = {}, {}
    ids = np.asarray(sample_ids)
    if not len(ids):
        raise ValueError('Empty scans are unsupported')
    for name, cutoff in profile['thresholds'].items():
        values = np.asarray(scores[name])
        if np.any((values < 0) | (values > 1)):
            raise ValueError('These detector scores must be in [0, 1]')
        raw = {'sample_ids': ids, 'scores': values, 'flags': values > cutoff}
        results[name] = detector_result(name, '1.0', raw,
            dict(profile, detector_threshold=cutoff), expected_sample_ids=ids)
        votes[name] = results[name]['flags'].copy()
    counts = np.column_stack(list(votes.values())).sum(axis=1)
    return {'schema_version': '1.0', 'pipeline_version': 'calibrated-1.0',
            'sample_ids': ids.copy(), 'detectors': results,
            'combination_votes': votes, 'flag_count': counts,
            'review_flags': counts >= 2, 'combination_settings': profile}


def scan_calibrated_label_flips(inputs, *, encoder, batch_size=256, progress=None):
    """Caller supplies CIFAR-10 raw embeddings from the named frozen encoder.

    Different encoders must be scanned separately. This is not a dataset-agnostic
    default. Output plugs into assess_label_flips and existing cleaning decisions.
    """
    profile = calibrated_profile(encoder)
    features = np.asarray(inputs.X)
    if features.ndim != 2 or features.shape[1] != profile['feature_dimension']:
        raise ValueError('Feature dimension does not match the selected encoder')
    dependencies = {p: version(p) for p in profile['dependencies']}
    if dependencies != profile['dependencies']:
        raise ValueError('Use the dependency versions recorded in the calibrated profile')
    models = {'knn': KNNLabelAgreement(k=20, batch_size=batch_size),
              'class_distance': ClassDistance(),
              'confident_learning': ConfidentLearning(**profile['confident_learning'])}
    scores = {}
    for index, (name, model) in enumerate(models.items(), 1):
        if progress:
            progress(f'{index} of 3: {name} ({profile["encoder"]})')
        raw = model.analyze(inputs, progress=progress) if name == 'confident_learning' else model.analyze(inputs)
        if not np.array_equal(raw['sample_ids'], inputs.sample_ids):
            raise ValueError('Detector output IDs are misaligned')
        scores[name] = raw['scores']
    return combine_calibrated_scores(inputs.sample_ids, scores, encoder=encoder)

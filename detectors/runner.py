"""Explicit plugin runner; feature and pixel inputs use the same output contract."""
from .output_connector import detector_result


def run_detectors(inputs, detectors, *, progress=None):
    """detectors maps a unique name to a callable(inputs) returning a connector dict.

    Use separate calls for different feature spaces or pixel inputs. Exceptions
    propagate: a failed detector is never replaced with an all-clear result.
    """
    if not detectors:
        raise ValueError('At least one detector is required')
    results = {}
    for index, (name, run) in enumerate(detectors.items(), 1):
        if progress is not None:
            progress(f'{index} of {len(detectors)}: {name}')
        result = run(inputs)
        if result['detector_name'] != name:
            raise ValueError('Registered name must match detector_name')
        raw = {key: result[key] for key in ('sample_ids', 'scores', 'flags')}
        checked = detector_result(name, result['version'], raw, result['settings'],
                                  expected_sample_ids=inputs.sample_ids)
        from copy import deepcopy
        checked['evidence'] = deepcopy(result.get('evidence', {}))
        results[name] = checked
    return {'schema_version': '1.0', 'sample_ids': inputs.sample_ids.copy(),
            'detectors': results}

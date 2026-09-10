"""Shared image-noise adapter for scanning, review and saved results."""
import numpy as np
from threadpoolctl import threadpool_limits
from .pipeline import scan_as_connector_result


def scan_noise(images, progress):
    # Leila: use the teammate scanner for both image datasets, without attack truth.
    progress(6.95, 'Stage 3 of 3 · Blended-injection checks\nCheck 1 of 1 · Shared noise scanner')
    with threadpool_limits(limits=4):
        result = scan_as_connector_result(images)
    evidence = result['evidence']
    # Leila: an infinite peer ratio means no peer signal; JSON uses null for it.
    for key in ('class_ratio', 'contrast'):
        if key in evidence and not np.isfinite(evidence[key]):
            evidence[key] = None
    # Leila: JSON object keys must match freshly returned and restored results.
    for key, value in list(evidence.items()):
        if isinstance(value, dict):
            evidence[key] = {str(k): v for k, v in value.items()}
    selected = np.flatnonzero(result['flags'])
    selected = selected[np.argsort(-result['scores'][selected], kind='stable')][:24]
    ui = dict(applicable=True, status='completed', method=evidence.get('method', 'residual-signature'),
        flagged=int(result['flags'].sum()), settings=result['settings'], evidence=evidence,
        examples=[dict(sample_id=str(images.sample_ids[i]), label=int(images.labels[i]),
                       score=float(result['scores'][i])) for i in selected])
    return result, ui

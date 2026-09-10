"""Patch scan presentation; flags join review through the shared assessment."""
import numpy as np
from .repeated_patch import RepeatedPatchDetector


# Leila: frozen clean-only CIFAR calibration; no attack parameters select this profile.
CIFAR_PATCH_SETTINGS = dict(intensity_bins=256, min_count=10, min_purity=.7,
                            min_lift=3.0, min_spatial_lift=3.0)


def scan_patch(images, label_assessment, progress, *, dataset="mnist"):
    if dataset not in ("mnist", "cifar10"):
        raise ValueError("Unsupported patch scan dataset")
    if not np.array_equal(images.sample_ids, label_assessment['sample_ids']):
        raise ValueError('Patch pixels and label-flip assessment IDs must match')
    progress(6, 'Checking repeated patches after label-flip scanning')
    def report(done, total):
        progress(6 + .9 * done / total, f'Patch scan: {done} of {total} positions checked')
    # Leila: preserve MNIST defaults and record the CIFAR profile in connector settings.
    settings = CIFAR_PATCH_SETTINGS if dataset == "cifar10" else {}
    result = RepeatedPatchDetector(**settings).analyze(images, progress=report)
    if dataset == "cifar10":
        result['settings']['profile'] = 'cifar-bright-patch-v1'
        result['settings']['calibration'] = (
            'clean-only calibration 20260910-051458-750811; limited held-out validation; '
            'missed 1% white and black patches')
    flags = result['flags']
    label_flags = np.asarray(label_assessment['flags'], dtype=bool)
    ui = dict(flagged=int(flags.sum()), rate=float(flags.mean()),
              overlap=int((flags & label_flags).sum()), unique_flagged=int((flags | label_flags).sum()),
              patterns=result['evidence']['patterns'],
              examples=[dict(sample_id=images.sample_ids[i], label=int(images.labels[i]),
                             score=float(result['scores'][i])) for i in np.flatnonzero(flags)[:24]],
              limitation='Provisional bright 2x2/3x3 patch heuristic. Findings request review, not proof of poisoning. '
                         'Patch findings join human review; unresolved samples are excluded in new training preparations.')
    return result, ui

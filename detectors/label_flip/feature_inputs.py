"""Join the two feature bundles at the scanner boundary, without poison truth."""
import numpy as np

from poison_features import FeatureBundle, detector_input


def paired_feature_inputs(resnet_bundle, dino_bundle):
    """Return separate DetectorInput objects after checking row identity.

    Matching IDs and labels do not prove identical pixels. The extraction
    service must generate both bundles from the same dataset version.
    This validates inputs only: it neither scans nor chooses thresholds.
    """
    bundles = {'resnet18': resnet_bundle, 'dinov2_vits14': dino_bundle}
    dimensions = {'resnet18': 512, 'dinov2_vits14': 384}
    for name, bundle in bundles.items():
        accepted_names = {'dinov2', 'dinov2_vits14'} if name == 'dinov2_vits14' else {name}
        if bundle.encoder not in accepted_names or bundle.modality != 'image':
            raise ValueError(f'Expected an image FeatureBundle from {name}')
        X = np.asarray(bundle.features)
        ids, labels = np.asarray(bundle.sample_ids), np.asarray(bundle.labels)
        if (X.ndim != 2 or not len(X) or X.shape[1] != dimensions[name]
                or X.dtype.kind not in 'fiu' or not np.isfinite(X).all()
                or np.any(np.all(X == 0, axis=1))):
            raise ValueError(f'{name} needs finite nonzero {dimensions[name]}-dimensional feature rows')
        if ids.shape != (len(X),) or ids.dtype.kind not in 'iuUS' or len(np.unique(ids)) != len(ids):
            raise ValueError('Provide one unique integer or string sample ID per feature row')
        if labels.shape != (len(X),) or labels.dtype.kind not in 'biufUS':
            raise ValueError('Provide one supplied class label per feature row')
        if labels.dtype.kind in 'biuf' and (not np.isfinite(labels).all() or np.any(labels == -1)):
            raise ValueError('Supplied labels must be finite and known')
    if resnet_bundle.dataset_name != dino_bundle.dataset_name:
        raise ValueError('The feature bundles name different datasets')
    if not np.array_equal(resnet_bundle.sample_ids, dino_bundle.sample_ids):
        raise ValueError('Both feature files must contain identical sample IDs in the same order')
    if not np.array_equal(resnet_bundle.labels, dino_bundle.labels):
        raise ValueError('Both feature files must contain identical supplied labels')
    return {name: detector_input(bundle, representation='raw', label_aware=True)
            for name, bundle in bundles.items()}


def load_paired_feature_inputs(resnet_path, dino_path):
    """Load trusted local FeatureBundle files and validate them for scanning.

    FeatureBundle uses pickle-backed metadata; do not load untrusted uploads.
    File names are not prescribed. Encoder identities are checked in metadata.
    """
    return paired_feature_inputs(FeatureBundle.load(resnet_path), FeatureBundle.load(dino_path))

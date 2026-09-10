"""Evaluation-only poison truth; never imported by selection or training code."""
from pathlib import Path
import numpy as np

from cleaning import human_review
from .preparation import ARTIFACTS, load_preparation, file_hash


def evaluate_prepared(version):
    """Compare frozen selection with locally generated benchmark attack metadata."""
    manifest = load_preparation(version)
    try:
        _, scan, digest = human_review.record(manifest['scan_id'])
        if digest != manifest['scan_sha256']:
            raise ValueError('The scan evidence changed after preparation.')
        source = human_review.text_feature_path(scan) if manifest.get('dataset')=='imdb' else human_review.image_path(scan)
        if str(source) != manifest['source_images'] or file_hash(source) != manifest['source_sha256']:
            raise ValueError('The original image bundle changed after preparation.')
        feature = Path(scan['feature_files'][0])
        if not feature.is_absolute(): feature = ARTIFACTS.parent / feature
        if feature.resolve().parent != ARTIFACTS.resolve():
            raise ValueError('The benchmark feature file is outside local artifacts.')
        # Leila: only existing local extractor archives use pickled metadata dictionaries.
        # Read row truth after selection is frozen; do not load large feature matrices.
        with np.load(source, allow_pickle=False) as images, np.load(feature, allow_pickle=True) as bundle:
            ids = bundle['sample_ids']
            if (ids.tolist() != manifest['sample_ids']
                    or not np.array_equal(ids, images['sample_ids'])
                    or not np.array_equal(bundle['labels'], images['labels'])):
                raise ValueError('Benchmark rows or labels do not match this prepared version.')
            mask = bundle['is_poisoned'] if 'is_poisoned' in bundle else np.empty(0,dtype=bool)
            metadata = bundle['metadata'].item() if 'metadata' in bundle else {}
        # Leila: pixels-only MNIST stores truth separately, read only after selection is frozen.
        if manifest.get('dataset') == 'mnist' and mask.size == 0:
            truth_path = source.with_name(source.name.replace('-images.npz','-evaluation.npz'))
            with np.load(truth_path,allow_pickle=False) as truth:
                if not np.array_equal(truth['sample_ids'],ids):
                    raise ValueError('MNIST truth IDs do not match the frozen selection.')
                mask = truth['is_poisoned']
        # Leila: verify legacy clean IMDB labels against official rows before supplying clean truth.
        # This evaluation-only fallback never changes scanning or selection.
        if manifest.get('dataset') == 'imdb' and mask.size == 0 and '-none-' in feature.name:
            from poison_features import FeatureBundle, load_imdb_dataset
            from poison_features.imdb_identity import imdb_training_indices
            legacy = FeatureBundle.load(feature)
            clean = load_imdb_dataset(split='train', cache_dir=str(ARTIFACTS.parent/'data'))
            indices = imdb_training_indices(legacy, feature, clean)
            expected = np.array([clean[int(i)]['label'] for i in indices])
            if not np.array_equal(legacy.labels, expected):
                raise ValueError('The clean IMDB archive contains changed labels.')
            mask = np.zeros(len(ids), dtype=bool)
        if not isinstance(metadata,dict):
            raise ValueError('Benchmark metadata is invalid.')
        if mask.size == 0:
            # Missing truth is not evidence of clean data; require an explicit clean run.
            if not (metadata.get('attack') == 'none' and metadata.get('poison_rate') == 0
                    and metadata.get('poison_count') == 0):
                raise ValueError('Known poison identities are unavailable for this dataset.')
            mask = np.zeros(len(ids),dtype=bool)
        if mask.shape != ids.shape or mask.dtype.kind not in 'biu' or not np.isin(mask,[0,1]).all():
            raise ValueError('Known poison identities are invalid or incomplete.')
        mask = mask.astype(bool)
        if ('poison_count' in metadata and metadata['poison_count'] != int(mask.sum())):
            raise ValueError('The saved poison count disagrees with the row identities.')
        actions = np.asarray(manifest['actions'])
        if actions.shape != mask.shape or not np.isin(actions,['keep','quarantine','human_review']).all():
            raise ValueError('The prepared selection is invalid.')
        kept = actions == 'keep'
        def counts(rows):
            return dict(clean=int((rows & ~mask).sum()), poisoned=int((rows & mask).sum()), total=int(rows.sum()))
        return dict(available=True, version=version, original=counts(np.ones(len(ids),dtype=bool)),
                    kept=counts(kept), removed=counts(~kept))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return dict(available=False, version=version, reason=str(exc))

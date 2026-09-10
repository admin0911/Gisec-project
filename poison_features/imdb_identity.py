"""Resolve official IMDB rows, including verified archives from the original builder."""
import re
from pathlib import Path
import numpy as np


def imdb_training_indices(bundle,path,clean):
    ids=np.asarray(bundle.sample_ids)
    parsed=[re.fullmatch(r'imdb-train:(0|[1-9][0-9]*)',str(sid)) for sid in ids]
    if all(parsed):
        indices=np.array([int(m[1]) for m in parsed])
    else:
        # Leila: legacy builder used range(total) IDs after deterministic random row selection.
        if not np.array_equal(ids,np.arange(len(ids))) or ids.dtype.kind not in 'iu':
            raise ValueError('Unknown IMDB IDs. Rebuild the dataset.')
        seed=re.search(r'-seed(\d+)-features\.npz$',Path(path).name)
        if seed is None or len(ids)>len(clean): raise ValueError('Cannot verify legacy IMDB row mapping.')
        indices=np.random.default_rng(int(seed[1])).choice(len(clean),size=len(ids),replace=False)
    if len(np.unique(indices))!=len(indices) or np.any(indices>=len(clean)):
        raise ValueError('Invalid official IMDB row mapping.')
    expected=np.array([clean[int(i)]['label'] for i in indices])
    originals=bundle.original_labels
    if originals is None and '-none-' in Path(path).name: originals=bundle.labels
    if originals is None or not np.array_equal(originals,expected):
        raise ValueError('IMDB reference labels do not match the original builder rows. Rebuild the dataset.')
    return indices

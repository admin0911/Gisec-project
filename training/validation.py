"""Shared clean demo holdout; never reads detector scores or poison identities."""
import numpy as np
from torch.utils.data import Subset
from .connector import training_input

def validation_split(reference, arms, fraction=.1, seed=2026):
    n=len(reference.sample_ids)
    if n<2: raise ValueError('At least two selected samples are needed for validation')
    count=max(1,int(np.ceil(n*fraction)))
    canonical=np.argsort(reference.sample_ids.astype(str))
    rows=np.sort(np.random.default_rng(seed).choice(canonical,count,replace=False))
    ids=reference.sample_ids[rows]; held=set(ids.tolist())
    validation=training_input(Subset(reference.dataset,rows.tolist()),ids,dataset_version=reference.dataset_version+'/validation',split='validation')
    result=[]
    for name,source in arms:
        keep=[i for i,sid in enumerate(source.sample_ids) if sid not in held]
        if not keep: raise ValueError('No training samples remain after reserving validation; review more samples or build a larger dataset')
        result.append((name,training_input(Subset(source.dataset,keep),source.sample_ids[keep],dataset_version=source.dataset_version+'/validation-excluded',split='train')))
    return validation,result,dict(fraction=fraction,seed=seed,samples=count,sample_ids=ids.tolist(),
        source='Original clean versions of selected training IDs; demo-only validation, not official test data',
        limitation='Diagnostic holdout for classifier fitting. Detectors previously scanned the submitted dataset; not an independent validation of detector selection.')

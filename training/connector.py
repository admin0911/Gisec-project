"""Only selected data and identity cross the cleaning-to-training boundary."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TrainingInput:
    dataset: object
    sample_ids: np.ndarray
    dataset_version: str
    split: str


def training_input(dataset, sample_ids, *, dataset_version, split):
    """Dataset items must be (input tensor, current label); labels are not repaired.

    Pass a keep DatasetView from partition_dataset. Decisions, detector scores
    and poison identities are intentionally absent. IDs must be split-qualified.
    """
    ids = np.asarray(sample_ids)
    if (ids.shape != (len(dataset),) or not len(ids) or ids.dtype.kind not in 'iuUS'
            or len(np.unique(ids)) != len(ids)):
        raise ValueError('Nonempty dataset with one unique ID per row required')
    if not isinstance(dataset_version, str) or not dataset_version.strip():
        raise ValueError('Record the dataset version')
    if split not in ('train', 'test'):
        raise ValueError('split must be train or test')
    if hasattr(dataset, 'sample_ids') and not np.array_equal(dataset.sample_ids, ids):
        raise ValueError('IDs must match the dataset view order')
    ids = ids.copy()
    ids.setflags(write=False)
    return TrainingInput(dataset, ids, dataset_version, split)

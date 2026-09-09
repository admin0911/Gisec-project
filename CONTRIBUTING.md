# Contributing a detector

This repository is the shared feature-layer contract. Detector work should be
isolated in a branch and merged through a pull request.

## Before coding

```powershell
git checkout main
git pull --ff-only origin main
git checkout -b detector/<short-name>
```

## Detector boundary

```python
from poison_features import detector_input

inputs = detector_input(bundle, representation="reduced", label_aware=False)
scores = detector.score(inputs)
```

`inputs.X` contains features only. `inputs.sample_ids` preserves identity.
`inputs.y` is present only for label-aware detectors. Evaluation-only poison
metadata must not be read during scoring.

### Pixel-based image detectors

Detectors that need raw pixels must use the separate image connector:

```python
from poison_features import load_image_inputs

pixels = load_image_inputs(dataset, sample_ids=my_ids)
scores = detector.score(pixels.images)
```

`pixels.images` is a float32 `NCHW` array in the original image dimensions,
while `pixels.sample_ids` and `pixels.labels` remain separate and aligned.
Pass a `PoisonedImageDataset` to receive post-attack pixels. Do not put pixels
into `FeatureBundle` vectors or use evaluation-only poison metadata while
scoring.

## Creating extraction runs

The frontend's **Extraction scope** controls workload:

- **Quick sample** uses the requested sample count and is the normal choice
  while developing or debugging a detector.
- **Whole training split** hides the sample-count control and processes every
  row in the selected training split. The API equivalent is
  `{"full_training": true}`.

Choose **Clean**, **Label flip**, or **Backdoor patch**. For either attack,
use only the provided 1%, 3%, 5%, or 10% poison rates. Clean runs have no
poisoned rows. Completed image runs create matching `*-features.npz` and
`*-images.npz` files under `artifacts/`; load them with
`FeatureBundle.load(...)` and `ImageInputBundle.load(...)` rather than
re-encoding the dataset.

The feature bundle and image bundle use the same stable `sample_ids`.
Backdoor image bundles contain the patched post-attack pixels. Labels remain
separate, and `is_poisoned`/`poison_type` are evaluation-only values: detector
scoring must not read them.

## Pull request checklist

- [ ] New detector is in a new file under `detectors/`
- [ ] Tests are in `tests/`
- [ ] Scores are finite and one-dimensional with one value per sample
- [ ] Sample IDs are returned unchanged
- [ ] Pixel detectors use `load_image_inputs` rather than embedding pixels
- [ ] No datasets, virtual environments, weights, or generated outputs committed
- [ ] `python -m unittest discover -s tests -v` passes
- [ ] Pull request targets `main`
- [ ] Description documents representation, threshold, runtime, and limitations

Keep changes narrow. Do not reformat or rewrite shared files while adding a
detector. This is the primary protection against merge conflicts before the
competition submission.

# Contributing a detector

This repository is the shared feature-layer contract. Detector work should be
isolated in a branch and merged through a pull request.

## Required AI-assisted workflow

All GitHub collaboration for this project must use an AI coding assistant for
repository operations and code changes. Use it to inspect the current branch,
create or update detector files, run the tests, review the diff, synchronize
with `main`, and prepare the pull request. Do not make unreviewed manual edits
directly on `main` or resolve conflicts by copying files blindly. The human
author remains responsible for checking the generated diff and test results.

This policy keeps detector contributions consistent and makes merge conflicts
visible before submission.

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

Choose **Clean**, **Label flip**, **Backdoor patch**, or **Blended noise
injection**. For attacks, use only the provided 1%, 3%, 5%, or 10% poison
rates. Clean runs have no poisoned rows. Blended injection uses a shared
low-amplitude noise pattern, default `alpha=0.10`, and target label `0`
(airplane for CIFAR-10); a full 3% CIFAR-10 run is approximately 1,500 rows.
Completed image runs create matching `*-features.npz` and
`*-images.npz` files under `artifacts/`; load them with
`FeatureBundle.load(...)` and `ImageInputBundle.load(...)` rather than
re-encoding the dataset.

### Comparing ResNet-18 and DINOv2

For image detectors, CIFAR-10 automatically produces both `resnet18` (512D)
and `dinov2` (384D); MNIST uses the **Image features** selector to choose one.
Keep the dataset split, sample IDs,
attack, poison rate, seed, and detector settings identical when comparing
them. The two runs are saved as separate artifacts because their feature
spaces are not interchangeable. DINOv2 downloads its pretrained weights on
first use and requires internet access then.

Label-flip runs reuse matching clean feature and image artifacts and only
replace labels plus evaluation metadata. Pixel-changing attacks such as
backdoors and blended injection must be re-encoded.

The feature bundle and image bundle use the same stable `sample_ids`.
Backdoor image bundles contain the patched post-attack pixels. Labels remain
separate, and `is_poisoned`/`poison_type` are evaluation-only values: detector
scoring must not read them.

For blended-injection detectors, use the saved post-injection pixels as well
as embeddings. Look for weak shared residual or frequency signals across
many samples and their association with the target label. Do not expect a
single image or a single ResNet coordinate to reveal a 10% blend reliably.

### Blended-injection detector checklist

- [ ] Create branch `detector/blended-injection`
- [ ] Begin with a 3% CIFAR-10 blended run, target `0`, `alpha=0.10`
- [ ] Load matching `*-features.npz` and `*-images.npz` artifacts
- [ ] Use pixels, embeddings, current labels, and stable sample IDs
- [ ] Search for shared weak residual or frequency signals across samples
- [ ] Return one finite suspicion score per sample
- [ ] Preserve sample IDs and keep poison metadata out of scoring
- [ ] Test clean data and 1%, 3%, 5%, and 10% blended runs
- [ ] Add implementation under `detectors/` and tests under `tests/`
- [ ] Run `python -m unittest discover -s tests -v`
- [ ] Open a pull request into `main`

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
- [ ] AI assistant was used to inspect, test, and review the contribution

Keep changes narrow. Do not reformat or rewrite shared files while adding a
detector. This is the primary protection against merge conflicts before the
competition submission.

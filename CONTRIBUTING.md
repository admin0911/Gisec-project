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

## Runs and integration

Use the [README](README.md) for setup, supported scenarios and verification.
The [feature API](docs/FEATURES.md) covers custom inputs; the
[connector guide](docs/CONNECTORS.md) covers detector outputs and identity checks.
Use [Evaluation](docs/EVALUATION.md) for calibration and benchmark requirements.
Keep input IDs, dataset version and encoder configuration fixed when comparing
methods. Never pass synthetic poison truth into scoring or selection.

## Publish and synchronize

Review the diff and test results, commit the intended files, then push your
feature branch and open a pull request into main. To bring main into an existing
shared branch, fetch and merge origin/main, resolve conflicts preserving both
features, and rerun checks before pushing. Do not blindly accept one side of a
shared-file conflict. A merge avoids rewriting published branch history.

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

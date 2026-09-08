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

## Pull request checklist

- [ ] New detector is in a new file under `detectors/`
- [ ] Tests are in `tests/`
- [ ] Scores are finite and one-dimensional with one value per sample
- [ ] Sample IDs are returned unchanged
- [ ] No datasets, virtual environments, weights, or generated outputs committed
- [ ] `python -m unittest discover -s tests -v` passes
- [ ] Pull request targets `main`
- [ ] Description documents representation, threshold, runtime, and limitations

Keep changes narrow. Do not reformat or rewrite shared files while adding a
detector. This is the primary protection against merge conflicts before the
competition submission.

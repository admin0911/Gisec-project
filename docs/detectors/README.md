# Detector code guide

## Detection and integration: `detectors/`

- `label_flip/knn_label_agreement.py`: neighbour label disagreement.
- `label_flip/class_distance.py`: distance to supplied and alternative class centres.
- `label_flip/confident_learning.py`: out-of-fold label-quality checks.
- `label_flip/pipeline.py`: runs all three and combines review votes.
- `output_connector.py`: shared result dictionary and JSON conversion.
- `backdoor/`: reserved for teammates' backdoor detectors; currently empty apart from its package marker.

Start with [CONNECTOR.md](CONNECTOR.md) for integration, or
[CONFIDENT_LEARNING.md](label_flip/CONFIDENT_LEARNING.md) for label-flip experiment setup.

## Running saved-feature experiments: `experiments/`

Run from the repository root with the appropriate Python environment:

```powershell
python -m experiments.run_knn artifacts/NAME-features.npz
python -m experiments.run_class_distance artifacts/NAME-features.npz
python -m experiments.benchmark_confident_learning
```

Replace NAME with an existing feature bundle. The benchmark requires the
specific saved bundles and neighbours documented in the experiment guide.
The previous `python -m detectors.run_*` commands have moved to `experiments`.
Import the pipeline with `from detectors.label_flip.pipeline import scan_label_flips`.
Label-flip detector imports now start with `detectors.label_flip`.
The shared connector remains `detectors.output_connector`.

Tests remain in `tests/`; generated results remain in `artifacts/`.
The benchmark fingerprints its source, so this reorganization changes its
cache signature. Previous result files remain available, but the next full
benchmark will recompute predictions. Do not rename caches to bypass this.

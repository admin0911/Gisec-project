# Detector code guide

## Detection and integration: `detectors/`

- `label_flip/knn_label_agreement.py`: neighbour label disagreement.
- `label_flip/class_distance.py`: distance to supplied and alternative class centres.
- `label_flip/confident_learning.py`: out-of-fold label-quality checks.
- `label_flip/pipeline.py`: runs all three and combines review votes.
- `output_connector.py`: shared result dictionary and JSON conversion.
- `backdoor/repeated_patch.py`: repeated bright pixel-patch heuristic, integrated into image web scans.
- `backdoor/spectral_signature.py`: experimental within-class spectral feature outliers.
- `backdoor/activation_clustering.py`: experimental minority feature-cluster analysis.
- `backdoor/feature_pipeline.py`: both feature methods with separate results and review candidates.
- `blended_injection/`: shared residual-signature detection and its pipeline.

See [FEATURE_DETECTORS.md](backdoor/FEATURE_DETECTORS.md) for the experimental
backdoor feature methods, command-line runs, thresholds, and limitations.
To scan saved image/feature pairs together and open a visual HTML report, use
[RUN_IMAGE_SCAN.md](backdoor/RUN_IMAGE_SCAN.md). Pixel flags remain review
candidates even when feature detectors return no flags.
The command's revised default and measured limitations are documented in
[PATCH_FIX.md](backdoor/PATCH_FIX.md). The legacy detector remains available
as a separately labelled comparison.

Start with [CONNECTOR.md](CONNECTOR.md) for integration, or
[CONFIDENT_LEARNING.md](label_flip/CONFIDENT_LEARNING.md) for label-flip experiment setup.
For a dataset-neutral NPZ or in-memory feature matrix, use
[FEATURE_BUNDLE_INPUT.md](FEATURE_BUNDLE_INPUT.md) and
`python -m experiments.scan_feature_bundle`.

For the optional DINOv2 extractor and paired ResNet18 comparison, see
[FEATURE_COMPARISON.md](label_flip/FEATURE_COMPARISON.md).

For post-scan actions, dataset views, and quarantine manifests, see
[CLEANING.md](label_flip/CLEANING.md).

For the current two-encoder label-flip assessment (before final cleaning), see
[ASSESSMENT.md](label_flip/ASSESSMENT.md).

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

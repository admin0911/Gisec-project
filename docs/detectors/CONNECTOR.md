# Shared detector output

The connector lives in `detectors/output_connector.py` and
`detectors/label_flip/pipeline.py`, with tests in `tests/test_output_connector.py`.
Install the project's combined `requirements.txt` in your Python 3.12 environment.

```python
import json
from poison_features import FeatureBundle, detector_input
from detectors.label_flip.pipeline import scan_label_flips
from detectors.output_connector import to_jsonable

bundle = FeatureBundle.load("artifacts/cifar10-train-full-none-000-seed0-features.npz")
inputs = detector_input(bundle, representation="raw", label_aware=True)
result = scan_label_flips(inputs, progress=print)
with open("detector_results.json", "w", encoding="utf-8") as output:
    json.dump(to_jsonable(result), output, allow_nan=False)
```

Each entry under `result['detectors']` contains detector_name, version,
sample_ids, scores, flags, settings, and optional evidence. Arrays preserve
input order. Scores are not poisoning probabilities and must not be averaged.
Evidence can include large neighbour arrays; omit evidence from dashboard
summary responses if the UI only needs scores and flags.

`flag_count` counts the **stricter combination votes**, not the individual
detector flags. `combination_votes` exposes each vote. `review_flags` means
at least two votes. Individual class-distance flags use threshold
0.024119124718243068; combination votes use 0.029466908865236896. Confident
Learning votes require both its own flag and score >= 0.90. kNN requires
at least 19 of 20 neighbours to disagree. No detector logic was changed.

This is the previously evaluated CIFAR-10 development preset using frozen raw
embeddings. It is not independently validated or transferable automatically
to other feature extractors or datasets. Unflagged does not mean known clean.
It does not remove or relabel samples. Ground truth and evaluation metrics are
excluded from the connector. Dependency versions are included in settings.

Backdoor detectors can call `detector_result(...)` with their own outputs and
settings. They must supply only detector evidence, never poison ground truth.
Their combined decisions remain separate from this label-flip preset.

## Receiving both image feature files

```python
from detectors.label_flip.feature_inputs import load_paired_feature_inputs

inputs = load_paired_feature_inputs("path/to/resnet-features.npz",
                                   "path/to/dino-features.npz")
resnet_inputs = inputs["resnet18"]
dino_inputs = inputs["dinov2_vits14"]
```

Both paths must be trusted local `FeatureBundle.save()` outputs. This adapter
checks the encoders, feature dimensions, dataset name, IDs, row order and supplied
labels. It returns separate raw feature inputs; it never merges vectors or
passes known poison identities to detectors. Matching IDs cannot verify the
pixels themselves: extraction must use the same dataset version for both files.
The filename convention is up to the extraction service.
The adapter accepts both `dinov2` (frontend) and `dinov2_vits14` (earlier Python
experiments) as encoder names; the returned dictionary uses `dinov2_vits14`.

This prepares inputs only. DINOv2 scoring needs its own calibrated thresholds;
do not pass DINOv2 inputs to the fixed ResNet18 pipeline preset.

Run the connector tests with:
```powershell
python -m unittest discover -s tests -p test_output_connector.py -v
```

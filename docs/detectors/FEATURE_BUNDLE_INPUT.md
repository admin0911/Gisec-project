# Dataset-neutral feature input

The detector layer can consume aligned numeric features produced by another
part of the application without routing on CIFAR-10, MNIST or IMDB names.
This is a feature-only interface for supported numeric embeddings; it does not
make raw-pixel or raw-text detectors interchangeable.

## Minimal NPZ contract

Save one two-dimensional feature matrix and one supplied class label per row.
Stable IDs are strongly recommended:

```python
import numpy as np

np.savez_compressed(
    "artifacts/my-features.npz",
    features=my_features,     # shape: samples x dimensions
    labels=my_labels,         # shape: samples
    sample_ids=my_ids,        # shape: samples; unique integers or strings
)
```

The safe loader also recognizes these aliases:

| Meaning | Accepted keys |
|---|---|
| Feature matrix | `features`, `X`, `embeddings` |
| Current labels | `labels`, `y`, `targets` |
| Stable IDs | `sample_ids`, `ids`, `row_ids` |

If IDs are absent, the adapter creates `0..N-1`. Labels are required by the
currently connected feature tracks. Poison truth, original labels and attack
types are deliberately ignored by the external adapter during scoring.

## Run both feature tracks

From the repository root in the VS Code terminal:

```powershell
& ".\.venv\Scripts\python.exe" -m experiments.scan_feature_bundle `
  artifacts\my-features.npz `
  --dataset-name my_dataset `
  --modality text `
  --encoder my_encoder `
  --output artifacts\my-feature-scan.json
```

For an IMDB MiniLM bundle, replace the file and metadata values; the scanner
does not require the dataset name to be `imdb`. Use `--track backdoor` or
`--track label_inconsistency` to run only one track. Add
`--include-cleanlab` for the slower cross-validated label-quality stage.

The JSON result contains separate `backdoor` and `label_inconsistency` tracks.
Each detector preserves the input IDs and returns finite scores and Boolean
flags. Track candidate flags mean review, not automatic deletion. Thresholds
are provisional and must be validated on an independent clean sample for each
encoder and deployment dataset.

## Direct Python connection

```python
from poison_features import feature_bundle_from_arrays
from detectors.feature_pipeline import scan_feature_bundle

bundle = feature_bundle_from_arrays(
    my_features,
    labels=my_labels,
    sample_ids=my_ids,
    dataset_name="my_dataset",
    modality="text",
    encoder="my_encoder",
)

result = scan_feature_bundle(bundle)
```

The two feature-compatible backdoor methods are spectral signatures and
activation clustering. The label-inconsistency track runs k-nearest-neighbour
label agreement and class distance, with Confident Learning optional. Image
patch detection still requires aligned raw pixels, and phrase-frequency
detection still requires aligned raw text.

# Gisec Project - PoisonGuard Feature Layer

This is a standalone module created for merging into the wider PoisonGuard
competition project. It intentionally does not import the existing
`PoisonGuard-main` feature extractor.

## Setup on Windows

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

## Verify the module

```powershell
& ".\.venv\Scripts\python.exe" -m unittest discover -s tests -v
```

## Download real image datasets

```powershell
& ".\.venv\Scripts\python.exe" download_datasets.py
```

This downloads CIFAR-10 and MNIST train/test splits into `data/`. The files
remain local and are excluded from Git.

To extract a real 100-sample CIFAR-10 subset:

```powershell
@'
from poison_features import UniversalFeatureExtractor, load_image_dataset

dataset = load_image_dataset("cifar10", train=True)
bundle = UniversalFeatureExtractor(batch_size=32).extract_images(
    dataset, labels=[dataset[i][1] for i in range(100)],
    sample_ids=range(100), dataset_name="cifar10"
)
print(bundle.features.shape, bundle.labels.shape, bundle.sample_ids.shape)
bundle.save("outputs/cifar10_first100.npz")
'@ | & ".\.venv\Scripts\python.exe" -
```

The first real ResNet-18 extraction downloads ImageNet weights into the local
cache. Data, weights, virtual environments, and generated `.npz` bundles are
not committed.

## Run the frontend

```powershell
& ".\.venv\Scripts\python.exe" serve_frontend.py
```

Open http://127.0.0.1:8787.

## Core API

```python
from poison_features import UniversalFeatureExtractor

bundle = UniversalFeatureExtractor().extract_images(
    dataset, dataset_name="cifar10", sample_ids=range(len(dataset))
)
X_unsupervised = bundle.scaled_features
y_label_aware = bundle.labels
bundle.save("outputs/cifar10_resnet18.npz")
```

Labels and poison ground truth are never concatenated into the feature arrays.

## How dataset routing works

The extractor chooses an adapter from the explicit dataset/modality entry
point; it does not guess from arbitrary file contents:

```text
CIFAR-10 / MNIST images ──> ResNet-18 ───────┐
IMDB review text ─────────> MiniLM (384D) ────┼─> FeatureBundle
CSV/JSON network flows ──> engineered fields ┘       │
                                                    ├─ features
                                                    ├─ scaled_features
                                                    ├─ reduced_features
                                                    └─ visual_features (PCA-2D)
```

Image datasets use `UniversalFeatureExtractor.extract_images`. IMDB uses
`extract_text`, and structured packet/flow records use
`extract_packet_features`. The detector layer should not select encoders; it
only consumes a `DetectorInput`.

## Detector connector

Detector authors connect at one small boundary:

```python
from poison_features import detector_input

inputs = detector_input(bundle, representation="reduced", label_aware=False)
scores = my_detector.score(inputs)  # one score per sample
sample_ids = inputs.sample_ids       # maps scores back to source rows
```

For label-aware detectors, use `label_aware=True`. This supplies `y` separately
and never appends labels to `X`. Unsupervised detectors should receive
`label_aware=False`.

Expected detector contract:

```python
class MyDetector:
    def score(self, inputs):
        # inputs.X: (n_samples, n_features)
        # inputs.sample_ids: stable row identifiers
        # inputs.y: labels or None
        return scores  # shape (n_samples,), higher = more suspicious
```

The detector may then return scores to the future consensus layer. It must not
read `bundle.is_poisoned`, `bundle.poison_type`, or `bundle.original_labels`;
those fields are evaluation-only.

## Frontend progress

The frontend starts extraction as a background job. It polls
`GET /api/jobs/{job_id}` and shows percentage, encoded sample count, PCA
preparation, completion, or an explicit error. This prevents the browser from
appearing frozen during a 50,000-sample run.

```text
POST /api/extract
        │ 202 + job_id
        ▼
background extraction ──> GET /api/jobs/{job_id} ──> progress bar
        │
        └──────────────────────────────────────────> PCA chart + summary
```

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

## What do the extracted features mean?

For images, a feature is a learned numeric measurement produced by the
penultimate layer of a frozen ImageNet ResNet-18. It is not a human-readable
field such as "amount of red" or "number of edges". Each image becomes a
512-value vector that summarizes visual patterns the network learned to
recognize. The values are useful for comparing samples, clustering them, and
finding outliers; individual coordinates should not be interpreted as labels
or proof of poisoning.

The representations are:

| Representation | Meaning | Typical detector use |
|---|---|---|
| `features` | Original ResNet-18 embedding, 512 values per image | Full-resolution analysis |
| `scaled_features` | Each embedding coordinate standardized across the selected dataset | Distances, LOF, Isolation Forest, SVM |
| `reduced_features` | PCA projection, at most 64 dimensions | Faster clustering, kNN, Mahalanobis |
| `visual_features` | PCA projection to 2 dimensions | Charts only; do not use as the primary detector input |

For IMDB, MiniLM produces 384-dimensional sentence embeddings. For network
flows, the features are explicit engineered measurements such as duration,
packet count, byte count, ports, protocol, packet-length statistics, and TCP
flags. All modalities still preserve `sample_ids` so a detector score can be
mapped back to the original sample.

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

## Detector developer workflow

Use one branch per detector. Do not edit the extractor internals unless the
detector genuinely requires a new shared representation.

### 1. Clone and create a branch

```powershell
git clone https://github.com/admin0911/Gisec-project.git
cd Gisec-project
git checkout -b detector/my-detector
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

### 2. Implement only the detector

Create a file such as `detectors/my_detector.py`:

```python
import numpy as np
from poison_features import detector_input


class MyDetector:
    def score_bundle(self, bundle):
        inputs = detector_input(
            bundle,
            representation="reduced",
            label_aware=False,
        )
        # Replace this example with the real detector.
        center = inputs.X.mean(axis=0)
        scores = np.linalg.norm(inputs.X - center, axis=1)
        return {
            "sample_ids": inputs.sample_ids,
            "scores": scores,
        }
```

Rules:

- Return exactly one score per input sample.
- Preserve `sample_ids` in the result.
- Higher scores should mean "more suspicious".
- Use `scaled_features` or `reduced_features` as appropriate.
- Set `label_aware=True` only when the algorithm is allowed to use current
  labels; labels remain separate in `inputs.y`.
- Never use `is_poisoned`, `poison_type`, or `original_labels` while scoring.
- Do not commit `data/`, `.venv/`, model weights, or generated `.npz` files.

### 3. Add a focused test

Add `tests/test_my_detector.py` using a small synthetic `FeatureBundle`.
Verify output length, finite scores, stable sample IDs, and that the detector
works with both a small and a normal-sized input.

Run before pushing:

```powershell
& ".\.venv\Scripts\python.exe" -m unittest discover -s tests -v
```

### 4. Commit and push your branch

```powershell
git add detectors/my_detector.py tests/test_my_detector.py
git commit -m "Add my detector"
git push -u origin detector/my-detector
```

### 5. Open a pull request

On GitHub, open a pull request from `detector/my-detector` into `main`.
Describe:

- Which representation the detector consumes
- Whether it is unsupervised or label-aware
- The score meaning and threshold
- Dataset/subset used for testing
- Runtime and memory expectations
- Test command and result

Keep the pull request limited to detector files and tests. This minimizes
merge conflicts. The maintainer should merge detector pull requests into
`main`; do not copy files manually on submission day.

### 6. Keep your branch current

Before requesting review:

```powershell
git fetch origin
git rebase origin/main
& ".\.venv\Scripts\python.exe" -m unittest discover -s tests -v
git push --force-with-lease
```

If GitHub reports conflicts, stop and ask the maintainer before resolving them
in shared files. Prefer adding a new detector file over changing
`universal.py`, `bundle.py`, or frontend files.

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

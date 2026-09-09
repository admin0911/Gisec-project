# Gisec Project - PoisonGuard Feature Layer

For label-flip detectors, the shared output connector, and saved-feature run
commands, see the [detector code guide](docs/detectors/README.md).

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

For a real-data end-to-end smoke test (without downloading anything):

```powershell
& ".\.venv\Scripts\python.exe" verify_extractor.py
```

Expected output includes:

```text
cifar10: PASS; raw=(8, 512), PCA=(8, 7), visual=(8, 2)
mnist: PASS; raw=(8, 512), PCA=(8, 7), visual=(8, 2)
```

The exact PCA dimension is `min(64, samples - 1, feature_dimension)`, so
small smoke tests intentionally produce fewer than 64 PCA columns. For a
normal 100-sample run, the PCA shape is `(100, 64)`.

## Download real image datasets

```powershell
& ".\.venv\Scripts\python.exe" download_datasets.py
```

This downloads CIFAR-10 and MNIST train/test splits into `data/`. The files
remain local and are excluded from Git.

## Use your own datasets

The built-in dataset names are only convenience adapters. You can use your
own data without changing the extractor.

### Custom image folders

Arrange images into class directories:

```text
my_images/
├── cat/
│   ├── cat_001.jpg
│   └── cat_002.png
└── dog/
    └── dog_001.jpg
```

Then load and extract them:

```python
from poison_features import UniversalFeatureExtractor, load_image_folder

dataset = load_image_folder("data/my_images")
bundle = UniversalFeatureExtractor().extract_images(
    dataset,
    dataset_name="my_images",
    sample_ids=[path for path, _ in dataset.samples],
)
```

`ImageFolder` assigns labels from the alphabetically ordered class folders.
For unlabeled images, provide a small custom dataset that returns `(image, -1)`
and pass stable IDs explicitly.

### Custom text CSV

Create a CSV with at least a `text` column:

```csv
text,label
"This review is excellent",1
"This review is poor",0
```

Load it with MiniLM:

```python
from poison_features import extract_text, load_text_table

texts, labels, sample_ids = load_text_table(
    "data/my_reviews.csv",
    text_column="text",
    label_column="label",
)
bundle = extract_text(
    texts,
    labels=labels,
    sample_ids=sample_ids,
    dataset_name="my_reviews",
)
```

`.json` and `.jsonl` files are also supported. JSON rows should look like
`{"text": "...", "label": 1}`. If no label column exists, labels are set to
`-1`; this is suitable for unsupervised detectors but not label-aware ones.

### Custom network-flow data

Use `extract_packet_features` with a list of dictionaries containing the
documented flow fields (`duration`, `packet_count`, `byte_count`, `protocol`,
and so on). This supports structured CSV/JSON records, not raw PCAP files.

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

## Pixel inputs for image detectors

Feature vectors are not the only detector input. Pixel-based detectors such as
MNIST pixel kNN or repeated-patch scanning can use the separate image-input
adapter:

```python
from poison_features import load_image_inputs, load_image_dataset

dataset = load_image_dataset("mnist", train=True, download=False)
pixels = load_image_inputs(
    dataset,
    sample_ids=[f"mnist-train:{i}" for i in range(len(dataset))],
)

images = pixels.images       # float32 array: N, C, H, W; pixels in [0, 1]
labels = pixels.labels       # current labels, separate from images
sample_ids = pixels.sample_ids
```

The adapter does not resize, normalize, flatten, or concatenate pixels with
learned embeddings. It returns the original image dimensions, so MNIST is
`(N, 1, 28, 28)` and CIFAR-10 is `(N, 3, 32, 32)`. The three arrays are
positionally aligned:

```text
images[i] ↔ labels[i] ↔ sample_ids[i]
```

For an attacked dataset, pass the attacked wrapper instead:

```python
from poison_features.attacks import poison_dataset

attacked = poison_dataset(dataset, "backdoor", poison_rate=0.05, seed=42)
pixels = load_image_inputs(attacked, sample_ids=range(len(attacked)))
```

`pixels.images` now contains the post-attack pixels, including the backdoor
patch. This makes the same API suitable for clean and poisoned MNIST/CIFAR-10
inputs. Pixel detectors must use `sample_ids` when returning scores so their
results can be joined with embedding-based detectors.

### Full training runs, poison rates, and saved artifacts

The local frontend offers an **Extraction scope** selector. **Quick sample**
shows a sample-count field for fast checks. **Whole training split** removes
that field and processes every row in the selected training split; there is no
hidden sample limit in this mode.
For experiments, choose **Clean**, **1%**, **3%**, **5%**, or **10%** poisoning
for either label flipping or the backdoor patch. Clean runs have no poisoned
rows; attack metadata remains evaluation-only and is not included in detector
inputs.

Each completed run is saved under `artifacts/` as:

```text
*-features.npz  # FeatureBundle: embeddings, scaled/PCA data, labels, IDs
*-images.npz    # ImageInputBundle: post-attack NCHW pixels, labels, IDs
```

Repeat the same request to load the existing feature bundle instead of
re-encoding it. You can also load files directly:

```python
from poison_features import FeatureBundle, ImageInputBundle

features = FeatureBundle.load("artifacts/<run>-features.npz")
pixels = ImageInputBundle.load("artifacts/<run>-images.npz")
```

`features.sample_ids` and `pixels.sample_ids` are identical, and labels are
stored separately from both representations. For backdoor runs, the saved
images include the white trigger patches; poison ground truth is available
only on `features.is_poisoned`/`features.poison_type` for evaluation.

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

For IMDB, MiniLM produces 384-dimensional sentence embeddings. The frontend
samples IMDB rows across the split instead of taking only the first rows,
because the original IMDB ordering groups negative reviews before positive
reviews. For network flows, the features are explicit engineered measurements such as duration,
packet count, byte count, ports, protocol, packet-length statistics, and TCP
flags. All modalities still preserve `sample_ids` so a detector score can be
mapped back to the original sample.

### IMDB split ordering

The labeled IMDB training split contains 25,000 reviews in this order:

```text
Rows 0–12,499      negative reviews (label 0)
Rows 12,500–24,999 positive reviews (label 1)
```

The test split follows the same convention. Therefore, a quick test that uses
only `range(100)` will contain only negative reviews and its chart will appear
to have one class. For a meaningful smoke test, sample from both ranges:

```python
from poison_features import load_imdb_dataset, extract_text

data = load_imdb_dataset(split="train")
indices = list(range(50)) + list(range(12_500, 12_550))
texts = [data[i]["text"] for i in indices]
labels = [data[i]["label"] for i in indices]
bundle = extract_text(
    texts,
    labels=labels,
    sample_ids=indices,
    dataset_name="imdb",
)
print(bundle.features.shape)  # (100, 384)
print(bundle.labels[:3], bundle.labels[-3:])  # 0s, then 1s
```

The frontend deliberately samples rows across the split for this reason.

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

### Text poisoning modes

The IMDB frontend supports:

- `label_flip`: selected review labels change from 0 to 1 or 1 to 0.
- `backdoor`: the phrase `excellent cinematic signal` is appended to selected
  reviews and their current labels are set to the target class.

Original labels, current labels, poison flags, and poison types remain
evaluation-only metadata. They are never included in MiniLM vectors or passed
to detectors.

### Synthetic attacks versus real poisoned data

The attack controls in the frontend create **synthetic experiments**. They are
useful because the expected poisoned rows are known:

- `label_flip` changes selected labels from 0 to 1 or 1 to 0.
- `backdoor` appends a known phrase to selected text and assigns a target label.

This is not the same as discovering an unknown attack in a real dataset. When
the extractor receives a real text dataset, it does not alter or “correct”
labels. It encodes the supplied text and preserves the supplied current labels:

```python
bundle.features
bundle.labels       # current labels from the input dataset
bundle.sample_ids
```

The extractor cannot know that a label is wrong by itself. A detector must
identify suspicious rows using the embeddings and current labels. Ground truth
can be evaluated only when an independent `original_label`, `is_poisoned`, or
`poison_type` field is available. Those fields must never be passed into the
detector while it is scoring samples.

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

Project policy requires an AI-assisted GitHub workflow for repository
operations and code changes. Contributors should use an AI coding assistant to
inspect the branch, make changes, run tests, review the diff, synchronize with
`main`, and prepare the pull request. Contributors must still inspect the
result and confirm that the tests pass; AI assistance does not replace human
review.

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

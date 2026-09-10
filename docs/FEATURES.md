# Feature extraction API

These Python examples support custom datasets. The web workflow is described in the [README](../README.md). Dataset-specific detector thresholds do not transfer automatically to custom inputs.

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
CIFAR-10 / MNIST images ──> ResNet-18 or DINOv2 ─┐
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

### Comparing image representations

Image extraction supports two interchangeable encoders:

```python
resnet_bundle = UniversalFeatureExtractor().extract_images(
    dataset, dataset_name="cifar10", encoder="resnet18",
    sample_ids=sample_ids,
)
dinov2_bundle = UniversalFeatureExtractor().extract_images(
    dataset, dataset_name="cifar10", encoder="dinov2",
    sample_ids=sample_ids,
)
```

ResNet-18 produces 512-dimensional ImageNet features. DINOv2 uses the
pretrained ViT-S/14 model and produces 384-dimensional features. Both return
the same `FeatureBundle` fields, preserve row order and IDs, and keep labels
and poison metadata separate. DINOv2 weights are downloaded and cached by
PyTorch Hub on first use, so internet access is required for that first run.

For CIFAR-10, the frontend automatically extracts both encoders; no encoder
selection is required. MNIST web scans use pixels and skip encoder extraction; the Python API still supports both encoders. Saved artifact names include the encoder, for example:

```text
cifar10-train-100-resnet18-...-features.npz
cifar10-train-100-dinov2-...-features.npz
```

For a fair comparison, use the same split, sample IDs, attack, poison rate,
seed, and detector settings with each bundle. Their feature spaces are
different and must not be mixed.

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

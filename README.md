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

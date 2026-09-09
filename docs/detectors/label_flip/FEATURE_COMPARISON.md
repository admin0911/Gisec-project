# Optional DINOv2 features

ResNet18 stays the default. To use frozen DINOv2 Small through the existing
feature connector:

```python
bundle = extractor.extract_images(
    dataset, labels=labels, sample_ids=sample_ids,
    encoder="dinov2_vits14", visual=False,
)
```

The extractor returns 384 raw embedding values per sample, preserves ordering,
and uses full-image 224x224 bicubic resizing and ImageNet normalization. It uses
the official facebookresearch/dinov2 code pinned to revision
7764ea0f912e53c92e82eb78a2a1631e92725fc8. First use downloads code and pretrained
weights into the standard torch.hub cache (configurable through TORCH_HOME).
There are no extra requirements beyond the combined requirements.txt.
The extraction web page now has an Image features selector: choose ResNet-18 or
DINOv2. It saves one selected encoder per run, with the encoder in the filename.
Run twice with identical dataset and attack settings to produce both bundles.
The frontend uses `encoder="dinov2"`; Python also accepts `"dinov2_vits14"`.
Both names use the same implementation in `poison_features/image.py`.
Import `DINOv2ImageEncoder` from `poison_features.image`; the redundant
compatibility module has been removed.

Run the paired development experiment from the repository root:

```powershell
python -m experiments.compare_image_encoders --batch-size 32
```

It expects the saved full clean seed-0 CIFAR ResNet18 feature and image bundles.
It picks 500 images per class, splitting them into 2,500 calibration and 2,500
evaluation images. Both encoders use identical IDs, supplied labels, and attack
seed. ResNet18 features are reused; DINOv2 features are extracted and cached.
It checks clean data and 5% cyclic label flipping.

Each detector's cutoff is its clean calibration score's 99th percentile; scores
must strictly exceed it. Discrete kNN scores may make this more conservative
than 1%. Confident Learning uses its continuous score here, not its native flag.
Two-of-three is a separate comparison rule, not a change to the app pipeline.
Compare the actual evaluation false-positive rates, not just nominal targets.

Results are written to artifacts/encoder_comparison/report.json and comparison.md.
This small comparison cannot establish final performance. CIFAR-10 has already
been explored during development. The existing pipeline thresholds were developed
on ResNet18 and must not be assumed valid for DINOv2.

# Calibrating clean DINOv2 label-flip scores

From the repository root, with the combined requirements installed:

```powershell
python -m experiments.calibrate_dinov2 artifacts/cifar10-train-full-dinov2-none-a0.10-t0-nrate-000-seed0-features.npz --known-clean
```

Only declare data clean when that is independently known. The command refuses
saved metadata that explicitly indicates poisoned samples. No poison metadata
enters the detectors.

The experiment uses a stratified 25,000/25,000 split, seed 20260910. Each half
is scanned separately: neighbours, class centres, and diagnostic classifiers
use only that half. The first half sets each score cutoff to the clean 99th
percentile. A flag requires a score strictly greater than the cutoff; ties
make the cutoff conservative. The second half measures false alarms with those
cutoffs frozen. Two of three flags forms the combined review rule.

Confident Learning uses 5-fold out-of-fold logistic predictions, seed 2026.
Its flag in this experiment uses the calibrated score threshold, not Cleanlab's
native issue mask. This distinction is recorded in the profile.

Results are saved in artifacts/dinov2_calibration/SIGNATURE/:
- profile.json: encoder, thresholds, comparison operator, versions and provenance.
- report.json and report.md: calibration and validation false-positive rates.
- separate NPZ score caches: aligned IDs and scores for every detector and half.

The signature includes source features, labels, IDs, detector code and package
versions. A repeated identical run can reuse scores. This does not update the
main pipeline or ResNet18 defaults. Dataset size affects these detectors; this
is a 25,000-row development calibration, not a validated full-50,000 preset.
The CIFAR-10 pool was previously explored, even though the two halves here do
not overlap. A later poisoned-data test must freeze these cutoffs and report
precision/recall and clean damage separately. Clean data alone cannot measure
poison detection recall.

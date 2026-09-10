# Experimental backdoor feature detectors

These two label-aware detectors accept the existing `DetectorInput`: finite
2D features, current known labels, and unique stable sample IDs. They never
read original labels, poison identities, attack type, trigger position, or
poisoning rate. Each returns the shared `detector_result` format. `score(inputs)`
also implements the repository's one-score-per-row interface.

## Methods and thresholds

| Method | Score and review rule | Limitations |
| --- | --- | --- |
| `SpectralSignatureDetector` | Center features per supplied class, find its leading right singular vector, and score absolute projections by positive deviation from their median, divided by `max(1.4826 * MAD, 1e-12)`. Flag scores >= 6.0. | A leading natural variation can hide a trigger; legitimate feature outliers can be flagged. |
| `ActivationClusteringDetector` | PCA to at most 10 dimensions per class, followed by deterministic two-means (10 initializations). Score minority members by centroid distance / pooled RMS cluster radius; other scores are zero. Flag only when minority count >= 3, minority fraction <= 0.35, and separation >= 3.0. | Rare legitimate subclasses look suspicious too. Majority poison, overlapping groups, and multiple triggers can be missed. |

All thresholds are provisional heuristics. They are not probabilities, a
known poison budget, or guarantees of clean false-positive rates. No fixed
fraction is automatically flagged. Clustering's fraction gate is **within
the supplied class**, not the entire dataset. Even 10% poisoning overall may
make the target class nearly half poisoned in a ten-class dataset.

Classes smaller than 10 samples or with constant features are skipped with
zero scores, false flags, an `evaluated` mask and explicit per-class reasons.
This means insufficient evidence, not a clean verdict. Cluster IDs are local
to each class. The spectral evidence includes squared projection energy in
the internally scalar-rescaled feature space; energies must not be compared
across classes. Default random seed is 0.

The algorithms are inspired by [Tran, Li and Madry's spectral signatures](https://arxiv.org/abs/1811.00636)
and [Chen et al.'s activation clustering](https://arxiv.org/abs/1811.03728).
These implementations are adaptations, not reproductions of the published
defences: the spectral threshold replaces budget-based removal, and the
clustering method omits exclusionary reclassification. In particular, frozen
ResNet-18, DINOv2, or MiniLM features may not encode an injected trigger the
way a model trained on poisoned data does. No published detection rate
transfers to these implementations.

## Run a saved feature bundle

From the repository root, using the environment from the main README:

```powershell
python -m experiments.scan_backdoor_features artifacts/NAME-features.npz --output artifacts/backdoor-scan.json
```

The default representation is `raw`. `--representation scaled` and
`--representation reduced` are available, but require separate validation.
Only load trusted local files produced by `FeatureBundle.save()`; its loader
uses NumPy pickle support for metadata.

```python
from poison_features import FeatureBundle, detector_input
from detectors.backdoor.feature_pipeline import scan_backdoor_features

bundle = FeatureBundle.load("artifacts/NAME-features.npz")
inputs = detector_input(bundle, representation="raw", label_aware=True)
result = scan_backdoor_features(inputs)
```

`detectors` preserves separate method outputs. `candidate_flags` is their
union for human inspection; `agreement_flags` requires both flags. These
methods use the same features and are not independent confirmations. Scores
are not averaged. The feature pipeline does not alter label-flip voting,
the existing pixel detector, web scans, training selection, or cleaning.

If the saved experiment includes boolean `is_poisoned`, add `--evaluate`.
The CLI scores first, then writes a separate `evaluation` section with TP,
FP, FN, TN, precision, recall, and false-positive rate. Undefined ratios are
JSON `null`. The report also records encoder, representation, runtime, and
dependency versions. Evaluation does not select a winning detector or tune
thresholds. Detection metrics do not measure model attack success rate (ASR).

## Tests and reproducible smoke benchmark

```powershell
python -m unittest discover -s tests -p test_backdoor_features.py -v
python -m experiments.benchmark_backdoor_features
python -m unittest discover -s tests -v
```

The first command checks input validation, finite scores, small and constant
classes, extreme finite values, clean and separated-cluster fixtures, stable
IDs, deterministic results, no input mutation, no truth access, and the CLI.
The existing full suite also requires downloaded MNIST data for its pixel test.

The benchmark uses 1,000 synthetic feature rows, 16 dimensions, two classes,
and seeds 11/23/37. It adds a large shared feature shift to 1%, 3%, 5%, 7%,
or 10% of all rows, plus clean controls. It tests feature geometry only:
there are no images, text triggers, classifier training, or ASR measurements.
It must not be cited as real-world backdoor detection accuracy.

## Runtime and validation still needed

Classes are processed sequentially. Randomized SVD uses seven power
iterations; clustering uses at most 10 projected dimensions. Approximate
dominant costs are O(n*d*k) for fixed-iteration projection and O(n*k*i) for
two-means, where k <= 10 and i <= 300 per initialization. Memory is O(n*d)
for float64 input and per-class work arrays; no n-by-n distance matrix is
built. Actual runtime depends on class count, feature size and BLAS threads.

Before web/cleaning integration, calibrate thresholds on independent clean
data for each encoder and dataset. Freeze settings before evaluating held-out
clean data and attacked data at 1%, 3%, 5%, 7%, and 10%, with multiple seeds,
targets, bright/dark/coloured patches, blended patterns and text phrases.
Report poison recall and clean false positives, then train/evaluate clean,
poisoned, and cleaned models on the same held-out test set. Measure clean
accuracy and triggered non-target ASR. Until those experiments are done,
neither detector establishes that an unknown dataset contains a backdoor.

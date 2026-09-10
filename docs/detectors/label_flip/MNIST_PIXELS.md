# MNIST pixel scanning experiment

From the repository root, with the project Python environment activated:

```powershell
python -m experiments.scan_mnist_pixels --poison-rate 0.05 --clean-baseline
```

Uses the downloaded MNIST training split in `data/MNIST` (60,000 images).
No test images or pretrained encoder are used. Omit `--clean-baseline` to scan
only the poisoned case. `--limit 2000` selects a reproducible random subset for
a quick test; results from this subset are not full-dataset results.

The existing image connector provides post-attack images as float32 values
in [0, 1]. Flattening gives 784 pixel features. The three existing detector
classes then apply their own row L2 normalization. Pixels are passed through
`DetectorInput`, not stored inside an embedding `FeatureBundle`.

The attack follows the shared project's label flip: randomly choose exactly
5% of rows, then set each selected label to `(label + 1) % 10`. Thus 5% of
60,000 means 3,000 changed labels. This is cyclic flipping on randomly chosen
samples, not uniformly random replacement labels. Default attack seed: 0.

## Rules fixed before this experiment

- kNN: cosine similarity, k=20, exclude self, flag at least 19/20 disagreeing neighbours.
- Class distance: leave-one-out class centre, default score threshold 0.1.
- Confident Learning: five-fold out-of-fold logistic regression, diagnostic
  seed 2026, cleanlab `prune_by_noise_rate`. Scores are 1 minus self-confidence.

These are provisional detector defaults, **not MNIST-calibrated thresholds**.
The CIFAR-10 calibrated pipeline and its combined rule are not used.
The summary additionally reports any / two-of-three / all-three detector votes
as exploratory comparisons. None automatically removes or relabels samples.

## Outputs

Each run writes a new folder under `artifacts/mnist_pixel_scans/`:

- `summary.json`: dataset, software versions, settings status, timings, and
  precision, recall, false-positive rate and confusion counts per rule/scenario.
- Per scenario, each detector has an `.npz` containing aligned `sample_ids`,
  `scores`, `flags` and evidence arrays, plus a `.json` describing its name,
  version and settings. Neighbour indices refer to the same ordered sample IDs.
- `review_votes.npz`: counts of individual detector flags per row.
- `evaluation_truth.npz`: poison identities, original labels and supplied
  labels, kept separate from detector inputs and outputs.

No precision or recall is invented when its denominator is zero; JSON uses
`null`. A flag means review, not proof of malicious poisoning. A single seed
is a baseline; use separate calibration and held-out configurations before
choosing thresholds. The experiment does not alter the web app or train models
for before/after accuracy comparison.

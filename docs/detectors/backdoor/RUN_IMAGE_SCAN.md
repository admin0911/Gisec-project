# Run the combined image backdoor scan

**Update:** the saved-file command now defaults to `--patch-profile contrast`.
It reports the original bright-patch detector as comparison-only. See
[PATCH_FIX.md](PATCH_FIX.md) for revised results and limitations. The legacy
results below describe the previous version and remain reproducible with
`--patch-profile legacy`.

The earlier `scan_backdoor_features` command scans embeddings only. On the
submitted 1,000-row CIFAR-10 patch experiment it missed all 54 poisoned rows
with both ResNet-18 and DINOv2. That failed result remains valid.

The new command adds the existing pixel detector, using its unchanged
`cifar-bright-patch-v1` profile. A pixel flag remains a review candidate even
when both feature methods return zero flags. This fixes incomplete routing;
it does not repair the feature methods' sensitivity or remove false positives.

## Windows / VS Code

Put the update files into your existing project folder, alongside `README.md`.
Keep your existing `artifacts` directory. Open a terminal in that folder.
Run with the environment already created on your laptop:

```powershell
C:\Users\hp\.venv\Scripts\python.exe -m experiments.scan_backdoor_images --folder artifacts --evaluate --open
```

For every `*-features.npz` file in that folder, the scanner requires the
matching `*-images.npz`. It scans every pair; do not mix text feature bundles
into this folder. Each encoder remains a separate report. This may repeat
the same pixel scan across two encoders, so candidate counts across reports
must not be added as if they were different images.

No downloads, re-extraction, server, or model training are needed for saved
files. Dependencies are NumPy, scikit-learn and Pillow. Report folders are
timestamped under `artifacts/backdoor-reports`. The index opens in your
browser; if it does not, open the printed `index.html` path manually.

Each report includes:

- Separate pixel, spectral and clustering results, and union review candidates.
- Inferred patch positions, sizes, support and label purity.
- Up to 24 original flagged images for visual review.
- Complete JSON and CSV scores for all rows.
- Precision, recall and clean false-positive rate only when saved truth exists.

`--evaluate` reads experiment truth only after scoring. A file named "clean"
does not establish ground truth. Without saved truth, evaluation is N/A.
For a single independently known clean image bundle, assert that explicitly:

```powershell
python -m experiments.scan_backdoor_images --images artifacts/CLEAN-images.npz --dataset cifar10 --known-clean --open
```

For a single feature bundle with the normal matching image filename:

```powershell
python -m experiments.scan_backdoor_images --features artifacts/NAME-features.npz --evaluate --open
```

If an image file was renamed on download, specify it explicitly:

```powershell
python -m experiments.scan_backdoor_images --features artifacts/NAME-features.npz --images "artifacts/NAME-images(1).npz" --evaluate --open
```

IDs, order and current labels must match; the scanner rejects mismatches.
Matching IDs alone cannot prove provenance: use the same extraction run.
Only load trusted local bundles produced by this project, whose feature
loader uses pickle support for stored metadata.

## How to interpret this experiment

With the existing CIFAR pixel profile on the submitted files:

| Dataset | Poison caught | Poison missed | Clean flagged |
| --- | ---: | ---: | ---: |
| Clean 1,000 images | N/A | N/A | 0 |
| Attacked 1,000 images | 54/54 | 0 | 7/946 |

Backdoor precision is 88.52%, recall 100%, and clean FPR 0.74% for this
single experiment. All 61 findings require review; neither the seven clean
matches nor the 54 poisons can be distinguished from ground truth during
scoring. No automatic removal or label correction is performed.

Keep the failed feature results and these fixed-profile results. Additional
seeds, trigger positions/colours, and independently held-out clean data are
needed before broader reliability claims. Thresholds must not be tuned to
these known poison identities. Model clean accuracy, triggered attack success
rate, and recovery after cleaning still need separate training experiments.

The HTML report is a standalone local report. The existing web app and its
training/cleaning policies are unchanged by this command.

## Executed checks for this update

`python -m unittest discover -s tests -v`: **172 tests passed**, including
nine new image-pipeline/report regressions. Checks cover pixel findings with
constant uninformative features, alignment rejection before scanning,
truth-free scoring, post-scoring evaluation, missing ground truth, CLI folder
runs, HTML/CSV output, explicit clean assertions, and escaped report content.
The new command was also executed directly on the user's uploaded clean
images and matching backdoor image/feature bundles, reproducing the counts
above. No previously generated feature scores or detector thresholds changed.

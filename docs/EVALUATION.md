# Calibration and evaluation

## Separate threshold selection from evaluation

Choose a false-positive budget before inspecting held-out attack results. Fit
thresholds on a separate calibration subset, freeze the settings and sample IDs,
then report detection precision, recall and clean-row false-positive rate on
disjoint evaluation samples. Calibration requires a defensible reference; an
unmodified public dataset is not guaranteed to be perfectly labeled or clean.
Distribution changes require fresh validation.

The CIFAR profile was calibrated using 25,000 rows. Full-50,000 results include
calibration rows and must be described as development evaluation. Do not remove
that limitation by renaming a status field. Check saved split IDs and report
provenance before claiming independent validation.

- [Threshold definitions and provenance](detectors/label_flip/THRESHOLDS.md)
- [DINO calibration](detectors/label_flip/DINO_CALIBRATION.md)
- [Consensus benchmark](detectors/label_flip/CONSENSUS_BENCHMARK.md)

## IMDB candidate calibration

The candidate experiment splits a saved clean-reference feature bundle into
disjoint calibration and evaluation halves, saves split IDs and a frozen
profile, and evaluates label-flip cases separately. It does not automatically
replace the app's provisional thresholds. Previously inspected public data
remains development evidence even when the new split is disjoint.

```powershell
python -m experiments.calibrate_imdb_thresholds --clean artifacts/YOUR-CLEAN-features.npz --output artifacts/imdb_threshold_calibration/YOUR-RUN --budget 0.02
```

Use `--help` for optional phrase inputs. Preserve the input identities,
`split_ids.npz`, `frozen_profile.json` and result files with the experiment.
A lower false-positive budget can reduce recall: report both rather than
choosing a threshold from the best-looking attacked test case.

## Evidence for the report

Record dataset, sample count, attack type, poison rate, trigger settings,
detector/profile version and evaluation split. Detection metrics need known
attack identities; those identities never enter scoring or selection.
Report clean rows removed as well as poisoned rows caught.

For downstream impact, use the three-seed [training comparison](TRAINING.md)
with matched steps and show clean accuracy alongside ASR where implemented.
Keep per-seed reports and include standard deviations. A synthetic unit test,
a quick sample and an independent full-data benchmark are different evidence.
Do not present one as another or omit a poor result to imply universal success.

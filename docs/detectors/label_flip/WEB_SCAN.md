# Local label-flip scan

Run `python serve_frontend.py` from the project folder and open
http://127.0.0.1:8787/ (or your configured `GISEC_PORT`). Restart an already
running server after adding the scan API.

Build or select a saved CIFAR-10 dataset on the scan page. CIFAR builds prepare
both encoders automatically. Scanning runs six label checks followed by patch
and blended-noise checks, with separate findings. Saved matching results may
reopen from cache. Human review is optional; training preparation is a separate
step described in [Training](../../TRAINING.md).

The service passes `DetectorInput` to the three detectors. Known poison
identities and original labels are not used in detection or assessment.
Every detector's scores and calibrated flags use the shared output connector.
Full results are saved under `artifacts/label_flip_scans/<job_id>/results.json`.
The browser gets a smaller summary. Job progress is in memory; saved results
remain on disk when the server restarts.

## Frozen development profile

The UI calls `scan_calibrated_label_flips` from `calibrated_pipeline.py`.
`detectors/label_flip/thresholds.py` is the single source of its separately
calibrated cutoffs from the existing paired clean-25k experiment.
All six flags use **score strictly greater than cutoff**. Confident Learning
uses its calibrated score rather than requiring its native cleanlab flag.
The saved ResNet18 kNN cutoff is 1.0, so that detector contributes no votes
under this profile. It is displayed explicitly, not silently relaxed.
Each encoder requires two of three votes. Both encoder flags means suspected;
one means uncertain; neither means not flagged.

These are experimental CIFAR-10 settings, not validated universal thresholds.
Different scan sizes and poison rates can change performance. The app reports
findings rather than a clean/poisoned verdict. MNIST and IMDB use their own scan adapters; this paired profile is CIFAR-specific. No new thresholds are fitted from the dataset being scanned.

Teammate-file edits are marked with `Leila:` comments. The scan adapter,
service, frontend controller, page and styles are separate files.
# Reusing completed scans

Clicking **Scan label flips** first checks for a completed matching scan.
Matching uses SHA-256 hashes of both feature files and both image files,
the detector profile, implementation files and dependency versions. A match
reopens the original scan ID, preserving saved human review decisions.
Changed inputs or settings run a new scan. Failed or corrupted results are
never reused. Input hashes are checked again before saving a completed scan.

Older scans without cache identity records remain viewable but need one new
scan to establish a verified reusable result. Results folders are created and
write access checked before the detectors run.

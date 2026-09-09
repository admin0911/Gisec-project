# Local label-flip scan

Run `python serve_frontend.py` from the project folder and open
http://127.0.0.1:8787/ (or your configured `GISEC_PORT`). Restart an already
running server after adding the scan API.

Already have features? Choose a pair under **Saved features for scanning** and
press **Scan label flips**. Available pairs load automatically when the page
opens; the selected saved file is the scan input, separate from extraction controls.

To create a new pair:

1. Choose CIFAR-10, scope and attack settings.
2. Extract ResNet18 features, then DINOv2 with the same settings. Existing
   matching artifacts are reused by the extraction page.
3. Press **Scan label flips** below extraction. The separate results page
   shows six stages, per-detector flags, paired assessment and a saved file path.

Changing controls disables Scan until extraction completes for those settings.
The server checks IDs, supplied labels and identical saved image pixels before
scanning. Missing or failed checks are errors. One scan runs at a time.
Human review remains disabled; scanning does not clean or train anything.

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
findings rather than a clean/poisoned verdict. MNIST and IMDB feature extraction
still work, but this paired scan is disabled for them until suitable profiles
exist. No new thresholds are fitted from the dataset being scanned.

Teammate-file edits are marked with `Leila:` comments. The scan adapter,
service, frontend controller, page and styles are separate files.

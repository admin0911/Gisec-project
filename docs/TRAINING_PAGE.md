# Prepare and compare training in the local app

At the bottom of a completed Scan results page, click **Prepare training
dataset**. A separate Training page prepares a new immutable selection version
from that scan and the current saved human review revision. Preparation does
not train anything. A refresh reopens the same version; return to Scan results
and prepare again to include later review decisions.

The final selection is:

| Evidence/choice | Action |
|---|---|
| Scanner not flagged | Keep |
| Saved human Keep | Keep |
| Saved human Quarantine | Quarantine |
| Saved Unsure | Unresolved, held out |
| Scanner suspected label flip, not reviewed | Quarantine |
| Scanner uncertain, not reviewed | Unresolved, held out |

Saved human choices override scanner defaults. Quarantine is exclusion for
suspicion, not proof of poisoning. Policy version 2.0 records this routing;
previously prepared versions and their training results remain unchanged.

Both uncertain and suspected scanner groups require an explicit human Keep
to enter training. A human Keep is a label-flip review choice, not clearance
from other attack detectors. No automatic relabeling is performed. No poison
identities or original clean labels are consulted.

`artifacts/training_preparations/<version>/manifest.json` records membership,
reasons, supplied source image path and SHA-256, scan hash, review revision and
saved-choice snapshot. ID arrays are also saved. The dataset version is an
index selection over the original saved images, not another large pixel copy.
Training rejects a source image file whose hash changed after preparation.

The page shows two cards: **Kept for training** and **Removed from training**.
Removed combines quarantined and unresolved rows, with both counts shown below.
The original images remain saved.

The separate **Demo evaluation** table shows known clean and poisoned counts
in the original, kept and removed rows. `/api/training/evaluation` reads the
local benchmark poison identities only after the selection is frozen; it does
not feed ground truth to scanning, selection or training. Empty poison masks
are treated as clean only when the metadata explicitly records a clean run
with zero poison rate and count. Missing or inconsistent truth is displayed
as unavailable. The table also works for existing prepared versions.

**Train & compare** runs three fresh `small_cnn_v1` models using the shared
`training.trainer.train_classifier`: clean reference, original input and kept rows. All
use seed 42, Adam at 0.001, batch size 128 and the selected epoch count (default
5). The official local CIFAR-10 test split is loaded with download=False and
used after training. There is no hyperparameter selection on test results.
The app uses CPU with four threads and allows one comparison at a time.

For input whose images and labels exactly match the corresponding official
clean reference rows, the duplicate before-cleaning model is skipped. Only
clean reference and after cleaning are trained, with progress shown as 1 of 2
and 2 of 2. Accuracy change is then relative to the clean reference. The
before-cleaning result row is also hidden for known-clean older demo runs;
their saved results are not changed.

Each arm saves its checkpoint, training sample IDs, test predictions and
metrics. `artifacts/web_training/<job_id>/comparison.json` holds the comparison.
The clean reference loads the official clean training split and selects the same
sample IDs, in the same order, as the original input. It does not read injected
poison identities or feed clean labels into scanning or cleaning. This reference
is possible for this benchmark because its original clean dataset is available.
The Training page displays test accuracy for all three arms, with the
percentage-point change under More information. Older runs show Not run for
the clean reference until a new comparison is started.
Backdoor ASR is not measured by this label-flip comparison. It is one seed;
same epochs imply different optimizer-step counts when filtering removes rows.

The URL retains the prepared version and training job so refreshing does not
restart work. Completed jobs can reopen after a server restart. A running job
interrupted by restart is reported as interrupted, not complete.

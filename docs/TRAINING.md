# Training and comparison

## Web workflow and selection

Open the Training page from a completed scan and prepare a selection. Preparation
records the source hash, sample IDs, actions, reasons and saved review revision
in `artifacts/training_preparations/<version>/manifest.json`. Original data is
retained. Prepare a new version to include later review decisions.

Human review is optional. Saved Keep or Quarantine choices override scanner
defaults; saved Unsure remains unresolved and excluded. Unreviewed findings
routed for exclusion are quarantined. Only Keep rows enter training. No automatic
label repair or poison-truth lookup decides selection. Dataset-specific voting
rules determine which findings reach this policy; individual detector votes are
not themselves proof of poisoning.

The demo evaluation table reads known poison identities separately after
selection. Missing truth is unavailable, not evidence that all samples are clean.

## Models and defaults

| Dataset | Classifier | Default epochs |
|---|---|---|
| CIFAR-10 | Small CNN on pixels | 15 |
| MNIST | Small single-channel CNN on pixels | 15 |
| IMDB | Linear classifier on frozen MiniLM embeddings | 20 |

Quick mode uses seed 42. Benchmark mode uses paired seeds 42, 43 and 44 and an
equal optimizer-step budget across arms, derived from the reference training
size and selected epochs. A smaller cleaned dataset can therefore traverse
more epochs. Defaults use Adam, learning rate 0.001 and batch size 128.

The three arms are clean reference, before cleaning and after cleaning. A matched
10% validation subset is excluded from training. The final checkpoint is evaluated;
there is no automatic early stopping or best-validation checkpoint selection.
Use validation rather than official test scores to choose training settings.
The clean reference is available for synthetic benchmarks, not required by the
real scanner. It is not independent validation of detector thresholds.

## Metrics and ASR

Accuracy, macro F1 and macro precision measure classification on the separate
official test split. They are different from poison-detection precision/recall.
Benchmark tables show mean and standard deviation across three seeds; expanded
details and loss curves show seed 42. Rates are percentages and standard
deviations are percentage points.

ASR is the fraction of originally non-target test samples predicted as the
target after applying the attack trigger. Lower is better; also inspect clean
accuracy and the untriggered target-prediction rate.

| Scenario | ASR in the web comparison |
|---|---|
| CIFAR-10 patch or single blended-noise attack | Supported using the verified trigger recipe |
| IMDB phrase backdoor | Supported for the app's silver lantern prefix, target 1 |
| Label flips | Not applicable |
| MNIST patch/blended and mixed image attacks | Supported; mixed triggers evaluated separately, main ASR is their unweighted mean |

For a compatible saved CIFAR-10 or MNIST comparison, ASR can be added without retraining:

```powershell
python -m experiments.add_image_asr artifacts/web_training/YOUR-JOB/comparison.json
```

## Saved results and reconnecting

Completed web reports are under `artifacts/web_training/<job_id>/comparison.json`,
with per-arm checkpoints, predictions, training IDs and reports. Benchmark runs
also retain per-seed outputs. The chart download summarizes the comparison.
Dataset identity, attack settings and preparation version must match when
comparing reports.

Refresh can reconnect to a running job while the server remains running.
Restarting the server interrupts active work; completed reports remain on disk.
A bare `/train` URL is not a shareable identifier for a specific result: retain
the preparation/scan/run identifiers or the report artifact.

## Python training API

# Training connector

`training/connector.py` accepts a dataset view, stable sample IDs, dataset version
and split. Each item is `(input_tensor, current_label)`. No detector scores,
thresholds or poison identities enter training. Use the actual post-attack images
and supplied labels. A teammate can consume TrainingInput in another trainer.

```python
from cleaning.label_flip import partition_dataset
from training import training_input, train_classifier, TrainConfig
from training.models import small_cnn

views = partition_dataset(dataset, sample_ids, decisions)
kept = views['keep']
train = training_input(kept, kept.sample_ids,
                       dataset_version='cifar10-run-001', split='train')
test = training_input(test_dataset, test_ids,
                      dataset_version='cifar10-official-test', split='test')
report = train_classifier(train, test,
    model_factory=lambda: small_cnn(channels=3, num_classes=10),
    model_name='small_cnn_v1', num_classes=10,
    config=TrainConfig(epochs=10, seed=42), output_root='artifacts/training',
    progress=print)
```

Variables dataset, IDs, decisions and test_dataset come from the caller. The
model factory must construct a new untrained torch module returning class logits.
For MNIST use channels=1. Inputs to the provided CNN are original-sized float32
pixels in [0,1], not ResNet/DINO embeddings. Small CNN is a baseline, not a claim
of optimal CIFAR-10 accuracy. Existing extractors stay frozen and separate.

The trainer supports epoch and optimizer-step budgets, Adam and cross-entropy. The web benchmark uses matched optimizer steps across arms. Use validation data in separate experiments for tuning; do not choose
epochs or thresholds using test results. Seeds are recorded, but exact matching
across hardware is not guaranteed. ID overlap is rejected; the caller must also
ensure IDs are honest, split-qualified and correspond to distinct source data.

## CLI using saved files

```powershell
python -m training.run --images artifacts/YOUR-images.npz --decisions artifacts/YOUR-RUN/decisions.json --dataset cifar10 --dataset-version YOUR-RUN --epochs 10
```

Replace placeholders with matching artifacts from the same dataset version.
The manifest must cover every image ID. Only keep rows enter training; all other
rows remain saved outside the training view. The official test dataset must
already be downloaded under data/. Use --device cuda if available.

## Standard training result

The returned dictionary has schema_version, trainer_name, version, model_name,
num_classes, settings, dataset_versions, training_samples, test_samples, history,
metrics, seconds, torch_version and artifacts. Metrics include accuracy (0 to 1),
per_class_accuracy, class_support and a confusion matrix (true rows, predicted
columns). Missing classes receive null accuracy. No poison detection metrics are
claimed by this trainer.

Each run saves report.json, model.pt (state dictionary), training_ids.npy and
test_predictions.npz with sample_ids, predictions and labels. Reconstruct the
recorded model architecture before loading weights. Never load untrusted files.

To compare filtering policies, construct different dataset views externally and
call the same trainer with matching model/settings/seeds/test split. A clean
reference run belongs to benchmark code, not to the real cleaning decision.


# Legacy saved-score comparison CLI

Run from the project root with the project virtual environment active:

```powershell
python -m experiments.compare_training --benchmark-dir artifacts/consensus_policy_benchmark/22bd13c20957bd60 --rate 0.05 --attack-seed 20260911 --epochs 10
```

This uses existing local label-flip scores/decisions; no extraction or detector
rerun. The benchmark artifacts and official CIFAR-10 train/test data must exist
locally; generated artifacts are not included in a Git clone.

Three NEW small CNN models use the same initialization seed and training settings:

1. Original clean images and labels, for the exact 25,000 benchmark rows.
2. The same 25,000 images with saved poisoned labels, before filtering.
3. Only keep rows, with their saved supplied labels. Quarantine and human review
   are excluded; original labels are not used to repair the training input.

All models evaluate on the same official 10,000-image clean test split. Clean
reference labels exist only in this benchmark; real cleaning does not require
them. Saved current labels are checked against the benchmark configuration to
prevent accidentally mixing cases. The script does not change decisions.

Use --device cuda if available. Defaults are CPU, four threads, batch size 128,
training seed 42. Change --rate to 0.01, 0.03 or 0.10 and --attack-seed to another
previously scanned seed to compare existing cases. A one-epoch run checks wiring
only; it is not sufficient evidence of cleaning effectiveness. Do not tune the
training schedule against the test scores.

Results are saved under artifacts/training_comparison/<timestamp>/:
comparison.json, comparison.md, and each model's report, checkpoint, training IDs
and test predictions. The JSON is updated after each completed arm, so completed
results survive interruption, but automatic resume is not implemented.

Filtering changes the number of batches per epoch. These are equal-epoch
comparisons, not equal optimizer-step comparisons. Single-seed differences are
preliminary; repeat training seeds for stronger evidence. This benchmark covers
label flips and clean-test classification only, not backdoor ASR.

The shared training module is unchanged. Other teammates can call
compare_training with their own aligned clean/poisoned datasets, decisions and
model factory. No frontend or GitHub integration is added by this script.

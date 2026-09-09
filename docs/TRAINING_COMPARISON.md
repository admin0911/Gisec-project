# Clean, poisoned and filtered training

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

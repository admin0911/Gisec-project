# Testing the two-encoder consensus policy

Run from the repository root with the same dependency versions used in calibration:

```powershell
python -m experiments.benchmark_consensus_policy artifacts/paired_encoder_comparison/20260909-130922-199408 artifacts/dinov2_calibration/93437e0cec649c3c
```

This reuses the clean ResNet18 and DINOv2 feature files referenced by the reports.
No image extraction occurs. It evaluates 1%, 3%, 5%, and 10% cyclic label flips
at seeds 20260911, 20260912, and 20260913 on the same 25,000 validation rows.
Both models retain their own frozen clean-calibrated score thresholds. Each
model's combined flag requires two of three detector votes. Confident Learning
uses calibrated score thresholds, not its native issue mask, in this experiment.

The proposed policy is:
- Neither model flags: keep in training.
- Both models flag: quarantine from training, preserving the source.
- Models disagree: hold out for human review.

The benchmark saves sample IDs, per-detector scores, model flags, policy actions,
and separate evaluation summaries under artifacts/consensus_policy_benchmark.
Every completed case updates report.json. The final report.md shows averages
across seeds; report.json also contains every individual case and precision ranges.
Caches are keyed by data, source code, versions, profile, and thresholds.
Re-running the same command reuses completed scores and neighbours.

Watch these quantities separately: quarantine precision, clean images wrongly
quarantined, clean images held for review, human-review workload, and poisoned
images mistakenly kept. A review case is not an automatically repaired image.

This is development evidence on a previously explored CIFAR-10 pool, not final
independent validation. ResNet18's calibrated kNN cutoff is 1.0 with strict >,
so that detector contributes no votes. No thresholds or app policies are changed
by this experiment. Higher quarantine precision alone does not establish that
the cleaned training dataset improves downstream model performance.

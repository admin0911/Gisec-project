# Label-flip assessment

```python
from detectors.label_flip.assessment import assess_label_flips

assessment = assess_label_flips(resnet_scan, dino_scan)
```

Each complete scan supplies sample_ids, combination_votes (boolean arrays named
knn, class_distance, confident_learning), and combination_settings recording
the thresholds used. Both scans must cover the same images and supplied labels;
use the paired-feature adapter before scanning. This function joins scan results
by sample ID and returns them in ResNet18 order. Missing scans/votes are errors.

Each encoder is flagged at two or more detector votes. Then:

| ResNet18 | DINOv2 | Assessment |
|---|---|---|
| Not flagged | Not flagged | not_flagged |
| Flagged | Flagged | suspected_label_flip |
| Different flags | Different flags | uncertain |

Output includes assessment, reasons, per-encoder flags, vote counts, individual
votes, settings, and summary counts. `flags` is a review signal for either
suspected or uncertain rows. It is NOT an instruction to remove data.
No synthetic probability is made by averaging scores.

This assessment does not choose thresholds or run extraction. Use separately
calibrated thresholds for each encoder; never apply the fixed ResNet preset
to DINOv2 features. It accepts the existing ResNet pipeline's combination_votes
and equivalent DINO vote records from calibrated scores.

The final decision/cleaning step can later map suspected rows to quarantine,
and uncertain rows to human review (or uncertain quarantine when review is off).
That decision is separate, so backdoor evidence can be considered as well.
The old `cleaning.route_label_flips` is an earlier sequential experiment, not
this two-encoder assessment. Nothing is deleted, trained, or relabeled here.

The rule is experimental. Both unflagged can still include single-detector
warnings; the returned individual votes retain them. Agreement is not proof
of poisoning, and absence of a flag does not establish clean data.

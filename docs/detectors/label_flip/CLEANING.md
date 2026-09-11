# Post-scan routing and reversible quarantine

The examples below describe standalone policies and historical experiments. For the active web workflow, see [Training](../../TRAINING.md).

## Standalone policy API (review disabled by default)

```python
from cleaning import decide_label_flip_actions, partition_dataset, save_decisions

decisions = decide_label_flip_actions(assessment)  # human_review_enabled=False
parts = partition_dataset(post_attack_dataset, dataset_sample_ids, decisions)
training_dataset = parts['keep']
save_decisions(decisions, 'artifacts/cleaning', source_description='dataset version')
```

The assessment comes from `assess_label_flips(resnet_scan, dino_scan)`.
Not flagged -> keep; suspected -> quarantine; uncertain -> quarantine with an
explicit uncertainty reason. No human input is required. Training uses only
the keep view. Original pixels and supplied labels remain unchanged. These
decisions address label flipping only; they do not override backdoor evidence.
The function neither trains a model nor changes the frontend automatically.

Pass `human_review_enabled=True` to hold uncertain samples for human review.
This option never returns unresolved samples to training automatically.

## Earlier sequential experiment (retained for reproducibility)

This experimental policy is separate from the shared detector dictionary.
`flags` and `review_flags` keep their existing meanings; neither confirms poison.
The new output uses `actions`, reasons, and boolean masks for the dashboard.

| Stage | Votes | Action |
|---|---|---|
| ResNet18 | 0 | Keep |
| ResNet18 | 1 or 2 | Request second check |
| ResNet18 | 3 | Quarantine |
| DINOv2, requested rows only | 0 | Keep |
| DINOv2, requested rows only | 1 or 2 | Human review |
| DINOv2, requested rows only | 3 | Quarantine |

These are operational decisions, not statements that data is clean or poisoned.
Unanimous detectors can share mistakes. The zero-vote route can miss poisons.
Human-review and pending rows stay out of the training view until resolved.
Nothing is deleted or automatically relabeled.

```python
from cleaning import route_label_flips, partition_dataset, save_decisions

# primary_scan is the existing ResNet18 pipeline output.
pending = route_label_flips(primary_scan)
requested_ids = pending['sample_ids'][pending['needs_second_opinion']]

# Obtain DINOv2 detector results with separately calibrated DINO thresholds.
# secondary_scan must provide sample_ids, combination_votes (three boolean
# arrays), and combination_settings. It may cover all rows or requested IDs.
# Do NOT apply the fixed ResNet18 scan_label_flips thresholds to DINO features.
decisions = route_label_flips(primary_scan, secondary_scan)
parts = partition_dataset(post_attack_dataset, dataset_sample_ids, decisions)
training_dataset = parts['keep']
quarantine_dataset = parts['quarantine']
human_review_dataset = parts['human_review']
save_decisions(decisions, 'artifacts/cleaning', source_description='your dataset version')
```

`partition_dataset` makes index views preserving the dataset's images and
current labels. IDs must cover exactly the same dataset; order can differ.
Saved JSON contains reasons, vote counts, policy/threshold settings and IDs;
separate ID lists let downstream code recreate each view from the source.
The original dataset must be retained. The web app supports saved review choices; see [Training](../../TRAINING.md) for its current selection policy.

This module routes existing results. It does not call DINOv2 or alter the web
app. A second-stage service must provide representative reference features for
kNN and class centres: scanning only suspicious images in isolation is invalid.
Until selective-reference scanning is validated, compute the full secondary
scan and use its results for requested IDs. Do not claim compute savings yet.

To test on already saved feature-comparison results:
```powershell
python -m experiments.evaluate_routing
```
This writes decisions and evaluation separately under artifacts/routing_results.
Poison identities are used only in that evaluation script, never in routing.

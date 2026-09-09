# Frozen benchmark thresholds

The source of truth is detectors/label_flip/thresholds.py. These are the exact
cutoffs used in the full 50,000-image CIFAR-10 benchmark, not newly tuned values.

| Detector | ResNet18 | DINOv2 |
|---|---:|---:|
| kNN disagreement | > 1.0 (inactive) | > 0.8 (17 or more out of 20 disagree) |
| Class distance | > 0.024618882985227353 | > 0.04322571427866734 |
| Confident Learning score | > 0.9513872983583611 | > 0.7984671745392444 |

All comparisons are strict greater-than. CL uses 1-self-confidence from five
out-of-fold predictions and does not require Cleanlab's native flag. At least
two of three detector votes flag an encoder scan. ResNet kNN cannot exceed 1.0;
therefore both class distance and CL must vote for a ResNet combined flag. This
is intentional fidelity to the benchmark; do not silently change it to >=1.0.

```python
from detectors.label_flip.calibrated_pipeline import scan_calibrated_label_flips
from detectors.label_flip.assessment import assess_label_flips
from cleaning.decision import decide_label_flip_actions

resnet_scan = scan_calibrated_label_flips(resnet_inputs, encoder='resnet18', progress=print)
dino_scan = scan_calibrated_label_flips(dino_inputs, encoder='dinov2', progress=print)
assessment = assess_label_flips(resnet_scan, dino_scan)
decisions = decide_label_flip_actions(assessment, human_review_enabled=True)
```

Use paired raw feature inputs for the same dataset version. Both encoders flag:
suspected label flip/quarantine. Exactly one: uncertain/human review. Neither:
not flagged/keep. Both quarantine and human review stay out of the training view.
This does not confirm poisoning. Calibration was on 25k clean CIFAR-10 rows;
the full 50k benchmark includes these rows and is not independent validation.

The old scan_label_flips entry point in pipeline.py remains a legacy preset.
Use the explicit calibrated entry point above to reproduce these results.
The frontend is not automatically connected to it. No benchmark data or model
extraction is rerun just by loading the profile. Scores are not probabilities.

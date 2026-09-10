# Repeated-patch detector

Input: the shared `ImageInputBundle` containing post-attack NCHW pixels in [0,1], current labels and unique sample IDs. No feature extraction or poison metadata is needed.

```python
from poison_features.image_inputs import ImageInputBundle
from detectors.backdoor import RepeatedPatchDetector
from detectors.output_connector import to_jsonable

pixels = ImageInputBundle.load("artifacts/your-dataset-images.npz")
result = RepeatedPatchDetector().analyze(pixels)
# result has detector_name, version, sample_ids, scores, flags, settings, evidence
```

`score_bundle(pixels)` is also supported. `score_bundle(features, pixels)` validates matching IDs and labels before using pixels.

The scanner searches every position and all observed classes for quantized 2x2/3x3 patterns. Defaults require at least max(30, 0.5% of rows) matches, at most 20% support, label purity >=90%, label lift >=3, and spatial lift >=5. These are heuristic support settings, not knowledge of the injected poison rate.

Scores are evidence strengths, not poisoning probabilities. Evidence includes patch coordinates, size, quantized pattern, support, dominant label, purity and spatial lift. A row's pattern_id refers to its highest-scoring accepted pattern, or -1 if none.

Flags mean review, not confirmed poisoning. Smaller patch fragments may match innocent images. Bright fixed-position patches are the intended scope; dark, variable-position, larger, blended and text triggers may be missed. RGB support is experimental and thresholds are not calibrated for CIFAR-10. Synthetic tests verify connector behavior and inserted-trigger detection, not real-world accuracy.

MNIST and CIFAR web scans run this detector after label-flip scanning on the same image connector. Patch findings and overlap counts are saved and shown separately. Existing label-flip votes, human review and training selection are unchanged at this stage. IMDB skips the image patch stage. Benchmark against separate clean and attacked datasets before integration; use known poison identities only afterward for evaluation.


## CIFAR matching comparison

`python -m experiments.compare_cifar_patches` compares 4, 16 and 256 intensity bins. 256 matches complete uint8 patches exactly, with collision-free byte keys for RGB. All other acceptance checks remain fixed. The web default remains 4 bins.

On the saved 1,000-row backdoor input (54 poisoned rows), four and sixteen bins caught zero; exact matching caught all 54 with 7 false positives (88.52% precision). All resolutions flagged zero on the corresponding clean input.

`python -m experiments.compare_cifar_patches --validation-bins 256` tested three separate 1,000-row blocks with 50 poisoned rows each, new seeds, labels and positions. Exact matching caught 0/50 top-left target 7, 50/50 centre target 2 (zero false positives), and 50/50 off-centre target 9 (3 false positives). All three clean blocks had zero flags. The top-left exact group had 89.29% label purity, below the unchanged 90% threshold. These results are exploratory, not full calibration, and do not justify claiming a general CIFAR fix. Known poison identities are used only after detection for metrics.


## Clean-only calibration experiment

Run `python -m experiments.calibrate_cifar_patches` to reproduce the fixed candidate search. It uses three disjoint 5,000-row clean blocks (training rows 4000:19000), excluding the previously inspected first 4,000 rows. All 18 candidates passed the predefined per-block 0.1% false-positive budget with zero flags. The predeclared sensitivity order therefore selected exact uint8 matching, minimum count 10, purity 0.7, label lift 3 and spatial lift 3. This is an empirical choice among tied clean results, not proof of optimality or a guaranteed false-alarm bound. Other defaults remain unchanged, including minimum fraction 0.005 and brightness filtering.

The profile was written to frozen_profile.json before inspecting any held-out attacks. On separate training rows 19000:36500, all six clean-only runs (17,500 unique images) had zero flags. Attacked runs caught 250/250 red centre patches with no false positives; 500/500 white off-centre patches with 2 false positives; 50/50 white top-right patches with 6 false positives; and 25/25 centre patches in a 500-row test with no false positives. It missed all 50 poisons in a 1% top-left white-patch test and all 50 poisons in a black-patch test. Black patches are outside the current brightness-filter scope.

Artifacts: artifacts/cifar_patch_calibration/20260910-051458-750811. These are single configurations at each rate/size, not broad validation. No settings were changed after the failures. No target-model training or ASR evaluation was performed. The web default remains unchanged pending a decision about the supported trigger scope.


## Web profile activation

CIFAR web scans now explicitly use `cifar-bright-patch-v1`: exact uint8 matching, min_count 10, min_purity 0.7, min_lift 3 and min_spatial_lift 3. The minimum 0.5% support and other defaults remain in force. MNIST retains its original settings. Connector settings record the profile and limited calibration scope. Patch-only flags enter Needs review and unresolved samples are excluded from new training preparations. These flags are not proof of poisoning. Source hashing invalidates old scan cache entries; saved historical results are preserved. Earlier web-default descriptions above describe the pre-activation experiments.

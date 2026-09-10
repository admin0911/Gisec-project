# Team integration contracts

Use the existing feature and output connectors. Add teammate implementations in
their own files; do not modify everyone else's algorithms or thresholds.

| Stage | Input | Public connector / output |
|---|---|---|
| Dataset and extraction | Actual current images and supplied labels | `FeatureBundle` and separate `ImageInputBundle`, same stable sample IDs |
| Feature detector input | FeatureBundle | `poison_features.detector_input(bundle, representation="raw", label_aware=True)` returns X, sample_ids, y |
| Pixel detector input | Post-attack dataset or saved image bundle | `load_image_inputs(...)` / `ImageInputBundle.load(...)` returns images, sample_ids, labels |
| Individual detector | Appropriate feature or pixel input | `detectors.output_connector.detector_result(...)` |
| Run teammate detectors | Mapping of names to callable adapters | `detectors.runner.run_detectors(...)` returns individual validated results |
| Attack-specific assessment | Detector evidence using recorded calibrated rules | Existing label-flip `assess_label_flips(resnet_scan, dino_scan)`; other teams own their assessment logic |
| Attack-specific cleaning policy | Assessment, no ground truth | Existing `decide_label_flip_actions(...)`; other teams return sample_ids, actions, settings |
| Combine attack checks | Complete decisions from explicitly required checks | `cleaning.connector.combine_decisions(...)` |
| Training handoff | Actual dataset, IDs, combined decisions | `cleaning.label_flip.partition_dataset(...)`; train on views['keep'] |
| Dashboard/export | Detector results, assessments, decisions | `to_jsonable(...)` for JSON; `save_decisions(...)` for manifests |
| Evaluation | Predictions/decisions plus separately held known poison identities | Separate experiment code; never passed to detector or cleaning calls |

## Add a detector

```python
from detectors.output_connector import detector_result
from detectors.runner import run_detectors

def my_detector(inputs):
    raw = MyDetector().analyze(inputs)  # teammate implementation
    # raw contains aligned sample_ids, finite scores and boolean flags
    return detector_result("my_detector", "1.0", raw,
                           {"threshold": 0.8},
                           expected_sample_ids=inputs.sample_ids)

results = run_detectors(inputs, {"my_detector": my_detector}, progress=print)
```

The snippet is an adapter template, not an implemented MyDetector. A score-only
detector must explicitly apply its own calibrated threshold to create flags.
Scores mean higher suspicion; they are not probabilities and are not averaged
across detectors. Existing label-flip scan outputs already contain a detectors
dictionary in this format. Run pixel and different encoder inputs separately.
The runner propagates failures and checks names, order, unique IDs, finite scores,
and boolean flags. It does not choose thresholds or combine votes.

## Add an attack-specific cleaning result

Each team supplies this minimum dictionary after its own assessment/policy:

```python
decision = {
    "sample_ids": sample_ids,
    "actions": actions,  # one of keep, human_review, second_check, quarantine
    "settings": {"policy_name": "my_attack_policy", "policy_version": "1.0"},
}
```

Use recorded evidence to choose actions, never the selected attack setting or
known poison IDs. Preserve uncertain evidence in the assessment even if the
cleaning policy excludes those rows.

```python
from cleaning.connector import combine_decisions
from cleaning.label_flip import partition_dataset

decisions = combine_decisions(sample_ids, {
    "label_flip": label_flip_decision,
    "backdoor": backdoor_decision,
}, required_checks=["label_flip", "backdoor"])
views = partition_dataset(dataset, sample_ids, decisions)
training_data = views["keep"]
```

Backdoor decisions above must be supplied by the backdoor team; no fake all-clear
placeholder is provided. Missing/partial checks fail rather than clear data.
Checks are joined by ID. Quarantine takes priority over pending second check,
then human review, then keep. All non-keep rows are held out without deletion.
Keep means permitted by the listed checks, not proof of clean data. This combining
policy is experimental and has not been validated across multiple attack types.

These generic helpers are integration APIs; the web selection workflow uses cleaning.selection.merge_choices and frozen preparation manifests (see TRAINING.md).
The training teammate can consume DatasetView with their normal data loader and
owns model initialization, training and untouched-test evaluation. Do not feed
DINOv2 into the old fixed ResNet label-flip preset; use encoder-specific calibration.

## Verify

```powershell
python -m unittest discover -s tests -p test_team_connectors.py -v
```


# Legacy single-encoder preset and paired feature loading

The connector lives in `detectors/output_connector.py` and
`detectors/label_flip/pipeline.py`, with tests in `tests/test_output_connector.py`.
Install the project's combined `requirements.txt` in your Python 3.12 environment.

```python
import json
from poison_features import FeatureBundle, detector_input
from detectors.label_flip.pipeline import scan_label_flips
from detectors.output_connector import to_jsonable

bundle = FeatureBundle.load("artifacts/cifar10-train-full-none-000-seed0-features.npz")
inputs = detector_input(bundle, representation="raw", label_aware=True)
result = scan_label_flips(inputs, progress=print)
with open("detector_results.json", "w", encoding="utf-8") as output:
    json.dump(to_jsonable(result), output, allow_nan=False)
```

Each entry under `result['detectors']` contains detector_name, version,
sample_ids, scores, flags, settings, and optional evidence. Arrays preserve
input order. Scores are not poisoning probabilities and must not be averaged.
Evidence can include large neighbour arrays; omit evidence from dashboard
summary responses if the UI only needs scores and flags.

`flag_count` counts the **stricter combination votes**, not the individual
detector flags. `combination_votes` exposes each vote. `review_flags` means
at least two votes. Individual class-distance flags use threshold
0.024119124718243068; combination votes use 0.029466908865236896. Confident
Learning votes require both its own flag and score >= 0.90. kNN requires
at least 19 of 20 neighbours to disagree. No detector logic was changed.

This is the previously evaluated CIFAR-10 development preset using frozen raw
embeddings. It is not independently validated or transferable automatically
to other feature extractors or datasets. Unflagged does not mean known clean.
It does not remove or relabel samples. Ground truth and evaluation metrics are
excluded from the connector. Dependency versions are included in settings.

Backdoor detectors can call `detector_result(...)` with their own outputs and
settings. They must supply only detector evidence, never poison ground truth.
Their combined decisions remain separate from this label-flip preset.

## Receiving both image feature files

```python
from detectors.label_flip.feature_inputs import load_paired_feature_inputs

inputs = load_paired_feature_inputs("path/to/resnet-features.npz",
                                   "path/to/dino-features.npz")
resnet_inputs = inputs["resnet18"]
dino_inputs = inputs["dinov2_vits14"]
```

Both paths must be trusted local `FeatureBundle.save()` outputs. This adapter
checks the encoders, feature dimensions, dataset name, IDs, row order and supplied
labels. It returns separate raw feature inputs; it never merges vectors or
passes known poison identities to detectors. Matching IDs cannot verify the
pixels themselves: extraction must use the same dataset version for both files.
The filename convention is up to the extraction service.
The adapter accepts both `dinov2` (frontend) and `dinov2_vits14` (earlier Python
experiments) as encoder names; the returned dictionary uses `dinov2_vits14`.

This prepares inputs only. DINOv2 scoring needs its own calibrated thresholds;
do not pass DINOv2 inputs to the fixed ResNet18 pipeline preset.

Run the connector tests with:
```powershell
python -m unittest discover -s tests -p test_output_connector.py -v
```


## Active web routing

CIFAR uses paired encoder inputs; MNIST uses pixel inputs; IMDB uses MiniLM and saved text. Patch and blended checks consume post-attack pixels. Scoring never receives poison truth. Saved human choices and detector assessments enter the web selection connector before an immutable preparation is created. See [Training](TRAINING.md) and [Detectors](DETECTORS.md). The legacy preset above is not the calibrated two-encoder web policy.

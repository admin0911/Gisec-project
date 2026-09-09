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

The frontend and training runner are not automatically wired to these new helpers.
The training teammate can consume DatasetView with their normal data loader and
owns model initialization, training and untouched-test evaluation. Do not feed
DINOv2 into the old fixed ResNet label-flip preset; use encoder-specific calibration.

## Verify

```powershell
python -m unittest discover -s tests -p test_team_connectors.py -v
```

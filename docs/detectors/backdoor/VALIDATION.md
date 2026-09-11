# Contribution review and validation — 10 September 2026

Reviewed source: GitHub `admin0911/Gisec-project`, commit
`4b5719dba0580f65e5dc35f30c69f65e16099bd8`, which matches the uploaded ZIP.
Read `README.md`, `CONTRIBUTING.md`, the detector connector guide, and the
existing backdoor detector implementation and tests before implementation.

## Repository assessment

- The feature and pixel connectors preserve aligned IDs and keep known poison
  identities outside scoring. New feature detectors can use the existing
  boundary without changing extraction or attack generation.
- Backdoor detection already includes a repeated bright 2x2/3x3 patch
  heuristic. A separate blended-injection residual detector also exists.
  The detector index incorrectly described the backdoor directory as empty;
  this contribution corrects that index.
- The existing patch method explicitly limits its scope. Feature-based
  methods provide additional evidence when a trigger changes the embedding,
  but frozen encoders can suppress a small trigger completely.
- Results must be treated as review candidates. Feature anomalies alone do
  not distinguish attacks from legitimate rare examples.

## Changes

Added spectral-signature scoring, minority feature-cluster analysis, a shared
truth-free input validator, and a pipeline with separate outputs. Added a
saved-feature JSON CLI, a reproducible synthetic geometry benchmark, twelve
test methods, and method documentation. No extraction, attack generation,
frontend, automatic cleaning, or training policy files were changed.

## Executed verification

Environment: Python 3.12.14; NumPy 1.26.4; scikit-learn 1.7.2;
Cleanlab 2.7.1; CPU PyTorch 2.14.0; torchvision 0.29.0.

| Check | Result |
| --- | --- |
| `python -m unittest discover -s tests -v` | **163 tests passed**, 12.541 seconds; includes all 12 new test methods. |
| Saved-feature CLI | Executed by an integration test, both with and without post-scoring evaluation. |
| `python -m experiments.benchmark_backdoor_features` | 18 runs completed; approximately 0.96 seconds total detector time on this environment. |
| `node --test tests/*.cjs` | 4 passed, 4 failed; the same four failures reproduce in the untouched ZIP baseline. |

The baseline first lacked PyTorch/Cleanlab and the MNIST fixture. Installing
the required dependencies and downloading MNIST resolved the Python failures.
No baseline tests were weakened or skipped to obtain the passing result.

Existing frontend failures are `test_build_page.cjs` (missing mock
`source-control`) and `test_imdb_scan_page.cjs`, `test_mnist_scan_page.cjs`,
`test_patch_scan_page.cjs` (undefined `scan-examples` node). These files and
their application code are unchanged by this contribution.

## Synthetic geometry results

Each rate aggregates three seeds, 1,000 rows each. Both methods produced the
same counts for this deliberately easy shared-shift fixture:

| Shifted fraction of all rows | Shifted rows caught | Shifted rows missed | Clean rows flagged |
| --- | ---: | ---: | ---: |
| 0% | 0 | 0 | 0 / 3,000 |
| 1% | 30 | 0 | 0 / 2,970 |
| 3% | 90 | 0 | 0 / 2,910 |
| 5% | 150 | 0 | 0 / 2,850 |
| 7% | 210 | 0 | 0 / 2,790 |
| 10% | 300 | 0 | 0 / 2,700 |

These are synthetic feature shifts, not measured image/text poisoning
performance. There are only two classes and a large shift of 8 against
per-coordinate noise standard deviation 0.25. No classifier was trained,
no real CIFAR/IMDB trigger was tested, and neither attack success rate nor
post-clean model quality was measured. Thresholds remain uncalibrated.
The separate regression fixture also demonstrates that a clean false
positive is possible; perfect benchmark counts are not a general guarantee.

## Next validation milestone

Use matching real clean and attacked saved feature bundles, validate each
encoder separately, and freeze thresholds using independent clean data.
Then report held-out detection recall, clean false positives, clean-test
model accuracy, and triggered non-target ASR before and after cleaning.
Keep the feature pipeline experimental until those results support a web
integration decision. See `FEATURE_DETECTORS.md` for commands and limitations.

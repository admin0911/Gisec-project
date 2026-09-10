## Active web scanner

The MNIST and CIFAR-10 scan adapters now use the teammate pipeline through
`web_noise.scan_noise` and `scan_as_connector_result`, after patch scanning.
All supplied classes are scanned; attack target and poison identities are not passed.
The residual-signature method falls back to background-lift when no residual finding is accepted.
Shared flags enter optional human review and training preparation. Saved Keep overrides flags;
unreviewed flags are quarantined; saved Unsure stays unresolved and excluded.
Cached scans are invalidated by changes to the noise implementation.

The consensus-pixel implementation below is retained for historical experiments only;
it is no longer used in web scanning.

Quick integration check: 1,000 randomly selected training images per dataset, seed 2026,
5% shared blended noise, alpha 0.1, attack seed 0, target 0. MNIST caught 50/50 with
0 false positives; CIFAR caught 0/50 with 0 false positives. Both clean checks had
0 flags. These are development checks, not held-out validation or ASR measurements.

# MNIST consensus-pixel detector

Experimental background-anomaly detector for MNIST-like images. Uses the submitted pixels only; labels and evaluation poison identities do not influence scores.

```python
from poison_features.image_inputs import ImageInputBundle
from detectors.blended_injection import ConsensusPixelDetector
from detectors.output_connector import to_jsonable

inputs = ImageInputBundle.load("path/to/submitted-images.npz")
result = ConsensusPixelDetector().analyze(inputs)
if not result["evidence"]["applicable"]:
    print("Inconclusive: too few consensus pixels")
else:
    review_ids = result["sample_ids"][result["flags"]]
serializable = to_jsonable(result)
```

Output uses the shared detector_name/version/sample_ids/scores/flags/settings/evidence contract. Scores are fractions of consensus pixels disturbed, not poisoning probabilities. A score of zero in an inconclusive run must not be interpreted as evidence of clean data. Extra evidence contains status, applicable, consensus_pixel_count, consensus_pixel_indices (flattened CHW positions), and image_shape. The optional progress callback takes (completed, total), matching pixel scan callbacks.

Defaults: consensus 0.90, tolerance 1/255, flag_fraction 0.50 (strict greater), minimum 50 consensus pixels. Flags are review requests. The MNIST web scan runs this check after patch scanning. Applicable flags enter the shared review queue as uncertain; unresolved samples are excluded from training. Other datasets do not run it.

Limitations: benign background changes may be flagged; faint or foreground-only changes can be missed; higher contamination can eliminate the consensus reference. Not validated for natural photographs. Development tests using the application blend attack found all 3,000 injected samples in full MNIST at 5% poisoning and alpha 0.10 with zero clean flags. These are development results, not independent final validation or measured attack success after training.

Run connector and edge-case tests:

```powershell
python -m unittest tests.test_consensus_pixels
```

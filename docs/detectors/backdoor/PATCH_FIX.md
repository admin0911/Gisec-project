# Contrast patch detector update

This update addresses three observed weaknesses of the old pixel detector:
dark patches were excluded, natural white regions diluted low-rate groups,
and fragments of those regions caused false positives. The spectral and
clustering detectors remain unchanged and experimental; no improvement in
their real-attack sensitivity is claimed.

## Run on your existing artifacts

Merge the update folders into the existing project, keeping `artifacts`,
`data` and your Python environment. From the project directory in VS Code:

```powershell
C:\Users\hp\.venv\Scripts\python.exe -m experiments.scan_backdoor_images --folder artifacts --evaluate --open
```

The command now uses `contrast-patch-v1` for active pixel decisions. It also
runs the original detector, visibly labelled **comparison only**, so old and
new outcomes can be assessed without mixing their flags. Experimental feature
flags remain active review evidence if feature bundles are supplied. No
scores are averaged, and nothing is automatically deleted or relabelled.

To reproduce the original active decisions, add `--patch-profile legacy`.
For individual or renamed image files, the `--features`, `--images`, and
`--known-clean` options documented in `RUN_IMAGE_SCAN.md` remain available.

Each timestamped report includes image previews, all-sample CSV/JSON,
inferred patch positions, and candidate rejection diagnostics. Poison truth
is used only for optional evaluation after scoring. Clean bundles with no
truth retain N/A evaluation metrics unless explicitly asserted clean.

The existing extraction web page and its existing automatic review/training
policies are unchanged. Use the command above for the revised scan. The
Python image-pipeline API keeps its legacy default for existing callers;
new callers must request `patch_profile="contrast"` explicitly.

## Algorithm and fixed settings

The new detector searches every location for exact quantized uint8 2x2/3x3
patches. It does not receive a trigger colour, location, target label, clean
counterpart, or poison identity. At each image/pattern match, it computes the
mean pixel difference from recovered integer pixel values across each visible
boundary and requires the minimum
of those edge means to be at least 1/255. At least two edges must be visible.
Fragments inside larger flat regions fail this check, regardless of colour.

Only boundary-qualified matches enter group support and label-association
checks. Minimum support is max(6, 0.5% of samples), maximum support 20%, label
purity >=70%, label lift >=3, and spatial lift >=3 against neighbouring
locations with the same boundary check. Reducing minimum count from 10 to 6
allows a 1% group to survive when some of its members have no visible edge.
These are development settings, not probabilities or formal guarantees.

Candidate diagnostics report raw support, boundary-qualified support and
reasons such as insufficient boundary support, low label purity, or low
spatial lift. Details are capped at 500 groups by default; totals and a
truncation marker disclose omissions. Scores remain evidence strengths.

## Executed development and reserved-block evaluation

Development used the supplied training-image pair and the already-inspected
CIFAR test rows 0-999. On the original supplied attack, the revised detector
caught 54/54 with zero false positives; the supplied clean control had zero
flags. This is development evidence, not independent validation.

An initial comparison on rows 1000-1999 exposed an exact-threshold rounding
bug in normalized float arithmetic. It was corrected using integer pixel
differences and a regression test. Those rows then became development data.
The corrected implementation was fixed before scanning test rows **2000-2999**.
The plan and source hashes were saved before scoring. We tested a clean control and
54 attack conditions: white/red/black 3x3 patches, bottom-right/centre, rates
1%/5%/10%, seeds 17/29/43, target 0. All classes were eligible for injection.
The same 1,000 base images were reused; 2,880 poison instances below are
repeated observations across conditions, not 2,880 unique poisoned images.

| Aggregate over attack conditions | Legacy | Contrast |
| --- | ---: | ---: |
| Poison instances caught | 1,890 / 2,880 | 2,867 / 2,880 |
| Poison instances missed | 990 | 13 |
| Detection recall | 65.63% | 99.55% |
| Clean false-positive instances | 56 | 0 |
| Flags in the separate clean control | 0 | 0 |

Both detectors together took about 53 seconds in the test environment.
The contrast method still missed some weak-boundary patches. In particular,
white corner recall was 30/30 at 1%, 147/150 at 5%, and 296/300 at 10%,
aggregating three seeds. Red patches were all detected in this matrix.
Black-patch recall was 954/960 across its conditions. The 13 residual miss
instances concern five unique images, all below the fixed boundary threshold;
two miss instances involved no actual pixel change after injection. No thresholds were
changed after inspecting these reserved-block results.

The update's full Python suite passed **187 tests**, including eleven new
contrast-detector regressions. It tests dark/bright/coloured patches, clean
backgrounds, support dilution, boundary fragments, input alignment,
determinism, diagnostics, truth separation and comparison-only routing.
The prior four frontend test failures are unrelated and remain unresolved;
the web frontend was not modified by this detector contribution.

## Reproduce and interpret the limits

```powershell
C:\Users\hp\.venv\Scripts\python.exe -m experiments.validate_patch_fix --open
```

This defaults to rows 2000-2999 and reproduces the evaluation above. It writes
`plan.json`, `summary.json`, and `index.html` under `artifacts/patch-fix-validation`.
The dataset is loaded from the existing cache; `--download` enables download
if missing. No encoders or model weights are required for this pixel test.

Use a fresh block, for example `--start 3000`, for later independent checks.
Do not retune on one block and continue calling that same block held out.

This is not a complete defence against every backdoor. Exact matching can
miss compressed, noisy, blended, variable-size and moving triggers. A patch
matching its surroundings can be visually unidentifiable; low support and
legitimate class-specific graphics remain challenging. Current results do
not establish text backdoor detection, model attack success rate, clean
model accuracy or recovery after retraining. Those need separate experiments.

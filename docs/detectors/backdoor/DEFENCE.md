# Backdoor quarantine and retraining defence

The defence command turns the revised patch scan into a reversible quarantine,
trains three fresh classifiers, and evaluates both normal test accuracy and
targeted backdoor attack success rate (ASR). It supports the controlled square
patch used by the CIFAR-10 and MNIST backdoor experiments.

## Run from VS Code on Windows

Use the poisoned ResNet18 feature bundle, its matching post-attack image bundle,
and the aligned clean image bundle from before the attack:

```powershell
C:\Users\hp\.venv\Scripts\python.exe -m experiments.defend_backdoor `
  --features artifacts\cifar10-train-1000-resnet18-backdoor-s1-a0.10-t0-nrate-005-seed0-features.npz `
  --images artifacts\cifar10-train-1000-resnet18-backdoor-s1-a0.10-t0-nrate-005-seed0-images.npz `
  --clean-images artifacts\cifar10-train-1000-resnet18-none-a0.00-t0-nrate-000-seed0-images.npz `
  --epochs 10 --threads 4 --open
```

Add `--download` once if the official test split is not already under `data/`.
Use `--device cuda` only when the installed PyTorch build can access an NVIDIA
GPU. The output is saved under `artifacts/backdoor-defence/<timestamp>/`.

## Decision policy

The detector receives only pixels, features, current labels, and stable sample
IDs. Poison identities, the clean reference, the target label, and the trigger
configuration are not supplied to detection or selection.

- A `contrast_patch` flag is quarantined.
- A spectral-signature or activation-clustering finding without a patch finding
  is held for human review.
- The comparison-only legacy patch result is ignored.
- Every non-`keep` row is excluded from defended training without deleting or
  changing the source bundle.
- Missing, partial, misaligned, or unknown active detector results stop the run.

The complete decisions and separate ID lists are saved before training. Saved
poison identities are read afterward only to report removal precision, poison
removal recall, and clean retention.

## Three-arm comparison

The command initializes a fresh `small_cnn_v1` with the same seed and settings
for each arm:

| Arm | Training rows |
|---|---|
| Clean reference | Aligned pre-attack images and labels |
| Before defence | Post-attack pixels and supplied labels |
| After defence | Post-attack input restricted to `keep` decisions |

All three models are evaluated on the same official clean test split. The same
test images are then copied and stamped with the configured evaluation patch.
ASR is the fraction of non-target-class test images predicted as the target.
Conditional ASR uses only non-target images the same model classified correctly
before the trigger. Clean target rate and ASR lift are also recorded so ordinary
target-class bias is not confused with a learned trigger effect.

## Reproduced 1,000-image CIFAR-10 run

With the saved 5% white bottom-right patch bundle, target label 0, ten epochs,
batch size 128, and training seed 42:

| Measure | Result |
|---|---:|
| Poisoned rows quarantined | 54 / 54 |
| Clean rows quarantined | 0 / 946 |
| Clean-data retention | 100% |
| Before-defence clean accuracy | 34.38% |
| After-defence clean accuracy | 34.81% |
| Before-defence triggered ASR | 27.70% |
| After-defence triggered ASR | 5.78% |
| Before-defence conditional ASR | 9.65% |
| After-defence conditional ASR | 0.06% |

The matched clean-reference model obtained 35.60% clean accuracy and 7.31%
triggered target rate. The defended model's remaining 5.78% target rate is below
that reference rate; its backdoor ASR lift was -0.50 percentage points. These
numbers demonstrate mitigation for this controlled run, not universal immunity.

## Limitations

This experiment uses 1,000 training images, one model architecture, one training
seed, and an exact square evaluation trigger. A stronger final study should use
the full training split, multiple seeds, and independent trigger configurations.
Adaptive, blended, moving, semantic and text triggers require their own detectors,
cleaning policies and ASR definitions. A zero finding never proves that a dataset
is clean.

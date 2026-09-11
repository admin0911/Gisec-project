# Next step: evaluate the fixed patch profile on new images

From the existing project directory, after copying this update:

```powershell
C:\Users\hp\.venv\Scripts\python.exe -m experiments.validate_patch_scope --open
```

The command loads the first 1,000 CIFAR-10 **test** images. Previous uploaded
runs used training images. It uses the existing local CIFAR cache; add
`--download` if that cache is missing. It does not download model weights or
extract embeddings.

The experiment plan is written before scoring, then results are saved after
each condition. Default run: one clean control plus 54 attack conditions:

- White, red and black 3x3 patches.
- Bottom-right and centre positions.
- Exactly 1%, 5%, and 10% of the selected subset.
- Seeds 17, 29 and 43; target label 0.

The same 1,000 base images are reused for each condition. This is **not**
55,000 independent test images. Sample selection is nested across rates and
matched across colours and positions for each seed. All original classes are
eligible, including rows already labelled target 0.

Current CIFAR thresholds remain fixed; no condition is used to choose a
detector or tune thresholds. Black patches are a deliberate out-of-scope
stress test for this bright-patch profile. Misses must remain in the results.
Two fixed positions do not test a trigger moving independently in every image.

Open `artifacts/patch-validation/<timestamp>/index.html` if the browser does
not open automatically. Upload `summary.json` for comparison. The report
includes TP, FP, FN, TN, precision, recall, clean FPR, and false-positive/missed
sample IDs for every condition. Source hashes and sample IDs are retained.

This tests **pixel detection only**. The spectral and clustering feature
methods are still experimental and were ineffective on the earlier uploaded
attack. No clean accuracy, attack success rate (ASR), or post-clean model
quality is measured here. Once results are examined, treat this subset as
development data for future changes; reserve a new subset (for example via
`--start 1000`) before testing any revised detector.

Implementation tests cover correct injection, stable IDs, no input mutation,
repeatable selections, truth separation, and report generation. Real CIFAR
results are produced when you run the command; no real-data performance is
assumed from those tests.

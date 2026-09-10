# Confident Learning experiment

This is diagnostic supervised training on supplied labels before final model training.
It never uses known poison identities during scoring. Five stratified folds produce
probabilities for each sample using a classifier trained on other samples only.
Raw frozen embeddings are normalized per sample; no dataset-wide scaler or PCA is
fitted before cross-validation. Logistic regression uses C=1 and a fixed fold seed.
The cleanlab filter is `prune_by_noise_rate`, with its default estimated noise counts;
the actual injected poison rate is not passed to it. Scores are one minus label
self-confidence and are not calibrated probabilities of poisoning.

## Shared environment

The combined `requirements.txt` includes Cleanlab and pins NumPy 1.26.4 and
scikit-learn 1.7.2. All components can use the same Python 3.12 environment.
Installing these requirements into an existing NumPy 2.x environment will
downgrade NumPy. From the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_confident_learning.py -v
.\.venv\Scripts\python.exe -m experiments.benchmark_confident_learning
```

The benchmark expects the existing full clean and 5% seed-0 CIFAR feature bundles
and the saved clean neighbour run `20260909-090644-524348`. It tests the clean split
once and cyclic label flips at 1%, 3%, 5%, 10% with attack seeds 0, 42, 123.
It recomputes class centres and trains fresh diagnostic classifiers for each label
configuration. kNN neighbours can be reused because these attacks change no pixels.

Results and out-of-fold probabilities are saved under
`artifacts/confident_learning_comparison`. Rerunning reuses completed cases only
when the code, data and package signature matches. Incomplete cases restart.
The report compares each detector and several Boolean combinations without changing
the app or existing detector defaults. Seeds 0 and 42 were already inspected in
previous experiments: this is an exploratory comparison, not independent validation.

References: https://docs.cleanlab.ai/v2.6.4/cleanlab/filter.html
and https://arxiv.org/abs/1911.00068.

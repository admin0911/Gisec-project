"""Compare three detectors on cached CIFAR features; no image extraction.
Run: python -m experiments.benchmark_confident_learning
Requires the project requirements.txt environment.
Models use only supplied labels. Attack truth is for evaluation after scoring.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from poison_features import FeatureBundle, DetectorInput
from detectors.label_flip.confident_learning import ConfidentLearning
from detectors.label_flip.class_distance import ClassDistance
from experiments.run_knn import evaluation

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.024119124718243068


def main():
    import cleanlab, sklearn
    clean_path = ROOT / 'artifacts/cifar10-train-full-none-000-seed0-features.npz'
    clean = FeatureBundle.load(clean_path)
    n = len(clean.labels)
    with np.load(ROOT / 'artifacts/knn_results/20260909-090644-524348/scores.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['sample_ids'], clean.sample_ids)
        neighbours = saved['neighbour_indices'].copy()
        assert neighbours.shape == (n, 20)
        assert np.all((neighbours >= 0) & (neighbours < n))
        assert not np.any(neighbours == np.arange(n)[:, None])
    old = FeatureBundle.load(ROOT / 'artifacts/cifar10-train-full-label_flip-005-seed0-features.npz')
    out = ROOT / 'artifacts/confident_learning_comparison'
    out.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256()
    for file in (Path(__file__), ROOT / 'detectors/label_flip/confident_learning.py'):
        signature.update(file.read_bytes())
    for arr in (clean.features, clean.labels, clean.sample_ids):
        signature.update(np.ascontiguousarray(arr).tobytes())
    signature.update(f'{cleanlab.__version__}/{sklearn.__version__}/{np.__version__}'.encode())
    digest = signature.hexdigest()
    cases = [(0., 0)] + [(r, s) for r in (.01, .03, .05, .1) for s in (0,42,123)]
    rows = []
    timings = []
    for rate, seed in cases:
        name = f'rate{rate:.2f}-seed{seed}'
        print(f'\n{name}', flush=True)
        labels = clean.labels.copy()
        selected = np.sort(np.random.default_rng(seed).choice(n, size=round(n * rate), replace=False))
        labels[selected] = (labels[selected] + 1) % 10
        if rate == .05 and seed == 0:
            np.testing.assert_array_equal(labels, old.labels)
            np.testing.assert_array_equal(clean.sample_ids, old.sample_ids)
        inputs = DetectorInput(clean.features, clean.sample_ids, labels)
        cache = out / f'{name}-{digest[:12]}.npz'
        start = time.perf_counter()
        if cache.exists():
            with np.load(cache, allow_pickle=False) as data:
                result = {key: data[key] for key in data.files}
            np.testing.assert_array_equal(result['sample_ids'], clean.sample_ids)
            print('Loaded cached out-of-fold results', flush=True)
        else:
            result = ConfidentLearning().analyze(inputs, progress=lambda s: print(s, flush=True))
            np.savez_compressed(cache, **result)
        elapsed = time.perf_counter() - start
        timings.append(dict(rate=rate, seed=seed, seconds=elapsed))
        knn = np.mean(labels[neighbours] != labels[:, None], axis=1) >= .95
        distance = ClassDistance(THRESHOLD).analyze(inputs)['flags']
        cl = result['flags']
        votes = knn.astype(int) + distance.astype(int) + cl.astype(int)
        masks = dict(knn=knn, class_distance=distance, confident_learning=cl,
                     either_original=knn | distance, both_original=knn & distance,
                     any_three=votes >= 1, two_of_three=votes >= 2, all_three=votes == 3)
        truth = np.zeros(n, dtype=bool)
        truth[selected] = True
        for method, flags in masks.items():
            rows.append(dict(rate=rate, seed=seed, method=method,
                             flagged=int(flags.sum()), **evaluation(flags, truth)))
        report = dict(settings=dict(folds=5, fold_seed=2026, classifier='L2 normalized embeddings; LogisticRegression C=1 lbfgs',
                      cleanlab_filter='prune_by_noise_rate', class_threshold=THRESHOLD, knn_threshold=.95),
                      versions=dict(cleanlab=cleanlab.__version__, sklearn=sklearn.__version__, numpy=np.__version__),
                      signature=digest, limitation='Exploratory on the same image pool used for clean calibration; class-conditional cyclic label flips only.',
                      completed_cases=len(timings), total_cases=len(cases), timings=timings, runs=rows)
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(f"Finished {name}: CL flags={cl.sum()}, seconds={elapsed:.1f}", flush=True)
    lines = ['# Three-detector comparison', '', 'Mean results across seeds 0, 42, 123; clean evaluated once.', '',
             '| Poison rate | Rule | Precision | Recall | False-positive rate |', '|---|---|---:|---:|---:|']
    for rate in (0., .01, .03, .05, .1):
        for method in masks:
            chosen = [r for r in rows if r['rate'] == rate and r['method'] == method]
            values = []
            for metric in ('precision','recall','false_positive_rate'):
                numbers = [r[metric] for r in chosen if r[metric] is not None]
                values.append(f'{np.mean(numbers):.2%}' if numbers else 'N/A')
            lines.append(f'| {rate:.0%} | {method} | ' + ' | '.join(values) + ' |')
    (out / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()

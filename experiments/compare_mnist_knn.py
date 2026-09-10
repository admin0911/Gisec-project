"""Compare 17/20, 18/20 and 19/20 rules on saved MNIST pixel scans.

python -m experiments.compare_mnist_knn artifacts/mnist_pixel_scans/RUN [MORE_RUNS]
Development comparison only: does not change thresholds or existing scans.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from experiments.run_knn import evaluation


def compare_scenario(folder):
    with np.load(folder / 'knn.npz', allow_pickle=False) as saved:
        ids = saved['sample_ids']
        counts = saved['disagreement_counts']
        neighbours = saved['neighbour_indices']
        if neighbours.shape != (len(ids), 20):
            raise ValueError('Expected k=20 saved neighbours.')
        if counts.shape != ids.shape or not np.isin(counts, np.arange(21)).all():
            raise ValueError('Invalid disagreement counts.')
        if np.any(neighbours < 0) or np.any(neighbours >= len(ids)):
            raise ValueError('Invalid neighbour indices.')
        if np.any(neighbours == np.arange(len(ids))[:, None]):
            raise ValueError('Self-neighbours must be excluded.')
        np.testing.assert_allclose(saved['scores'], counts / 20)
        knn_settings = json.loads((folder / 'knn.json').read_text())['settings']
        np.testing.assert_array_equal(saved['flags'], counts / 20 >= knn_settings['threshold'])
    other_flags = []
    for name in ('class_distance', 'confident_learning'):
        with np.load(folder / f'{name}.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['sample_ids'], ids)
            if saved['flags'].shape != ids.shape or saved['flags'].dtype.kind != 'b':
                raise ValueError('Flags must be aligned boolean arrays.')
            other_flags.append(saved['flags'])
    # All candidate decisions use detector evidence only, before consulting truth.
    rules = {}
    for threshold in (17, 18, 19):
        knn = counts >= threshold
        votes = np.sum(np.stack([knn, *other_flags]), axis=0)
        rules[threshold] = {'knn': knn, 'two_of_three': votes >= 2}
    with np.load(folder / 'evaluation_truth.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['sample_ids'], ids)
        labels, truth = saved['supplied_labels'], saved['is_poisoned']
        np.testing.assert_array_equal(counts, np.sum(labels[neighbours] != labels[:, None], axis=1))
    return [dict(disagreeing_neighbours=threshold, k=20, rule=name,
                 flagged=int(flags.sum()), **evaluation(flags, truth))
            for threshold, variants in rules.items() for name, flags in variants.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    rows = []
    for root in args.runs:
        summary = json.loads((root / 'summary.json').read_text())
        if summary['dataset'] != 'mnist' or summary['feature_dimensions'] != 784:
            raise ValueError('Expected a MNIST pixel scan.')
        for scenario, original in summary['scenarios'].items():
            comparisons = compare_scenario(root / scenario)
            # The unchanged rule must reproduce the existing benchmark exactly.
            for item in comparisons:
                if item['disagreeing_neighbours'] == 19:
                    for key, value in original['results'][item['rule']].items():
                        if item[key] != value:
                            raise ValueError(f'Saved 19/20 results do not match: {key}')
            rows.extend(dict(source=str(root.resolve()), scenario=scenario,
                             attack_seed=summary['attack_seed'], **item) for item in comparisons)
    output = Path(__file__).resolve().parents[1] / 'artifacts/mnist_knn_comparisons' / datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='development_comparison_not_held_out_validation',
                  unchanged_detectors=['class_distance', 'confident_learning'], results=rows)
    (output / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# MNIST pixel kNN threshold comparison', '',
             'Development results from existing runs. No defaults or saved scan outputs were changed.', '',
             '| Run / scenario | Rule | Disagreement | Precision | Recall | False positives | FPR |',
             '|---|---|---:|---:|---:|---:|---:|']
    percent = lambda n: 'N/A' if n is None else f'{100*n:.2f}%'
    for row in rows:
        lines.append(f"| {Path(row['source']).name} / {row['scenario']} | {row['rule']} | {row['disagreeing_neighbours']}/20 | {percent(row['precision'])} | {percent(row['recall'])} | {row['false_positives']} | {percent(row['false_positive_rate'])} |")
    (output / 'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f'\nSaved: {output}')


if __name__ == '__main__':
    main()

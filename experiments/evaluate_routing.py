"""Evaluate routing on saved encoder scores; does not re-extract any images."""
import json
from pathlib import Path

import numpy as np

from cleaning.label_flip import NAMES, route_label_flips, save_decisions
from detectors.output_connector import to_jsonable


ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'artifacts/encoder_comparison'
    config = json.loads((source / 'report.json').read_text())
    reports = []
    for rate in (0.0, 0.05):
        scans = []
        for encoder in ('resnet18', 'dinov2_vits14'):
            with np.load(source / f'{encoder}-rate{rate:.2f}-scores.npz', allow_pickle=False) as archive:
                ids = archive['sample_ids'].copy()
                cuts = config['thresholds'][encoder]
                scans.append(dict(sample_ids=ids,
                    combination_votes={name: archive[name] > cuts[name] for name in NAMES},
                    combination_settings=dict(encoder=encoder, thresholds=cuts, comparison='strict_greater',
                        preset='small_sample_clean_percentile_development',
                        source_signature=config['source_signature'],
                        confident_learning_requires_individual_flag=False)))
        np.testing.assert_array_equal(scans[0]['sample_ids'], scans[1]['sample_ids'])
        routed = route_label_flips(scans[0], scans[1])
        # A one-stage DINO policy: zero -> keep, three -> quarantine, 1/2 -> review.
        dino_only = route_label_flips(scans[1], scans[1])
        # Poison truth is reconstructed only after every routing decision.
        truth = np.zeros(len(ids), dtype=bool)
        indices = np.random.default_rng(config['attack_seed']).choice(len(ids), round(len(ids)*rate), replace=False)
        truth[indices] = True
        for name, result in [('resnet_then_dino', routed), ('dino_only', dino_only)]:
            groups = {}
            for action in ('keep', 'quarantine', 'human_review', 'second_check'):
                mask = result['actions'] == action
                groups[action] = dict(total=int(mask.sum()), poisoned=int(np.sum(mask & truth)), clean=int(np.sum(mask & ~truth)))
            path = save_decisions(result, ROOT / 'artifacts/routing_results',
                                  source_description=f'{source}; rate={rate}; policy={name}')
            reports.append(dict(rate=rate, policy=name, groups=groups,
                                second_check_requested=int(result['second_check_requested'].sum()),
                                manifest=str(path)))
    report = dict(runs=reports,
                  limitation='Same development images and tuned feature-comparison cutoffs. Full DINO scores already existed: routing savings have NOT been measured. ResNet kNN is inactive at its calibrated cutoff in this sample.')
    out = ROOT / 'artifacts/routing_results/comparison.json'
    out.write_text(json.dumps(to_jsonable(report), indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

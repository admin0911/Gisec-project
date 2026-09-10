"""Offline geometry benchmark. NOT images, trained backdoors, or calibration."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from poison_features import DetectorInput
from detectors.backdoor.feature_pipeline import scan_backdoor_features
from experiments.scan_backdoor_features import detection_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/backdoor-feature-smoke.json"))
    args = parser.parse_args()
    report = dict(scope="synthetic feature geometry only; no model training, ASR, or post-clean quality",
                  rows=1000, dimensions=16, shift=8.0, noise_std=0.25,
                  thresholds="fixed detector defaults; not tuned by this benchmark", runs=[])
    for seed in (11, 23, 37):
        for rate in (0, .01, .03, .05, .07, .10):
            rng = np.random.default_rng(seed)
            X = rng.normal(0, .25, (1000, 16))
            y = np.arange(1000) % 2
            truth = np.zeros(1000, bool)
            rows = rng.choice(np.flatnonzero(y == 0), size=round(1000 * rate), replace=False)
            X[rows, 0] += 8
            truth[rows] = True
            start = perf_counter()
            result = scan_backdoor_features(DetectorInput(X, np.arange(1000), y))
            report["runs"].append(dict(
                seed=seed, fraction_of_all_rows=rate, fraction_of_target_class=2 * rate,
                runtime_seconds=perf_counter() - start,
                detectors={name: detection_metrics(item["flags"], truth)
                           for name, item in result["detectors"].items()},
            ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {len(report['runs'])} synthetic feature experiments to {args.output}")


if __name__ == "__main__":
    main()

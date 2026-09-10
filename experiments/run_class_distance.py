"""Evaluate class distance on a saved feature bundle.
Example: python -m experiments.run_class_distance artifacts/NAME-features.npz
Optional --knn-results artifacts/knn_results/RUN compares aligned saved kNN flags.
--known-clean is an explicit evaluation assumption, not a detector input.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from poison_features import FeatureBundle, detector_input
from detectors.label_flip.class_distance import ClassDistance
from experiments.run_knn import evaluation


def run(path, threshold=0.1, known_clean=False, knn_results=None):
    path = Path(path)
    bundle = FeatureBundle.load(path)
    if known_clean and bundle.is_poisoned is not None and np.any(bundle.is_poisoned):
        raise ValueError("--known-clean conflicts with poison identities")
    start = time.perf_counter()
    result = ClassDistance(threshold).analyze(
        detector_input(bundle, representation="raw", label_aware=True))
    elapsed = time.perf_counter() - start
    rules = {"class_distance": result["flags"]}
    if knn_results is not None:
        directory = Path(knn_results)
        summary = json.loads((directory / "summary.json").read_text())
        if Path(summary["source"]).resolve() != path.resolve():
            raise ValueError("kNN results must come from this same feature file")
        with np.load(directory / "scores.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["sample_ids"], bundle.sample_ids)
            neighbours = saved["neighbour_indices"]
            if neighbours.shape != (len(bundle.labels), 20):
                raise ValueError("Expected saved k=20 neighbours")
            if np.any(neighbours < 0) or np.any(neighbours >= len(bundle.labels)):
                raise ValueError("Invalid neighbour indices")
            recomputed = np.mean(bundle.labels[neighbours] != bundle.labels[:, None], axis=1)
            np.testing.assert_allclose(recomputed, saved["scores"])
            knn = saved["flags"].copy()
            np.testing.assert_array_equal(knn, recomputed >= summary["threshold"])
        rules["knn"] = knn
        rules["both_detectors"] = knn & result["flags"]
        rules["either_detector"] = knn | result["flags"]
    # Evaluation metadata is kept out of every detector and combination rule.
    truth = np.zeros(len(bundle.labels), dtype=bool) if known_clean else bundle.is_poisoned
    comparisons = {}
    for name, flags in rules.items():
        metrics = evaluation(flags, truth) if truth is not None else None
        comparisons[name] = dict(flagged=int(flags.sum()), evaluation=metrics)
    report = dict(source=str(path.resolve()), dataset=bundle.dataset_name,
                  samples=len(bundle.labels), threshold=threshold, representation="raw",
                  score="max(0, own_cosine_distance - alternative_cosine_distance) / 2",
                  threshold_status="provisional, not calibrated",
                  detector_seconds=elapsed,
                  evaluation_basis="user-declared clean" if known_clean else "saved poison identities" if truth is not None else "unavailable",
                  results=comparisons)
    output = Path("artifacts/class_distance_results") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "scores.npz", **result)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {output.resolve()}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--known-clean", action="store_true")
    parser.add_argument("--knn-results", type=Path)
    args = parser.parse_args()
    run(args.features, args.threshold, args.known_clean, args.knn_results)


if __name__ == "__main__":
    main()

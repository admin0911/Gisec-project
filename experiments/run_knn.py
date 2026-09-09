"""Run kNN on a locally generated FeatureBundle; no extraction or training.

From the repository root:
python -m experiments.run_knn artifacts/NAME-features.npz
Use --known-clean only when you independently know the input is clean.
Default raw means encoder embeddings. Threshold 0.95 is provisional.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from poison_features import FeatureBundle, detector_input
from detectors.label_flip.knn_label_agreement import KNNLabelAgreement


def evaluation(flags, truth):
    flags = np.asarray(flags, dtype=bool)
    truth = np.asarray(truth)
    if truth.shape != flags.shape or not np.isin(truth, [0, 1]).all():
        raise ValueError("Poison identities must be an aligned boolean array")
    truth = truth.astype(bool)
    tp = int(np.sum(flags & truth))
    fp = int(np.sum(flags & ~truth))
    fn = int(np.sum(~flags & truth))
    tn = int(np.sum(~flags & ~truth))
    return dict(true_positives=tp, false_positives=fp,
                false_negatives=fn, true_negatives=tn,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                false_positive_rate=fp / (fp + tn) if fp + tn else None)


def run(path, *, known_clean=False, output_root=Path("artifacts/knn_results")):
    path = Path(path)
    print(f"Loading {path}", flush=True)
    bundle = FeatureBundle.load(path)
    if known_clean and bundle.is_poisoned is not None and np.any(bundle.is_poisoned):
        raise ValueError("--known-clean conflicts with saved poison identities")
    inputs = detector_input(bundle, representation="raw", label_aware=True)
    print(f"Comparing all {len(inputs.X):,} samples with k=20; full scans can take time.", flush=True)
    result = KNNLabelAgreement().analyze(inputs)
    flagged = int(result["flags"].sum())
    summary = dict(source=str(path.resolve()), samples=len(inputs.X),
                   dataset=bundle.dataset_name, encoder=bundle.encoder,
                   representation="raw", k=20, threshold=0.95,
                   threshold_status="provisional, not calibrated for this bundle",
                   flagged=flagged, flagged_fraction=flagged / len(inputs.X),
                   evaluation=None)
    # Ground truth is accessed for evaluation only AFTER detector scoring.
    truth = bundle.is_poisoned
    if known_clean:
        truth = np.zeros(len(inputs.X), dtype=bool)
    if truth is not None:
        summary["evaluation"] = evaluation(result["flags"], truth)
        summary["evaluation_basis"] = "user-declared clean" if known_clean else "saved poison identities"
    output = Path(output_root) / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "scores.npz", **result)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Flagged: {flagged:,}/{len(inputs.X):,} ({100 * flagged / len(inputs.X):.2f}%)")
    if summary["evaluation"] is None:
        print("Precision/recall unavailable: no known poison identities supplied.")
    else:
        print(json.dumps(summary["evaluation"], indent=2))
    print(f"Saved results: {output.resolve()}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path, help="Locally generated *-features.npz file")
    parser.add_argument("--known-clean", action="store_true")
    args = parser.parse_args()
    try:
        run(args.features, known_clean=args.known_clean)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Cannot run kNN: {exc}\n")


if __name__ == "__main__":
    main()

"""Scan a trusted local FeatureBundle; optionally evaluate AFTER scoring."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from poison_features import FeatureBundle, detector_input
from detectors.backdoor.feature_pipeline import scan_backdoor_features
from detectors.output_connector import to_jsonable


def detection_metrics(flags, truth):
    flags = np.asarray(flags)
    truth = np.asarray(truth)
    if (truth.shape != flags.shape or truth.ndim != 1 or truth.dtype.kind != "b"
            or flags.dtype.kind != "b"):
        raise ValueError("Evaluation requires aligned boolean flags and poison truth")
    tp = int(np.sum(flags & truth)); fp = int(np.sum(flags & ~truth))
    fn = int(np.sum(~flags & truth)); tn = int(np.sum(~flags & ~truth))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                false_positive_rate=fp / (fp + tn) if fp + tn else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path)
    parser.add_argument("--representation", choices=("raw", "scaled", "reduced"), default="raw")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluate", action="store_true", help="Use saved poison truth only after scoring")
    args = parser.parse_args()
    if args.output.resolve() == args.features.resolve():
        parser.error("Output must not overwrite the feature bundle")
    bundle = FeatureBundle.load(args.features)
    start = perf_counter()
    result = scan_backdoor_features(detector_input(
        bundle, representation=args.representation, label_aware=True,
    ))
    result["runtime_seconds"] = perf_counter() - start
    result["input"] = dict(encoder=bundle.encoder, dataset_name=bundle.dataset_name,
                           representation=args.representation, dimensions=bundle.original_feature_dim
                           if args.representation == "raw" else
                           int(getattr(bundle, args.representation + "_features").shape[1]))
    if args.evaluate:
        if bundle.is_poisoned is None:
            parser.error("Evaluation requested but the bundle has no poison truth")
        result["evaluation"] = {name: detection_metrics(item["flags"], bundle.is_poisoned)
                                for name, item in result["detectors"].items()}
        result["evaluation"]["candidates"] = detection_metrics(result["candidate_flags"], bundle.is_poisoned)
        result["evaluation"]["agreement"] = detection_metrics(result["agreement_flags"], bundle.is_poisoned)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_jsonable(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Scanned {len(bundle.sample_ids)} rows in {result['runtime_seconds']:.3f}s; "
          f"{int(result['candidate_flags'].sum())} review candidates. Saved {args.output}")


if __name__ == "__main__":
    main()

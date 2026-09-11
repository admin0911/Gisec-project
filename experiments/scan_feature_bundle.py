"""Scan an aligned NPZ feature matrix without dataset-specific routing."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from detectors.feature_pipeline import SUPPORTED_TRACKS, scan_feature_bundle
from detectors.output_connector import to_jsonable
from poison_features import load_external_feature_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path, help="NPZ containing features/X/embeddings")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features-key")
    parser.add_argument("--labels-key")
    parser.add_argument("--sample-ids-key")
    parser.add_argument("--dataset-name")
    parser.add_argument("--modality")
    parser.add_argument("--encoder")
    parser.add_argument("--representation", choices=("raw", "scaled", "reduced"), default="raw")
    parser.add_argument("--track", action="append", choices=SUPPORTED_TRACKS,
                        help="Repeat to select tracks; default runs both")
    parser.add_argument("--include-cleanlab", action="store_true",
                        help="Add the slower cross-validated label-quality detector")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.output.resolve() == args.features.resolve():
        parser.error("Output must not overwrite the input bundle")
    try:
        bundle = load_external_feature_bundle(
            args.features,
            features_key=args.features_key,
            labels_key=args.labels_key,
            sample_ids_key=args.sample_ids_key,
            dataset_name=args.dataset_name,
            modality=args.modality,
            encoder=args.encoder,
        )
        started = perf_counter()
        result = scan_feature_bundle(
            bundle,
            tracks=args.track or SUPPORTED_TRACKS,
            representation=args.representation,
            include_cleanlab=args.include_cleanlab,
            k=args.k,
            batch_size=args.batch_size,
            progress=print,
        )
        result["runtime_seconds"] = perf_counter() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(to_jsonable(result), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Cannot scan feature bundle: {exc}\n")
    counts = {
        name: int(track["candidate_flags"].sum())
        for name, track in result["tracks"].items()
    }
    print(f"Scanned {len(bundle.sample_ids):,} rows; candidates by track: {counts}")
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()

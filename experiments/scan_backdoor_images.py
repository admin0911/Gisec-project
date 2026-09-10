"""Run image backdoor scans and save an offline HTML report, JSON and CSV."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
import webbrowser

import numpy as np

from poison_features import FeatureBundle, ImageInputBundle, detector_input
from detectors.backdoor.image_pipeline import scan_backdoor_images
from detectors.output_connector import to_jsonable
from experiments.backdoor_report import report_html, index_html, write_sample_csv
from experiments.scan_backdoor_features import detection_metrics


def paired_images(features):
    features = Path(features)
    if not features.name.endswith("-features.npz"):
        raise ValueError("Use a saved file ending in -features.npz or provide --images explicitly")
    path = features.with_name(features.name[:-len("-features.npz")] + "-images.npz")
    if not path.is_file():
        raise FileNotFoundError(f"Missing matching image bundle: {path}. Use --images for a renamed file.")
    return path


def run_scan(image_file, *, feature_file=None, dataset=None, evaluate=False, known_clean=False):
    pixels = ImageInputBundle.load(image_file)
    bundle = FeatureBundle.load(feature_file) if feature_file else None
    if bundle is not None and bundle.modality != "image":
        raise ValueError("Image scanner requires an image feature bundle")
    if bundle is not None and dataset is not None and dataset != bundle.dataset_name:
        raise ValueError("Explicit dataset does not match feature bundle metadata")
    dataset = dataset or (bundle.dataset_name if bundle else None)
    inputs = detector_input(bundle, representation="raw", label_aware=True) if bundle else None
    start = perf_counter()
    result = scan_backdoor_images(pixels, dataset=dataset, features=inputs, progress=print)
    result["runtime_seconds"] = perf_counter() - start
    result["input"] = dict(feature_file=str(feature_file) if feature_file else None,
                           image_file=str(image_file), dataset=dataset,
                           encoder=bundle.encoder if bundle else None,
                           representation="raw" if bundle else "pixels_only")
    result["evaluation_note"] = "Evaluation disabled; poisoning status is unknown to this report."
    if evaluate or known_clean:
        # Truth is accessed only after the entire detection pipeline returns.
        truth = bundle.is_poisoned if bundle is not None else None
        if known_clean:
            if truth is not None and np.any(truth):
                raise ValueError("--known-clean contradicts saved poison truth")
            truth = np.zeros(len(pixels.sample_ids), dtype=bool)
        if truth is not None:
            result["evaluation"] = {name: detection_metrics(item["flags"], truth)
                                    for name, item in result["detectors"].items()}
            result["evaluation"]["candidates"] = detection_metrics(result["candidate_flags"], truth)
            result["evaluation_note"] = ("Evaluation uses the user's explicit clean-data assertion."
                                         if known_clean else "Evaluation uses saved experiment truth only after scoring.")
        else:
            result["evaluation_note"] = "No saved poison truth: recall, precision and false-positive rate are unavailable. Filename is not ground truth."
    return result, pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--folder", type=Path, help="Scan every feature file with its matching image bundle")
    mode.add_argument("--features", type=Path)
    parser.add_argument("--images", type=Path, help="Single image bundle or explicit companion to --features")
    parser.add_argument("--dataset", choices=("cifar10", "mnist"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/backdoor-reports"))
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--known-clean", action="store_true", help="Explicitly assert a single dataset is clean for evaluation")
    parser.add_argument("--open", action="store_true", help="Open the generated index in the default browser")
    args = parser.parse_args()
    if args.folder and (args.images or args.known_clean):
        parser.error("--images and --known-clean are only for single-dataset scans")
    if not (args.folder or args.features or args.images):
        parser.error("Specify --folder, --features or --images")
    if args.images and not args.features and not args.dataset:
        parser.error("Pixel-only scans require --dataset")
    if args.folder:
        feature_files = sorted(args.folder.glob("*-features.npz"))
        if not feature_files:
            parser.error("No *-features.npz files found. Run from the project folder after extraction.")
        try:
            jobs = [(path, paired_images(path)) for path in feature_files]
        except (ValueError, FileNotFoundError) as exc:
            parser.error(str(exc))
    else:
        try:
            jobs = [(args.features, args.images or paired_images(args.features))]
        except (ValueError, FileNotFoundError) as exc:
            parser.error(str(exc))
    # Append-only run folders preserve all historical evidence and source data.
    output = args.output / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    entries = []
    for index, (features, images) in enumerate(jobs):
        try:
            result, pixels = run_scan(images, feature_file=features, dataset=args.dataset,
                                      evaluate=args.evaluate, known_clean=args.known_clean)
        except (ValueError, TypeError, FileNotFoundError) as exc:
            parser.error(f"{features or images}: {exc}")
        folder = output / f"scan-{index + 1:03d}"
        folder.mkdir(parents=True)
        (folder / "results.json").write_text(json.dumps(to_jsonable(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        write_sample_csv(folder / "samples.csv", result)
        (folder / "report.html").write_text(report_html(result, pixels), encoding="utf-8")
        flagged = int(np.sum(result["candidate_flags"]))
        entries.append(dict(name=(features or images).name, link=f"scan-{index + 1:03d}/report.html",
                            rows=len(pixels.sample_ids), flagged=flagged))
        (output / "index.html").write_text(index_html(entries), encoding="utf-8")
        print(f"{(features or images).name}: {flagged}/{len(pixels.sample_ids)} review candidates")
    index = (output / "index.html").resolve()
    print(f"Open report: {index}")
    if args.open:
        webbrowser.open(index.as_uri())


if __name__ == "__main__":
    main()

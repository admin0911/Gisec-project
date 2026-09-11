"""Fixed-profile stress tests on a separate CIFAR-10 test subset.

This evaluates pixel detection only. It does not train models or measure ASR.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from time import perf_counter
import webbrowser

import numpy as np

from poison_features import ImageInputBundle, load_image_dataset, load_image_inputs
from detectors.backdoor.image_pipeline import scan_backdoor_images
from detectors.backdoor.web_patch import CIFAR_PATCH_SETTINGS
from experiments.backdoor_report import page, rate
from experiments.scan_backdoor_features import detection_metrics


COLOURS = {"white": (1., 1., 1.), "red": (1., 0., 0.), "black": (0., 0., 0.)}
POSITIONS = ("bottom_right", "centre")
RATES = (.01, .05, .10)


def injected_pixels(clean, *, fraction, colour, position, seed, target=0):
    if clean.images.shape[1] != 3 or min(clean.images.shape[2:]) < 3:
        raise ValueError("Need RGB images at least 3x3")
    if colour not in COLOURS or position not in POSITIONS:
        raise ValueError("Unsupported colour or position")
    if not np.isfinite(fraction) or not 0 <= fraction <= .1:
        raise ValueError("Fraction must be between 0 and 0.1")
    h, w = clean.images.shape[2:]
    row, col = (h - 3, w - 3) if position == "bottom_right" else ((h - 3) // 2, (w - 3) // 2)
    # Same shuffled ordering across rates gives nested selections, independent of colour/position.
    selected = np.random.default_rng(seed).permutation(len(clean.images))[:round(len(clean.images) * fraction)]
    truth = np.zeros(len(clean.images), dtype=bool); truth[selected] = True
    images = clean.images.copy(); labels = clean.labels.copy()
    images[selected, :, row:row+3, col:col+3] = np.array(COLOURS[colour], dtype=np.float32)[None, :, None, None]
    labels[selected] = target
    return ImageInputBundle(images, labels, clean.sample_ids.copy()), truth


def scan_case(pixels, truth):
    start = perf_counter()
    result = scan_backdoor_images(pixels, dataset="cifar10")
    elapsed = perf_counter() - start
    flags = result["candidate_flags"]
    # Ground truth enters only here, after the detector has returned.
    return dict(metrics=detection_metrics(flags, truth), runtime_seconds=elapsed,
                flagged_ids=pixels.sample_ids[flags].tolist(),
                false_positive_ids=pixels.sample_ids[flags & ~truth].tolist(),
                missed_poison_ids=pixels.sample_ids[~flags & truth].tolist())


def summary_html(report):
    body = ("<h1>Patch detector validation</h1>"
            f"<p>{report['unique_base_images']:,} unique CIFAR-10 test images; {len(report['runs'])} completed conditions.</p>"
            "<p class='notice'>Same images reused across conditions: do not count these as independent datasets. "
            "Pixel detector only, with unchanged thresholds. No model training or ASR evaluation. "
            "Black triggers are an intentional test outside the bright-patch profile's advertised scope.</p>"
            "<p><a href='summary.json'>Full results JSON</a> · <a href='plan.json'>Frozen experiment plan</a></p>"
            "<div class='scroll'><table><tr><th>Colour</th><th>Position</th><th>Rate</th><th>Seed</th>"
            "<th>Caught</th><th>Missed</th><th>Clean flagged</th><th>Recall</th><th>Precision</th><th>Clean FPR</th></tr>")
    for run in report["runs"]:
        m = run["metrics"]
        values = [run["colour"], run["position"], rate(run["fraction"]), run["seed"],
                  f"{m['tp']}/{m['tp']+m['fn']}", m["fn"], m["fp"], rate(m["recall"]),
                  rate(m["precision"]), rate(m["false_positive_rate"])]
        body += "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in values) + "</tr>"
    return page("Patch validation", body + "</table></div><p>Keep misses and false positives in the report. Once examined, these images are development data for future changes; reserve a new subset for the next final evaluation.</p>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--download", action="store_true", help="Download CIFAR-10 if not cached")
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/patch-validation"))
    args = parser.parse_args()
    if args.samples < 100 or args.start < 0 or not args.seeds or any(s < 0 for s in args.seeds):
        parser.error("Use at least 100 samples, a nonnegative start and nonnegative seeds")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique")
    try:
        dataset = load_image_dataset("cifar10", root=args.data_root, train=False, download=args.download)
    except RuntimeError as exc:
        parser.error(f"{exc}. If the CIFAR-10 cache is missing, rerun with --download.")
    if args.start + args.samples > len(dataset):
        parser.error("Requested subset exceeds the CIFAR-10 test split")
    class SelectedRows:
        def __len__(self): return args.samples
        def __getitem__(self, i):
            if not 0 <= i < args.samples: raise IndexError(i)
            return dataset[args.start + i]
    ids = np.array([f"cifar10-test:{i}" for i in range(args.start, args.start + args.samples)])
    clean = load_image_inputs(SelectedRows(), sample_ids=ids)
    source_root = Path(__file__).resolve().parents[1]
    source_paths = ["detectors/backdoor/repeated_patch.py", "detectors/backdoor/web_patch.py",
                    "detectors/backdoor/image_pipeline.py", "experiments/validate_patch_scope.py"]
    report = dict(dataset="cifar10", split="test", start=args.start, unique_base_images=args.samples,
                  sample_ids=ids.tolist(), seeds=args.seeds, target=0, patch_size=3,
                  colours=COLOURS, positions=POSITIONS, fractions=RATES,
                  profile="cifar-bright-patch-v1", profile_overrides=CIFAR_PATCH_SETTINGS,
                  source_sha256={p: sha256((source_root / p).read_bytes()).hexdigest() for p in source_paths},
                  selection="All classes eligible, including samples already labelled target 0; nested rates",
                  scope="Pixel detection only; synthetic triggers on real held-out-split images. No threshold tuning or ASR.",
                  runs=[])
    output = args.output / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    def record(metadata, pixels, truth):
        run = dict(metadata, **scan_case(pixels, truth)); report["runs"].append(run)
        (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        (output / "index.html").write_text(summary_html(report), encoding="utf-8")
        m = run["metrics"]
        print(f"{len(report['runs'])}/{1 + len(args.seeds)*len(COLOURS)*len(POSITIONS)*len(RATES)} "
              f"{run['colour']} {run['position']} {run['fraction']:.0%} seed={run['seed']}: "
              f"caught {m['tp']}/{m['tp']+m['fn']}, clean flagged {m['fp']}", flush=True)
    record(dict(colour="clean", position="none", fraction=0., seed=None), clean, np.zeros(args.samples, bool))
    for seed in args.seeds:
        for colour in COLOURS:
            for position in POSITIONS:
                for fraction in RATES:
                    pixels, truth = injected_pixels(clean, fraction=fraction, colour=colour,
                                                    position=position, seed=seed)
                    record(dict(colour=colour, position=position, fraction=fraction, seed=seed), pixels, truth)
    index = (output / "index.html").resolve()
    print(f"Results: {index}")
    if args.open: webbrowser.open(index.as_uri())


if __name__ == "__main__":
    main()

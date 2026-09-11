"""Compare fixed legacy and contrast profiles on a reserved CIFAR test block."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from time import perf_counter
import webbrowser

import numpy as np
from poison_features import load_image_dataset, load_image_inputs
from detectors.backdoor.contrast_patch import ContrastPatchDetector
from detectors.backdoor.repeated_patch import RepeatedPatchDetector
from detectors.backdoor.web_patch import CIFAR_PATCH_SETTINGS
from experiments.validate_patch_scope import injected_pixels, COLOURS, POSITIONS, RATES
from experiments.backdoor_report import page, rate
from experiments.scan_backdoor_features import detection_metrics


def compare_case(pixels, truth):
    outputs = {}
    for name, detector in (("legacy", RepeatedPatchDetector(**CIFAR_PATCH_SETTINGS)),
                           ("contrast", ContrastPatchDetector())):
        start = perf_counter(); result = detector.analyze(pixels)
        elapsed = perf_counter() - start
        # Only public pixels/current labels/IDs enter either detector.
        flags = result["flags"]
        outputs[name] = dict(metrics=detection_metrics(flags, truth), runtime_seconds=elapsed,
                             false_positive_ids=pixels.sample_ids[flags & ~truth].tolist(),
                             missed_poison_ids=pixels.sample_ids[~flags & truth].tolist(),
                             diagnostic_counts=result["evidence"].get("diagnostic_counts", {}))
    return outputs


def comparison_html(report):
    body = ("<h1>Patch detector comparison</h1>"
            f"<p>{report['samples']:,} unique CIFAR-10 test images, starting at row {report['start']}. "
            f"{len(report['runs'])} conditions completed.</p>"
            "<p class='notice'>The same images are reused across conditions. These are fixed patch-injection tests, "
            "not model ASR or general backdoor guarantees. Both profiles were fixed before this run.</p>"
            "<p><a href='summary.json'>Full JSON</a> · <a href='plan.json'>Frozen plan</a></p>"
            "<div class='scroll'><table><tr><th>Condition</th><th>Rate</th><th>Seed</th><th>Profile</th>"
            "<th>Caught</th><th>Missed</th><th>Clean flagged</th><th>Recall</th><th>Precision</th></tr>")
    for run in report["runs"]:
        for name, result in run["detectors"].items():
            m = result["metrics"]
            values = [f"{run['colour']} / {run['position']}", rate(run["fraction"]), run["seed"], name,
                      f"{m['tp']}/{m['tp']+m['fn']}", m["fn"], m["fp"], rate(m["recall"]), rate(m["precision"])]
            body += "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in values) + "</tr>"
    return page("Patch profile comparison", body + "</table></div><p>Retain all misses and false positives. Reserve a fresh subset before evaluating further revisions.</p>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/patch-fix-validation"))
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if args.start < 0 or args.samples < 100 or len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error("Use nonnegative start, at least 100 images and distinct nonnegative seeds")
    if args.start < 2000:
        parser.error("Rows 0-1999 were inspected during development; select --start 2000 or later")
    try:
        dataset = load_image_dataset("cifar10", root=args.data_root, train=False, download=args.download)
    except RuntimeError as exc:
        parser.error(f"{exc}. Add --download if the cache is missing.")
    if args.start + args.samples > len(dataset): parser.error("Requested block exceeds test split")
    class Block:
        def __len__(self): return args.samples
        def __getitem__(self, i):
            if not 0 <= i < args.samples: raise IndexError(i)
            return dataset[args.start + i]
    ids = np.array([f"cifar10-test:{i}" for i in range(args.start, args.start+args.samples)])
    clean = load_image_inputs(Block(), sample_ids=ids)
    root = Path(__file__).resolve().parents[1]
    paths = ["detectors/backdoor/contrast_patch.py", "detectors/backdoor/repeated_patch.py",
             "detectors/backdoor/web_patch.py", "experiments/validate_patch_scope.py",
             "experiments/validate_patch_fix.py"]
    report = dict(dataset="cifar10", split="test", start=args.start, samples=args.samples,
                  unique_base_images=args.samples, sample_ids=ids.tolist(), seeds=args.seeds,
                  legacy_overrides=CIFAR_PATCH_SETTINGS, contrast_settings=ContrastPatchDetector().settings,
                  source_sha256={name: sha256((root/name).read_bytes()).hexdigest() for name in paths},
                  target=0, patch_size=3, colours=COLOURS, positions=POSITIONS, fractions=RATES,
                  scope="Fixed pixel detectors; repeated conditions on one reserved block. No classifier training or ASR.",
                  runs=[])
    output = args.output / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True)
    (output/"plan.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    def record(metadata, pixels, truth):
        report["runs"].append(dict(metadata, detectors=compare_case(pixels, truth)))
        (output/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        (output/"index.html").write_text(comparison_html(report), encoding="utf-8")
        latest = report["runs"][-1]
        metrics = latest["detectors"]["contrast"]["metrics"]
        print(f"{len(report['runs'])}/{1+len(args.seeds)*18}: {latest['colour']} {latest['position']} "
              f"{latest['fraction']:.0%}; revised caught {metrics['tp']}/{metrics['tp']+metrics['fn']}, "
              f"clean flagged {metrics['fp']}", flush=True)
    record(dict(colour="clean", position="none", fraction=0., seed=None), clean, np.zeros(args.samples, bool))
    for seed in args.seeds:
        for colour in COLOURS:
            for position in POSITIONS:
                for fraction in RATES:
                    pixels, truth = injected_pixels(clean, fraction=fraction, colour=colour, position=position, seed=seed)
                    record(dict(colour=colour, position=position, fraction=fraction, seed=seed), pixels, truth)
    print(f"Report: {(output/'index.html').resolve()}")
    if args.open: webbrowser.open((output/"index.html").resolve().as_uri())


if __name__ == "__main__": main()

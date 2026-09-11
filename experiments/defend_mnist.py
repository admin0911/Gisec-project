"""Run the complete 60,000-image MNIST patch-defence benchmark."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import webbrowser

import numpy as np
import torch

from cleaning import decide_backdoor_actions, save_decisions
from detectors.backdoor.contrast_patch import ContrastPatchDetector
from detectors.output_connector import to_jsonable
from experiments.compare_training import compare_training
from experiments.defend_backdoor import defence_effect, removal_metrics, report_html
from poison_features import load_image_dataset, load_image_inputs
from poison_features.attacks import poison_dataset
from training import TrainConfig, training_input
from training.backdoor_evaluation import TriggeredDataset, evaluate_backdoor_checkpoint
from training.models import small_cnn


def scan_pixels(pixels, progress=None):
    detector = ContrastPatchDetector()
    patch = detector.analyze(pixels, progress=progress)
    flags = patch["flags"].copy()
    return {
        "sample_ids": pixels.sample_ids.copy(),
        "detectors": {"contrast_patch": patch},
        "candidate_flags": flags,
        "flag_count": flags.astype(np.int64),
        "settings": {
            "candidate_rule": "contrast_patch flag requests quarantine",
            "active_detectors": ["contrast_patch"],
            "comparison_only": [],
            "patch_profile": "contrast",
            "features_used": False,
            "limitation": "Fixed exact 2x2/3x3 patch scan; zero flags is not proof of clean data.",
        },
    }


def run_full_mnist_defence(*, data_root="data", output_root="artifacts/mnist-backdoor-defence",
                           poison_rate=.05, attack_seed=0, train_seed=42, target_label=0,
                           epochs=3, batch_size=256, threads=4, device="cpu",
                           download=False, progress=print):
    if not 0 < poison_rate <= .10:
        raise ValueError("poison_rate must be greater than 0 and at most 0.10")
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be positive")
    if type(target_label) is not int or not 0 <= target_label < 10:
        raise ValueError("target_label must be from 0 to 9")
    torch.set_num_threads(threads)

    if progress:
        progress("1/6 Loading all 60,000 MNIST training images")
    clean_train = load_image_dataset("mnist", root=data_root, train=True, download=download)
    if len(clean_train) != 60000:
        raise ValueError("Expected the complete 60,000-image MNIST training split")
    ids = np.array([f"mnist-train:{index}" for index in range(len(clean_train))])
    poisoned_train = poison_dataset(
        clean_train, "backdoor", poison_rate=poison_rate,
        target_label=target_label, seed=attack_seed,
    )
    pixels = load_image_inputs(poisoned_train, sample_ids=ids)

    if progress:
        progress("2/6 Scanning pixels without poison truth or trigger settings")
    def scan_progress(done, total):
        if progress:
            progress(f"Pixel scan {done}/{total} locations")
    scan = scan_pixels(pixels, progress=scan_progress)
    decisions = decide_backdoor_actions(scan)
    del pixels

    # Truth is accessed only after the complete scan and defence decision exist.
    truth = poisoned_train.metadata.is_poisoned.copy()
    selection = removal_metrics(decisions, truth)
    output = Path(output_root) / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    (output / "scan.json").write_text(
        json.dumps(to_jsonable(scan), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    decision_folder = save_decisions(
        decisions, output / "decisions",
        source_description=f"MNIST train; backdoor rate={poison_rate}; seed={attack_seed}",
    )
    np.savez_compressed(
        output / "evaluation-truth.npz", sample_ids=ids, is_poisoned=truth,
        original_labels=poisoned_train.metadata.original_labels,
    )

    if progress:
        progress("3/6 Loading all 10,000 untouched MNIST test images")
    clean_test_data = load_image_dataset("mnist", root=data_root, train=False, download=download)
    if len(clean_test_data) != 10000:
        raise ValueError("Expected the complete 10,000-image MNIST test split")
    test_ids = np.array([f"mnist-test:{index}" for index in range(len(clean_test_data))])
    clean_test = training_input(
        clean_test_data, test_ids, dataset_version="mnist-official-test", split="test"
    )
    triggered_test = TriggeredDataset(
        clean_test_data, patch_size=3, position="bottom_right", colour="white"
    )

    if progress:
        progress("4/6 Training clean-reference, poisoned and defended models")
    config = TrainConfig(
        epochs=epochs, batch_size=batch_size, seed=train_seed, device=device
    )
    training_report = compare_training(
        clean_train, poisoned_train, ids, decisions, clean_test,
        model_factory=lambda: small_cnn(channels=1, num_classes=10),
        model_name="small_cnn_mnist_v1", num_classes=10,
        output_root=output / "training", dataset_version="mnist-full-backdoor-defence",
        config=config, progress=progress,
    )

    if progress:
        progress("5/6 Measuring clean accuracy and triggered attack success")
    backdoor = {}
    for name, run in training_report["runs"].items():
        backdoor[name] = evaluate_backdoor_checkpoint(
            run, test_ids, triggered_test,
            model_factory=lambda: small_cnn(channels=1, num_classes=10),
            target_label=target_label, output=output / f"{name}-triggered-predictions.npz",
            num_classes=10, batch_size=batch_size, device=device,
        )
    effect = defence_effect(backdoor)
    report = {
        "schema_version": "1.0",
        "status": "complete",
        "dataset": "mnist",
        "scope": {"training_images": 60000, "test_images": 10000},
        "attack": {
            "type": "backdoor", "poison_rate": poison_rate,
            "poison_count": int(truth.sum()), "attack_seed": attack_seed,
            "target_label": target_label, "patch_size": 3,
            "position": "bottom_right", "colour": "white",
        },
        "separation": (
            "Attack settings and poison truth are generated by the benchmark but are not "
            "passed to scanning or defence selection."
        ),
        "decision_manifest": str((decision_folder / "decisions.json").resolve()),
        "decision_summary": decisions["summary"],
        "selection_evaluation": selection,
        "evaluation_basis": "saved benchmark truth accessed only after the defence decision",
        "training": training_report,
        "backdoor_evaluation": backdoor,
        "defence_effect": effect,
        "trigger_evaluation": {
            "target_label": target_label, "patch_size": 3,
            "position": "bottom_right", "colour": "white",
            "basis": "controlled test setting; never supplied to scanning or selection",
        },
        "limitation": (
            "Complete MNIST train/test splits, one fixed patch attack, one model architecture, "
            "one attack seed and one training seed. Other trigger families require separate tests."
        ),
    }
    json_path, html_path = output / "defence.json", output / "defence.html"
    json_path.write_text(
        json.dumps(to_jsonable(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    html_path.write_text(report_html(report), encoding="utf-8")
    if progress:
        progress(f"6/6 Saved complete MNIST report: {html_path.resolve()}")
    return report, html_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="artifacts/mnist-backdoor-defence")
    parser.add_argument("--poison-rate", type=float, default=.05)
    parser.add_argument("--attack-seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    try:
        _, path = run_full_mnist_defence(
            data_root=args.data_root, output_root=args.output,
            poison_rate=args.poison_rate, attack_seed=args.attack_seed,
            train_seed=args.train_seed, target_label=args.target_label,
            epochs=args.epochs, batch_size=args.batch_size, threads=args.threads,
            device=args.device, download=args.download,
        )
    except (ValueError, TypeError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
    if args.open:
        webbrowser.open(path.resolve().as_uri())


if __name__ == "__main__":
    main()

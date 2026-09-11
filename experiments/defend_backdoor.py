"""Quarantine detected patch candidates, retrain fresh models, and measure ASR."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import html
import json
from pathlib import Path
import webbrowser

import numpy as np
import torch
from torch.utils.data import TensorDataset

from cleaning import decide_backdoor_actions, save_decisions
from detectors.output_connector import to_jsonable
from experiments.compare_training import compare_training
from experiments.scan_backdoor_images import paired_images, run_scan
from poison_features import FeatureBundle, ImageInputBundle, load_image_dataset
from training import TrainConfig, training_input
from training.backdoor_evaluation import TriggeredDataset, evaluate_backdoor_checkpoint
from training.models import small_cnn


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_dataset(bundle):
    return TensorDataset(
        torch.from_numpy(bundle.images).float(),
        torch.from_numpy(bundle.labels.astype(np.int64)),
    )


def validate_matched_bundles(clean, poisoned, features):
    if not (
        np.array_equal(clean.sample_ids, poisoned.sample_ids)
        and np.array_equal(poisoned.sample_ids, features.sample_ids)
        and np.array_equal(poisoned.labels, features.labels)
        and clean.images.shape == poisoned.images.shape
    ):
        raise ValueError("Clean, poisoned and feature bundles must contain the same aligned rows")
    if clean.labels.dtype.kind not in "iu" or poisoned.labels.dtype.kind not in "iu":
        raise ValueError("Training labels must be integer class indices")


def removal_metrics(decisions, truth):
    poison = np.asarray(truth)
    actions = np.asarray(decisions["actions"])
    if poison.shape != actions.shape or poison.dtype.kind != "b":
        raise ValueError("Saved poison truth must align with defence decisions")
    removed = actions != "keep"
    quarantined = actions == "quarantine"
    review = actions == "human_review"
    poisoned = int(poison.sum())
    clean = int((~poison).sum())
    poison_removed = int((poison & removed).sum())
    clean_removed = int((~poison & removed).sum())
    return {
        "known_poisoned": poisoned,
        "known_clean": clean,
        "removed_from_training": int(removed.sum()),
        "quarantined": int(quarantined.sum()),
        "held_for_review": int(review.sum()),
        "poisoned_removed": poison_removed,
        "poisoned_retained": int((poison & ~removed).sum()),
        "clean_removed": clean_removed,
        "clean_retained": int((~poison & ~removed).sum()),
        "removal_precision": poison_removed / int(removed.sum()) if removed.any() else None,
        "poison_removal_recall": poison_removed / poisoned if poisoned else None,
        "clean_retention": int((~poison & ~removed).sum()) / clean if clean else None,
    }


def defence_effect(backdoor):
    before = backdoor["poisoned_baseline"]["metrics"]
    after = backdoor["after_cleaning"]["metrics"]
    reference = backdoor["clean_baseline"]["metrics"]
    asr_reduction = before["attack_success_rate"] - after["attack_success_rate"]
    conditional_reduction = (
        None if before["conditional_attack_success_rate"] is None
        or after["conditional_attack_success_rate"] is None
        else before["conditional_attack_success_rate"] - after["conditional_attack_success_rate"]
    )
    return {
        "asr_before_defence": before["attack_success_rate"],
        "asr_after_defence": after["attack_success_rate"],
        "absolute_asr_reduction": asr_reduction,
        "relative_asr_reduction": (
            asr_reduction / before["attack_success_rate"]
            if before["attack_success_rate"] else None
        ),
        "backdoor_lift_before_defence": before["attack_success_lift"],
        "backdoor_lift_after_defence": after["attack_success_lift"],
        "conditional_asr_reduction": conditional_reduction,
        "clean_accuracy_change_vs_poisoned": (
            after["clean_accuracy"] - before["clean_accuracy"]
        ),
        "clean_accuracy_gap_vs_reference": (
            after["clean_accuracy"] - reference["clean_accuracy"]
        ),
    }


def report_html(report):
    selection = report.get("selection_evaluation") or {}
    effect = report["defence_effect"]
    rows = []
    labels = {
        "clean_baseline": "Clean reference",
        "poisoned_baseline": "Before defence",
        "after_cleaning": "After defence",
    }
    def percent(value):
        return "N/A" if value is None else f"{value:.2%}"
    for name in ("clean_baseline", "poisoned_baseline", "after_cleaning"):
        run = report["training"]["runs"][name]
        metrics = report["backdoor_evaluation"][name]["metrics"]
        rows.append(
            "<tr>"
            f"<td>{labels[name]}</td><td>{run['training_samples']:,}</td>"
            f"<td>{metrics['clean_accuracy']:.2%}</td>"
            f"<td>{metrics['attack_success_rate']:.2%}</td>"
            f"<td>{percent(metrics['conditional_attack_success_rate'])}</td>"
            "</tr>"
        )
    value = lambda key: "N/A" if selection.get(key) is None else (
        f"{selection[key]:.2%}" if isinstance(selection[key], float) else f"{selection[key]:,}"
    )
    limitation = html.escape(report["limitation"])
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Backdoor defence comparison</title>
<style>
body{{font:16px system-ui;background:#0f141d;color:#e7eefc;max-width:1050px;margin:40px auto;padding:0 20px}}
h1{{font-size:42px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}
.card,table{{background:#171f2c;border:1px solid #35435b;border-radius:12px}}.card{{padding:18px}}
.card b{{display:block;font-size:30px;color:#69e6c1}}table{{width:100%;border-collapse:collapse;margin:24px 0}}
th,td{{padding:13px;text-align:left;border-bottom:1px solid #35435b}}th{{color:#a9c7ff}}
.note{{color:#b9c6da;line-height:1.55}}@media(max-width:700px){{.cards{{grid-template-columns:1fr}}}}
</style><body><h1>Backdoor defence comparison</h1>
<div class="cards"><div class="card">Poison removal recall<b>{value('poison_removal_recall')}</b></div>
<div class="card">Removal precision<b>{value('removal_precision')}</b></div>
<div class="card">ASR reduction<b>{percent(effect['absolute_asr_reduction'])}</b></div></div>
<table><thead><tr><th>Training arm</th><th>Rows</th><th>Clean accuracy</th><th>Triggered ASR</th><th>Conditional ASR</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p class="note">ASR excludes test images whose true class is already the target. Conditional ASR uses only eligible images the same model classified correctly before adding the trigger.</p>
<p class="note">{limitation}</p></body></html>"""


def run_defence(*, feature_file, clean_image_file, image_file=None, data_root="data",
                output_root="artifacts/backdoor-defence", epochs=10, batch_size=128,
                train_seed=42, device="cpu", threads=4, download=False,
                target_label=None, patch_size=3, position="bottom_right", colour="white",
                progress=print):
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    feature_file = Path(feature_file)
    image_file = Path(image_file) if image_file else paired_images(feature_file)
    clean_image_file = Path(clean_image_file)
    features = FeatureBundle.load(feature_file)
    poisoned = ImageInputBundle.load(image_file)
    clean = ImageInputBundle.load(clean_image_file)
    if features.modality != "image" or features.dataset_name not in ("cifar10", "mnist"):
        raise ValueError("Backdoor defence currently supports saved CIFAR-10 or MNIST image bundles")
    validate_matched_bundles(clean, poisoned, features)

    if progress:
        progress("1/5 Scanning without poison truth or attack settings")
    scan, _ = run_scan(image_file, feature_file=feature_file, dataset=features.dataset_name,
                       evaluate=False, patch_profile="contrast")
    decisions = decide_backdoor_actions(scan)

    # Evaluation metadata is first accessed after detection and the defence decision.
    truth = features.is_poisoned
    selection = removal_metrics(decisions, truth) if truth is not None else None
    metadata = features.metadata or {}
    target_label = int(metadata.get("target_label", 0)) if target_label is None else target_label
    if type(target_label) is not int or not 0 <= target_label < 10:
        raise ValueError("target_label must be an integer class index from 0 to 9")
    if features.original_labels is not None and not np.array_equal(clean.labels, features.original_labels):
        raise ValueError("Clean reference labels do not match saved original labels")

    output = Path(output_root) / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    (output / "scan.json").write_text(
        json.dumps(to_jsonable(scan), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    decision_folder = save_decisions(
        decisions, output / "decisions", source_description=str(image_file)
    )

    if progress:
        progress("2/5 Loading the separate official test split")
    test_data = load_image_dataset(
        features.dataset_name, root=data_root, train=False, download=download
    )
    test_ids = np.array(
        [f"{features.dataset_name}-test:{index}" for index in range(len(test_data))]
    )
    clean_test = training_input(
        test_data, test_ids,
        dataset_version=f"{features.dataset_name}-official-test", split="test"
    )
    triggered_test = TriggeredDataset(
        test_data, patch_size=patch_size, position=position, colour=colour
    )

    if progress:
        progress("3/5 Training clean-reference, poisoned and defended models")
    config = TrainConfig(
        epochs=epochs, batch_size=batch_size, seed=train_seed, device=device
    )
    training_report = compare_training(
        tensor_dataset(clean), tensor_dataset(poisoned), poisoned.sample_ids, decisions, clean_test,
        model_factory=lambda: small_cnn(channels=poisoned.images.shape[1], num_classes=10),
        model_name="small_cnn_v1", num_classes=10, output_root=output / "training",
        dataset_version=f"{features.dataset_name}-backdoor-defence",
        config=config, progress=progress,
    )

    if progress:
        progress("4/5 Measuring clean accuracy and triggered attack success")
    backdoor = {}
    for name, run in training_report["runs"].items():
        backdoor[name] = evaluate_backdoor_checkpoint(
            run, test_ids, triggered_test,
            model_factory=lambda: small_cnn(channels=poisoned.images.shape[1], num_classes=10),
            target_label=target_label, output=output / f"{name}-triggered-predictions.npz",
            num_classes=10, batch_size=batch_size, device=device,
        )
    effect = defence_effect(backdoor)

    report = {
        "schema_version": "1.0",
        "status": "complete",
        "dataset": features.dataset_name,
        "inputs": {
            "features": str(feature_file),
            "poisoned_images": str(image_file),
            "clean_images": str(clean_image_file),
            "sha256": {
                "features": file_sha256(feature_file),
                "poisoned_images": file_sha256(image_file),
                "clean_images": file_sha256(clean_image_file),
            },
        },
        "decision_manifest": str((decision_folder / "decisions.json").resolve()),
        "decision_summary": decisions["summary"],
        "selection_evaluation": selection,
        "evaluation_basis": (
            "saved poison identities accessed only after the defence decision"
            if truth is not None else "unavailable"
        ),
        "training": training_report,
        "backdoor_evaluation": backdoor,
        "defence_effect": effect,
        "trigger_evaluation": {
            "target_label": target_label,
            "patch_size": patch_size,
            "position": position,
            "colour": colour,
            "basis": "controlled evaluation setting; never supplied to scanning or selection",
        },
        "limitation": (
            "Controlled exact square-patch benchmark with one model architecture and training seed. "
            "It does not guarantee defence against adaptive, blended, moving, semantic or text triggers."
        ),
    }
    json_path = output / "defence.json"
    html_path = output / "defence.html"
    json_path.write_text(
        json.dumps(to_jsonable(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    html_path.write_text(report_html(report), encoding="utf-8")
    if progress:
        progress(f"5/5 Saved defence report: {html_path.resolve()}")
    return report, html_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="Poisoned *-features.npz")
    parser.add_argument("--images", type=Path, help="Matching poisoned image bundle")
    parser.add_argument("--clean-images", type=Path, required=True, help="Matched pre-attack image bundle")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", default="artifacts/backdoor-defence")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--download", action="store_true", help="Download the official test set if absent")
    parser.add_argument("--target-label", type=int)
    parser.add_argument("--patch-size", type=int, default=3)
    parser.add_argument("--position", choices=("bottom_right", "centre", "top_left", "top_right", "bottom_left"), default="bottom_right")
    parser.add_argument("--colour", choices=("white", "red", "black"), default="white")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        _, path = run_defence(
            feature_file=args.features, image_file=args.images,
            clean_image_file=args.clean_images, data_root=args.data_root,
            output_root=args.output, epochs=args.epochs, batch_size=args.batch_size,
            train_seed=args.train_seed, device=args.device, threads=args.threads,
            download=args.download, target_label=args.target_label,
            patch_size=args.patch_size, position=args.position, colour=args.colour,
        )
    except (ValueError, TypeError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
    if args.open:
        webbrowser.open(path.resolve().as_uri())


if __name__ == "__main__":
    main()

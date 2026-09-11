"""Backdoor trigger evaluation for locally trained image classifiers."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


COLOURS = {
    "white": (1.0, 1.0, 1.0),
    "red": (1.0, 0.0, 0.0),
    "black": (0.0, 0.0, 0.0),
}
POSITIONS = ("bottom_right", "centre", "top_left", "top_right", "bottom_left")


def apply_square_trigger(image, *, patch_size=3, position="bottom_right", colour="white"):
    """Return a triggered tensor without modifying the source image."""
    if not torch.is_tensor(image) or image.ndim != 3 or not torch.isfinite(image).all():
        raise ValueError("Expected one finite CHW tensor image")
    if image.shape[0] not in (1, 3) or image.min() < 0 or image.max() > 1:
        raise ValueError("Expected one- or three-channel image pixels in [0, 1]")
    if type(patch_size) is not int or patch_size < 1 or patch_size > min(image.shape[-2:]):
        raise ValueError("patch_size must fit within the image")
    if position not in POSITIONS:
        raise ValueError(f"position must be one of {', '.join(POSITIONS)}")
    if colour not in COLOURS:
        raise ValueError(f"colour must be one of {', '.join(COLOURS)}")

    height, width = image.shape[-2:]
    if position == "bottom_right":
        row, column = height - patch_size, width - patch_size
    elif position == "centre":
        row, column = (height - patch_size) // 2, (width - patch_size) // 2
    elif position == "top_left":
        row, column = 0, 0
    elif position == "top_right":
        row, column = 0, width - patch_size
    else:
        row, column = height - patch_size, 0

    values = COLOURS[colour][: image.shape[0]]
    if image.shape[0] == 1:
        values = (float(np.mean(COLOURS[colour])),)
    triggered = image.clone()
    fill = torch.tensor(values, dtype=triggered.dtype, device=triggered.device)[:, None, None]
    triggered[:, row : row + patch_size, column : column + patch_size] = fill
    return triggered


class TriggeredDataset:
    """Read-only view applying the evaluation trigger to every image."""

    def __init__(self, dataset, *, patch_size=3, position="bottom_right", colour="white"):
        if len(dataset) < 1:
            raise ValueError("Evaluation dataset must not be empty")
        self.dataset = dataset
        self.settings = {
            "patch_size": patch_size,
            "position": position,
            "colour": colour,
        }
        # Validate settings and the first row before a model is loaded.
        image, _ = dataset[0]
        apply_square_trigger(image, **self.settings)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return apply_square_trigger(image, **self.settings), label


def predict_checkpoint(dataset, *, checkpoint, model_factory, num_classes=10,
                       batch_size=128, device="cpu"):
    """Load a local state dictionary and return aligned class predictions."""
    if type(num_classes) is not int or num_classes < 2 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Invalid prediction settings")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    model = model_factory().to(target_device)
    state = torch.load(Path(checkpoint), map_location=target_device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    predictions, labels = [], []
    with torch.inference_mode():
        for images, target in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            if not torch.is_tensor(images) or not torch.isfinite(images).all():
                raise ValueError("Evaluation images must be finite tensors")
            target = torch.as_tensor(target)
            if target.ndim != 1 or target.dtype not in (torch.int32, torch.int64):
                raise ValueError("Evaluation labels must be integer class indices")
            logits = model(images.to(target_device))
            if logits.shape != (len(target), num_classes) or not torch.isfinite(logits).all():
                raise ValueError("Model returned invalid evaluation logits")
            predictions.append(logits.argmax(1).cpu().numpy())
            labels.append(target.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(labels)


def backdoor_metrics(clean_predictions, triggered_predictions, labels, *, target_label):
    """Measure targeted ASR without counting true target-class examples."""
    clean = np.asarray(clean_predictions)
    triggered = np.asarray(triggered_predictions)
    truth = np.asarray(labels)
    if (
        clean.ndim != 1
        or triggered.shape != clean.shape
        or truth.shape != clean.shape
        or clean.dtype.kind not in "iu"
        or triggered.dtype.kind not in "iu"
        or truth.dtype.kind not in "iu"
        or type(target_label) is not int
    ):
        raise ValueError("Aligned integer predictions, labels and target_label are required")
    eligible = truth != target_label
    if not eligible.any():
        raise ValueError("No non-target test samples are available for ASR")
    clean_correct = eligible & (clean == truth)
    clean_target = clean[eligible] == target_label
    triggered_target = triggered[eligible] == target_label
    conditional = triggered[clean_correct] == target_label
    return {
        "target_label": target_label,
        "test_samples": int(len(truth)),
        "eligible_non_target_samples": int(eligible.sum()),
        "clean_correct_eligible_samples": int(clean_correct.sum()),
        "clean_accuracy": float(np.mean(clean == truth)),
        "triggered_accuracy": float(np.mean(triggered == truth)),
        "attack_success_rate": float(np.mean(triggered_target)),
        "clean_target_rate": float(np.mean(clean_target)),
        "attack_success_lift": float(np.mean(triggered_target) - np.mean(clean_target)),
        "prediction_flip_to_target_rate": float(np.mean(triggered_target & ~clean_target)),
        "conditional_attack_success_rate": (
            float(np.mean(conditional)) if len(conditional) else None
        ),
    }


def evaluate_backdoor_checkpoint(run, clean_test_ids, triggered_test, *, model_factory,
                                 target_label, output, num_classes=10, batch_size=128,
                                 device="cpu"):
    """Evaluate one training run on the paired clean and triggered test rows."""
    with np.load(run["artifacts"]["predictions"], allow_pickle=False) as saved:
        ids = saved["sample_ids"].copy()
        clean_predictions = saved["predictions"].copy()
        labels = saved["labels"].copy()
    expected_ids = np.asarray(clean_test_ids)
    if not np.array_equal(ids, expected_ids) or len(triggered_test) != len(ids):
        raise ValueError("Clean and triggered test rows must have identical IDs and order")
    triggered_predictions, triggered_labels = predict_checkpoint(
        triggered_test,
        checkpoint=run["artifacts"]["checkpoint"],
        model_factory=model_factory,
        num_classes=num_classes,
        batch_size=batch_size,
        device=device,
    )
    if not np.array_equal(labels, triggered_labels):
        raise ValueError("Triggered evaluation changed the ground-truth labels")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        sample_ids=ids,
        clean_predictions=clean_predictions,
        triggered_predictions=triggered_predictions,
        labels=labels,
    )
    return {
        "metrics": backdoor_metrics(
            clean_predictions, triggered_predictions, labels, target_label=target_label
        ),
        "predictions": str(output.resolve()),
        "trigger": dict(triggered_test.settings),
    }

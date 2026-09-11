import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from cleaning.backdoor import decide_backdoor_actions
from experiments.defend_backdoor import defence_effect, removal_metrics, validate_matched_bundles
from poison_features import ImageInputBundle
from training.backdoor_evaluation import (
    TriggeredDataset,
    apply_square_trigger,
    backdoor_metrics,
    evaluate_backdoor_checkpoint,
)


def result(ids, flags, version="1"):
    return {
        "sample_ids": np.asarray(ids),
        "flags": np.asarray(flags, dtype=bool),
        "scores": np.asarray(flags, dtype=float),
        "version": version,
        "settings": {"threshold": 1},
    }


class BackdoorDefenceTests(unittest.TestCase):
    def scan(self):
        ids = np.array(["a", "b", "c", "d"])
        detectors = {
            "contrast_patch": result(ids, [0, 1, 0, 0]),
            "spectral_signature": result(ids, [0, 0, 1, 0]),
            "activation_clustering": result(ids, [0, 0, 0, 0]),
            "repeated_patch": result(ids, [0, 0, 0, 1]),
        }
        return {
            "sample_ids": ids,
            "detectors": detectors,
            "candidate_flags": np.array([0, 1, 1, 0], dtype=bool),
            "settings": {
                "active_detectors": [
                    "contrast_patch", "spectral_signature", "activation_clustering"
                ],
                "comparison_only": ["repeated_patch"],
                "patch_profile": "contrast",
            },
            "evaluation": {"candidates": {"tp": 999}},
        }

    def test_policy_quarantines_patch_and_holds_feature_only(self):
        scan = self.scan()
        before = scan["evaluation"].copy()
        decision = decide_backdoor_actions(scan)
        self.assertEqual(
            decision["actions"].tolist(),
            ["keep", "quarantine", "human_review", "keep"],
        )
        self.assertEqual(decision["summary"]["quarantine"], 1)
        self.assertEqual(decision["summary"]["human_review"], 1)
        self.assertEqual(scan["evaluation"], before)
        self.assertEqual(decision["settings"]["comparison_only_ignored"], ["repeated_patch"])

    def test_policy_rejects_missing_rules_alignment_and_candidate_drift(self):
        scan = self.scan()
        scan["candidate_flags"][0] = True
        with self.assertRaisesRegex(ValueError, "union"):
            decide_backdoor_actions(scan)
        scan = self.scan()
        scan["detectors"]["contrast_patch"]["sample_ids"] = scan["sample_ids"][::-1]
        with self.assertRaisesRegex(ValueError, "aligned"):
            decide_backdoor_actions(scan)
        scan = self.scan()
        scan["settings"]["active_detectors"].append("new_detector")
        with self.assertRaisesRegex(ValueError, "explicit"):
            decide_backdoor_actions(scan)

    def test_trigger_is_positioned_and_source_is_unchanged(self):
        image = torch.zeros(3, 8, 9)
        original = image.clone()
        triggered = apply_square_trigger(
            image, patch_size=3, position="centre", colour="red"
        )
        self.assertTrue(torch.equal(image, original))
        self.assertTrue(torch.equal(triggered[0, 2:5, 3:6], torch.ones(3, 3)))
        self.assertFalse(triggered[1:].any())
        with self.assertRaises(ValueError):
            apply_square_trigger(image, patch_size=10)

    def test_metrics_exclude_true_target_class_and_report_lift(self):
        labels = np.array([0, 1, 1, 2])
        clean = np.array([0, 1, 2, 1])
        triggered = np.array([0, 0, 0, 1])
        metrics = backdoor_metrics(clean, triggered, labels, target_label=0)
        self.assertEqual(metrics["eligible_non_target_samples"], 3)
        self.assertAlmostEqual(metrics["attack_success_rate"], 2 / 3)
        self.assertEqual(metrics["clean_target_rate"], 0)
        self.assertEqual(metrics["conditional_attack_success_rate"], 1)

    def test_checkpoint_evaluation_uses_paired_test_rows(self):
        class PatchModel(nn.Module):
            def forward(self, values):
                patch = values[:, :, -1, -1].mean(1)
                logits = torch.zeros(len(values), 3)
                logits[:, 0] = patch * 10
                logits[:, 1] = (1 - patch) * 10
                return logits

        images = torch.zeros(4, 3, 5, 5)
        labels = torch.tensor([0, 1, 1, 2])
        dataset = TensorDataset(images, labels)
        triggered = TriggeredDataset(dataset)
        ids = np.array(["t0", "t1", "t2", "t3"])
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            checkpoint = root / "model.pt"
            predictions = root / "clean.npz"
            torch.save(PatchModel().state_dict(), checkpoint)
            np.savez_compressed(
                predictions, sample_ids=ids, predictions=np.ones(4, dtype=int), labels=labels.numpy()
            )
            evaluated = evaluate_backdoor_checkpoint(
                {"artifacts": {"checkpoint": str(checkpoint), "predictions": str(predictions)}},
                ids, triggered, model_factory=PatchModel, target_label=0,
                output=root / "triggered.npz", num_classes=3, batch_size=2,
            )
            self.assertEqual(evaluated["metrics"]["attack_success_rate"], 1)
            self.assertEqual(evaluated["metrics"]["prediction_flip_to_target_rate"], 1)
            self.assertTrue(Path(evaluated["predictions"]).is_file())

    def test_removal_metrics_and_matched_bundle_checks(self):
        decision = decide_backdoor_actions(self.scan())
        metrics = removal_metrics(decision, np.array([0, 1, 1, 0], dtype=bool))
        self.assertEqual(metrics["poisoned_removed"], 2)
        self.assertEqual(metrics["clean_removed"], 0)
        self.assertEqual(metrics["poison_removal_recall"], 1)

        ids = np.array(["a", "b"])
        images = np.zeros((2, 3, 4, 4), dtype=np.float32)
        clean = ImageInputBundle(images, np.array([1, 2]), ids)
        poisoned = ImageInputBundle(images.copy(), np.array([0, 2]), ids.copy())
        class Features:
            sample_ids = ids.copy()
            labels = np.array([0, 2])
        validate_matched_bundles(clean, poisoned, Features())
        bad = ImageInputBundle(images.copy(), np.array([0, 2]), ids[::-1])
        with self.assertRaisesRegex(ValueError, "same aligned rows"):
            validate_matched_bundles(clean, bad, Features())

    def test_defence_effect_compares_identical_training_arms(self):
        def arm(asr, conditional, accuracy, lift):
            return {"metrics": {"attack_success_rate": asr,
                "conditional_attack_success_rate": conditional,
                "clean_accuracy": accuracy, "attack_success_lift": lift}}
        effect = defence_effect({
            "clean_baseline": arm(.1, .01, .8, 0),
            "poisoned_baseline": arm(.8, .7, .7, .7),
            "after_cleaning": arm(.2, .1, .78, .1),
        })
        self.assertAlmostEqual(effect["absolute_asr_reduction"], .6)
        self.assertAlmostEqual(effect["relative_asr_reduction"], .75)
        self.assertAlmostEqual(effect["clean_accuracy_gap_vs_reference"], -.02)


if __name__ == "__main__":
    unittest.main()

"""Experiment construction and evaluation boundaries, not scope calibration."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json

import numpy as np
from PIL import Image

from poison_features import ImageInputBundle
from experiments.validate_patch_scope import injected_pixels, main, scan_case


class PatchValidationTests(unittest.TestCase):
    def clean(self):
        return ImageInputBundle(np.full((200, 3, 8, 8), .4, np.float32),
                                np.arange(200) % 10, np.arange(200))

    def test_colours_positions_counts_and_no_mutation(self):
        clean = self.clean()
        for colour, values in [("white", [1, 1, 1]), ("red", [1, 0, 0]), ("black", [0, 0, 0])]:
            for position, start in [("bottom_right", 5), ("centre", 2)]:
                pixels, truth = injected_pixels(clean, fraction=.1, colour=colour, position=position, seed=17)
                self.assertEqual(int(truth.sum()), 20)
                np.testing.assert_array_equal((pixels.images != clean.images).any(axis=(1, 2, 3)), truth)
                np.testing.assert_array_equal(pixels.sample_ids, clean.sample_ids)
                self.assertTrue(np.all(pixels.labels[truth] == 0))
                np.testing.assert_allclose(pixels.images[truth, :, start, start], np.tile(values, (20, 1)))
        self.assertTrue(np.all(clean.images == .4))
        np.testing.assert_array_equal(clean.labels, np.arange(200) % 10)

    def test_seeds_reproducible_and_rates_nested(self):
        clean = self.clean()
        _, small = injected_pixels(clean, fraction=.01, colour="white", position="centre", seed=1)
        _, big = injected_pixels(clean, fraction=.10, colour="black", position="bottom_right", seed=1)
        self.assertTrue(np.all(big[small]))
        _, repeated = injected_pixels(clean, fraction=.10, colour="white", position="centre", seed=1)
        np.testing.assert_array_equal(big, repeated)

    def test_truth_only_used_after_scoring(self):
        clean = self.clean()
        fake = dict(candidate_flags=np.zeros(200, bool))
        with patch("experiments.validate_patch_scope.scan_backdoor_images", return_value=fake) as scanner:
            scan_case(clean, np.ones(200, bool))
            scanner.assert_called_once_with(clean, dataset="cifar10")

    def test_report_cli_uses_test_split_and_saves_plan_before_scoring(self):
        dataset = [(Image.fromarray(np.full((8, 8, 3), 102, np.uint8)), i % 10) for i in range(200)]
        with tempfile.TemporaryDirectory() as tmp:
            def fake_scan(pixels, truth):
                self.assertTrue(list(Path(tmp).glob("*/plan.json")))
                self.assertTrue(str(pixels.sample_ids[0]).startswith("cifar10-test:"))
                count = int(truth.sum())
                return dict(metrics=dict(tp=0, fp=0, fn=count, tn=len(truth)-count,
                                         recall=0. if count else None, precision=None, false_positive_rate=0.),
                            runtime_seconds=0, flagged_ids=[], false_positive_ids=[], missed_poison_ids=[])
            with patch("experiments.validate_patch_scope.load_image_dataset", return_value=dataset) as loader, \
                 patch("experiments.validate_patch_scope.scan_case", side_effect=fake_scan), \
                 patch("sys.argv", ["validate_patch_scope", "--samples", "100", "--seeds", "17", "--output", tmp]):
                main()
            self.assertFalse(loader.call_args.kwargs["train"])
            report = json.loads(next(Path(tmp).glob("*/summary.json")).read_text())
            self.assertEqual(len(report["runs"]), 19)
            self.assertEqual(report["unique_base_images"], 100)
            self.assertTrue(next(Path(tmp).glob("*/index.html")).is_file())


if __name__ == "__main__":
    unittest.main()

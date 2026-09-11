"""Contrast detector geometry, false-positive controls and routing regressions."""
import json
import unittest
from unittest.mock import patch

import numpy as np
from poison_features import ImageInputBundle
from detectors.backdoor import ContrastPatchDetector
from detectors.backdoor.contrast_patch import boundary_contrast
from detectors.backdoor.image_pipeline import scan_backdoor_images
from detectors.output_connector import to_jsonable
from experiments.validate_patch_fix import compare_case


class ContrastPatchTests(unittest.TestCase):
    def test_public_package_export(self):
        self.assertEqual(ContrastPatchDetector.__name__, "ContrastPatchDetector")

    def clean(self, n=1000):
        images = np.random.default_rng(8).uniform(.2, .8, (n, 3, 10, 10)).astype(np.float32)
        return ImageInputBundle(images, np.arange(n) % 10, np.array([f"sample:{i}" for i in range(n)]))

    def test_colours_and_positions(self):
        for colour in ([0, 0, 0], [1, 1, 1], [1, 0, 0], [.2, .2, .2]):
            for row, col in ((0, 0), (7, 7), (3, 4)):
                source = self.clean()
                source.images[:10, :, row:row+3, col:col+3] = np.array(colour)[None, :, None, None]
                source.labels[:10] = 0
                result = ContrastPatchDetector().analyze(source)
                self.assertTrue(result["flags"][:10].all())
                self.assertFalse(result["flags"][10:].any())
                self.assertEqual(result["scores"].shape, (1000,))
                self.assertTrue(np.isfinite(result["scores"]).all())
                np.testing.assert_array_equal(result["sample_ids"], source.sample_ids)
                json.dumps(to_jsonable(result), allow_nan=False)

    def test_context_filter_prevents_low_rate_dilution(self):
        source = self.clean()
        source.images[10:110, :, 4:, 4:] = 1  # larger innocent regions
        source.images[:10, :, 7:, 7:] = 1
        source.labels[:10] = 0
        result = ContrastPatchDetector().analyze(source)
        self.assertTrue(result["flags"][:10].all())
        self.assertFalse(result["flags"][10:].any())
        evidence = [p for p in result["evidence"]["patterns"] if p["size"] == 3]
        self.assertEqual(evidence[0]["raw_support"], 110)
        self.assertEqual(evidence[0]["boundary_support"], 10)

    def test_constant_backgrounds_and_clean_random_data(self):
        for value in (None, 0., 1.):
            source = self.clean(200)
            if value is not None: source.images[:] = value
            result = ContrastPatchDetector().analyze(source)
            self.assertFalse(result["flags"].any())

    def test_known_limit_patch_identical_to_surroundings(self):
        source = self.clean()
        source.images[:10] = 1
        source.labels[:10] = 0
        self.assertFalse(ContrastPatchDetector().analyze(source)["flags"].any())

    def test_reproducible_no_input_mutation_and_no_truth_access(self):
        source = self.clean(200)
        source.images[:10, :, -3:, -3:] = 0
        source.labels[:10] = 0
        originals = [a.copy() for a in (source.images, source.labels, source.sample_ids)]
        class GuardedBundle(ImageInputBundle):
            def __getattr__(self, name): raise AssertionError(f"Unexpected truth access {name}")
        guarded = GuardedBundle(source.images, source.labels, source.sample_ids)
        detector = ContrastPatchDetector()
        a = detector.analyze(guarded); b = detector.analyze(guarded)
        np.testing.assert_array_equal(a["scores"], b["scores"])
        for before, after in zip(originals, (source.images, source.labels, source.sample_ids)):
            np.testing.assert_array_equal(before, after)

    def test_invalid_settings_and_alignment(self):
        for kwargs in (dict(min_count=1), dict(patch_sizes=(2, 2)), dict(patch_sizes=(4,)),
                       dict(min_contrast=0), dict(min_contrast=np.nan), dict(min_lift=1),
                       dict(diagnostic_limit=-1), dict(min_purity=.5)):
            with self.assertRaises(ValueError): ContrastPatchDetector(**kwargs)
        source = self.clean(200); source.sample_ids[1] = source.sample_ids[0]
        with self.assertRaises(ValueError): ContrastPatchDetector().analyze(source)

    def test_progress_and_diagnostics(self):
        source = self.clean(200); source.images[:] = 0
        calls = []
        result = ContrastPatchDetector(diagnostic_limit=1).analyze(source, lambda d,t: calls.append((d,t)))
        self.assertEqual(calls[-1][0], calls[-1][1])
        self.assertEqual(len(result["evidence"]["diagnostics"]), 1)
        self.assertTrue(result["evidence"]["diagnostics_truncated"])
        self.assertIn("insufficient_boundary_support", result["evidence"]["diagnostic_counts"])

    def test_large_patch_fragment_has_no_complete_boundary(self):
        source = self.clean(200); source.images[:10, :, 4:, 4:] = 1
        self.assertTrue(np.all(boundary_contrast(source.images[:10], 7, 7, 3) == 0))

    def test_one_uint8_level_boundary_is_not_rounded_below_threshold(self):
        for level in (1, 128, 205, 255):
            images = np.full((1, 3, 10, 10), (level-1)/255, dtype=np.float32)
            images[:, :, -3:, -3:] = level/255
            self.assertEqual(float(boundary_contrast(images, 7, 7, 3)[0]), 1/255)

    def test_legacy_comparison_flags_do_not_enter_revised_queue(self):
        source = self.clean(200)
        legacy = dict(flags=np.ones(200, bool), scores=np.ones(200), sample_ids=source.sample_ids,
                      settings={}, evidence={})
        with patch("detectors.backdoor.image_pipeline.RepeatedPatchDetector.analyze", return_value=legacy):
            result = scan_backdoor_images(source, dataset="cifar10", patch_profile="contrast")
        self.assertTrue(result["detectors"]["repeated_patch"]["flags"].all())
        self.assertFalse(result["candidate_flags"].any())
        self.assertEqual(result["settings"]["active_detectors"], ["contrast_patch"])
        self.assertEqual(result["settings"]["comparison_only"], ["repeated_patch"])

    def test_comparison_evaluates_after_both_separate_scans(self):
        source = self.clean(200)
        result = dict(flags=np.zeros(200, bool), evidence={})
        with patch("experiments.validate_patch_fix.RepeatedPatchDetector.analyze", return_value=result) as old, \
             patch("experiments.validate_patch_fix.ContrastPatchDetector.analyze", return_value=result) as new:
            metrics = compare_case(source, np.ones(200, bool))
            old.assert_called_once_with(source); new.assert_called_once_with(source)
        self.assertEqual(metrics["legacy"]["metrics"]["fn"], 200)
        self.assertEqual(metrics["contrast"]["metrics"]["fn"], 200)


if __name__ == "__main__": unittest.main()

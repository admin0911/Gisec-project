"""Routing, evaluation separation and offline report integration."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from poison_features import DetectorInput, FeatureBundle, ImageInputBundle
from detectors.backdoor.image_pipeline import scan_backdoor_images
from experiments.backdoor_report import report_html, write_sample_csv
from experiments.scan_backdoor_images import paired_images, run_scan


class BackdoorImagePipelineTests(unittest.TestCase):
    def inputs(self, attack=True):
        rng = np.random.default_rng(5)
        images = rng.uniform(0, .7, (200, 3, 8, 8)).astype(np.float32)
        labels = np.arange(200) % 10
        truth = np.zeros(200, bool)
        if attack:
            truth[:20] = True
            images[:20, :, -3:, -3:] = 1
            labels[:20] = 0
        pixels = ImageInputBundle(images, labels, np.array([f"row:{i}" for i in range(200)]))
        # Deliberately constant features: pixel evidence must survive.
        features = np.zeros((200, 8))
        bundle = FeatureBundle(features, features.copy(), features.copy(), labels,
                               pixels.sample_ids.copy(), "image", "synthetic", dataset_name="cifar10",
                               is_poisoned=truth)
        return pixels, bundle

    def test_feature_nonfindings_do_not_veto_pixel_candidates(self):
        pixels, bundle = self.inputs()
        inputs = DetectorInput(bundle.features, bundle.sample_ids, bundle.labels)
        result = scan_backdoor_images(pixels, dataset="cifar10", features=inputs)
        self.assertTrue(result["candidate_flags"][:20].all())
        self.assertFalse(result["candidate_flags"][20:].any())
        self.assertFalse(result["detectors"]["spectral_signature"]["flags"].any())
        self.assertFalse(result["detectors"]["activation_clustering"]["flags"].any())
        np.testing.assert_array_equal(result["sample_ids"], pixels.sample_ids)
        self.assertEqual(result["detectors"]["repeated_patch"]["settings"]["profile"], "cifar-bright-patch-v1")

    def test_misalignment_fails_before_scanning(self):
        pixels, bundle = self.inputs()
        for inputs in (DetectorInput(bundle.features, bundle.sample_ids[::-1], bundle.labels),
                       DetectorInput(bundle.features, bundle.sample_ids, bundle.labels + 1)):
            with patch("detectors.backdoor.image_pipeline.RepeatedPatchDetector.analyze") as detector:
                with self.assertRaisesRegex(ValueError, "must match"):
                    scan_backdoor_images(pixels, dataset="cifar10", features=inputs)
                detector.assert_not_called()

    def test_clean_pixel_only_and_unknown_dataset(self):
        pixels, _ = self.inputs(False)
        result = scan_backdoor_images(pixels, dataset="cifar10")
        self.assertFalse(result["candidate_flags"].any())
        self.assertEqual(list(result["detectors"]), ["repeated_patch"])
        with self.assertRaises(ValueError): scan_backdoor_images(pixels, dataset="unknown")

    def test_scoring_does_not_access_truth(self):
        pixels, bundle = self.inputs()
        class GuardedInput:
            X = bundle.features
            y = bundle.labels
            sample_ids = bundle.sample_ids
            def __getattr__(self, name):
                raise AssertionError(f"Unexpected access: {name}")
        scan_backdoor_images(pixels, dataset="cifar10", features=GuardedInput())

    def test_cli_folder_and_report_exports(self):
        pixels, bundle = self.inputs()
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input"; source.mkdir()
            output = Path(tmp) / "reports"
            bundle.save(source / "test-features.npz")
            pixels.save(source / "test-images.npz")
            process = subprocess.run([sys.executable, "-m", "experiments.scan_backdoor_images",
                                      "--folder", str(source), "--output", str(output), "--evaluate"],
                                     capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            result_path = next(output.glob("*/scan-001/results.json"))
            result = json.loads(result_path.read_text())
            self.assertEqual(result["evaluation"]["candidates"]["tp"], 20)
            self.assertEqual(result["evaluation"]["candidates"]["fp"], 0)
            self.assertTrue(result_path.with_name("samples.csv").exists())
            html = result_path.with_name("report.html").read_text()
            self.assertIn("20/20", html)
            self.assertIn("data:image/png;base64,", html)
            self.assertIn("spectral_signature", html)
            self.assertIn("100.00%", html)
            self.assertTrue(result_path.parent.parent.joinpath("index.html").exists())

    def test_evaluation_does_not_change_detection_and_missing_truth_is_unknown(self):
        pixels, bundle = self.inputs()
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "test-features.npz"; p = Path(tmp) / "test-images.npz"
            pixels.save(p); bundle.save(f)
            plain, _ = run_scan(p, feature_file=f)
            evaluated, _ = run_scan(p, feature_file=f, evaluate=True)
            self.assertNotIn("evaluation", plain)
            np.testing.assert_array_equal(plain["candidate_flags"], evaluated["candidate_flags"])
            bundle.is_poisoned = None; bundle.save(f)
            unknown, _ = run_scan(p, feature_file=f, evaluate=True)
            self.assertNotIn("evaluation", unknown)
            self.assertIn("No saved poison truth", unknown["evaluation_note"])

    def test_known_clean_cannot_contradict_saved_truth(self):
        pixels, bundle = self.inputs()
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "test-features.npz"; p = Path(tmp) / "test-images.npz"
            pixels.save(p); bundle.save(f)
            with self.assertRaisesRegex(ValueError, "contradicts"):
                run_scan(p, feature_file=f, known_clean=True)

    def test_missing_companion_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "test-features.npz"
            with self.assertRaises(FileNotFoundError): paired_images(f)
            with self.assertRaises(ValueError): paired_images(Path(tmp) / "unknown.npz")

    def test_report_escapes_untrusted_text(self):
        pixels, _ = self.inputs()
        pixels = ImageInputBundle(pixels.images, pixels.labels, pixels.sample_ids.astype("<U100"))
        pixels.sample_ids[0] = "<script>alert(1)</script>"
        pixels.sample_ids[1] = "=SUM(1,2)"
        result = scan_backdoor_images(pixels, dataset="cifar10")
        result.update(input=dict(feature_file="<img src=x onerror=alert(1)>", image_file=""),
                      runtime_seconds=1, evaluation_note="Unknown")
        html = report_html(result, pixels)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<img src=x", html)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "samples.csv"
            write_sample_csv(p, result)
            self.assertIn("'=SUM", p.read_text())


if __name__ == "__main__":
    unittest.main()

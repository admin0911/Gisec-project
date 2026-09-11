"""Feature geometry and integration regressions, not real-attack calibration."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from poison_features import DetectorInput, FeatureBundle, detector_input
from detectors.backdoor.activation_clustering import ActivationClusteringDetector
from detectors.backdoor.spectral_signature import SpectralSignatureDetector
from detectors.backdoor.feature_pipeline import scan_backdoor_features
from detectors.output_connector import to_jsonable


class BackdoorFeatureTests(unittest.TestCase):
    def inputs(self, shifted=False):
        rng = np.random.default_rng(42)
        X = rng.normal(0, .25, (1000, 16))
        y = np.arange(len(X)) % 2
        poison = np.arange(0, 100, 2)
        if shifted:
            X[poison, 0] += 8
        return DetectorInput(X, np.array([f"row:{i}" for i in range(len(X))]), y), poison

    def detectors(self):
        return (SpectralSignatureDetector(), ActivationClusteringDetector())

    def test_separated_minority_and_contract(self):
        inputs, poison = self.inputs(True)
        originals = [v.copy() for v in (inputs.X, inputs.y, inputs.sample_ids)]
        for detector in self.detectors():
            with self.subTest(detector=type(detector).__name__):
                result = detector.analyze(inputs)
                np.testing.assert_array_equal(result["sample_ids"], inputs.sample_ids)
                self.assertEqual(result["scores"].shape, (len(inputs.X),))
                self.assertTrue(np.isfinite(result["scores"]).all())
                self.assertTrue(result["flags"][poison].all())
                self.assertLessEqual(result["flags"].sum() - len(poison), 5)
                self.assertEqual(result["flags"].dtype, bool)
                json.dumps(to_jsonable(result), allow_nan=False)
                np.testing.assert_allclose(detector.score(inputs), result["scores"])
        for before, after in zip(originals, (inputs.X, inputs.y, inputs.sample_ids)):
            np.testing.assert_array_equal(before, after)

    def test_clean_fixture_has_low_false_alarms_without_forced_quota(self):
        inputs, _ = self.inputs()
        for detector in self.detectors():
            self.assertLessEqual(detector.analyze(inputs)["flags"].sum(), 5)

    def test_constant_and_small_classes_are_explicitly_unevaluated(self):
        for X, y in [(np.zeros((20, 3)), np.zeros(20, int)),
                     (np.ones((3, 1)), np.array([0, 0, 1]))]:
            for detector in self.detectors():
                result = detector.analyze(DetectorInput(X, np.arange(len(X)), y))
                self.assertFalse(result["flags"].any())
                self.assertFalse(result["evidence"]["evaluated"].any())
                self.assertEqual(result["scores"].sum(), 0)

    def test_string_labels_one_dimension_and_large_finite_values(self):
        X = np.r_[np.ones(90), np.full(10, -1.)][:, None] * 1e300
        for detector in self.detectors():
            result = detector.analyze(DetectorInput(X, np.arange(100), np.full(100, "cat")))
            self.assertTrue(np.isfinite(result["scores"]).all())
            json.dumps(to_jsonable(result), allow_nan=False)

    def test_validation(self):
        X = np.ones((20, 3)); ids = np.arange(20); y = np.zeros(20, int)
        cases = [(X, ids, None), (X, ids, np.full(20, -1)),
                 (X, np.zeros(20, int), y), (X, ids[:-1], y),
                 (X, ids, y[:-1]), (np.full((20, 3), np.nan), ids, y),
                 (np.full((20, 3), np.inf), ids, y), (X[:, :0], ids, y),
                 (X, ids, np.full(20, "")), (X, ids, np.zeros(20, float)),
                 (X, ids.astype(float), y), (np.zeros(20), ids, y)]
        for detector in self.detectors():
            for values in cases:
                with self.subTest(detector=type(detector).__name__, shapes=[np.shape(v) for v in values]):
                    with self.assertRaises(ValueError):
                        detector.analyze(DetectorInput(*values))

    def test_invalid_settings(self):
        for kwargs in ({"threshold": 0}, {"threshold": np.inf}, {"min_class_size": True},
                       {"random_state": -1}, {"random_state": 2**32}):
            with self.assertRaises(ValueError): SpectralSignatureDetector(**kwargs)
        for kwargs in ({"n_components": 0}, {"max_cluster_fraction": .5},
                       {"min_separation": np.nan}, {"min_cluster_size": 1},
                       {"random_state": -1}, {"random_state": 2**32}):
            with self.assertRaises(ValueError): ActivationClusteringDetector(**kwargs)

    def test_balanced_clusters_are_not_assumed_poisoned(self):
        X = np.r_[np.zeros((50, 2)), np.ones((50, 2))]
        inputs = DetectorInput(X, np.arange(100), np.zeros(100, int))
        for detector in self.detectors():
            self.assertFalse(detector.analyze(inputs)["flags"].any())

    def test_row_permutation_preserves_separated_findings(self):
        inputs, _ = self.inputs(True)
        order = np.random.default_rng(5).permutation(len(inputs.X))
        shuffled = DetectorInput(inputs.X[order], inputs.sample_ids[order], inputs.y[order])
        for detector in self.detectors():
            before = detector.analyze(inputs)
            after = detector.analyze(shuffled)
            np.testing.assert_array_equal(after["sample_ids"], inputs.sample_ids[order])
            np.testing.assert_array_equal(after["flags"], before["flags"][order])

    def test_seed_reproducibility_and_scalar_rescaling(self):
        inputs, _ = self.inputs(True)
        scaled = DetectorInput(inputs.X * 100, inputs.sample_ids, inputs.y)
        for detector in self.detectors():
            result = detector.analyze(inputs)
            np.testing.assert_array_equal(result["flags"], detector.analyze(inputs)["flags"])
            np.testing.assert_allclose(result["scores"], detector.score(scaled), rtol=1e-6, atol=1e-8)

    def test_pipeline_never_reads_evaluation_metadata(self):
        inputs, _ = self.inputs(True)
        class GuardedInput:
            X = inputs.X
            y = inputs.y
            sample_ids = inputs.sample_ids
            def __getattr__(self, name):
                raise AssertionError(f"Unexpected access to {name}")
        result = scan_backdoor_features(GuardedInput())
        votes = sum(item["flags"].astype(int) for item in result["detectors"].values())
        np.testing.assert_array_equal(votes, result["flag_count"])
        np.testing.assert_array_equal(votes > 0, result["candidate_flags"])
        np.testing.assert_array_equal(votes == 2, result["agreement_flags"])
        json.dumps(to_jsonable(result), allow_nan=False)

    def test_bundle_truth_changes_cannot_change_detection(self):
        inputs, _ = self.inputs(True)
        bundle = FeatureBundle(inputs.X, inputs.X.copy(), inputs.X.copy(), inputs.y,
                               inputs.sample_ids, "image", "test", is_poisoned=np.zeros(1000, bool))
        before = scan_backdoor_features(detector_input(bundle, representation="raw", label_aware=True))
        bundle.is_poisoned[:] = True
        bundle.original_labels = np.full(1000, -99)
        bundle.poison_type = np.full(1000, "irrelevant")
        after = scan_backdoor_features(detector_input(bundle, representation="raw", label_aware=True))
        for name in before["detectors"]:
            np.testing.assert_array_equal(before["detectors"][name]["scores"], after["detectors"][name]["scores"])

    def test_saved_feature_cli(self):
        inputs, _ = self.inputs(True)
        bundle = FeatureBundle(inputs.X, inputs.X, inputs.X, inputs.y,
                               inputs.sample_ids, "image", "synthetic-test")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "features.npz"
            output = Path(tmp) / "scan.json"
            bundle.save(source)
            subprocess.run([sys.executable, "-m", "experiments.scan_backdoor_features",
                            str(source), "--output", str(output)], check=True, capture_output=True)
            result = json.loads(output.read_text())
            self.assertNotIn("evaluation", result)
            self.assertEqual(result["sample_ids"], inputs.sample_ids.tolist())
            self.assertEqual(result["input"]["representation"], "raw")
            bundle.is_poisoned = np.zeros(1000, bool)
            bundle.is_poisoned[:10] = True
            bundle.save(source)
            subprocess.run([sys.executable, "-m", "experiments.scan_backdoor_features",
                            str(source), "--output", str(output), "--evaluate"],
                           check=True, capture_output=True)
            evaluated = json.loads(output.read_text())
            self.assertIn("evaluation", evaluated)
            for name in result["detectors"]:
                self.assertEqual(result["detectors"][name]["scores"], evaluated["detectors"][name]["scores"])


if __name__ == "__main__":
    unittest.main()

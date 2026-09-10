import json
import unittest
from importlib.util import find_spec
from types import SimpleNamespace

import numpy as np

from detectors.output_connector import detector_result, to_jsonable
from detectors.label_flip.pipeline import combine_label_results, scan_label_flips


class OutputConnectorTests(unittest.TestCase):
    def result(self, scores, flags, name="test"):
        return detector_result(name, "1.0", {
            "sample_ids": np.array(["b", "a", "c"]),
            "scores": np.array(scores), "flags": np.array(flags),
        }, {"k": 20}, expected_sample_ids=["b", "a", "c"])

    def test_alignment_and_boolean_flags_are_enforced(self):
        raw = dict(sample_ids=np.array(["a", "b"]),
                   scores=np.array([0.1, 0.9]), flags=np.array([False, True]))
        with self.assertRaises(ValueError):
            detector_result("test", "1.0", raw, {}, expected_sample_ids=["b", "a"])
        raw["flags"] = np.array([0, 1])
        with self.assertRaises(ValueError):
            detector_result("test", "1.0", raw, {}, expected_sample_ids=["a", "b"])
        raw["flags"] = np.array([False, True])
        raw["scores"][0] = np.nan
        with self.assertRaises(ValueError):
            detector_result("test", "1.0", raw, {}, expected_sample_ids=["a", "b"])

    def test_combination_preserves_individual_flags_and_requires_cl_flag(self):
        results = {
            "knn": self.result([0.95, 0.9, 0.95], [True, False, True]),
            "class_distance": self.result([0.025, 0.04, 0.01], [True, True, False]),
            "confident_learning": self.result([0.95, 0.95, 0.95], [False, True, True]),
        }
        output = combine_label_results(results)
        np.testing.assert_array_equal(output["flag_count"], [1, 2, 2])
        np.testing.assert_array_equal(output["review_flags"], [False, True, True])
        self.assertTrue(output["detectors"]["class_distance"]["flags"][0])
        decoded = json.loads(json.dumps(to_jsonable(output), allow_nan=False))
        self.assertEqual(decoded["sample_ids"], ["b", "a", "c"])
        self.assertEqual(decoded["review_flags"], [False, True, True])

    @unittest.skipUnless(find_spec("cleanlab"), "Optional cleanlab dependency missing")
    def test_real_three_detector_scan(self):
        rng = np.random.default_rng(7)
        labels = np.repeat(np.arange(3), 25)
        X = np.eye(3)[labels] + rng.normal(0, 0.03, (75, 3))
        labels[0] = 1
        inputs = SimpleNamespace(X=X, y=labels, sample_ids=np.array([f"i-{i}" for i in range(75)]))
        progress = []
        output = scan_label_flips(inputs, batch_size=16, progress=progress.append)
        self.assertEqual(len(output["detectors"]), 3)
        for result in output["detectors"].values():
            np.testing.assert_array_equal(result["sample_ids"], inputs.sample_ids)
            self.assertEqual(result["scores"].shape, (75,))
            self.assertEqual(result["flags"].dtype, np.dtype(bool))
        self.assertTrue(output["review_flags"][0])
        self.assertEqual(sum("of 3:" in message for message in progress), 3)
        json.dumps(to_jsonable(output), allow_nan=False)


if __name__ == "__main__":
    unittest.main()

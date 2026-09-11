import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from detectors.feature_pipeline import scan_feature_bundle
from detectors.output_connector import to_jsonable
from poison_features import (
    FeatureBundle,
    feature_bundle_from_arrays,
    load_external_feature_bundle,
)


class ExternalFeatureBundleTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(7)
        labels = np.arange(120) % 2
        features = rng.normal(0, .2, (120, 12)).astype(np.float32)
        features[:12:2, 0] += 7
        ids = np.array([f"review:{i}" for i in range(120)])
        return features, labels, ids

    def test_array_adapter_is_dataset_neutral(self):
        features, labels, ids = self.fixture()
        for modality in ("image", "text", "network", "tabular"):
            with self.subTest(modality=modality):
                bundle = feature_bundle_from_arrays(
                    features, labels=labels, sample_ids=ids,
                    modality=modality, encoder="my-program-v2",
                    dataset_name="custom-data",
                )
                self.assertEqual(bundle.features.shape, (120, 12))
                self.assertEqual(bundle.scaled_features.shape, (120, 12))
                self.assertLessEqual(bundle.reduced_features.shape[1], 64)
                self.assertIsNone(bundle.visual_features)
                np.testing.assert_array_equal(bundle.labels, labels)
                np.testing.assert_array_equal(bundle.sample_ids, ids)

    def test_safe_npz_aliases_and_metadata(self):
        features, labels, ids = self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "export.npz"
            np.savez_compressed(
                path, embeddings=features, targets=labels, row_ids=ids,
                metadata_json=np.array(json.dumps({
                    "dataset_name": "imdb", "modality": "text", "encoder": "minilm",
                })),
            )
            bundle = load_external_feature_bundle(path)
        self.assertEqual(bundle.dataset_name, "imdb")
        self.assertEqual(bundle.modality, "text")
        self.assertEqual(bundle.encoder, "minilm")
        self.assertEqual(bundle.metadata["source_format"], "external_npz")
        np.testing.assert_array_equal(bundle.features, features)

    def test_integral_float_labels_are_normalized_for_all_feature_detectors(self):
        features, labels, ids = self.fixture()
        bundle = feature_bundle_from_arrays(features, labels=labels.astype(float), sample_ids=ids)
        self.assertEqual(bundle.labels.dtype.kind, "i")
        result = scan_feature_bundle(bundle, tracks=("backdoor",), k=5)
        self.assertIn("backdoor", result["tracks"])

    def test_current_bundle_is_safe_external_input_and_truth_is_not_imported(self):
        features, labels, ids = self.fixture()
        bundle = FeatureBundle(
            features, features.copy(), features[:, :5].copy(), labels, ids,
            "text", "custom", "reviews", is_poisoned=np.ones(120, dtype=bool),
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "features.npz"
            bundle.save(path)
            loaded = load_external_feature_bundle(path)
        self.assertEqual(loaded.dataset_name, "reviews")
        self.assertIsNone(loaded.is_poisoned)
        np.testing.assert_array_equal(loaded.reduced_features, features[:, :5])

    def test_invalid_external_arrays_are_rejected(self):
        features, labels, ids = self.fixture()
        cases = [
            dict(features=np.full_like(features, np.nan), labels=labels, sample_ids=ids),
            dict(features=features, labels=labels[:-1], sample_ids=ids),
            dict(features=features, labels=labels, sample_ids=np.zeros(len(ids), int)),
        ]
        for case in cases:
            with self.subTest(case=tuple(np.shape(v) for v in case.values())):
                with self.assertRaises(ValueError):
                    feature_bundle_from_arrays(**case)

    def test_tracks_remain_separate_and_do_not_read_truth(self):
        features, labels, ids = self.fixture()
        bundle = feature_bundle_from_arrays(
            features, labels=labels, sample_ids=ids,
            modality="text", encoder="external", dataset_name="imdb-like",
        )
        bundle.is_poisoned = np.ones(len(labels), dtype=bool)
        before = scan_feature_bundle(bundle, k=5, batch_size=32)
        bundle.is_poisoned[:] = False
        after = scan_feature_bundle(bundle, k=5, batch_size=32)
        self.assertEqual(set(before["tracks"]), {"backdoor", "label_inconsistency"})
        self.assertFalse(before["settings"]["automatic_cleaning"])
        self.assertFalse(before["settings"]["truth_used_for_scoring"])
        for track in before["tracks"]:
            np.testing.assert_array_equal(
                before["tracks"][track]["candidate_flags"],
                after["tracks"][track]["candidate_flags"],
            )
        json.dumps(to_jsonable(before), allow_nan=False)

    def test_cli_accepts_program_style_keys(self):
        features, labels, ids = self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "program-output.npz"
            output = Path(folder) / "scan.json"
            np.savez_compressed(source, X=features, y=labels, ids=ids)
            completed = subprocess.run(
                [sys.executable, "-m", "experiments.scan_feature_bundle", str(source),
                 "--dataset-name", "custom", "--modality", "text", "--encoder", "mine",
                 "--track", "backdoor", "--output", str(output)],
                check=True, capture_output=True, text=True,
            )
            result = json.loads(output.read_text(encoding="utf-8"))
        self.assertIn("Scanned 120 rows", completed.stdout)
        self.assertEqual(result["input"]["dataset_name"], "custom")
        self.assertEqual(set(result["tracks"]), {"backdoor"})
        self.assertNotIn("evaluation", result)

    def test_known_labels_are_required_for_current_tracks(self):
        features, _, ids = self.fixture()
        bundle = feature_bundle_from_arrays(features, sample_ids=ids)
        with self.assertRaisesRegex(ValueError, "known class label"):
            scan_feature_bundle(bundle)


if __name__ == "__main__":
    unittest.main()

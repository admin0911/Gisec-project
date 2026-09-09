"""Unit tests for the blended injection detector."""

import unittest
import numpy as np
from poison_features.image_inputs import ImageInputBundle
from detectors.blended_injection import (
    BlendedInjectionDetector,
    flag_samples,
    scan_blended_injection,
    DETECTOR_NAME,
    DETECTOR_VERSION,
)
from detectors.blended_injection.residual_signature import (
    _otsu_threshold,
    _high_pass,
)

TARGET_CLASS = 0
ALPHA = 0.10


def _smooth_image_batch(rng, n_samples, size=32, coarse=4):
    """Generate image-like data: smooth, low-frequency, with fine grain.

    Real photographs are dominated by low-frequency content - adjacent pixels
    are strongly correlated. Uniform random noise is not a valid stand-in for
    an image dataset and would misrepresent any frequency-domain detector,
    since high-pass filtering relies on there being smooth content to remove.
    """
    blocks = rng.rand(n_samples, 3, coarse, coarse).astype(np.float32)
    scale = size // coarse
    images = np.repeat(np.repeat(blocks, scale, axis=2), scale, axis=3)

    for _ in range(3):  # blur away the block edges
        padded = np.pad(images, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="reflect")
        blurred = np.zeros_like(images)
        for dy in range(3):
            for dx in range(3):
                blurred += padded[:, :, dy:dy + size, dx:dx + size]
        images = blurred / 9.0

    images += 0.02 * rng.randn(n_samples, 3, size, size).astype(np.float32)
    return np.clip(images, 0.0, 1.0).astype(np.float32)


def _make_bundle(n_samples=600, poison_rate=0.05, seed=42):
    """Synthetic ImageInputBundle carrying a blended injection attack.

    Mirrors the real attack: poisoned rows are drawn from non-target classes,
    have the shared trigger blended in, and are relabelled to the target class.
    """
    rng = np.random.RandomState(seed)
    images = _smooth_image_batch(rng, n_samples)
    labels = np.array([i % 10 for i in range(n_samples)])
    sample_ids = np.arange(n_samples)

    trigger = np.clip(rng.normal(0.5, 0.2, (3, 32, 32)), 0.0, 1.0).astype(np.float32)
    n_poison = int(round(n_samples * poison_rate))
    is_poisoned = np.zeros(n_samples, dtype=bool)

    if n_poison > 0:
        candidates = np.where(labels != TARGET_CLASS)[0]
        for idx in rng.choice(candidates, n_poison, replace=False):
            images[idx] = np.clip(
                (1 - ALPHA) * images[idx] + ALPHA * trigger, 0.0, 1.0)
            labels[idx] = TARGET_CLASS
            is_poisoned[idx] = True

    bundle = ImageInputBundle(images=images, labels=labels, sample_ids=sample_ids)
    return bundle, is_poisoned


class TestOutputContract(unittest.TestCase):
    """The detector must satisfy the documented detector contract."""

    @classmethod
    def setUpClass(cls):
        cls.bundle, cls.truth = _make_bundle()
        cls.result = BlendedInjectionDetector(TARGET_CLASS).score_bundle(None, cls.bundle)
        cls.scores = cls.result["scores"]

    def test_required_fields_present(self):
        self.assertIn("sample_ids", self.result)
        self.assertIn("scores", self.result)

    def test_one_score_per_sample(self):
        self.assertEqual(len(self.scores), len(self.bundle.images))
        self.assertEqual(len(self.result["sample_ids"]), len(self.bundle.images))

    def test_sample_ids_preserved_in_order(self):
        np.testing.assert_array_equal(self.result["sample_ids"], self.bundle.sample_ids)

    def test_scores_are_finite(self):
        self.assertTrue(np.isfinite(self.scores).all())

    def test_higher_means_more_suspicious(self):
        """Poisoned rows must outrank clean rows on average."""
        self.assertGreater(self.scores[self.truth].mean(),
                           self.scores[~self.truth].mean())

    def test_only_target_class_is_scored(self):
        """Blended injection relabels every poisoned row to the target class."""
        off_target = self.bundle.labels != TARGET_CLASS
        self.assertTrue(np.all(self.scores[off_target] == 0.0))


class TestDetectionQuality(unittest.TestCase):

    def _measure(self, poison_rate, n_samples=600, seed=42):
        bundle, truth = _make_bundle(n_samples, poison_rate, seed)
        result = BlendedInjectionDetector(TARGET_CLASS).score_bundle(None, bundle)
        flags = flag_samples(result, bundle.labels, TARGET_CLASS)
        tp = int(np.sum(flags & truth))
        fp = int(np.sum(flags & ~truth))
        fn = int(np.sum(~flags & truth))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return precision, recall, int(flags.sum())

    def test_clean_data_flags_nothing(self):
        """The critical case: no attack present means no rows removed.

        Any threshold rule will happily cut the tail of a clean distribution
        and report poison that does not exist. The detector must decline.
        """
        _, _, flagged = self._measure(poison_rate=0.0)
        self.assertEqual(flagged, 0, "flagged clean samples as poisoned")

    def test_detects_at_realistic_poison_rates(self):
        for rate in (0.03, 0.05, 0.10):
            with self.subTest(poison_rate=rate):
                precision, recall, _ = self._measure(rate)
                self.assertGreater(recall, 0.80, f"recall {recall:.1%}")
                self.assertGreater(precision, 0.80, f"precision {precision:.1%}")

    def test_stable_across_seeds(self):
        for seed in (1, 7, 99):
            with self.subTest(seed=seed):
                precision, recall, _ = self._measure(0.05, seed=seed)
                self.assertGreater(recall, 0.80, f"recall {recall:.1%}")
                self.assertGreater(precision, 0.80, f"precision {precision:.1%}")

    def test_declines_rather_than_guesses_when_signal_is_weak(self):
        """With too few poisoned rows the honest answer is to report nothing.

        Precision must not collapse: either the attack is found, or no rows
        are flagged. A long list of wrong guesses is the failure mode being
        guarded against here.
        """
        precision, _, flagged = self._measure(poison_rate=0.01, n_samples=300)
        if flagged:
            self.assertGreater(precision, 0.50, f"precision {precision:.1%}")


class TestPipelineOutput(unittest.TestCase):
    """The shared output connector requires sample_ids, scores and flags."""

    @classmethod
    def setUpClass(cls):
        cls.bundle, cls.truth = _make_bundle()
        cls.result = scan_blended_injection(cls.bundle, target_class=TARGET_CLASS)

    def test_carries_connector_required_keys(self):
        for key in ("sample_ids", "scores", "flags"):
            self.assertIn(key, self.result)

    def test_flags_are_boolean_one_per_sample(self):
        flags = self.result["flags"]
        self.assertEqual(flags.dtype, np.dtype(bool))
        self.assertEqual(len(flags), len(self.bundle.sample_ids))

    def test_sample_ids_match_input_order(self):
        np.testing.assert_array_equal(self.result["sample_ids"], self.bundle.sample_ids)

    def test_scores_finite_one_per_sample(self):
        scores = self.result["scores"]
        self.assertEqual(len(scores), len(self.bundle.sample_ids))
        self.assertTrue(np.isfinite(scores).all())

    def test_evidence_values_present(self):
        self.assertIn("contrast", self.result)
        self.assertIn("group_size", self.result)

    def test_detector_is_named_and_versioned(self):
        self.assertTrue(DETECTOR_NAME)
        self.assertTrue(DETECTOR_VERSION)

    def test_clean_scan_flags_nothing(self):
        bundle, _ = _make_bundle(poison_rate=0.0)
        self.assertEqual(int(scan_blended_injection(bundle)["flags"].sum()), 0)


class TestRobustness(unittest.TestCase):

    def setUp(self):
        self.detector = BlendedInjectionDetector(TARGET_CLASS)

    def test_small_input_does_not_crash(self):
        bundle, _ = _make_bundle(n_samples=20, poison_rate=0.10)
        result = self.detector.score_bundle(None, bundle)
        self.assertEqual(len(result["scores"]), 20)
        self.assertTrue(np.isfinite(result["scores"]).all())

    def test_too_few_target_rows_returns_zeros(self):
        images = _smooth_image_batch(np.random.RandomState(0), 3)
        bundle = ImageInputBundle(images=images, labels=np.array([0, 1, 2]),
                                  sample_ids=np.arange(3))
        result = self.detector.score_bundle(None, bundle)
        self.assertTrue(np.all(result["scores"] == 0.0))

    def test_no_target_class_present(self):
        images = _smooth_image_batch(np.random.RandomState(0), 12)
        labels = np.array([1 + (i % 5) for i in range(12)])
        bundle = ImageInputBundle(images=images, labels=labels,
                                  sample_ids=np.arange(12))
        result = self.detector.score_bundle(None, bundle)
        self.assertTrue(np.all(result["scores"] == 0.0))
        flags = flag_samples(result, labels, TARGET_CLASS)
        self.assertFalse(flags.any())

    def test_flags_are_boolean_and_target_only(self):
        bundle, _ = _make_bundle()
        result = self.detector.score_bundle(None, bundle)
        flags = flag_samples(result, bundle.labels, TARGET_CLASS)
        self.assertEqual(flags.dtype, np.dtype(bool))
        self.assertFalse(flags[bundle.labels != TARGET_CLASS].any())

    def test_accepts_single_argument_call(self):
        """The documented contract passes one bundle; both styles must work."""
        bundle, _ = _make_bundle()
        one = self.detector.score_bundle(bundle)["scores"]
        two = self.detector.score_bundle(None, bundle)["scores"]
        np.testing.assert_array_equal(one, two)

    def test_missing_pixels_raises_clear_error(self):
        with self.assertRaises(TypeError):
            self.detector.score_bundle(None)

    def test_repeated_runs_are_deterministic(self):
        bundle, _ = _make_bundle()
        first = self.detector.score_bundle(None, bundle)["scores"]
        second = self.detector.score_bundle(None, bundle)["scores"]
        np.testing.assert_array_equal(first, second)


class TestComponents(unittest.TestCase):

    def test_high_pass_removes_smooth_content(self):
        """A constant image has no high frequencies, so its residual is zero."""
        flat = np.full((2, 3, 32, 32), 0.5, dtype=np.float32)
        self.assertLess(np.abs(_high_pass(flat)).max(), 1e-9)

    def test_high_pass_keeps_noise(self):
        noise = np.random.RandomState(0).rand(2, 3, 32, 32).astype(np.float32)
        self.assertGreater(np.abs(_high_pass(noise)).max(), 0.1)

    def test_otsu_splits_two_clear_groups(self):
        values = np.concatenate([np.full(50, 0.1), np.full(10, 0.9)])
        threshold = _otsu_threshold(values)
        self.assertGreater(threshold, 0.1)
        self.assertLess(threshold, 0.9)

    def test_otsu_handles_constant_input(self):
        self.assertTrue(np.isfinite(_otsu_threshold(np.full(20, 0.4))))


if __name__ == "__main__":
    unittest.main()

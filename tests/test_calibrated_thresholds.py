import unittest
import numpy as np
from detectors.label_flip.thresholds import calibrated_profile
from detectors.label_flip.calibrated_pipeline import combine_calibrated_scores


class CalibratedThresholdTests(unittest.TestCase):
    def test_strict_boundaries_and_two_votes(self):
        p = calibrated_profile('dinov2')
        scores = {n: np.array([v, np.nextafter(v, 1.), v]) for n,v in p['thresholds'].items()}
        scores['knn'][2] = 1.
        result = combine_calibrated_scores(['a','b','c'], scores, encoder='dinov2')
        self.assertEqual(result['flag_count'].tolist(), [0,3,1])
        self.assertEqual(result['review_flags'].tolist(), [False,True,False])

    def test_resnet_knn_is_inactive(self):
        result = combine_calibrated_scores(['a'],
            {'knn': [1.], 'class_distance': [.1], 'confident_learning': [.99]}, encoder='resnet18')
        self.assertFalse(result['detectors']['knn']['flags'][0])
        self.assertTrue(result['review_flags'][0])

    def test_profiles_are_independent_and_alias_supported(self):
        p = calibrated_profile('dinov2_vits14')
        p['thresholds']['knn'] = 0
        self.assertEqual(calibrated_profile('dinov2')['thresholds']['knn'], .8)
        with self.assertRaises(ValueError):
            calibrated_profile('mnist')

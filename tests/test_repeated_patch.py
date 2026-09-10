"""Connector and trigger-location regression checks; synthetic data is not calibration."""
import json
import unittest
from types import SimpleNamespace
import numpy as np
from poison_features.image_inputs import ImageInputBundle
from detectors.backdoor import RepeatedPatchDetector
from detectors.output_connector import to_jsonable

class RepeatedPatchTests(unittest.TestCase):
    def bundle(self, channels=1):
        rng=np.random.default_rng(18)
        images=rng.integers(0,256,(400,channels,8,8),dtype=np.uint8).astype(np.float32)/255
        return ImageInputBundle(images,np.arange(400)%10,np.array([f'row:{i}' for i in range(400)]))

    def test_clean_random_images(self):
        result=RepeatedPatchDetector().analyze(self.bundle())
        self.assertFalse(result['flags'].any())
        json.dumps(to_jsonable(result),allow_nan=False)

    def test_different_targets_positions_channels_and_sizes(self):
        for channels,size,row,col,target in [(1,2,0,0,7),(1,3,5,5,2),(3,3,0,5,9)]:
            with self.subTest(channels=channels,size=size,target=target):
                source=self.bundle(channels)
                poisoned=np.flatnonzero(source.labels!=target)[:40]
                source.images[poisoned,:,row:row+size,col:col+size]=1
                source.labels[poisoned]=target
                before=source.images.copy()
                progress=[]
                result=RepeatedPatchDetector().analyze(source,lambda done,total:progress.append((done,total)))
                self.assertTrue(result['flags'][poisoned].all())
                # Smaller patch fragments can also match clean rows; do not promise perfect precision.
                self.assertLess(result['flags'].sum(),len(source.labels))
                np.testing.assert_array_equal(result['sample_ids'],source.sample_ids)
                np.testing.assert_array_equal(before,source.images)
                self.assertEqual(progress[-1][0],progress[-1][1])
                self.assertTrue(any(p['dominant_label']==target for p in result['evidence']['patterns']))

    def test_common_background_is_not_flagged(self):
        source=self.bundle();source.images[:]=0
        self.assertFalse(RepeatedPatchDetector().analyze(source)['flags'].any())

    def test_connector_rejects_misalignment(self):
        source=self.bundle()
        features=SimpleNamespace(sample_ids=source.sample_ids[::-1],labels=source.labels)
        with self.assertRaises(ValueError): RepeatedPatchDetector().score_bundle(features,source)
        source.sample_ids[1]=source.sample_ids[0]
        with self.assertRaises(ValueError): RepeatedPatchDetector().analyze(source)

    def test_invalid_settings(self):
        with self.assertRaises(ValueError): RepeatedPatchDetector(patch_sizes=(4,)).analyze(self.bundle())

    def test_exact_rgb_separates_bright_distractors(self):
        source=self.bundle(3)
        source.images[:40,:,0:3,0:3]=1
        source.labels[:40]=7
        source.images[40:80,:,0:3,0:3]=210/255
        source.labels[40:80]=np.arange(40)%10
        coarse=RepeatedPatchDetector().analyze(source)
        exact=RepeatedPatchDetector(intensity_bins=256).analyze(source)
        self.assertFalse(coarse['flags'][:40].any())
        self.assertTrue(exact['flags'][:40].all())
        self.assertFalse(exact['flags'][40:80].any())
        self.assertEqual(exact['settings']['intensity_bins'],256)

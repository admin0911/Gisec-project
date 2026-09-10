import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from poison_features import ImageInputBundle
from detectors.label_flip import web_scan, web_mnist
from cleaning import human_review


class WebMNISTTests(unittest.TestCase):
    def test_pixel_connector_assessment_restoration_and_reviews(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            feature = root/'mnist-train-30-resnet18-label_flip-features.npz'
            ids = np.array([f'mnist-train:{i}' for i in range(30)])
            labels = np.arange(30)%3
            pixels = np.ones((30,1,28,28),dtype='float32')*.5
            # Encoder vectors and conflicting poison identities must never enter the detector.
            np.savez(feature,sample_ids=ids,labels=labels,features=np.full((30,512),999),is_poisoned=np.ones(30))
            image = feature.with_name(feature.name.replace('-features','-images'))
            ImageInputBundle(pixels,labels,ids).save(image)
            self.assertEqual(web_scan.feature_pair(str(feature),root),(feature,))
            self.assertEqual(len(web_scan.saved_feature_pairs(root)),1)
            def score(inputs,**kwargs):
                np.testing.assert_array_equal(inputs.X,pixels.reshape(30,784))
                np.testing.assert_array_equal(inputs.y,labels)
                self.assertFalse(hasattr(inputs,'is_poisoned'))
                result = {}
                for index,name in enumerate(('knn','class_distance','confident_learning')):
                    flags = np.zeros(30,dtype=bool); flags[:index+1]=True
                    result[name]=dict(detector_name=name,version='1.0',sample_ids=ids,
                        scores=flags.astype(float),flags=flags,settings={},evidence={})
                return result,{}
            job = 'b'*32
            with patch.object(web_mnist,'scan_pixels',side_effect=score):
                result=web_mnist.run(feature,root/'label_flip_scans'/job,lambda *args:None)
            self.assertEqual(result['summary'],dict(not_flagged=27,uncertain=1,suspected_label_flip=2))
            self.assertTrue(result['training_enabled'])
            with patch.object(human_review,'ARTIFACTS',root):
                restored=human_review.restored_scan_job(job)
                self.assertEqual(restored['result'],result)
                review=human_review.review_page(job,'suspected_label_flip')
                self.assertEqual(review['items'][0]['class_name'],'0')
                self.assertEqual(review['items'][0]['pixel_votes'],3)
                self.assertIsNone(review['items'][0]['resnet_votes'])
            np.savez(feature,sample_ids=ids,labels=labels+1)
            with self.assertRaisesRegex(ValueError,'do not match'):
                web_mnist.run(feature,root/'different',lambda *args:None)


if __name__=='__main__': unittest.main()

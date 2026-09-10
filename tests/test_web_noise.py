import json
import unittest
from unittest.mock import patch
import numpy as np
from poison_features import ImageInputBundle
from detectors.blended_injection.web_noise import scan_noise
from detectors.output_connector import detector_result, to_jsonable
from cleaning.review_assessment import review_assessment
from cleaning.selection import merge_choices

class WebNoiseTests(unittest.TestCase):
    def test_connector_review_and_restoration_both_datasets(self):
        for dataset in ('mnist', 'cifar10'):
            ids = np.array([f'{dataset}-train:{i}' for i in range(4)])
            images = ImageInputBundle(np.zeros((4,1,28,28),np.float32),np.array([0,0,1,1]),ids)
            raw = dict(sample_ids=ids,scores=np.array([.9,.8,0.,0.]),flags=np.array([True,True,False,False]),class_ratio=float('inf'),class_contrasts={0:2.0},method='background-lift')
            result = detector_result('blended-injection-residual','1.1.0',raw,{},expected_sample_ids=ids)
            with patch('detectors.blended_injection.web_noise.scan_as_connector_result',return_value=result) as scanner:
                result, ui = scan_noise(images,lambda *args:None)
                scanner.assert_called_once_with(images)
            self.assertEqual(json.loads(json.dumps(to_jsonable(ui),allow_nan=False)),to_jsonable(ui))
            scan = dict(dataset=dataset,blended_scan=result,assessment=dict(sample_ids=ids.tolist(),assessment=['not_flagged']*4))
            assessment = review_assessment(scan)
            self.assertEqual(assessment['summary']['uncertain'],2)
            selection = merge_choices(assessment,{'decisions':{ids[0]:{'decision':'keep'}}})
            self.assertEqual(selection['summary']['kept'],3)
            self.assertEqual(selection['summary']['quarantined'],1)

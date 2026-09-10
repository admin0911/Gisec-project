import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from cleaning.scan_evaluation import evaluate_scan

class ScanEvaluationTests(unittest.TestCase):
    def test_metrics_alignment_and_clean_recall(self):
        with tempfile.TemporaryDirectory() as folder:
            image=Path(folder)/'mnist-train-images.npz'
            truth=image.with_name('mnist-train-evaluation.npz')
            ids=['mnist-train:0','mnist-train:1','mnist-train:2']
            data=dict(dataset='mnist',assessment=dict(sample_ids=ids,
                assessment=['suspected_label_flip','suspected_label_flip','not_flagged']),
                scans={'pixels':{'detectors':{}}})
            with patch('cleaning.scan_evaluation.record',return_value=(None,data,None)), patch('cleaning.scan_evaluation.image_path',return_value=image):
                np.savez(truth,sample_ids=ids,is_poisoned=[1,0,1])
                row=evaluate_scan('job')['rows'][0]
                self.assertEqual(row['precision'],.5)
                self.assertEqual(row['recall'],.5)
                np.savez(truth,sample_ids=ids,is_poisoned=[0,0,0])
                self.assertIsNone(evaluate_scan('job')['rows'][0]['recall'])
                np.savez(truth,sample_ids=ids[::-1],is_poisoned=[0,0,0])
                with self.assertRaisesRegex(ValueError,'IDs'):
                    evaluate_scan('job')

    def test_uncertain_excluded_and_human_decisions_override(self):
        with tempfile.TemporaryDirectory() as folder:
            image=Path(folder)/'mnist-train-images.npz'
            ids=['mnist-train:0','mnist-train:1','mnist-train:2']
            np.savez(image.with_name('mnist-train-evaluation.npz'),sample_ids=ids,is_poisoned=[1,0,0])
            data=dict(dataset='mnist',assessment=dict(sample_ids=ids,
                assessment=['suspected_label_flip','uncertain','not_flagged']),
                scans={'pixels':{'detectors':{}}})
            with patch('cleaning.scan_evaluation.record',return_value=(image,data,'hash')), patch('cleaning.scan_evaluation.image_path',return_value=image), patch('cleaning.scan_evaluation._reviews',return_value={'decisions':{}}) as reviews:
                result=evaluate_scan('job')
                self.assertEqual(result['removed'],dict(clean=1,poisoned=1,total=2))
                self.assertEqual(result['kept']['total'],1)
                reviews.return_value={'decisions':{ids[0]:{'decision':'keep'},ids[1]:{'decision':'quarantine'}}}
                result=evaluate_scan('job')
                self.assertEqual(result['removed'],dict(clean=1,poisoned=0,total=1))

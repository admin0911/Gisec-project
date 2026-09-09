import json
import unittest
from unittest.mock import patch
import numpy as np

import test_training_preparation
from training import preparation, demo_evaluation


class DemoEvaluationTests(unittest.TestCase):
    setUp = test_training_preparation.PreparationTests.setUp

    def evaluate(self):
        with patch.object(demo_evaluation,'ARTIFACTS',self.root):
            return demo_evaluation.evaluate_prepared(self.version)

    def write_truth(self, mask, metadata=None, ids=None):
        np.savez_compressed(self.feature,sample_ids=self.ids if ids is None else ids,
            labels=self.labels,is_poisoned=mask,metadata=np.array(metadata or {},dtype=object))

    def prepare(self):
        self.version=preparation.prepare_dataset(self.job)['version']

    def test_mixed_truth_counts_frozen_selection_and_never_changes_it(self):
        mask=np.zeros(20,dtype=bool); mask[[0,11,12]]=True
        self.write_truth(mask); self.prepare()
        before=preparation.load_preparation(self.version)
        result=self.evaluate()
        self.assertTrue(result['available'])
        self.assertEqual(result['original'],dict(clean=17,poisoned=3,total=20))
        self.assertEqual(result['kept'],dict(clean=10,poisoned=1,total=11))
        self.assertEqual(result['removed'],dict(clean=7,poisoned=2,total=9))
        self.write_truth(~mask)
        self.assertEqual(self.evaluate()['original']['poisoned'],17)
        self.assertEqual(preparation.load_preparation(self.version),before)

    def test_explicit_clean_metadata_and_missing_truth(self):
        self.write_truth(np.empty(0,dtype=bool),dict(attack='none',poison_rate=0.,poison_count=0))
        self.prepare()
        self.assertEqual(self.evaluate()['removed'],dict(clean=9,poisoned=0,total=9))
        self.write_truth(np.empty(0,dtype=bool))
        self.assertFalse(self.evaluate()['available'])

    def test_invalid_mask_misaligned_ids_and_conflicting_count(self):
        self.write_truth(np.zeros(20,dtype=bool)); self.prepare()
        for mask,metadata,ids in [
            (np.zeros(19,dtype=bool),{},None),
            (np.full(20,2),{},None),
            (np.zeros(20,dtype=bool),{'poison_count':1},None),
            (np.zeros(20,dtype=bool),{},self.ids[::-1]),
        ]:
            self.write_truth(mask,metadata,ids)
            self.assertFalse(self.evaluate()['available'])

    def test_changed_scan_or_source_cannot_supply_truth(self):
        self.write_truth(np.zeros(20,dtype=bool)); self.prepare()
        self.source.write_bytes(b'changed')
        self.assertFalse(self.evaluate()['available'])
        scan=json.loads(self.scan.read_text()); scan['changed']=True
        self.scan.write_text(json.dumps(scan))
        self.assertFalse(self.evaluate()['available'])


if __name__ == '__main__': unittest.main()

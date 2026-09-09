import unittest

import numpy as np

from cleaning import decide_label_flip_actions, partition_dataset
from detectors.label_flip.assessment import assess_label_flips, DETECTORS


def scan(ids, counts):
    return {'sample_ids': np.array(ids),
            'combination_votes': {name: np.array(counts) > i for i,name in enumerate(DETECTORS)},
            'combination_settings': {'preset': 'test'}}


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.assessment = assess_label_flips(scan(['a','b','c','d'], [0,2,3,0]),
                                             scan(['a','b','c','d'], [1,3,0,2]))

    def test_review_off_by_default_and_partition_keeps_current_labels(self):
        result = decide_label_flip_actions(self.assessment)
        self.assertEqual(result['actions'].tolist(), ['keep','quarantine','quarantine','quarantine'])
        self.assertEqual(result['summary']['human_review'], 0)
        self.assertFalse(result['settings']['human_review_enabled'])
        dataset = [('pixels-d',9),('pixels-c',7),('pixels-b',3),('pixels-a',1)]
        parts = partition_dataset(dataset, ['d','c','b','a'], result)
        self.assertEqual(parts['keep'][0], ('pixels-a',1))
        self.assertEqual(len(parts['quarantine']), 3)
        self.assertEqual(len(dataset), 4)
        self.assertNotIn('actions', self.assessment)

    def test_review_can_be_enabled_later(self):
        result = decide_label_flip_actions(self.assessment, human_review_enabled=True)
        self.assertEqual(result['actions'].tolist(), ['keep','quarantine','human_review','human_review'])
        self.assertEqual(result['training_keep_mask'].tolist(), [True,False,False,False])

    def test_invalid_state_and_string_switch_rejected(self):
        with self.assertRaises(ValueError):
            decide_label_flip_actions(self.assessment, human_review_enabled='false')
        self.assessment['assessment'][0] = 'unknown'
        with self.assertRaises(ValueError):
            decide_label_flip_actions(self.assessment)


if __name__ == '__main__':
    unittest.main()

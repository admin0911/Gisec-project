import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cleaning.label_flip import NAMES, route_label_flips, partition_dataset, save_decisions


def scan(ids, counts):
    return dict(sample_ids=np.array(ids),
                combination_votes={name: np.array(counts) > i for i, name in enumerate(NAMES)},
                combination_settings={'preset': 'unit_test'})


class CleaningTests(unittest.TestCase):
    def test_primary_pending_and_original_scan_unchanged(self):
        primary = scan(['a','b','c','d'], [0,1,2,3])
        result = route_label_flips(primary)
        self.assertEqual(result['actions'].tolist(), ['keep','second_check','second_check','quarantine'])
        self.assertEqual(result['needs_second_opinion'].tolist(), [False,True,True,False])
        self.assertNotIn('actions', primary)

    def test_secondary_joins_ids_and_all_outcomes(self):
        primary = scan(['a','b','c','d','e','f'], [0,1,2,1,2,3])
        second = scan(['e','d','c','b','a','f'], [3,2,1,0,3,0])
        result = route_label_flips(primary, second)
        self.assertEqual(result['actions'].tolist(), ['keep','keep','human_review','human_review','quarantine','quarantine'])
        # Unrequested rows ignore second-stage scores, even if they disagree.
        self.assertEqual(result['secondary_vote_count'].tolist(), [-1,0,1,2,3,-1])

    def test_bad_or_incomplete_votes_rejected(self):
        primary = scan(['a','b'], [1,2])
        with self.assertRaises(ValueError):
            route_label_flips(primary, scan(['a'], [1]))
        with self.assertRaises(ValueError):
            route_label_flips(scan(['a','a'], [0,1]))
        primary['combination_votes']['knn'] = np.array([0,1])
        with self.assertRaises(ValueError):
            route_label_flips(primary)

    def test_partition_preserves_pixels_labels_and_original(self):
        dataset = [('original-b-pixels', 7), ('original-a-pixels', 1), ('original-c-pixels', 9)]
        result = route_label_flips(scan(['a','b','c'], [0,3,1]))
        parts = partition_dataset(dataset, ['b','a','c'], result)
        self.assertEqual(parts['keep'][0], ('original-a-pixels', 1))
        self.assertEqual(parts['quarantine'][0], ('original-b-pixels', 7))
        self.assertEqual(parts['second_check'][0], ('original-c-pixels', 9))
        self.assertEqual(len(dataset), 3)
        with self.assertRaises(ValueError):
            partition_dataset(dataset, ['b','a','unknown'], result)
        with tempfile.TemporaryDirectory() as tmp:
            folder = save_decisions(result, tmp, source_description='test dataset')
            saved = json.loads((folder / 'decisions.json').read_text())
            self.assertEqual(saved['actions'], result['actions'].tolist())
            np.testing.assert_array_equal(np.load(folder / 'keep_ids.npy', allow_pickle=False), ['a'])


if __name__ == '__main__':
    unittest.main()

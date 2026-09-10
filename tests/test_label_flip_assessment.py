import json
import unittest

import numpy as np

from detectors.label_flip.assessment import assess_label_flips, DETECTORS
from detectors.output_connector import to_jsonable


def scan(ids, counts):
    return dict(sample_ids=np.asarray(ids),
                combination_votes={name: np.asarray(counts) > i for i, name in enumerate(DETECTORS)},
                combination_settings={'test_thresholds': True})


class AssessmentTests(unittest.TestCase):
    def test_all_sixteen_vote_count_pairs(self):
        pairs = [(a,b) for a in range(4) for b in range(4)]
        ids = np.arange(16)
        a, b = scan(ids, [x for x,y in pairs]), scan(ids, [y for x,y in pairs])
        result = assess_label_flips(a,b)
        for i, (x,y) in enumerate(pairs):
            expected = ('suspected_label_flip' if x>=2 and y>=2 else
                        'uncertain' if (x>=2)!=(y>=2) else 'not_flagged')
            self.assertEqual(result['assessment'][i], expected)
            self.assertEqual(result['flags'][i], x>=2 or y>=2)
        self.assertEqual(sum(result['summary'].values()), 16)
        json.dumps(to_jsonable(result), allow_nan=False)
        result['detector_votes']['resnet18']['knn'][:] = False
        self.assertTrue(a['combination_votes']['knn'][-1])

    def test_alignment_is_by_id(self):
        result = assess_label_flips(scan(['b','a','c'],[2,0,1]), scan(['c','b','a'],[2,3,0]))
        self.assertEqual(result['sample_ids'].tolist(), ['b','a','c'])
        self.assertEqual(result['assessment'].tolist(), ['suspected_label_flip','not_flagged','uncertain'])
        self.assertEqual(result['vote_counts']['dinov2'].tolist(), [3,0,2])

    def test_incomplete_or_invalid_scans_fail(self):
        good = scan(['a','b'],[0,2])
        for bad in (scan(['a'],[0]), scan(['a','a'],[0,0]), scan(['a','x'],[0,0])):
            with self.assertRaises(ValueError):
                assess_label_flips(good, bad)
        bad = scan(['a','b'],[0,2])
        bad['combination_votes']['knn'] = np.array([0,1])
        with self.assertRaises(ValueError):
            assess_label_flips(good, bad)
        del bad['combination_votes']['knn']
        with self.assertRaises(ValueError):
            assess_label_flips(good, bad)

    def test_truth_and_cached_flag_counts_do_not_drive_assessment(self):
        a,b = scan([0,1],[0,2]),scan([0,1],[0,2])
        a.update(is_poisoned=np.ones(2), flag_count=np.array([3,0]))
        result = assess_label_flips(a,b)
        self.assertEqual(result['assessment'].tolist(), ['not_flagged','suspected_label_flip'])
        self.assertNotIn('is_poisoned', result)


if __name__ == '__main__':
    unittest.main()

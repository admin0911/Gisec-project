import unittest
from types import SimpleNamespace
import numpy as np
from detectors.runner import run_detectors
from detectors.output_connector import detector_result
from cleaning.connector import combine_decisions
from cleaning.label_flip import partition_dataset


class TeamConnectorTests(unittest.TestCase):
    def test_custom_detector_and_progress(self):
        inputs = SimpleNamespace(sample_ids=np.array(['a', 'b']))
        def custom(x):
            return detector_result('custom', '1', dict(sample_ids=x.sample_ids,
                scores=np.array([0.1, 0.9]), flags=np.array([False, True])),
                {'threshold': 0.8}, expected_sample_ids=x.sample_ids)
        progress = []
        result = run_detectors(inputs, {'custom': custom}, progress=progress.append)
        self.assertEqual(progress, ['1 of 1: custom'])
        self.assertTrue(result['detectors']['custom']['flags'][1])
        with self.assertRaises(ValueError):
            run_detectors(inputs, {'wrong': custom})

    def test_failed_detector_is_not_hidden(self):
        def fail(_):
            raise RuntimeError('failed')
        with self.assertRaises(RuntimeError):
            run_detectors(SimpleNamespace(sample_ids=np.array([1])), {'fail': fail})

    def decision(self, ids, actions):
        return dict(sample_ids=np.array(ids), actions=np.array(actions),
                    settings={'policy': 'test'})

    def test_id_join_and_training_partition(self):
        checks = {'label_flip': self.decision(['a', 'b', 'c'], ['keep', 'human_review', 'keep']),
                  'backdoor': self.decision(['c', 'b', 'a'], ['quarantine', 'keep', 'keep'])}
        result = combine_decisions(['a', 'b', 'c'], checks, required_checks=list(checks))
        self.assertEqual(result['actions'].tolist(), ['keep', 'human_review', 'quarantine'])
        views = partition_dataset([10, 20, 30], ['a', 'b', 'c'], result)
        self.assertEqual(views['keep'][0], 10)
        self.assertEqual(len(views['keep']), 1)

    def test_missing_partial_duplicate_and_invalid_checks_rejected(self):
        good = self.decision(['a', 'b'], ['keep', 'keep'])
        with self.assertRaises(ValueError):
            combine_decisions(['a', 'b'], {'one': good}, required_checks=['one', 'two'])
        for bad in [self.decision(['a'], ['keep']),
                    self.decision(['a', 'a'], ['keep', 'keep']),
                    self.decision(['a', 'b'], ['keep', 'clean'])]:
            with self.assertRaises(ValueError):
                combine_decisions(['a', 'b'], {'one': bad}, required_checks=['one'])

    def test_quarantine_wins_but_other_evidence_remains(self):
        result = combine_decisions([1], {
            'one': self.decision([1], ['human_review']),
            'two': self.decision([1], ['quarantine'])}, required_checks=['one', 'two'])
        self.assertEqual(result['actions'][0], 'quarantine')
        self.assertEqual(result['check_actions']['one'][0], 'human_review')


if __name__ == '__main__':
    unittest.main()

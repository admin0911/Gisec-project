import unittest
from importlib.util import find_spec
from unittest.mock import patch
import numpy as np
from poison_features import DetectorInput
from detectors.label_flip.confident_learning import ConfidentLearning


@unittest.skipUnless(find_spec('cleanlab'), 'Install requirements.txt')
class ConfidentLearningTests(unittest.TestCase):
    def test_fold_exclusion_and_alignment(self):
        seen = []
        class Model:
            def __init__(self, **kwargs):
                pass
            def fit(self, X, y):
                self.train = X.copy()
                self.classes_ = np.unique(y)
                self.n_iter_ = [1]
                return self
            def predict_proba(self, X):
                for row in X:
                    assert not np.any(np.all(self.train == row, axis=1))
                seen.extend(X.tolist())
                return np.tile([.6, .4], (len(X), 1))
        X = np.column_stack([np.ones(30), np.arange(1, 31)])
        inputs = DetectorInput(X, np.arange(100, 130), np.arange(30) % 2)
        with patch('sklearn.linear_model.LogisticRegression', Model):
            result = ConfidentLearning(folds=3).analyze(inputs)
        self.assertEqual(len(seen), 30)
        np.testing.assert_array_equal(result['sample_ids'], inputs.sample_ids)
        self.assertEqual(len(np.unique(result['fold_ids'])), 3)
        np.testing.assert_allclose(result['pred_probs'].sum(axis=1), 1)
        self.assertTrue(np.isfinite(result['scores']).all())

    def test_real_classifier_and_invalid_labels(self):
        rng = np.random.default_rng(7)
        X = np.vstack([rng.normal([3, 0], .1, (30, 2)), rng.normal([0, 3], .1, (30, 2))])
        y = np.repeat([0, 1], 30)
        y[0] = 1
        result = ConfidentLearning(folds=3).analyze(DetectorInput(X, np.arange(60), y))
        self.assertTrue(result['flags'][0])
        with self.assertRaises(ValueError):
            ConfidentLearning().analyze(DetectorInput(X, np.arange(60)))
        with self.assertRaises(ValueError):
            ConfidentLearning().analyze(DetectorInput(X[:6], np.arange(6), [0,0,0,1,1,1]))


if __name__ == '__main__':
    unittest.main()

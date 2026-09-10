import tempfile
import unittest
from pathlib import Path
import numpy as np
from poison_features import FeatureBundle
from detectors.label_flip.web_imdb import run
from detectors.label_flip.web_scan import feature_pair

class IMDBTests(unittest.TestCase):
    def test_real_scoring_connector_and_demo(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); path=root/'imdb-train-30-minilm-label_flip-features.npz'
            rng=np.random.default_rng(42); labels=np.arange(30)%2
            x=rng.normal(size=(30,384)).astype('float32'); x[:,0]+=labels*10
            FeatureBundle(x,x,x[:,:2],labels,np.array([f'imdb-train:{i}' for i in range(30)]),
                'text','sentence-transformers/all-MiniLM-L6-v2','imdb',is_poisoned=np.zeros(30,dtype=bool)).save(path)
            self.assertEqual(feature_pair(str(path),root),(path,))
            result=run(path,root/'scan',lambda *args:None)
            self.assertEqual(result['dataset'],'imdb')
            self.assertEqual(len(result['detectors']),3)
            self.assertEqual(sum(result['summary'].values()),30)
            self.assertIsNone(result['demo_evaluation']['recall'])
            self.assertTrue(result['training_enabled'])

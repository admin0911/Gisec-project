import unittest
import numpy as np
from experiments.calibrate_imdb_thresholds import thresholds,flags,metrics,NAMES
class IMDBCalibrationTests(unittest.TestCase):
    def test_budget_ties_and_cleanlab_gate(self):
        r={n:dict(scores=np.linspace(0,1,1200),flags=np.ones(1200,bool)) for n in NAMES}
        cuts=thresholds(r,.01)
        self.assertLessEqual(np.logical_or.reduce(list(flags(r,cuts).values())).mean(),.01)
        for n in NAMES:r[n]['scores'][:]=cuts[n]
        self.assertFalse(np.logical_or.reduce(list(flags(r,cuts).values())).any())
        r['confident_learning']['scores'][:]=1;r['confident_learning']['flags'][:]=False
        self.assertFalse(flags(r,cuts)['confident_learning'].any())
    def test_metrics(self):
        m=metrics(np.array([1,1,0,0],bool),np.array([1,0,1,0],bool))
        self.assertEqual((m['tp'],m['fp'],m['fn'],m['tn']),(1,1,1,1))
if __name__=='__main__':unittest.main()

import json
import unittest
import numpy as np
from poison_features.image_inputs import ImageInputBundle
from detectors.blended_injection import ConsensusPixelDetector
from detectors.output_connector import to_jsonable

class ConsensusPixelTests(unittest.TestCase):
    def bundle(self,n=100):
        return ImageInputBundle(np.zeros((n,1,28,28),np.float32),np.arange(n)%10,np.array([f'mnist-train:{i}' for i in range(n)]))
    def test_connector_and_label_independence(self):
        b=self.bundle();b.images[:5]=.1;original=b.images.copy();progress=[]
        r=ConsensusPixelDetector().analyze(b,progress=lambda a,z:progress.append((a,z)))
        self.assertEqual(set(r),{'detector_name','version','sample_ids','scores','flags','settings','evidence'})
        np.testing.assert_array_equal(r['flags'],np.arange(100)<5)
        np.testing.assert_array_equal(r['scores'],(np.arange(100)<5).astype(float))
        np.testing.assert_array_equal(r['sample_ids'],b.sample_ids)
        np.testing.assert_array_equal(b.images,original)
        b.labels[:]=9;other=ConsensusPixelDetector().analyze(b)
        np.testing.assert_array_equal(r['scores'],other['scores']);self.assertEqual(progress,[(1,1)])
        json.dumps(to_jsonable(r),allow_nan=False)
    def test_clean_and_inconclusive_are_distinct(self):
        b=self.bundle();clean=ConsensusPixelDetector().analyze(b)
        self.assertTrue(clean['evidence']['applicable']);self.assertFalse(clean['flags'].any())
        b.images[:20]=.1;r=ConsensusPixelDetector().analyze(b)
        self.assertFalse(r['evidence']['applicable']);self.assertIn('inconclusive',r['evidence']['status']);self.assertFalse(r['flags'].any())
    def test_faint_changes_missed_benign_changes_flagged(self):
        b=self.bundle();b.images[:5]=.002
        self.assertFalse(ConsensusPixelDetector().analyze(b)['flags'].any())
        b.images[:5]=.03
        self.assertEqual(int(ConsensusPixelDetector().analyze(b)['flags'].sum()),5)
    def test_validation(self):
        b=self.bundle();b.images[0]=2
        with self.assertRaisesRegex(ValueError,'pixels'):ConsensusPixelDetector().analyze(b)
        b=self.bundle();b.sample_ids[1]=b.sample_ids[0]
        with self.assertRaisesRegex(ValueError,'unique'):ConsensusPixelDetector().analyze(b)
        for settings in ({'consensus':.5},{'flag_fraction':1},{'min_consensus_pixels':True},{'tolerance':float('nan')}):
            with self.assertRaises(ValueError):ConsensusPixelDetector(**settings).analyze(self.bundle())

    def test_shared_review_routing(self):
        from cleaning.review_assessment import review_assessment
        from cleaning.selection import merge_choices
        b=self.bundle();b.images[:5]=.1;r=ConsensusPixelDetector().analyze(b)
        scan={'dataset':'mnist','assessment':{'sample_ids':b.sample_ids.tolist(),'assessment':['not_flagged']*100},'blended_scan':r}
        a=review_assessment(scan)
        self.assertEqual(a['summary']['uncertain'],5)
        selection=merge_choices(a,{'decisions':{}},keep_uncertain=False)
        self.assertEqual(selection['summary']['quarantined'],5)
        selection=merge_choices(a,{'decisions':{str(b.sample_ids[0]):{'decision':'keep'}}},keep_uncertain=False)
        self.assertEqual(selection['summary']['kept'],96)
        r['evidence']['applicable']=False
        self.assertEqual(review_assessment(scan)['summary']['uncertain'],0)

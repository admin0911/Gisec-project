import unittest
import numpy as np
from poison_features.text_inputs import TextInputBundle
from detectors.backdoor.repeated_phrase import RepeatedPhraseDetector

class PhraseTests(unittest.TestCase):
    def test_unknown_phrases_and_both_targets(self):
        for phrase,target in [('amber comet',0),('violet compass',1)]:
            texts=[f'unique{i} review{i} ending{i}' for i in range(200)]
            labels=np.arange(200)%2
            clean=TextInputBundle(tuple(texts),labels,np.arange(200))
            self.assertFalse(RepeatedPhraseDetector().analyze(clean)['flags'].any())
            for i in range(20): texts[i]=phrase+' '+texts[i]; labels[i]=target
            result=RepeatedPhraseDetector().analyze(TextInputBundle(tuple(texts),labels,np.arange(200)))
            self.assertEqual(np.flatnonzero(result['flags']).tolist(),list(range(20)))
            self.assertEqual(result['sample_ids'].tolist(),list(range(200)))
    def test_common_phrase_across_labels_not_flagged(self):
        texts=tuple('common phrase '+str(i) if i<20 else 'unique '+str(i) for i in range(200))
        result=RepeatedPhraseDetector().analyze(TextInputBundle(texts,np.arange(200)%2,np.arange(200)))
        self.assertFalse(result['flags'].any())
    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError): TextInputBundle(('a','b'),np.array([0,1]),np.array([1,1]))

import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from cleaning import human_review
from cleaning.review_assessment import review_assessment
from cleaning.selection import merge_choices
from poison_features import ImageInputBundle

class CombinedReviewTests(unittest.TestCase):
    def test_both_sources_review_save_and_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);job='a'*32;folder=root/'label_flip_scans'/job;folder.mkdir(parents=True)
            ids=['mnist-train:'+str(i) for i in range(4)]
            feature=root/'mnist-train-test-features.npz'
            ImageInputBundle(np.zeros((4,1,8,8),dtype=np.float32),np.arange(4),np.array(ids)).save(feature.with_name(feature.name.replace('features','images')))
            scan=dict(dataset='mnist',feature_files=[str(feature)],assessment=dict(sample_ids=ids,assessment=['not_flagged','uncertain','suspected_label_flip','not_flagged'],vote_counts={'pixels':[0,1,3,0]},flags=[False,True,True,False]),patch_scan=dict(sample_ids=ids,flags=[True,True,True,False]))
            (folder/'results.json').write_text(json.dumps(scan))
            with patch.object(human_review,'ARTIFACTS',root):
                page=human_review.review_page(job)
                self.assertEqual([i['sample_id'] for i in page['items']],ids[:2])
                self.assertTrue(page['items'][0]['patch_flagged'])
                self.assertEqual(human_review.review_summary(job)['total'],3)
                self.assertEqual(human_review.review_page(job,'suspected_label_flip')['total'],1)
                human_review.save_review(job,{ids[0]:'keep',ids[1]:'unsure',ids[2]:'quarantine'},0)
                path,_,digest=human_review.record(job)
                saved=human_review._reviews(path,digest)
                self.assertEqual(merge_choices(review_assessment(scan),saved)['actions'],['keep','human_review','quarantine','keep'])
                self.assertEqual(scan['assessment']['assessment'][0],'not_flagged')
                self.assertEqual(merge_choices(review_assessment(scan),{'decisions':{}})['actions'][0],'human_review')
    def test_bad_patch_alignment_rejected(self):
        with self.assertRaises(ValueError):
            review_assessment({'assessment':{'sample_ids':['a']},'patch_scan':{'sample_ids':['b'],'flags':[True]}})

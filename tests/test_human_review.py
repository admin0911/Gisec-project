"""Review persistence, sample alignment and restart recovery without model reruns."""
from hashlib import sha256
from io import BytesIO
import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from cleaning import human_review as review


class HumanReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patcher = patch.object(review, 'ARTIFACTS', self.root)
        patcher.start(); self.addCleanup(patcher.stop)
        self.job = 'a'*32
        self.folder = self.root/'label_flip_scans'/self.job
        self.folder.mkdir(parents=True)
        self.ids = [f'cifar10-train:{i}' for i in range(45)]
        self.states = ['uncertain']*21 + ['suspected_label_flip']*3 + ['not_flagged']*21
        images = np.stack([np.full((3,4,4), i/255, dtype='float32') for i in range(45)])
        self.image_file = self.root/'example-images.npz'
        np.savez_compressed(self.image_file,images=images,labels=np.arange(45)%10,sample_ids=self.ids)
        a = dict(sample_ids=self.ids,assessment=self.states,vote_counts={'resnet18':[2]*45,'dinov2':[0]*45},
                 summary={name:self.states.count(name) for name in set(self.states)})
        detector = dict(flags=[False]*45, settings={'detector_threshold':.9})
        self.data = dict(feature_files=[str(self.root/'example-features.npz')],assessment=a,
            scans={'resnet18':{'detectors':{'knn':detector}}},profile={'name':'test','limitation':'test'})
        self.result_file = self.folder/'results.json'
        self.result_file.write_text(json.dumps(self.data))
        self.original_hash = sha256(self.result_file.read_bytes()).hexdigest()

    def test_pagination_and_pixels_match_sample_ids(self):
        result = review.review_page(self.job)
        self.assertEqual(len(result['items']),20)
        self.assertEqual(result['pages'],2)
        self.assertEqual(result['unreviewed'],21)
        self.assertIsNone(result['items'][0]['decision'])
        last = review.review_page(self.job,page=1)['items'][0]
        self.assertEqual(last['sample_id'],self.ids[20])
        pixels = np.asarray(Image.open(BytesIO(base64.b64decode(last['image'].split(',')[1]))))
        self.assertTrue(np.all(pixels==20))
        self.assertEqual(len(review.review_page(self.job,page_size=50)['items']),21)
        self.assertEqual(len(review.review_page(self.job,page_size=100)['items']),21)
        self.assertFalse((self.folder/'human_review.json').exists())

    def test_saved_decisions_do_not_change_evidence_and_unsure_is_unresolved(self):
        image_hash = sha256(self.image_file.read_bytes()).hexdigest()
        review.save_review(self.job,{self.ids[0]:'keep',self.ids[1]:'quarantine',self.ids[2]:'unsure'},0)
        review._record.cache_clear(); review._thumbnails.cache_clear()
        result = review.review_page(self.job)
        self.assertEqual((result['resolved'],result['unsure'],result['unreviewed']),(2,1,18))
        self.assertEqual([i['decision'] for i in result['items'][:4]],['keep','quarantine','unsure',None])
        review.save_review(self.job,{self.ids[0]:'quarantine'},1)
        saved = json.loads((self.folder/'human_review.json').read_text())
        self.assertEqual(len(saved['history']),4)
        self.assertEqual(sha256(self.result_file.read_bytes()).hexdigest(),self.original_hash)
        self.assertEqual(sha256(self.image_file.read_bytes()).hexdigest(),image_hash)

    def test_invalid_inputs_and_conflicting_saves(self):
        for kwargs in ({'page_size':1000},{'page':-1},{'page':2},{'group':'invalid'}):
            with self.assertRaises(ValueError): review.review_page(self.job,**kwargs)
        for changes in ({self.ids[25]:'keep'},{'missing':'keep'},{self.ids[0]:'delete'},{}):
            with self.assertRaises(ValueError): review.save_review(self.job,changes,0)
        with self.assertRaises(ValueError): review.result_path('../outside')
        review.save_review(self.job,{self.ids[0]:'keep'},0)
        with self.assertRaises(review.ReviewConflict):
            review.save_review(self.job,{self.ids[1]:'keep'},0)
        self.assertIsNone(review.review_page(self.job)['items'][1]['decision'])

    def test_restore_completed_scan_after_restart(self):
        result = review.restored_scan_job(self.job)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(result['result']['samples'],45)
        self.assertEqual(result['result']['summary']['uncertain'],21)

    def test_summary_counts_current_saved_choices_across_groups(self):
        empty = review.review_summary(self.job)
        self.assertEqual((empty['saved'],empty['unreviewed']),(0,24))
        review.save_review(self.job,{self.ids[0]:'keep',self.ids[1]:'unsure',self.ids[21]:'quarantine'},0)
        self.assertEqual(review.review_summary(self.job),dict(keep=1,quarantine=1,unsure=1,unreviewed=21,total=24,saved=3,revision=1))
        review.save_review(self.job,{self.ids[0]:'quarantine'},1)
        result = review.review_summary(self.job)
        self.assertEqual((result['keep'],result['quarantine'],result['saved']),(0,2,3))
        self.assertEqual(sha256(self.result_file.read_bytes()).hexdigest(),self.original_hash)

    def test_changed_evidence_rejects_old_reviews(self):
        review.save_review(self.job,{self.ids[0]:'keep'},0)
        self.data['profile']['name'] = 'changed profile'
        self.result_file.write_text(json.dumps(self.data))
        with self.assertRaises(review.ReviewConflict): review.review_page(self.job)

    def test_misaligned_image_ids_rejected(self):
        np.savez_compressed(self.image_file,images=np.zeros((45,3,4,4)),labels=np.arange(45)%10,sample_ids=self.ids[::-1])
        with self.assertRaisesRegex(ValueError,'do not match'): review.review_page(self.job)


if __name__ == '__main__': unittest.main()

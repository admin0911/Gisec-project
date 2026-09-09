from pathlib import Path
import json
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

from cleaning import human_review
from training import preparation, web_comparison
import training_api


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for module,name,value in [(human_review,'ARTIFACTS',self.root),(preparation,'ARTIFACTS',self.root),
            (preparation,'PREPARATIONS',self.root/'training_preparations'),(web_comparison,'ARTIFACTS',self.root),
            (training_api,'OUTPUT',self.root/'web_training')]:
            p=patch.object(module,name,value); p.start(); self.addCleanup(p.stop)
        self.job = 'd'*32
        folder = self.root/'label_flip_scans'/self.job; folder.mkdir(parents=True)
        self.ids = [f'cifar10-train:{i}' for i in range(20)]
        self.labels = np.arange(20)%10
        self.images = np.random.default_rng(1).random((20,3,32,32),dtype=np.float32)
        self.source = self.root/'source-images.npz'
        np.savez_compressed(self.source,images=self.images,sample_ids=self.ids,labels=self.labels)
        self.feature = self.root/'source-features.npz'
        np.savez_compressed(self.feature,sample_ids=self.ids,labels=self.labels,is_poisoned=np.ones(20))
        self.assessment = dict(sample_ids=self.ids,assessment=['not_flagged']*10+['uncertain','suspected_label_flip']*5)
        self.scan = folder/'results.json'
        self.scan.write_text(json.dumps(dict(assessment=self.assessment,feature_files=[str(self.feature)])))
        human_review.save_review(self.job,{self.ids[10]:'keep',self.ids[11]:'quarantine',self.ids[12]:'unsure'},0)

    def test_snapshot_policy_and_later_reviews(self):
        original = preparation.file_hash(self.source)
        prepared = preparation.prepare_dataset(self.job)
        self.assertEqual(prepared['summary'],dict(kept=11,quarantined=5,unresolved=4,total=20))
        manifest = preparation.load_preparation(prepared['version'])
        self.assertEqual(manifest['actions'][10:14],['keep','quarantine','human_review','quarantine'])
        self.assertEqual(manifest['reasons'][13],'scanner_suspected')
        self.assertEqual(prepared['policy_version'],'2.0')
        human_review.save_review(self.job,{self.ids[13]:'keep'},1)
        self.assertEqual(preparation.load_preparation(prepared['version'])['summary']['kept'],11)
        latest = preparation.prepare_dataset(self.job)
        self.assertNotEqual(latest['version'],prepared['version'])
        self.assertEqual(latest['summary']['kept'],12)
        self.assertEqual(latest['summary']['quarantined'],4)
        self.assertEqual(preparation.load_preparation(prepared['version'])['summary']['quarantined'],5)
        self.assertEqual(preparation.file_hash(self.source),original)

    def test_no_reviews_quarantines_suspects_and_holds_uncertain(self):
        selection = preparation.merge_choices(self.assessment,{'decisions':{}})
        self.assertEqual(selection['summary'],dict(kept=10,quarantined=5,unresolved=5,total=20))

    def test_human_unsure_overrides_suspect_and_legacy_policy_is_preserved(self):
        selection = preparation.merge_choices(self.assessment,{'decisions':{
            self.ids[11]:{'decision':'unsure'}, self.ids[13]:{'decision':'keep'}}})
        self.assertEqual(selection['summary'],dict(kept=11,quarantined=3,unresolved=6,total=20))
        self.assertEqual(selection['reasons'][11],'human_unsure')
        data = preparation.prepare_dataset(self.job)
        del data['policy_version']
        self.assertEqual(preparation.preparation_summary(data)['policy_version'],'1.0')

    def test_reject_missing_ids_and_modified_labels(self):
        with self.assertRaises(ValueError): preparation.merge_choices(self.assessment,{'decisions':{'unknown':{'decision':'keep'}}})
        np.savez_compressed(self.feature,sample_ids=self.ids,labels=(self.labels+1)%10)
        with self.assertRaisesRegex(ValueError,'no longer match'): preparation.prepare_dataset(self.job)
        with self.assertRaises(ValueError): preparation.load_preparation('../outside')

    def test_real_three_model_training_uses_clean_reference_and_only_kept_rows(self):
        version = preparation.prepare_dataset(self.job)['version']
        test = TensorDataset(torch.from_numpy(self.images[:10]),torch.from_numpy(self.labels[:10]))
        clean = TensorDataset(torch.from_numpy(self.images),torch.from_numpy((self.labels+1)%10))
        def load_dataset(*args,**kwargs):
            return clean if kwargs['train'] else test
        initial = []
        factory = web_comparison.small_cnn
        def create_model():
            model = factory(); initial.append(model[0].weight.detach().clone()); return model
        torch.set_num_threads(2)
        progress = []
        with patch.object(web_comparison,'load_image_dataset',side_effect=load_dataset) as loader, patch.object(web_comparison,'small_cnn',create_model):
            report = web_comparison.train_comparison(version,1,self.root/'comparison',lambda value,message:progress.append(value))
        self.assertEqual(report['status'],'complete')
        self.assertEqual(len(initial),3)
        self.assertTrue(all(torch.equal(initial[0],weights) for weights in initial[1:]))
        self.assertEqual(report['runs']['clean_reference']['training_samples'],20)
        self.assertEqual(report['runs']['before_cleaning']['training_samples'],20)
        self.assertEqual(report['runs']['after_cleaning']['training_samples'],11)
        ids = np.load(report['runs']['after_cleaning']['artifacts']['training_ids'])
        self.assertEqual(ids.tolist(),self.ids[:11])
        self.assertFalse(loader.call_args.kwargs['download'])
        self.assertFalse(loader.call_args.kwargs['train'])
        self.assertEqual(report['runs']['before_cleaning']['settings'],report['runs']['after_cleaning']['settings'])
        self.assertEqual(report['runs']['clean_reference']['settings'],report['runs']['before_cleaning']['settings'])
        self.assertEqual([call.kwargs['train'] for call in loader.call_args_list],[True,False])
        self.assertTrue(all(not call.kwargs['download'] for call in loader.call_args_list))
        self.assertEqual(progress,sorted(progress))

    def test_reference_matches_subset_order_and_uses_official_labels(self):
        from types import SimpleNamespace
        ids = np.array(['cifar10-train:7','cifar10-train:2'],dtype='U40')
        manifest = dict(source_images=str(self.source),source_sha256=preparation.file_hash(self.source),
            sample_ids=ids.tolist(),scan_id=self.job,review_revision=0,summary={},actions=['keep','quarantine'])
        bundle = SimpleNamespace(sample_ids=ids,images=self.images[:2],labels=np.array([9,9]))
        clean = TensorDataset(torch.from_numpy(self.images),torch.from_numpy(self.labels))
        seen = []
        def train(inputs,test,**kwargs):
            seen.append((inputs.sample_ids.tolist(),[int(inputs.dataset[i][1]) for i in range(len(inputs.dataset))]))
            return {'metrics':{'accuracy':.5}}
        with patch.object(web_comparison,'load_preparation',return_value=manifest), \
             patch.object(web_comparison.ImageInputBundle,'load',return_value=bundle), \
             patch.object(web_comparison,'load_image_dataset',return_value=clean), \
             patch.object(web_comparison,'train_classifier',side_effect=train):
            web_comparison.train_comparison('unused',1,self.root/'subset',lambda *args:None)
            self.assertEqual(seen,[(ids.tolist(),[7,2]),(ids.tolist(),[9,9]),([ids[0]],[9])])
            ids[0]='cifar10-train:99'; manifest['sample_ids']=ids.tolist()
            with self.assertRaisesRegex(ValueError,'cannot be mapped'):
                web_comparison.train_comparison('unused',1,self.root/'bad',lambda *args:None)

    def test_clean_input_skips_duplicate_model_but_pixel_changes_do_not(self):
        version=preparation.prepare_dataset(self.job)['version']
        clean=TensorDataset(torch.from_numpy(self.images),torch.from_numpy(self.labels))
        test=TensorDataset(torch.from_numpy(self.images[:10]),torch.from_numpy(self.labels[:10]))
        progress=[]
        torch.set_num_threads(2)
        with patch.object(web_comparison,'load_image_dataset',side_effect=lambda *a,**kw: clean if kw['train'] else test):
            report=web_comparison.train_comparison(version,1,self.root/'clean',lambda p,m:progress.append(m))
        self.assertEqual(set(report['runs']),{'clean_reference','after_cleaning'})
        self.assertTrue(report['before_cleaning_skipped'])
        self.assertEqual(report['accuracy_change_reference'],'clean_reference')
        self.assertEqual(report['accuracy_change'],report['runs']['after_cleaning']['metrics']['accuracy']-report['runs']['clean_reference']['metrics']['accuracy'])
        self.assertTrue(any('Model 2 of 2' in m for m in progress))
        altered=self.images.copy(); altered[0,0,0,0]=1-altered[0,0,0,0]
        self.assertFalse(web_comparison.matches_clean_reference(TensorDataset(torch.from_numpy(altered),torch.from_numpy(self.labels)),clean))

    def test_training_rejects_changed_source_and_invalid_epochs(self):
        version = preparation.prepare_dataset(self.job)['version']
        self.source.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed after preparation'):
            web_comparison.train_comparison(version,1,self.root/'unused',lambda *args:None)
        for n in (0,51,True,1.5):
            with self.assertRaises(ValueError): web_comparison.validate_epochs(n)

    def test_job_failure_and_restart_recovery(self):
        version = preparation.prepare_dataset(self.job)['version']
        with patch.object(training_api,'train_comparison',side_effect=ValueError('Training failed')):
            job = training_api.start_training(version,1)['job_id']
            for _ in range(200):
                state = training_api.get_job(job)
                if state['status']=='error' and not training_api.RUN_LOCK.locked(): break
                time.sleep(.01)
        self.assertEqual(state['status'],'error')
        with training_api.JOBS_LOCK: training_api.JOBS.pop(job)
        self.assertEqual(training_api.get_job(job)['status'],'error')
        saved = training_api.OUTPUT/job/'job.json'
        saved.write_text(json.dumps(dict(state,status='running')))
        self.assertIn('interrupted',training_api.get_job(job)['message'])


if __name__ == '__main__': unittest.main()

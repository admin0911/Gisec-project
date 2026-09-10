import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from poison_features import FeatureBundle
from training import text_comparison as module

class IMDBTrainingTests(unittest.TestCase):
    def test_three_arms_and_separate_ids(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); path=root/'imdb-train-10-minilm-label_flip-features.npz'
            x=np.random.default_rng(0).normal(size=(10,384)).astype('float32'); y=np.arange(10)%2; bad=y.copy();bad[0]=1
            ids=np.array([f'imdb-train:{i}' for i in range(10)])
            def bundle(labels,sample_ids):
                return FeatureBundle(x,x,x[:,:2],labels,sample_ids,'text','sentence-transformers/all-MiniLM-L6-v2','imdb',original_labels=y)
            bundle(bad,ids).save(path)
            bundle(y,np.array([f'imdb-test:{i}' for i in range(10)])).save(root/'imdb-test-full-minilm-training-features.npz')
            manifest=dict(version='v',scan_id='s',source_images=str(path),source_sha256=module.file_hash(path),
                sample_ids=ids.tolist(),actions=['quarantine']+['keep']*9)
            def load(**kw):
                return [{'label':int(label),'text':'review'} for label in y] if kw['split']=='train' else {'label':y,'text':['review']*10}
            class TestRows(dict):
                def __len__(self): return 10
            def data(**kw):
                return load(**kw) if kw['split']=='train' else TestRows(load(**kw))
            with patch.object(module,'ARTIFACTS',root),patch.object(module,'load_imdb_dataset',side_effect=data):
                result=module.train_text_comparison(manifest,1,root/'run',lambda *args:None)
            self.assertEqual(result['status'],'complete')
            self.assertEqual([r['training_samples'] for r in result['runs'].values()],[9,9,8])
            for run in result['runs'].values():
                test=np.load(run['artifacts']['predictions'])
                train=np.load(run['artifacts']['training_ids'])
                self.assertFalse(set(test['sample_ids']) & set(train))
                test.close()

    def test_backdoor_clean_reference_and_trigger_metrics(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); path=root/'imdb-train-10-minilm-backdoor-phrase-v1-positive-start-features.npz'
            x=np.random.default_rng(0).normal(size=(10,384)).astype('float32'); y=np.arange(10)%2; bad=y.copy();bad[0]=1
            ids=np.array([f'imdb-train:{i}' for i in range(10)])
            def bundle(labels,sample_ids):
                return FeatureBundle(x,x,x[:,:2],labels,sample_ids,'text','sentence-transformers/all-MiniLM-L6-v2','imdb',original_labels=y)
            bundle(bad,ids).save(path)
            bundle(y,np.array([f'imdb-test:{i}' for i in range(10)])).save(root/'imdb-test-full-minilm-training-features.npz')
            manifest=dict(version='v',scan_id='s',source_images=str(path),source_sha256=module.file_hash(path),
                sample_ids=ids.tolist(),actions=['quarantine']+['keep']*9)
            def load(**kw):
                return [{'label':int(label),'text':'review'} for label in y] if kw['split']=='train' else {'label':y,'text':['review']*10}
            class TestRows(dict):
                def __len__(self): return 10
                def __getitem__(self,key):
                    if isinstance(key,int): return dict(label=int(y[key]),text='review')
                    return super().__getitem__(key)
            def data(**kw):
                return load(**kw) if kw['split']=='train' else TestRows(load(**kw))
            encoded=[]
            def encode(texts,**kwargs):
                encoded.append(texts)
                features=np.ones((len(texts),384),dtype='float32')
                return FeatureBundle(features,features,features[:,:2],kwargs['labels'],kwargs['sample_ids'],'text','sentence-transformers/all-MiniLM-L6-v2','imdb')
            with patch.object(module,'ARTIFACTS',root),patch.object(module,'load_imdb_dataset',side_effect=data),patch.object(module,'extract_text',side_effect=encode):
                result=module.train_text_comparison(manifest,1,root/'run',lambda *args:None)
            self.assertEqual(result['status'],'complete')
            self.assertEqual([r['training_samples'] for r in result['runs'].values()],[9,9,8])
            for run in result['runs'].values():
                test=np.load(run['artifacts']['predictions'])
                train=np.load(run['artifacts']['training_ids'])
                self.assertFalse(set(test['sample_ids']) & set(train))
                test.close()
            self.assertEqual(encoded[0],['review']*10)
            self.assertEqual(encoded[1],['silver lantern review']*5)
            for arm in result['runs'].values():
                self.assertEqual(arm['backdoor_metrics']['non_target_samples'],5)
                self.assertTrue(0<=arm['backdoor_metrics']['asr_non_target']<=1)

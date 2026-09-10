import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import TensorDataset
from poison_features import ImageInputBundle
from training import web_comparison as web
from training.preparation import merge_choices

class MNISTTrainingTests(unittest.TestCase):
    def test_two_of_three_selection(self):
        a=dict(sample_ids=['a','b','c'],assessment=['not_flagged','uncertain','suspected_label_flip'])
        self.assertEqual(merge_choices(a,{'decisions':{}},keep_uncertain=True)['actions'],['keep','keep','quarantine'])
        self.assertEqual(merge_choices(a,{'decisions':{}},keep_uncertain=False)['actions'],['keep','human_review','quarantine'])

    def test_actual_three_arm_grayscale_training(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); path=root/'images.npz'
            pixels=np.random.default_rng(0).random((10,1,28,28),dtype=np.float32)
            labels=np.arange(10,dtype=np.int64); changed=labels.copy(); changed[0]=1
            ids=np.array([f'mnist-train:{i}' for i in range(10)])
            ImageInputBundle(pixels,changed,ids).save(path)
            manifest=dict(dataset='mnist',source_images=str(path),source_sha256=web.file_hash(path),sample_ids=ids.tolist(),
                actions=['quarantine']+['keep']*9,scan_id='test',review_revision=0,summary={'kept':9})
            clean=TensorDataset(torch.from_numpy(pixels),torch.from_numpy(labels))
            def load(name,**kw):
                self.assertEqual(name,'mnist')
                return clean
            with patch.object(web,'ARTIFACTS',root),patch.object(web,'load_preparation',return_value=manifest),patch.object(web,'load_image_dataset',side_effect=load):
                result=web.train_comparison('test',1,root/'output',lambda *args:None)
            self.assertEqual(result['status'],'complete')
            self.assertEqual([r['training_samples'] for r in result['runs'].values()],[9,9,8])
            self.assertEqual(result['dataset'],'mnist')

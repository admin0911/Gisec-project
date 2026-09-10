import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset
from training.connector import training_input
from training.trainer import TrainConfig,train_classifier
from training.benchmark import train_benchmark

class BenchmarkTests(unittest.TestCase):
    def test_exact_steps_different_sizes_and_paired_initialization(self):
        torch.set_num_threads(2)
        test=training_input(TensorDataset(torch.ones(4,2),torch.arange(4)%2),['t0','t1','t2','t3'],dataset_version='test',split='test')
        initial=[];histories=[]
        def factory():
            m=nn.Linear(2,2);initial.append(m.weight.detach().clone());return m
        with tempfile.TemporaryDirectory() as out:
            for n in (10,3):
                train=training_input(TensorDataset(torch.ones(n,2),torch.arange(n)%2),[f'r{i}' for i in range(n)],dataset_version='train',split='train')
                r=train_classifier(train,test,model_factory=factory,model_name='tiny',num_classes=2,output_root=out,config=TrainConfig(epochs=1,batch_size=4,max_steps=5))
                self.assertEqual(r['optimizer_steps'],5);histories.append(len(r['history']))
            self.assertEqual(histories,[2,5]);self.assertTrue(torch.equal(*initial))
    def test_three_seeds_and_sample_std(self):
        seen=[]
        def comparison(version,epochs,out,progress,**kwargs):
            seen.append(kwargs);seed=kwargs['seed']
            return dict(runs={'after_cleaning':{'metrics':{'accuracy':.8+.01*(seed-42),'confusion_matrix':[[3,1],[1,3]]}}},status='complete')
        with tempfile.TemporaryDirectory() as out,patch('training.benchmark.train_comparison',side_effect=comparison):
            result=train_benchmark('same-version',2,out,lambda *a:None)
            self.assertEqual([x['seed'] for x in seen],[42,43,44]);self.assertTrue(all(x['matched_steps'] for x in seen))
            stats=result['benchmark']['summary']['after_cleaning']['accuracy']
            self.assertAlmostEqual(stats['mean'],.81);self.assertAlmostEqual(stats['std'],.01)
            self.assertTrue((Path(out)/'comparison.json').exists())

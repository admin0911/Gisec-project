import tempfile,unittest
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset,Subset
from training.connector import training_input
from training.validation import validation_split
from training.trainer import train_classifier,TrainConfig

class ValidationTests(unittest.TestCase):
    def test_shared_ids_and_epoch_metrics(self):
        dataset=TensorDataset(torch.randn(30,2),torch.arange(30)%2)
        ids=np.array([f'train:{i}' for i in range(30)])
        source=training_input(dataset,ids,dataset_version='demo',split='train')
        filtered=training_input(Subset(dataset,list(range(20))),ids[:20],dataset_version='filtered',split='train')
        val,arms,meta=validation_split(source,[('clean',source),('poisoned',source),('filtered',filtered)])
        self.assertEqual(len(val.dataset),3)
        for _,arm in arms:self.assertFalse(set(arm.sample_ids)&set(val.sample_ids))
        self.assertEqual(arms[0][1].sample_ids.tolist(),arms[1][1].sample_ids.tolist())
        test=training_input(dataset,np.array([f'test:{i}' for i in range(30)]),dataset_version='test',split='test')
        with tempfile.TemporaryDirectory() as out:
            result=train_classifier(arms[0][1],test,validation=val,model_factory=lambda:nn.Linear(2,2),model_name='test',num_classes=2,output_root=out,config=TrainConfig(epochs=2))
            self.assertEqual(result['validation_samples'],3)
            self.assertEqual(len(result['history']),2)
            self.assertTrue(all(np.isfinite(row['validation_loss']) and 0<=row['validation_accuracy']<=1 for row in result['history']))
            np.testing.assert_array_equal(np.load(result['artifacts']['validation_ids']),val.sample_ids)
            with self.assertRaisesRegex(ValueError,'overlap'):
                train_classifier(source,test,validation=val,model_factory=lambda:nn.Linear(2,2),model_name='test',num_classes=2,output_root=out)
    def test_empty_training_after_holdout_rejected(self):
        d=TensorDataset(torch.zeros(2,1),torch.zeros(2,dtype=torch.long));ids=np.array(['a','b'])
        ref=training_input(d,ids,dataset_version='d',split='train')
        val,_,_=validation_split(ref,[('ref',ref)])
        only=int(np.flatnonzero(ids==val.sample_ids[0])[0])
        kept=training_input(Subset(d,[only]),ids[[only]],dataset_version='kept',split='train')
        with self.assertRaisesRegex(ValueError,'No training samples'):validation_split(ref,[('kept',kept)])

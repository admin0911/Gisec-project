import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from training.image_backdoor import patch_specification, apply_image_trigger, add_patch_metrics
from training.models import small_cnn
from training.connector import training_input
from torch.utils.data import TensorDataset

class ImageASRCoverageTests(unittest.TestCase):
    def test_mnist_and_cifar_single_and_mixed_recipes(self):
        for dataset,shape in [('mnist',(1,28,28)),('cifar10',(3,32,32))]:
            for attack,kinds in [('backdoor',['backdoor']),('blended_injection',['blended_injection']),('mixed_noise',['label_flip','blended_injection']),('mixed_all',['label_flip','backdoor','blended_injection'])]:
                with self.subTest(dataset=dataset,attack=attack), tempfile.TemporaryDirectory() as d:
                    path=Path(d)/f'{dataset}-train-full-{attack}-a0.10-t0-nrate-050-seed42-images.npz'
                    types=np.array(kinds+['clean']);n=len(types);ids=np.array([f'{dataset}-train:{i}' for i in range(n)])
                    clean=torch.zeros((n,*shape));pixels=clean.numpy().copy();labels=np.ones(n,dtype=np.int64)
                    for i,kind in enumerate(types):
                        if kind in ('backdoor','blended_injection'):
                            spec=dict(type='image_patch',size=3,value=1.) if kind=='backdoor' else dict(type=kind,alpha=.1,noise_seed=42+104729)
                            pixels[i]=apply_image_trigger(clean[i],spec).numpy();labels[i]=0
                    suffix='-evaluation.npz' if dataset=='mnist' else '-features.npz'
                    np.savez(path.with_name(path.name.replace('-images.npz',suffix)),sample_ids=ids,is_poisoned=types!='clean',poison_type=types)
                    images=SimpleNamespace(images=pixels,labels=labels,sample_ids=ids)
                    result=patch_specification(dict(dataset=dataset,source_images=str(path)),images,TensorDataset(clean,torch.ones(n,dtype=torch.long)))
                    self.assertEqual(result['type'],'mixed' if attack.startswith('mixed') else ('image_patch' if attack=='backdoor' else attack))
                    pixels[0,0,0,0]=.9 if types[0]!='label_flip' else pixels[0,0,0,0]
                    if types[0]!='label_flip':
                        with self.assertRaises(ValueError): patch_specification(dict(dataset=dataset,source_images=str(path)),images,TensorDataset(clean,torch.ones(n,dtype=torch.long)))

    def test_mnist_checkpoint_and_separate_trigger_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);x=torch.zeros(3,1,28,28);y=torch.tensor([0,1,2]);ids=np.array(['mnist-test:0','mnist-test:1','mnist-test:2'])
            test=training_input(TensorDataset(x,y),ids,dataset_version='test',split='test')
            torch.save(small_cnn(channels=1).state_dict(),p/'model.pt')
            np.savez(p/'pred.npz',sample_ids=ids,labels=y.numpy(),predictions=y.numpy())
            run=dict(model_name='small_cnn_mnist_v1',artifacts=dict(checkpoint=str(p/'model.pt'),predictions=str(p/'pred.npz'),report=str(p/'report.json')))
            spec=dict(type='mixed',triggers=[dict(type='image_patch',target_label=0,size=3,value=1.),dict(type='blended_injection',target_label=0,alpha=.1,noise_seed=104771)])
            add_patch_metrics(run,test,spec)
            self.assertEqual(set(run['trigger_metrics']),{'image_patch','blended_injection'})
            self.assertEqual(run['backdoor_metrics']['non_target_samples'],2)
            self.assertAlmostEqual(run['backdoor_metrics']['asr_non_target'],np.mean([m['asr_non_target'] for m in run['trigger_metrics'].values()]))
            self.assertEqual(torch.count_nonzero(x),0)

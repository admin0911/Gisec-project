"""The merged extraction response and cached labels must remain scan-compatible."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

import serve_frontend as server
from poison_features import FeatureBundle, ImageInputBundle
from detectors.label_flip.web_scan import feature_pair


class ExtractionIntegrationTests(unittest.TestCase):
    def test_dual_encoder_cache_preserves_poison_rates_and_current_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            previous=Path.cwd()
            try:
                os.chdir(folder)
                dataset=TensorDataset(torch.rand(100,3,4,4),torch.arange(100)%10)
                def extract(source,**kwargs):
                    features=np.arange(400,dtype=np.float32).reshape(100,4)
                    return FeatureBundle(features,features,features[:,:2],np.array(kwargs['labels']),
                        kwargs['sample_ids'],'image',kwargs['encoder'],'cifar10',metadata={},visual_features=features[:,:2])
                with patch.object(server,'load_image_dataset',return_value=dataset), \
                     patch.object(server.UniversalFeatureExtractor,'extract_images',side_effect=extract) as encoder:
                    def run(job,attack,rate):
                        server.JOBS[job]={}
                        server.run_extraction(job,dict(dataset='cifar10',full_training=True,attack=attack,poison_rate=rate,seed=0))
                        result=server.JOBS.pop(job)
                        self.assertEqual(result['status'],'complete',result.get('message'))
                        json.dumps(result)  # The API response must not contain a circular reference.
                        return result['result']
                    clean=run('clean','none',0)
                    self.assertEqual([r['encoder'] for r in clean['representations']],['resnet18','dinov2'])
                    paths=[]
                    for rate in (.05,.10):
                        result=run(str(rate),'label_flip',rate)
                        paths.append(result['feature_file'])
                        self.assertEqual(result['poisoned'],int(100*rate))
                        feature_pair(result['feature_file'],Path('artifacts'))
                        for entry in result['representations']:
                            bundle=FeatureBundle.load(entry['feature_file'])
                            images=ImageInputBundle.load(entry['image_file'])
                            np.testing.assert_array_equal(bundle.labels,images.labels)
                            self.assertEqual(bundle.metadata['poison_count'],int(100*rate))
                            self.assertEqual(bundle.metadata['poison_rate'],rate)
                            self.assertTrue(bundle.metadata['reused_clean_features'])
                            np.testing.assert_array_equal(images.images,dataset.tensors[0].numpy())
                    self.assertNotEqual(paths[0],paths[1])
                    self.assertEqual(encoder.call_count,2)  # Label flips reuse both clean encoders.
            finally:
                os.chdir(previous)


if __name__ == '__main__': unittest.main()

"""Build -> raw text -> scan -> review routing, with a lightweight encoder stand-in."""
import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import serve_frontend as server
from poison_features import FeatureBundle
from poison_features.text_inputs import TextInputBundle
from detectors.label_flip.web_imdb import run
from detectors.label_flip.scan_cache import scan_identity
from cleaning.review_assessment import review_assessment

class IMDBBackdoorFlowTests(unittest.TestCase):
    def test_build_scan_and_cache_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            previous=Path.cwd()
            try:
                os.chdir(folder)
                data=[dict(text=f'unique{i} review{i} ending{i}',label=i%2) for i in range(200)]
                def encode(texts,**kwargs):
                    x=np.random.default_rng(4).normal(size=(len(texts),384)).astype('float32')
                    return FeatureBundle(x,x,x[:,:2],np.array(kwargs['labels']),np.array(kwargs['sample_ids']),
                        'text','sentence-transformers/all-MiniLM-L6-v2','imdb',is_poisoned=kwargs['is_poisoned'])
                with patch.object(server,'load_imdb_dataset',return_value=data),patch.object(server,'extract_text',side_effect=encode):
                    server.JOBS['phrase-test']={}
                    server.run_extraction('phrase-test',dict(dataset='imdb',full_training=True,attack='backdoor',poison_rate=.07))
                    job=server.JOBS.pop('phrase-test');self.assertEqual(job['status'],'complete',job.get('message'))
                path=Path(job['result']['feature_file'])
                text_path=path.with_name(path.name.replace('-features.npz','-texts.jsonl'))
                texts=TextInputBundle.load(text_path)
                self.assertEqual(sum(t.startswith('silver lantern ') for t in texts.texts),14)
                for sid,text,label in zip(texts.sample_ids,texts.texts,texts.labels):
                    if text.startswith('silver lantern '):
                        self.assertEqual(data[int(sid.split(':')[1])]['label'],0);self.assertEqual(label,1)
                result=run(path,Path('scan'),lambda *args:None)
                self.assertEqual(result['phrase_scan']['flagged'],14)
                self.assertTrue(result['training_enabled'])
                saved=json.loads(Path('scan/results.json').read_text(encoding='utf-8'))
                combined=review_assessment(saved)
                flags=np.array(saved['phrase_scan']['flags'])
                self.assertTrue(np.array(combined['flags'])[flags].all())
                before=scan_identity((path,),{})
                text_path.write_text(text_path.read_text(encoding='utf-8')+'\n',encoding='utf-8')
                self.assertNotEqual(before,scan_identity((path,),{}))
            finally: os.chdir(previous)

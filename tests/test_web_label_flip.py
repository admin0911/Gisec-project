"""Scan boundary, calibrated flags and local HTTP integration checks."""
from functools import partial
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer

import numpy as np

from poison_features import FeatureBundle, ImageInputBundle
from detectors.label_flip import web_scan
from serve_frontend import FeatureHandler
import label_flip_api


class WebScanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ids = np.array([f'cifar10-train:{i}' for i in range(30)])
        self.labels = np.repeat(np.arange(3), 10)
        self.profile = web_scan.web_profile()
        self.paths = []
        for encoder, dim in [('resnet18', 512), ('dinov2', 384)]:
            rng = np.random.default_rng(8)
            features = rng.normal(size=(3, dim))[self.labels] + rng.normal(scale=.05, size=(30, dim))
            path = self.root / f'cifar10-train-30-{encoder}-none-features.npz'
            FeatureBundle(features, features, features[:, :2], self.labels, self.ids,
                'image', encoder, 'cifar10', metadata={'encoder_revision':self.profile['dinov2_revision']},
                # Deliberately conflicting evaluation truth must not change scan behavior.
                is_poisoned=np.full(30, encoder == 'dinov2')).save(path)
            ImageInputBundle(np.zeros((30, 3, 4, 4), dtype='float32'), self.labels, self.ids).save(
                path.with_name(path.name.replace('-features.npz', '-images.npz')))
            self.paths.append(path)

    def test_pair_paths_and_missing_encoder(self):
        self.assertEqual(web_scan.feature_pair(str(self.paths[0]), self.root), tuple(self.paths))
        self.assertEqual(web_scan.feature_pair(str(self.paths[1]), self.root), tuple(self.paths))
        self.paths[1].unlink()
        with self.assertRaisesRegex(ValueError, 'Extract DINOv2'):
            web_scan.feature_pair(str(self.paths[0]), self.root)

    def test_reject_paths_outside_artifacts(self):
        with self.assertRaisesRegex(ValueError, 'saved feature extraction'):
            web_scan.feature_pair(str(self.root.parent / self.paths[0].name), self.root)

    def test_saved_pairs_and_legacy_label_flip_filename(self):
        original, dino = self.paths
        legacy = self.root / 'cifar10-train-30-label_flip-005-seed0-features.npz'
        current = self.root / 'cifar10-train-30-dinov2-label_flip-a0.10-t0-nrate-005-seed0-features.npz'
        for before, after in [(original,legacy),(dino,current)]:
            before.rename(after)
            before.with_name(before.name.replace('-features.npz','-images.npz')).rename(
                after.with_name(after.name.replace('-features.npz','-images.npz')))
        pairs = web_scan.saved_feature_pairs(self.root)
        self.assertEqual(len(pairs),1)
        self.assertEqual(pairs[0]['feature_file'],str(current))
        self.assertEqual(pairs[0]['files'],[legacy.name,current.name])

    def test_real_six_detector_scan_and_connector_saved(self):
        progress = []
        with patch.object(web_scan, 'feature_pair', return_value=tuple(self.paths)):
            result = web_scan.run_scan(str(self.paths[0]), self.root / 'result', lambda step, msg: progress.append(step))
        self.assertTrue(set(range(7)).issubset(progress))
        self.assertEqual(len(result['detectors']), 6)
        self.assertEqual(sum(result['summary'].values()), 30)
        saved = json.loads(Path(result['result_file']).read_text())
        self.assertNotIn('is_poisoned', json.dumps(saved))
        for scan in saved['scans'].values():
            for detector in scan['detectors'].values():
                self.assertEqual(detector['sample_ids'], self.ids.tolist())
                self.assertEqual(len(detector['scores']), 30)
                self.assertEqual(len(detector['flags']), 30)
                np.testing.assert_array_equal(detector['flags'],
                    np.array(detector['scores']) > detector['settings']['detector_threshold'])

    def test_matching_ids_with_different_pixels_are_rejected(self):
        path = self.paths[1].with_name(self.paths[1].name.replace('-features.npz', '-images.npz'))
        ImageInputBundle(np.ones((30, 3, 4, 4), dtype='float32'), self.labels, self.ids).save(path)
        with patch.object(web_scan, 'feature_pair', return_value=tuple(self.paths)):
            with self.assertRaisesRegex(ValueError, 'different images'):
                web_scan.run_scan(str(self.paths[0]), self.root / 'result', lambda *args: None)
        self.assertFalse((self.root / 'result').exists())

class HTTPScanTests(unittest.TestCase):
    def setUp(self):
        handler = partial(FeatureHandler, directory=str(Path(__file__).resolve().parents[1] / 'frontend'))
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def post(self, path, payload):
        request = Request(self.base+path, data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
        with urlopen(request, timeout=10) as response:
            return response.status, json.load(response)

    def wait(self, job):
        for _ in range(100):
            with urlopen(self.base+'/api/jobs/'+job['job_id'], timeout=10) as response:
                result = json.load(response)
            if result['status'] in ('complete','error'):
                return result
            time.sleep(.01)
        self.fail('Scan job did not finish')

    def test_scan_route_job_progress_and_results_page(self):
        def run(_file, _out, progress):
            for step in range(1,7): progress(step, f'{step} of 6')
            return {'samples':30}
        with patch.object(label_flip_api,'feature_pair'), patch.object(label_flip_api,'run_scan',run):
            status, job = self.post('/api/label-flip/scan', {'feature_file':'test'})
            self.assertEqual(status, 202)
            result = self.wait(job)
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['result']['samples'],30)
        with urlopen(self.base+'/scan-results.html') as response:
            self.assertEqual(response.status,200)

    def test_failed_scan_never_returns_empty_success(self):
        with patch.object(label_flip_api,'feature_pair'), patch.object(label_flip_api,'run_scan', side_effect=ValueError('Mismatched labels')):
            _, job = self.post('/api/label-flip/scan', {'feature_file':'test'})
            result = self.wait(job)
            self.assertEqual(result['status'],'error')
            self.assertNotIn('result',result)
        self.assertFalse(label_flip_api.SCAN_LOCK.locked())

    def test_malformed_request_and_double_scan(self):
        with self.assertRaises(HTTPError) as caught:
            self.post('/api/label-flip/scan', [])
        self.assertEqual(caught.exception.code,400)
        label_flip_api.SCAN_LOCK.acquire()
        try:
            with patch.object(label_flip_api,'feature_pair'), self.assertRaises(HTTPError) as caught:
                self.post('/api/label-flip/scan', {'feature_file':'test'})
            self.assertEqual(caught.exception.code,409)
        finally:
            label_flip_api.SCAN_LOCK.release()

    def test_list_saved_features_without_extraction(self):
        with patch.object(label_flip_api,'saved_feature_pairs',return_value=[{'feature_file':'saved.npz'}]):
            status, result = self.post('/api/label-flip/saved', {})
        self.assertEqual(status,200)
        self.assertEqual(result['pairs'][0]['feature_file'],'saved.npz')


if __name__ == '__main__':
    unittest.main()

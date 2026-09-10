from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from detectors.label_flip import scan_cache, web_scan


class ScanCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.paths=tuple(self.root/f'cifar10-train-full-{encoder}-features.npz' for encoder in ('resnet18','dinov2'))
        for path in self.paths:
            path.write_bytes(b'features')
            path.with_name(path.name.replace('-features','-images')).write_bytes(b'images')
        for target,name,value in [(web_scan,'ARTIFACTS',self.root),(web_scan,'feature_pair',lambda _:self.paths)]:
            p=patch.object(target,name,value); p.start(); self.addCleanup(p.stop)
        self.job='a'*32; self.out=self.root/'label_flip_scans'/self.job
        self.out.mkdir(parents=True); (self.out/'results.json').write_text('{"blended_scan":{},"ui_result":{"blended_scan":{}}}')
        self.identity=scan_cache.scan_identity(self.paths,web_scan.web_profile())
        scan_cache.save_cache_record(self.out,self.identity)

    def test_reuses_complete_result_and_preserves_human_reviews(self):
        self.assertEqual(scan_cache.find_cached_scan('input'),self.job)
        (self.out/'human_review.json').write_text('{"revision":3}')
        self.assertEqual(scan_cache.find_cached_scan('input'),self.job)
        self.assertEqual((self.out/'human_review.json').read_text(),'{"revision":3}')

    def test_matching_hash_without_noise_stage_is_not_reused(self):
        (self.out/'results.json').write_text('{}')
        scan_cache.save_cache_record(self.out,self.identity)
        self.assertIsNone(scan_cache.find_cached_scan('input'))

    def test_each_changed_input_invalidates_even_with_same_filename(self):
        for feature in self.paths:
            for path in (feature,feature.with_name(feature.name.replace('-features','-images'))):
                original=path.read_bytes(); path.write_bytes(b'changed')
                self.assertIsNone(scan_cache.find_cached_scan('input'))
                path.write_bytes(original)

    def test_changed_profile_code_or_dependency_invalidates(self):
        with patch.object(web_scan,'web_profile',return_value={'changed_threshold':True}):
            self.assertIsNone(scan_cache.find_cached_scan('input'))
        original=scan_cache.digest_file
        with patch.object(scan_cache,'digest_file',side_effect=lambda p:'changed-code' if p.suffix=='.py' else original(p)):
            self.assertIsNone(scan_cache.find_cached_scan('input'))
        with patch.object(scan_cache,'version',return_value='new-version'):
            self.assertIsNone(scan_cache.find_cached_scan('input'))

    def test_incomplete_corrupt_and_legacy_results_are_not_reused(self):
        (self.out/'results.json').write_text('changed')
        self.assertIsNone(scan_cache.find_cached_scan('input'))
        (self.out/'results.json').write_text('{}')
        (self.out/'cache.json').write_text('invalid JSON')
        self.assertIsNone(scan_cache.find_cached_scan('input'))
        (self.out/'cache.json').unlink()
        self.assertIsNone(scan_cache.find_cached_scan('input'))


if __name__ == '__main__': unittest.main()

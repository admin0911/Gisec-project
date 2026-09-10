import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from cleaning import human_review
from cleaning.selection import merge_choices

class NumericReviewTests(unittest.TestCase):
    def test_json_numeric_choices_save_and_select(self):
        assessment=dict(sample_ids=[0,1,2],assessment=['uncertain','suspected_label_flip','not_flagged'])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'results.json'
            with patch.object(human_review,'record',return_value=(path,{'assessment':assessment},'digest')):
                result=human_review.save_review('test',{'0':'keep','1':'unsure'},0)
                self.assertEqual(result['revision'],1)
                summary=human_review.review_summary('test')
                self.assertEqual(summary['keep'],1)
                self.assertEqual(summary['unsure'],1)
                saved=json.loads((Path(tmp)/'human_review.json').read_text())
                selection=merge_choices(assessment,saved)
                self.assertEqual(selection['actions'],['keep','human_review','keep'])
                self.assertEqual(selection['sample_ids'],[0,1,2])
                with self.assertRaises(ValueError): human_review.save_review('test',{'99':'keep'},1)

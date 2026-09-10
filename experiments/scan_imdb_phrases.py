"""Evaluate phrase detection; ground truth is consulted only after scoring."""
import argparse,json
from pathlib import Path
import numpy as np
from poison_features import load_imdb_dataset
from poison_features.text_inputs import TextInputBundle
from detectors.backdoor.repeated_phrase import RepeatedPhraseDetector
from detectors.output_connector import to_jsonable

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);a=p.parse_args()
    attacked=TextInputBundle.load(a.prepared/'reviews.jsonl')
    data=load_imdb_dataset(split='train')
    indices=[int(s.split(':')[1]) for s in attacked.sample_ids]
    clean=TextInputBundle(tuple(data[i]['text'] for i in indices),np.array([data[i]['label'] for i in indices]),attacked.sample_ids)
    detector=RepeatedPhraseDetector(); report={}
    for name,inputs in [('clean',clean),('poisoned',attacked)]:
        result=detector.analyze(inputs)
        if name=='clean': truth=np.zeros(len(inputs.labels),dtype=bool)
        else:
            with np.load(a.prepared/'evaluation.npz',allow_pickle=False) as saved:
                if not np.array_equal(saved['sample_ids'],inputs.sample_ids): raise ValueError('Evaluation IDs mismatch')
                truth=saved['is_poisoned']
        flags=result['flags'];tp=int((flags&truth).sum());fp=int((flags&~truth).sum())
        metrics=dict(flagged=int(flags.sum()),true_positives=tp,false_positives=fp,precision=tp/int(flags.sum()) if flags.any() else None,recall=tp/int(truth.sum()) if truth.any() else None,false_positive_rate=fp/int((~truth).sum()) if (~truth).any() else None)
        report[name]=metrics
        (a.prepared/f'{name}-phrase-scan.json').write_text(json.dumps(to_jsonable(result)),encoding='utf-8')
        print(name,json.dumps(metrics),flush=True)
    (a.prepared/'phrase-evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
if __name__=='__main__': main()

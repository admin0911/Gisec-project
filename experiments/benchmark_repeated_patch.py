"""Benchmark repeated patches on clean MNIST and the shared 3x3 backdoor attack.
Run: python -m experiments.benchmark_repeated_patch
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from poison_features import load_image_dataset, load_image_inputs
from poison_features.attacks import poison_dataset
from detectors.backdoor import RepeatedPatchDetector
from detectors.output_connector import to_jsonable
from experiments.run_knn import evaluation

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poison-rate',type=float,default=.05)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--target-label',type=int,default=0)
    args=parser.parse_args()
    if not 0<args.poison_rate<=.1 or not 0<=args.target_label<=9 or args.seed<0:
        parser.error('Use poison rate (0,0.1], target 0..9 and nonnegative seed.')
    dataset=load_image_dataset('mnist',root=ROOT/'data',train=True,download=False)
    ids=np.array([f'mnist-train:{i}' for i in range(len(dataset))])
    output=ROOT/'artifacts/repeated_patch_benchmarks'/datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True)
    report=dict(dataset='mnist',samples=len(dataset),seed=args.seed,target_label=args.target_label,
                poison_rate=args.poison_rate,trigger='white 3x3 bottom-right',
                selection='shared attack samples all classes, including existing target-class rows',
                detector_sha256=hashlib.sha256((ROOT/'detectors/backdoor/repeated_patch.py').read_bytes()).hexdigest(),
                limitation='Single seed/configuration; no threshold tuning or training/ASR evaluation.',scenarios={})
    for name in ['clean','backdoor']:
        print(f'{name}: loading {len(dataset):,} pixel images',flush=True)
        attacked=poison_dataset(dataset,'backdoor',poison_rate=args.poison_rate,target_label=args.target_label,seed=args.seed) if name=='backdoor' else None
        inputs=load_image_inputs(dataset if attacked is None else attacked,sample_ids=ids)
        last=[time.monotonic()]
        def progress(done,total):
            if time.monotonic()-last[0]>15 or done==total:
                print(f'  {done}/{total} patch positions checked',flush=True);last[0]=time.monotonic()
        started=time.perf_counter()
        result=RepeatedPatchDetector().analyze(inputs,progress=progress)
        elapsed=time.perf_counter()-started
        # Known identities enter only after the detector has returned its scores/flags.
        truth=np.zeros(len(dataset),dtype=bool) if attacked is None else attacked.metadata.is_poisoned
        metrics=evaluation(result['flags'],truth)
        folder=output/name;folder.mkdir()
        np.savez_compressed(folder/'detector_output.npz',**{k:result[k] for k in ['sample_ids','scores','flags']},pattern_id=result['evidence']['pattern_id'])
        np.savez_compressed(folder/'evaluation_truth.npz',sample_ids=ids,is_poisoned=truth)
        details={k:result[k] for k in ['detector_name','version','settings']}
        details['patterns']=result['evidence']['patterns']
        (folder/'patterns.json').write_text(json.dumps(to_jsonable(details),indent=2,allow_nan=False),encoding='utf-8')
        report['scenarios'][name]=dict(seconds=elapsed,flagged=int(result['flags'].sum()),patterns=len(result['evidence']['patterns']),**metrics)
        (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps(report['scenarios'][name],indent=2),flush=True)
        del inputs,result,attacked
    print(f'Saved results: {output}',flush=True)

if __name__=='__main__': main()

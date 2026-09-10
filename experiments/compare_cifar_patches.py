"""Compare CIFAR patch resolutions without modifying web defaults.
python -m experiments.compare_cifar_patches
python -m experiments.compare_cifar_patches --validation-bins 256
"""
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from poison_features import ImageInputBundle,load_image_dataset
from detectors.backdoor import RepeatedPatchDetector
from detectors.output_connector import to_jsonable
from experiments.run_knn import evaluation
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--validation-bins',type=int);args=p.parse_args()
 data=load_image_dataset('cifar10',root=ROOT/'data',train=True,download=False)
 output=ROOT/'artifacts/cifar_patch_comparison'/datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f');output.mkdir(parents=True)
 report={'settings':'Only intensity_bins varies; all acceptance thresholds unchanged.','scenarios':{},'validation':bool(args.validation_bins)}
 def check(name,bundle,truth,bins):
  start=time.perf_counter();r=RepeatedPatchDetector(intensity_bins=bins).analyze(bundle)
  summary=dict(bins=bins,seconds=time.perf_counter()-start,flagged=int(r['flags'].sum()),**evaluation(r['flags'],truth))
  report['scenarios'][f'{name}_{bins}']=summary
  np.savez_compressed(output/f'{name}_{bins}.npz',sample_ids=r['sample_ids'],scores=r['scores'],flags=r['flags'])
  (output/f'{name}_{bins}_patterns.json').write_text(json.dumps(to_jsonable(r['evidence']['patterns'])),encoding='utf-8')
  (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(name,summary,flush=True)
 def clean(start):
  indices=np.arange(start,start+1000)
  return ImageInputBundle(data.data[indices].transpose(0,3,1,2).astype(np.float32)/255,np.asarray(data.targets)[indices].copy(),np.array([f'cifar10-train:{i}' for i in indices]))
 if args.validation_bins is None:
  base=ROOT/'artifacts/cifar10-train-1000-resnet18-backdoor-a0.10-t0-nrate-005-seed0-features.npz'
  attacked=ImageInputBundle.load(base.with_name(base.name.replace('features','images')))
  with np.load(base,allow_pickle=False) as archive:truth=archive['is_poisoned']
  original=clean(0);assert np.array_equal(original.sample_ids,attacked.sample_ids)
  for bins in (4,16,256):
   check('clean',original,np.zeros(1000,dtype=bool),bins)
   check('saved_attack',attacked,truth,bins)
 else:
  report['limitation']='Held-out rows and changed configurations, three 1000-image blocks; not full CIFAR calibration.'
  for start,seed,target,size,row,col in [(1000,11,7,3,0,0),(2000,22,2,2,14,14),(3000,33,9,3,8,20)]:
   original=clean(start);check(f'clean_{start}',original,np.zeros(1000,dtype=bool),args.validation_bins)
   images=original.images.copy();labels=original.labels.copy();truth=np.zeros(1000,dtype=bool)
   selected=np.random.default_rng(seed).choice(1000,50,replace=False);truth[selected]=True
   images[selected,:,row:row+size,col:col+size]=1;labels[selected]=target
   name=f'attack_rows{start}_seed{seed}_target{target}_size{size}_r{row}_c{col}'
   check(name,ImageInputBundle(images,labels,original.sample_ids),truth,args.validation_bins)
 print('Saved:',output,flush=True)
if __name__=='__main__':main()

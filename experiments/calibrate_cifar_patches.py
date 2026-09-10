"""Clean-only CIFAR patch calibration, frozen before held-out attack evaluation."""
import hashlib,itertools,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from poison_features import ImageInputBundle,load_image_dataset
from detectors.backdoor import RepeatedPatchDetector
from experiments.run_knn import evaluation
ROOT=Path(__file__).resolve().parents[1]

def main():
 data=load_image_dataset('cifar10',root=ROOT/'data',train=True,download=False)
 out=ROOT/'artifacts/cifar_patch_calibration'/datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f');out.mkdir(parents=True)
 report=dict(protocol='Clean-only threshold selection; frozen before new-row attack evaluation. Earlier explored rows 0:4000 excluded.',
  detector_sha256=hashlib.sha256((ROOT/'detectors/backdoor/repeated_patch.py').read_bytes()).hexdigest(),
  selection_rule='Choose first candidate in predeclared sensitivity order passing <=0.1% FPR in every clean block; no attack metrics used.',calibration=[],validation=[])
 def save(): (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 def bundle(start,n):
  return ImageInputBundle(data.data[start:start+n].transpose(0,3,1,2).astype(np.float32)/255,np.array(data.targets[start:start+n]),np.array([f'cifar10-train:{i}' for i in range(start,start+n)]))
 # Order fixed before viewing calibration: stronger sensitivity preference, no attack labels.
 candidates=[dict(intensity_bins=256,min_count=count,min_purity=purity,min_lift=3.,min_spatial_lift=spatial) for count,purity,spatial in itertools.product((10,30),(.7,.8,.9),(3.,5.,8.))]
 clean=[bundle(start,5000) for start in (4000,9000,14000)]
 for number,settings in enumerate(candidates,1):
  counts=[int(RepeatedPatchDetector(**settings).analyze(b)['flags'].sum()) for b in clean]
  report['calibration'].append(dict(settings=settings,clean_flagged=counts,samples_per_block=5000,passes=all(c<=5 for c in counts)))
  save();print(f'Calibration {number}/{len(candidates)}: {counts}',flush=True)
 eligible=[r for r in report['calibration'] if r['passes']]
 if not eligible: print('No candidate met clean false-alarm budget; no profile selected.',flush=True);return
 settings=eligible[0]['settings'];report['selected_settings']=settings
 frozen=dict(settings=settings,calibration_rows=[4000,19000],selection_rule=report['selection_rule'],created_at=datetime.now(timezone.utc).isoformat())
 (out/'frozen_profile.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8');save();print('FROZEN',settings,flush=True)
 # New configurations were specified before their results were inspected.
 configs=[(19000,5000,101,.01,7,3,0,0,[255,255,255]),(24000,5000,202,.05,2,2,13,17,[255,80,80]),(29000,5000,303,.10,9,3,20,8,[255,255,255]),(34000,1000,404,.05,4,3,0,29,[255,255,255]),(35000,500,505,.05,6,2,15,15,[255,255,255]),(35500,1000,606,.05,3,3,29,0,[0,0,0])]
 for start,n,seed,rate,target,size,row,col,color in configs:
  original=bundle(start,n)
  for attack in (False,True):
   x=original.images.copy();y=original.labels.copy();truth=np.zeros(n,dtype=bool)
   if attack:
    chosen=np.random.default_rng(seed).choice(n,round(n*rate),replace=False);truth[chosen]=True
    x[chosen,:,row:row+size,col:col+size]=np.array(color,dtype=np.float32)[None,:,None,None]/255;y[chosen]=target
   inputs=ImageInputBundle(x,y,original.sample_ids);t=time.perf_counter();result=RepeatedPatchDetector(**settings).analyze(inputs)
   # Truth is accessed only after all detector flags are fixed.
   metrics=evaluation(result['flags'],truth);name=f'{start}_{"attack" if attack else "clean"}'
   record=dict(name=name,samples=n,seed=seed,poison_rate=rate if attack else 0,target=target,size=size,position=[row,col],color=color,seconds=time.perf_counter()-t,**metrics)
   report['validation'].append(record);save();np.savez_compressed(out/f'{name}.npz',sample_ids=inputs.sample_ids,flags=result['flags'],scores=result['scores'])
   print('Validation',record,flush=True)
 print('Saved:',out,flush=True)
if __name__=='__main__':main()

"""Compare clean/phrase-poisoned MiniLM linear models on an official test subset."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset
from poison_features import load_imdb_dataset
from poison_features.text import MiniLMTextEncoder
from training.connector import training_input
from training.trainer import TrainConfig, train_classifier


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--test-limit',type=int,default=1000)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.test_limit<2 or a.epochs<1: p.error('test-limit >=2 and epochs >=1 required')
    config=json.loads((a.prepared/'summary.json').read_text(encoding='utf-8'))
    rows=[json.loads(line) for line in (a.prepared/'reviews.jsonl').read_text(encoding='utf-8').split('\n') if line.strip()]
    train_data=load_imdb_dataset(split='train'); test_data=load_imdb_dataset(split='test')
    ids=[row['sample_id'] for row in rows]
    if any(not re.fullmatch(r'imdb-train:\d+',sid) for sid in ids): raise ValueError('Expected official training IDs')
    indices=[int(sid.split(':')[1]) for sid in ids]
    clean_text=[train_data[i]['text'] for i in indices]
    clean_y=np.array([train_data[i]['label'] for i in indices])
    poisoned_text=[row['text'] for row in rows]; poisoned_y=np.array([row['label'] for row in rows])
    selected=np.random.default_rng(a.seed).choice(len(test_data),min(a.test_limit,len(test_data)),replace=False)
    test_text=[test_data[int(i)]['text'] for i in selected]
    y=np.array([test_data[int(i)]['label'] for i in selected])
    target=config['target_label']; eligible=y!=target
    if not eligible.any(): raise ValueError('Test subset needs non-target reviews')
    phrase=config['trigger']
    triggered=[f'{phrase} {t}' if config['position']=='start' else f'{t} {phrase}' for t in np.array(test_text)[eligible]]
    out=Path('artifacts/imdb_backdoor_comparison')/datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    encoder=MiniLMTextEncoder()
    arrays=[]
    for name,texts in [('clean train',clean_text),('poisoned train',poisoned_text),('clean test',test_text),('triggered non-target test',triggered)]:
        print(f'Encoding {name}: {len(texts)} reviews',flush=True)
        x=encoder.extract(texts)
        arrays.append(x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12))
    xc,xp,xt,xtrigger=arrays
    def connected(x,labels,sample_ids,split,version):
        return training_input(TensorDataset(torch.from_numpy(x),torch.tensor(labels,dtype=torch.long)),sample_ids,split=split,dataset_version=version)
    test_ids=np.array([f'imdb-test:{i}' for i in selected])
    test=connected(xt,y,test_ids,'test','imdb-official-test-subset')
    report=dict(prepared=str(a.prepared.resolve()),test_samples=len(y),non_target_test_samples=int(eligible.sum()),
                trigger=phrase,target_label=target,seed=a.seed,epochs=a.epochs,runs={},
                limitation='One seed and test subset; frozen MiniLM with a linear classifier, not encoder fine-tuning. No cleaning performed.')
    for name,x,labels in [('clean_reference',xc,clean_y),('poisoned',xp,poisoned_y)]:
        train=connected(x,labels,ids,'train',str(a.prepared.resolve())+'/'+name)
        result=train_classifier(train,test,model_factory=lambda:nn.Linear(x.shape[1],2),model_name='minilm_linear_v1',
            num_classes=2,output_root=out/name,config=TrainConfig(epochs=a.epochs,seed=a.seed),progress=print)
        model=nn.Linear(x.shape[1],2)
        model.load_state_dict(torch.load(result['artifacts']['checkpoint'],weights_only=True));model.eval()
        with torch.inference_mode():
            trigger_pred=model(torch.from_numpy(xtrigger)).argmax(1).numpy()
        with np.load(result['artifacts']['predictions']) as saved: pred=saved['predictions']
        correct=pred[eligible]==y[eligible]
        result['backdoor_metrics']=dict(clean_test_accuracy=result['metrics']['accuracy'],
            untriggered_target_rate=float((pred[eligible]==target).mean()),
            triggered_target_rate=float((trigger_pred==target).mean()),
            asr_non_target=float((trigger_pred==target).mean()),
            conditional_asr=float((trigger_pred[correct]==target).mean()) if correct.any() else None)
        np.savez_compressed(out/f'{name}-triggered-predictions.npz',sample_ids=test_ids[eligible],predictions=trigger_pred,labels=y[eligible])
        report['runs'][name]=result
        (out/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v['backdoor_metrics'] for k,v in report['runs'].items()},indent=2))
    print(f'Saved: {out.resolve()}')

if __name__=='__main__': main()

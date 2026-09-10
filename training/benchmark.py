"""Leila: paired three-seed benchmark on one frozen preparation and holdout."""
import json
from pathlib import Path
import numpy as np
from .web_comparison import train_comparison


def summarize(reports):
    summary={}
    for arm in reports[0]['runs']:
        values={}
        for report in reports:
            run=report['runs'][arm];m=run['metrics'];matrix=np.asarray(m['confusion_matrix'],dtype=float)
            actual=matrix.sum(1);predicted=matrix.sum(0);diagonal=matrix.diagonal()
            measured={'accuracy':m['accuracy'],
                'macro_f1':float(np.divide(2*diagonal,actual+predicted,out=np.zeros_like(diagonal),where=actual+predicted>0).mean()),
                'macro_precision':float(np.divide(diagonal,predicted,out=np.zeros_like(diagonal),where=predicted>0).mean())}
            measured.update(run.get('backdoor_metrics',{}))
            for key,value in measured.items():
                if isinstance(value,(int,float)) and value is not None: values.setdefault(key,[]).append(value)
        summary[arm]={key:{'mean':float(np.mean(v)),'std':float(np.std(v,ddof=1)),'n':len(v)} for key,v in values.items() if len(v)==len(reports)}
    return summary


def train_benchmark(version,epochs,output,progress):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    reports=[];seeds=[42,43,44]
    for i,seed in enumerate(seeds):
        report=train_comparison(version,epochs,output/f'seed-{seed}',
            lambda p,m,i=i,seed=seed:progress((i*100+min(p,99))/3,f'Seed {seed} ({i+1}/3) · {m}'),
            seed=seed,matched_steps=True)
        reports.append(report)
        (output/'benchmark-progress.json').write_text(json.dumps({'completed_seeds':seeds[:i+1],'reports':reports},allow_nan=False),encoding='utf-8')
    result=dict(reports[0])
    result.update(benchmark={'seeds':seeds,'summary':summarize(reports),'reports':reports,
        'budget':'Same optimizer steps in every arm; budget = selected epochs × reference batches after validation exclusion'},
        limitation='Three paired training seeds on one fixed prepared dataset. Same validation IDs and optimizer-step budget. Variation describes training randomness, not attack-seed variation. Individual tables and loss curves below show seed 42.',
        result_file=str(output/'comparison.json'))
    (output/'comparison.json').write_text(json.dumps(result,allow_nan=False),encoding='utf-8')
    return result

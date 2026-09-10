"""Prepare IMDB phrase-attack rows; no detector or training is run.

python -m experiments.prepare_imdb_backdoor --limit 1000 --poison-rate .05
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from poison_features import load_imdb_dataset
from poison_features.attacks import poison_texts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--poison-rate', type=float, default=.05)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--trigger', default='silver lantern')
    parser.add_argument('--target-label', type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('limit must be positive')
    data = load_imdb_dataset(split='train')
    indices = np.random.default_rng(args.seed).choice(len(data), min(args.limit,len(data)), replace=False)
    texts = [data[int(i)]['text'] for i in indices]
    labels = [data[int(i)]['label'] for i in indices]
    attacked, current, truth = poison_texts(texts, labels, attack='backdoor',
        poison_rate=args.poison_rate, seed=args.seed, trigger=args.trigger,
        target_label=args.target_label, trigger_position='start', selection_policy='non_target')
    ids = [f'imdb-train:{int(i)}' for i in indices]
    mask = truth['is_poisoned']
    assert int(mask.sum()) == round(len(texts)*args.poison_rate)
    assert np.all(np.asarray(labels)[mask] != args.target_label)
    assert int((current != labels).sum()) == int(mask.sum())
    for i in range(len(texts)):
        assert attacked[i] == (f'{args.trigger} {texts[i]}' if mask[i] else texts[i])
        assert current[i] == (args.target_label if mask[i] else labels[i])
    folder = Path('artifacts/imdb_backdoor_preparation')/datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    folder.mkdir(parents=True)
    # Leila: scanner inputs contain observed text/labels/IDs, never poison identities.
    with (folder/'reviews.jsonl').open('w',encoding='utf-8') as stream:
        for sid, text, label in zip(ids, attacked, current):
            stream.write(json.dumps(dict(sample_id=sid,text=text,label=int(label)),ensure_ascii=False)+'\n')
    np.savez_compressed(folder/'evaluation.npz',sample_ids=np.array(ids),**truth)
    summary=dict(samples=len(ids),poisoned=int(mask.sum()),labels_changed=int((current!=labels).sum()),
                 trigger=args.trigger,target_label=args.target_label,position='start',seed=args.seed,
                 poison_rate=args.poison_rate,selection_policy='non_target',validation='Text insertion and labels verified; ASR not tested')
    (folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print(f'Saved: {folder.resolve()}')

if __name__ == '__main__':
    main()

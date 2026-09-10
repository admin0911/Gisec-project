"""Raw text connector: observed text, labels and stable IDs only."""
from dataclasses import dataclass
import json
import numpy as np

@dataclass(frozen=True)
class TextInputBundle:
    texts: tuple
    labels: np.ndarray
    sample_ids: np.ndarray

    def __post_init__(self):
        texts=tuple(self.texts); labels=np.asarray(self.labels); ids=np.asarray(self.sample_ids)
        if not texts or not all(isinstance(t,str) for t in texts): raise ValueError('Nonempty text rows required')
        if labels.shape!=(len(texts),) or labels.dtype.kind not in 'iu': raise ValueError('Aligned integer labels required')
        if ids.shape!=labels.shape or ids.dtype.kind not in 'iuUS' or len(np.unique(ids))!=len(ids): raise ValueError('Aligned unique IDs required')
        object.__setattr__(self,'texts',texts)
        object.__setattr__(self,'labels',labels.copy())
        object.__setattr__(self,'sample_ids',ids.copy())

    @classmethod
    def load(cls,path):
        # Iterate physical lines: reviews can contain Unicode line separators.
        with open(path,encoding='utf-8') as stream: rows=[json.loads(line) for line in stream if line.strip()]
        return cls(tuple(r['text'] for r in rows),np.array([r['label'] for r in rows]),np.array([r['sample_id'] for r in rows]))

    def save(self,path):
        with open(path,'w',encoding='utf-8') as stream:
            for sid,text,label in zip(self.sample_ids,self.texts,self.labels):
                stream.write(json.dumps(dict(sample_id=sid.item(),text=text,label=int(label)),ensure_ascii=False)+'\n')

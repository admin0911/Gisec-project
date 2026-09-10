"""Experimental repeated phrase + label association + position consistency scan.
No supplied trigger, target, clean reference or poison identities. Flags request review.
"""
from collections import Counter
import math
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from poison_features.text_inputs import TextInputBundle
from detectors.output_connector import detector_result

class RepeatedPhraseDetector:
    def __init__(self,min_count=10,min_fraction=.002,max_fraction=.2,min_purity=.95,min_lift=1.5,min_position_fraction=.8):
        if min_count<2 or not 0<min_fraction<=max_fraction<1 or not .5<min_purity<=1 or min_lift<=1 or not 0<min_position_fraction<=1:
            raise ValueError('Invalid phrase thresholds')
        self.settings=dict(min_count=min_count,min_fraction=min_fraction,max_fraction=max_fraction,min_purity=min_purity,min_lift=min_lift,min_position_fraction=min_position_fraction,
            ngram_range=[2,4],calibration='experimental; not calibrated',flag_meaning='needs review',position_rule='same token offset from start or end')

    def analyze(self,inputs):
        if not isinstance(inputs,TextInputBundle): raise TypeError('Expected TextInputBundle')
        texts=inputs.texts; y=inputs.labels; n=len(texts); s=self.settings
        scores=np.zeros(n); pattern_ids=np.full(n,-1); patterns=[]
        tokens=[re.findall(r'(?u)\b\w+\b',t.lower()) for t in texts]
        lower=max(s['min_count'],math.ceil(s['min_fraction']*n))
        if lower<=math.floor(s['max_fraction']*n):
            vectorizer=CountVectorizer(lowercase=True,token_pattern=r'(?u)\b\w+\b',ngram_range=(2,4),binary=True,min_df=lower,max_df=s['max_fraction'])
            try: matrix=vectorizer.fit_transform(texts).tocsc()
            except ValueError as exc:
                if not any(term in str(exc) for term in ('empty vocabulary','After pruning','max_df corresponds')): raise
                matrix=None
            if matrix is not None:
                for j,phrase in enumerate(vectorizer.get_feature_names_out()):
                    rows=matrix.indices[matrix.indptr[j]:matrix.indptr[j+1]]
                    labels,counts=np.unique(y[rows],return_counts=True); label=labels[counts.argmax()]
                    purity=float(counts.max()/len(rows)); lift=purity/float((y==label).mean())
                    if purity<s['min_purity'] or lift<s['min_lift']: continue
                    words=phrase.split(); width=len(words); locations=Counter()
                    for i in rows:
                        positions=[k for k in range(len(tokens[i])-width+1) if tokens[i][k:k+width]==words]
                        keys={('start',k) for k in positions}|{('end',len(tokens[i])-width-k) for k in positions}
                        locations.update(keys)
                    location,support=sorted(locations.items(),key=lambda item:(-item[1],item[0]))[0]
                    consistency=support/len(rows)
                    if consistency<s['min_position_fraction']: continue
                    score=purity*(1-1/lift)*consistency
                    pid=len(patterns); patterns.append(dict(phrase=phrase,support=len(rows),dominant_label=int(label),purity=purity,lift=lift,position=list(location),position_fraction=consistency,score=score))
                    better=rows[score>scores[rows]]; scores[better]=score; pattern_ids[better]=pid
        return detector_result('repeated_phrase','0.1.0',dict(sample_ids=inputs.sample_ids,scores=scores,flags=pattern_ids>=0,pattern_id=pattern_ids,patterns=patterns),s,expected_sample_ids=inputs.sample_ids)

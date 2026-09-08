"""Algebraic unit test, not a training experiment or empirical causal result."""
from __future__ import annotations
import json
import numpy as np

def novelty(v:np.ndarray)->float:
    if v.ndim!=2 or len(v)<2 or not np.isfinite(v).all():raise ValueError('invalid signatures')
    i,j=np.triu_indices(len(v),1)
    return float((1-(v@v.T)[i,j]).mean())

def general_identity(v:np.ndarray)->float:
    q=len(v)
    return float(1+(np.square(v).sum()-np.square(v.sum(0)).sum())/(q*(q-1)))

def main():
    rng=np.random.default_rng(913);errors=[]
    for _ in range(1000):
        raw=rng.normal(size=(4,12));raw[0]*=rng.choice([0.,1e-15,1.])
        v=raw/np.maximum(np.linalg.norm(raw,axis=1,keepdims=True),1e-12)
        errors.append(abs(novelty(v)-general_identity(v)))
    a=np.array([[1,0],[-1,0],[1,0],[-1,0]],float)
    b=np.array([[1,0],[-1,0],[0,1],[0,-1]],float)
    assert abs(novelty(a)-4/3)<1e-15 and novelty(a)==novelty(b)
    assert np.linalg.matrix_rank(a)==1 and np.linalg.matrix_rank(b)==2
    assert max(errors)<1e-14
    print(json.dumps({'max_general_identity_error':max(errors),'synthetic_trials':1000,
                      'rank1_novelty':novelty(a),'rank2_novelty':novelty(b),'rank1':1,'rank2':2,
                      'scope':'algebraic limitation of pairwise novelty; no trained-model conclusion'},indent=2))
if __name__=='__main__':main()

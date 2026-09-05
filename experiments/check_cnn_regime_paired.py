"""Independent paired-statistic checker for preregistered Issue #67."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
EXPECTED=tuple(range(1400,1430))

def est(x):
    x=np.asarray(x,float); n=len(x); m=float(x.mean()); se=float(stats.sem(x)); k=float(stats.t.ppf(.975,n-1))
    return dict(n=n,mean=m,se=se,ci95_low=m-k*se,ci95_high=m+k*se,p_one_sided=float(stats.ttest_1samp(x,0,alternative='greater').pvalue),positive=int((x>0).sum()))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',default='results'); a=ap.parse_args(); root=Path(a.input_dir)
    p=pd.read_csv(root/'cnn_regime_paired30.csv'); d=pd.read_csv(root/'cnn_regime_decision30.csv').iloc[0]
    if len(p)!=30 or p.rep.duplicated().any() or tuple(sorted(p.rep.astype(int)))!=EXPECTED: raise ValueError('paired grid')
    if not np.allclose(p.full_benefit-p.clean_benefit,p.clean_interaction,rtol=0,atol=2e-16): raise ValueError('clean interaction arithmetic')
    if not np.allclose(p.full_benefit-p.weak_benefit,p.weak_interaction,rtol=0,atol=2e-16): raise ValueError('weak interaction arithmetic')
    full=est(p.full_benefit); clean=est(p.clean_benefit); ci=est(p.clean_interaction); wi=est(p.weak_interaction)
    for prefix,q in [('full',full),('clean_interaction',ci),('weak_interaction',wi)]:
        for k in ('mean','se','ci95_low','ci95_high','p_one_sided'):
            if not np.isclose(q[k],float(d[f'{prefix}_{k}']),rtol=1e-12,atol=1e-15): raise ValueError(f'{prefix} {k}')
    expected='CNN FULL EFFECT REPLICATES / CLEAN INTERACTION DOES NOT'
    if not (full['mean']>0 and full['p_one_sided']<.05): raise ValueError('full frozen condition')
    if ci['mean']>0 and ci['p_one_sided']<.05: raise ValueError('interaction unexpectedly passes')
    if str(d.decision)!=expected: raise ValueError('decision label')
    print('FULL',full); print('CLEAN',clean); print('FULL-CLEAN',ci); print('FULL-WEAK',wi); print('DECISION',expected)
if __name__=='__main__': main()

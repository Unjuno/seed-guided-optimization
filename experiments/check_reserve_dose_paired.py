"""Independent paired-statistic checker for preregistered Issue #64.

This script reads only public paired/decision CSVs; it does not retrain models.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

EXPECTED_REPS=tuple(range(1300,1330))

def estimate(x):
    x=np.asarray(x,dtype=np.float64)
    n=len(x); mean=float(x.mean()); se=float(stats.sem(x)); k=float(stats.t.ppf(.975,n-1))
    p=float(stats.ttest_1samp(x,0,alternative='greater').pvalue)
    return dict(n=n,mean=mean,se=se,ci95_low=mean-k*se,ci95_high=mean+k*se,p_one_sided=p,positive=int((x>0).sum()))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',default='results'); a=ap.parse_args(); root=Path(a.input_dir)
    p=pd.read_csv(root/'reserve_dose_paired30.csv'); d=pd.read_csv(root/'reserve_dose_decision30.csv').iloc[0]
    if len(p)!=30 or p.rep.duplicated().any() or tuple(sorted(p.rep.astype(int)))!=EXPECTED_REPS:
        raise ValueError('paired grid mismatch')
    if not np.isfinite(p[['clean_benefit','full_benefit','interaction']].to_numpy()).all(): raise ValueError('nonfinite')
    if not np.allclose(p.full_benefit-p.clean_benefit,p.interaction,rtol=0,atol=2e-16): raise ValueError('interaction arithmetic mismatch')
    full=estimate(p.full_benefit); clean=estimate(p.clean_benefit); inter=estimate(p.interaction)
    for prefix,est in [('full',full),('interaction',inter)]:
        for key in ('mean','se','ci95_low','ci95_high','p_one_sided'):
            if not np.isclose(est[key],float(d[f'{prefix}_{key}']),rtol=1e-12,atol=1e-15):
                raise ValueError(f'{prefix} {key} mismatch')
    expected='DOSE-DEPENDENT BENEFIT REPLICATES ON RESERVE IMAGES'
    if not (full['mean']>0 and full['p_one_sided']<.05 and inter['mean']>0 and inter['p_one_sided']<.05): raise ValueError('frozen pass conditions do not hold')
    if str(d.decision)!=expected: raise ValueError('decision label mismatch')
    print('FULL',full)
    print('CLEAN secondary',clean)
    print('INTERACTION',inter)
    print('DECISION',expected)

if __name__=='__main__': main()

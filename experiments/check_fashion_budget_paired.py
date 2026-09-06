"""Recompute Issue #73 primary attenuation from public paired CSVs."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

REPS=tuple(range(30,60))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',default='results'); a=ap.parse_args(); root=Path(a.input_dir)
    x=pd.read_csv(root/'fashion_budget_attenuation30.csv')
    if len(x)!=30 or tuple(sorted(x.rep.astype(int)))!=REPS or x.rep.duplicated().any(): raise ValueError('rep grid')
    cols=['benefit_low','benefit_high','benefit_attenuation','slope_vs_coverage']
    if not np.isfinite(x[cols].to_numpy(float)).all(): raise ValueError('nonfinite')
    if not np.allclose(x.benefit_low-x.benefit_high,x.benefit_attenuation,rtol=0,atol=2e-16): raise ValueError('arithmetic')
    v=x.benefit_attenuation.to_numpy(float); mean=float(v.mean()); se=float(v.std(ddof=1)/np.sqrt(len(v))); k=float(stats.t.ppf(.975,len(v)-1)); p=float(stats.ttest_1samp(v,0,alternative='greater').pvalue)
    dec=pd.read_csv(root/'fashion_budget_decision30.csv').iloc[0]
    expected={'mean_benefit_attenuation':mean,'benefit_attenuation_se':se,'benefit_attenuation_ci95_low':mean-k*se,'benefit_attenuation_ci95_high':mean+k*se,'benefit_attenuation_p_one_sided':p}
    for key,val in expected.items():
        if not np.isclose(float(dec[key]),val,rtol=0,atol=5e-15): raise ValueError(key+' mismatch')
    passed=bool(mean>0 and p<.05 and bool(dec.q8_identity_pass) and int(dec.q8_mismatch_count)==0)
    label='FASHION TRANSFORMER FINITE-BUDGET COVERAGE REPLICATES' if passed else 'FAIL'
    print({'n':len(v),'mean_attenuation':mean,'se':se,'ci95':[mean-k*se,mean+k*se],'p_one_sided':p,'positive_pairs':int((v>0).sum()),'decision':label})
if __name__=='__main__': main()

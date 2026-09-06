"""Recompute Issue #70 primary attenuation from public paired CSVs without retraining."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

EXPECTED_REPS=tuple(range(1500,1530))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',default='results'); a=ap.parse_args(); root=Path(a.input_dir)
    x=pd.read_csv(root/'cnn_budget_attenuation30.csv')
    if len(x)!=30 or tuple(sorted(x.rep.astype(int)))!=EXPECTED_REPS or x.rep.duplicated().any(): raise ValueError('rep grid')
    if not np.isfinite(x[['benefit_low','benefit_high','benefit_attenuation','slope_vs_coverage']].to_numpy()).all(): raise ValueError('nonfinite')
    recomputed=x.benefit_low-x.benefit_high
    if not np.allclose(recomputed,x.benefit_attenuation,rtol=0,atol=2e-16): raise ValueError('attenuation arithmetic')
    v=x.benefit_attenuation.to_numpy(float); mean=float(v.mean()); se=float(v.std(ddof=1)/np.sqrt(len(v))); k=float(stats.t.ppf(.975,len(v)-1)); p=float(stats.ttest_1samp(v,0,alternative='greater').pvalue)
    dec=pd.read_csv(root/'cnn_budget_decision30.csv').iloc[0]
    expected=dict(mean_benefit_attenuation=mean,benefit_attenuation_se=se,benefit_attenuation_ci95_low=mean-k*se,benefit_attenuation_ci95_high=mean+k*se,benefit_attenuation_p_one_sided=p)
    for key,val in expected.items():
        if not np.isclose(float(dec[key]),val,rtol=0,atol=5e-15): raise ValueError(f'{key} mismatch')
    passed=bool(mean>0 and p<.05 and bool(dec.q16_identity_pass) and int(dec.q16_mismatch_count)==0)
    label='CNN FINITE-BUDGET COVERAGE REPLICATES' if passed else 'FAIL'
    print({'n':len(v),'mean_attenuation':mean,'se':se,'ci95':[mean-k*se,mean+k*se],'p_one_sided':p,'positive_pairs':int((v>0).sum()),'decision':label})
if __name__=='__main__': main()

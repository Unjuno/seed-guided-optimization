"""Independent paired-statistic check for Issue83; does not retrain or tune."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

def check(path: Path) -> dict:
    d=pd.read_csv(path,float_precision='round_trip')
    cols=['gradient_gap','hardness_diff','parameter_z_diff','heldout_benefit']
    if len(d)!=30 or d.rep.duplicated().any() or set(d.rep)!=set(range(1800,1830)):
        raise ValueError('replicate grid')
    if not np.isfinite(d[cols].to_numpy(float)).all(): raise ValueError('nonfinite data')
    out={}
    for col in cols:
        x=d[col].to_numpy(float);test=stats.ttest_1samp(x,0,alternative='greater')
        ci=stats.ttest_1samp(x,0).confidence_interval(.95)
        out[col]={'mean':float(x.mean()),'se':float(stats.sem(x)),
                  'ci95':[float(ci.low),float(ci.high)],'p_one_sided':float(test.pvalue)}
    gates=[]
    for col in ['hardness_diff','parameter_z_diff']:
        x=d[col];ci=stats.ttest_1samp(x,0).confidence_interval(.90)
        pl=float(stats.ttest_1samp(x,-.05,alternative='greater').pvalue)
        pu=float(stats.ttest_1samp(x,.05,alternative='less').pvalue)
        ok=ci.low>-.05 and ci.high<.05 and pl<.05 and pu<.05
        out[col]['tost_pass']=bool(ok); gates.append(ok)
    for col in ['gradient_gap','heldout_benefit']:
        gates.append(out[col]['mean']>0 and out[col]['p_one_sided']<.05)
    out['frozen_pass']=bool(all(gates));return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('results/parameter_matched_paired30.csv'));a=p.parse_args()
    print(json.dumps(check(a.input),indent=2,allow_nan=False))

"""Independent Issue86 paired t/TOST check; reads public data, never trains."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


def check(path: Path) -> dict:
    d=pd.read_csv(path,float_precision='round_trip')
    required=['g_mean_test_benefit','t_mean_test_benefit','g_gradient_gap',
              't_translation_gap','g_hardness_diff','g_parameter_z_diff',
              't_hardness_diff','t_parameter_z_diff']
    if len(d)!=30 or d.rep.duplicated().any() or set(d.rep)!=set(range(1900,1930)):
        raise ValueError('unregistered or incomplete replicate grid')
    if not np.isfinite(d[required].to_numpy(dtype=float)).all():
        raise ValueError('nonfinite input')
    report={'n':30,'tests':{},'balance':{}}
    for c in required:
        x=d[c].to_numpy(dtype=float)
        ci=stats.ttest_1samp(x,0).confidence_interval(.95)
        report['tests'][c]={'mean':float(x.mean()),'se':float(stats.sem(x)),
                           'ci95':[float(ci.low),float(ci.high)],
                           'p_one_sided':float(stats.ttest_1samp(x,0,alternative='greater').pvalue)}
    balance=True
    for f in ['g','t']:
        for name in ['hardness_diff','parameter_z_diff']:
            c=f+'_'+name;x=d[c];ci=stats.ttest_1samp(x,0).confidence_interval(.90)
            pl=float(stats.ttest_1samp(x,-.05,alternative='greater').pvalue)
            pu=float(stats.ttest_1samp(x,.05,alternative='less').pvalue)
            ok=bool(ci.low>-.05 and ci.high<.05 and pl<.05 and pu<.05)
            report['balance'][c]={'ci90':[float(ci.low),float(ci.high)],'p_lower':pl,'p_upper':pu,'pass':ok}
            balance &= ok
    def positive(c: str) -> bool:
        s=report['tests'][c]
        return s['mean']>0 and s['p_one_sided']<.05
    gates=balance and positive('g_gradient_gap') and positive('t_translation_gap')
    g=positive('g_mean_test_benefit');t=positive('t_mean_test_benefit')
    if not gates: decision='MATCH OR MANIPULATION FAILURE'
    elif g and t: decision='SPATIAL ALLOCATION ALTERNATIVE SUPPORTED'
    elif g: decision='GRADIENT REPLICATION ONLY / SPATIAL ALTERNATIVE NOT SUPPORTED'
    elif t: decision='SPATIAL EFFECT ONLY / GRADIENT REPLICATION FAILED'
    else: decision='NO PERFORMANCE REPLICATION'
    report['decision']=decision
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,default=Path('results/spatial_primary_paired30.csv'));a=ap.parse_args()
    print(json.dumps(check(a.input),indent=2,allow_nan=False))

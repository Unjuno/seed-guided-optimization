"""Prespecified secondary coordinate summaries; no confirmatory multiplicity claim."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

COORDS=('angle','dx','dy','blur','contrast','brightness','noise')
def run(path:Path,out:Path)->None:
    df=pd.read_csv(path,float_precision='round_trip')
    expected={(r,f) for r in range(2200,2230) for f in ('u','m')}
    if len(df)!=60 or df.duplicated(['rep','family']).any() or set(df[['rep','family']].itertuples(index=False,name=None))!=expected:
        raise ValueError('invalid reference grid')
    rows=[]
    for f in ('u','m'):
        for kind in ('mean','var'):
            for c in COORDS:
                vals=df.loc[df.family==f,f'{kind}_{c}_diff'].to_numpy(float)
                if not np.isfinite(vals).all():raise ValueError('nonfinite')
                mean=float(vals.mean());se=float(vals.std(ddof=1)/np.sqrt(len(vals)));k=float(stats.t.ppf(.975,len(vals)-1))
                rows.append({'family':f,'kind':kind,'coordinate':c,'n':len(vals),'mean':mean,'se':se,
                             'ci95_low_descriptive':mean-k*se,'ci95_high_descriptive':mean+k*se})
    out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    print(pd.DataFrame(rows).query("family=='m' and kind=='var'").to_string(index=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(Path(a.input),Path(a.output))

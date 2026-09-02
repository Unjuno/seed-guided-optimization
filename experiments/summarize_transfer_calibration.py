from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np, pandas as pd

EXPECTED_REPS=tuple(range(850,860))
ALPHAS=tuple(np.round(np.arange(0.10,0.901,0.05),2))


def load_parts(root:Path)->pd.DataFrame:
    files=sorted(root.rglob('transfer_calibration_*.csv'))
    if not files: raise FileNotFoundError('no transfer calibration files')
    return pd.concat([pd.read_csv(p) for p in files],ignore_index=True)


def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',required=True); ap.add_argument('--output-dir',required=True); a=ap.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); df=load_parts(Path(a.input_dir))
    expected=len(EXPECTED_REPS)*(1+len(ALPHAS))
    if len(df)!=expected: raise ValueError(f'expected {expected} rows got {len(df)}')
    shared=df[df.family=='shared']
    if len(shared)!=len(EXPECTED_REPS) or shared.duplicated(['rep']).any(): raise ValueError('shared calibration invalid')
    if set(shared.rep.astype(int))!=set(EXPECTED_REPS): raise ValueError('shared rep set mismatch')
    shared_mean=float(shared.mean_test.mean()); rows=[]
    for alpha in ALPHAS:
        part=df[(df.family=='nuisance') & np.isclose(df.alpha,alpha)]
        if len(part)!=len(EXPECTED_REPS) or part.duplicated(['rep']).any(): raise ValueError(f'nuisance alpha {alpha} invalid')
        if set(part.rep.astype(int))!=set(EXPECTED_REPS): raise ValueError(f'nuisance alpha {alpha} rep mismatch')
        nmean=float(part.mean_test.mean()); mismatch=abs(nmean-shared_mean)
        rows.append({'alpha':alpha,'shared_loss_hard_mean_accuracy':shared_mean,'nuisance_loss_hard_mean_accuracy':nmean,'accuracy_mismatch':mismatch})
    summary=pd.DataFrame(rows).sort_values(['accuracy_mismatch','alpha']).reset_index(drop=True); best=summary.iloc[0]; passed=bool(best.accuracy_mismatch<=0.01)
    decision=pd.DataFrame([{'selected_alpha':float(best.alpha),'calibration_shared_mean_accuracy':float(best.shared_loss_hard_mean_accuracy),'calibration_nuisance_mean_accuracy':float(best.nuisance_loss_hard_mean_accuracy),'calibration_accuracy_mismatch':float(best.accuracy_mismatch),'accuracy_tolerance':0.01,'calibration_pass':passed,'gradnov_used_for_calibration':False,'decision':'CALIBRATION DIFFICULTY PASS' if passed else 'CALIBRATION DIFFICULTY MATCH FAILURE'}])
    summary.to_csv(out/'transfer_calibration_summary.csv',index=False); decision.to_csv(out/'transfer_calibration_decision.csv',index=False); print(decision.to_string(index=False),flush=True)

if __name__=='__main__': main()

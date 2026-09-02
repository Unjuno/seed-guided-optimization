from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np, pandas as pd

EXPECTED_REPS=tuple(range(1050,1060))
LAMBDAS=(0.01,0.02,0.03,0.04,0.05,0.075,0.10,0.15,0.20)
ALPHAS=(0.005,0.01,0.015,0.02,0.03,0.04,0.05)


def load_parts(root:Path)->pd.DataFrame:
    files=sorted(root.rglob('dual_transfer_calibration_*.csv'))
    if not files: raise FileNotFoundError('no dual transfer calibration files')
    return pd.concat([pd.read_csv(p) for p in files],ignore_index=True)


def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',required=True); ap.add_argument('--output-dir',required=True); a=ap.parse_args()
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); df=load_parts(Path(a.input_dir))
    expected=len(EXPECTED_REPS)*(len(LAMBDAS)+len(ALPHAS))
    if len(df)!=expected: raise ValueError(f'expected {expected} rows got {len(df)}')
    rows=[]
    shared_means={}; nuisance_means={}
    for lam in LAMBDAS:
        part=df[(df.family=='shared') & np.isclose(df.lambda_strength,lam)]
        if len(part)!=len(EXPECTED_REPS) or part.duplicated(['rep']).any(): raise ValueError(f'shared lambda {lam} invalid')
        if set(part.rep.astype(int))!=set(EXPECTED_REPS): raise ValueError(f'shared lambda {lam} rep mismatch')
        shared_means[lam]=float(part.mean_test.mean())
    for alpha in ALPHAS:
        part=df[(df.family=='nuisance') & np.isclose(df.alpha,alpha)]
        if len(part)!=len(EXPECTED_REPS) or part.duplicated(['rep']).any(): raise ValueError(f'nuisance alpha {alpha} invalid')
        if set(part.rep.astype(int))!=set(EXPECTED_REPS): raise ValueError(f'nuisance alpha {alpha} rep mismatch')
        nuisance_means[alpha]=float(part.mean_test.mean())
    for lam in LAMBDAS:
        for alpha in ALPHAS:
            sm=shared_means[lam]; nm=nuisance_means[alpha]
            rows.append({'lambda_strength':lam,'alpha':alpha,'shared_loss_hard_mean_accuracy':sm,'nuisance_loss_hard_mean_accuracy':nm,'accuracy_mismatch':abs(sm-nm)})
    summary=pd.DataFrame(rows).sort_values(['accuracy_mismatch','lambda_strength','alpha']).reset_index(drop=True)
    best=summary.iloc[0]; passed=bool(best.accuracy_mismatch<=0.01)
    decision=pd.DataFrame([{'selected_lambda':float(best.lambda_strength),'selected_alpha':float(best.alpha),'calibration_shared_mean_accuracy':float(best.shared_loss_hard_mean_accuracy),'calibration_nuisance_mean_accuracy':float(best.nuisance_loss_hard_mean_accuracy),'calibration_accuracy_mismatch':float(best.accuracy_mismatch),'accuracy_tolerance':0.01,'calibration_pass':passed,'gradnov_used_for_calibration':False,'decision':'CALIBRATION DIFFICULTY PASS' if passed else 'CALIBRATION DIFFICULTY MATCH FAILURE'}])
    summary.to_csv(out/'dual_transfer_calibration_summary.csv',index=False); decision.to_csv(out/'dual_transfer_calibration_decision.csv',index=False); print(decision.to_string(index=False),flush=True)

if __name__=='__main__': main()

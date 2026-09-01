from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_REPS=tuple(range(650,660))
ALPHAS=(0.60,0.65,0.70,0.75)
BETAS=(0.03,0.06,0.09,0.12,0.15)


def load_parts(root:Path)->pd.DataFrame:
    files=sorted(root.rglob('two_axis_calibration_*.csv'))
    if not files: raise FileNotFoundError('no two-axis calibration files')
    return pd.concat([pd.read_csv(p) for p in files],ignore_index=True)


def summarize(frame:pd.DataFrame)->tuple[float,float]:
    novelty=[]; losses=[]
    for rep in EXPECTED_REPS:
        part=frame[frame.rep==rep].set_index('method')
        if set(part.index)!={'loss_hard','gradnov'}: raise ValueError(f'incomplete rep {rep}')
        novelty.append(float(part.loc['gradnov','selected_pairwise_novelty']-part.loc['loss_hard','selected_pairwise_novelty']))
        losses.append(float(part['mean_candidate_loss'].mean()))
    return float(np.mean(novelty)),float(np.mean(losses))


def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',required=True); ap.add_argument('--output-dir',required=True); a=ap.parse_args()
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); df=load_parts(Path(a.input_dir))
    expected=len(EXPECTED_REPS)*(1+len(ALPHAS)*len(BETAS))*2
    if len(df)!=expected: raise ValueError(f'expected {expected} rows got {len(df)}')
    if df.duplicated(['rep','family','alpha','beta','method'],keep=False).any() and False: pass
    structured=df[df.family=='structured']
    if len(structured)!=len(EXPECTED_REPS)*2: raise ValueError('structured calibration incomplete')
    s_nov,s_loss=summarize(structured)
    rows=[]
    for alpha in ALPHAS:
        for beta in BETAS:
            part=df[(df.family=='nuisance') & np.isclose(df.alpha,alpha) & np.isclose(df.beta,beta)]
            if len(part)!=len(EXPECTED_REPS)*2: raise ValueError(f'grid incomplete alpha={alpha} beta={beta}')
            n_nov,n_loss=summarize(part); nm=abs(n_nov-s_nov); lm=abs(n_loss-s_loss)
            rows.append({'alpha':alpha,'beta':beta,'structured_novelty_gain':s_nov,'nuisance_novelty_gain':n_nov,'novelty_mismatch':nm,'structured_candidate_loss':s_loss,'nuisance_candidate_loss':n_loss,'candidate_loss_mismatch':lm,'calibration_score':nm/0.03+lm/0.10})
    summary=pd.DataFrame(rows).sort_values(['calibration_score','alpha','beta']).reset_index(drop=True); best=summary.iloc[0]
    passed=bool(best.novelty_mismatch<=0.03 and best.candidate_loss_mismatch<=0.10)
    decision=pd.DataFrame([{'selected_alpha':float(best.alpha),'selected_beta':float(best.beta),'calibration_novelty_mismatch':float(best.novelty_mismatch),'novelty_tolerance':0.03,'calibration_candidate_loss_mismatch':float(best.candidate_loss_mismatch),'candidate_loss_tolerance':0.10,'calibration_pass':passed,'heldout_used':False,'decision':'CALIBRATION GATE PASS' if passed else 'CALIBRATION MATCH FAILURE'}])
    summary.to_csv(out/'two_axis_calibration_summary.csv',index=False); decision.to_csv(out/'two_axis_calibration_decision.csv',index=False); print(decision.to_string(index=False),flush=True)


if __name__=='__main__': main()

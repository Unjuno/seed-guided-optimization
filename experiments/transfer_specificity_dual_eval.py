from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch
import transfer_specificity as base

LAMBDAS=(0.01,0.02,0.03,0.04,0.05,0.075,0.10,0.15,0.20)
ALPHAS=(0.005,0.01,0.015,0.02,0.03,0.04,0.05)
CAL_TRAIN_SEEDS=np.arange(36000,36064,dtype=int)
CAL_SHARED_SEEDS=np.arange(37000,37040,dtype=int)
CAL_NUISANCE_SEEDS=np.arange(37100,37140,dtype=int)
CONF_TRAIN_SEEDS=np.arange(38000,38064,dtype=int)
CONF_SHARED_SEEDS=np.arange(39000,39080,dtype=int)
CONF_NUISANCE_SEEDS=np.arange(40000,40080,dtype=int)


def shared_envs(images,seeds,lam):
    x=np.asarray(images,np.float32)
    out=[]
    for seed in seeds:
        g=np.asarray(base.geometric_environment(x,int(seed)),np.float32)
        y=np.clip((1.0-float(lam))*x+float(lam)*g,0,1).astype(np.float32)
        out.append(torch.tensor(y))
    return out


def nuisance_envs(images,seeds,alpha):
    return [torch.tensor(base.nuisance_environment(images,int(s),float(alpha))) for s in seeds]


def calibrate(start,end,outdir):
    base.configure_determinism(1); outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    xtr,ytr,xte,yte=base.load_digits_split(); ty=torch.tensor(ytr); ey=torch.tensor(yte); clean=torch.tensor(xte).flatten(1)
    train=base.geometric_envs(xtr,CAL_TRAIN_SEEDS)
    shared={lam:shared_envs(xte,CAL_SHARED_SEEDS,lam) for lam in LAMBDAS}
    nuisance={alpha:nuisance_envs(xte,CAL_NUISANCE_SEEDS,alpha) for alpha in ALPHAS}
    rows=[]
    for rep in range(start,end):
        seed=1_510_000_000+rep*4099; sched=base.schedule(len(ty),seed,CAL_TRAIN_SEEDS); model,diag=base.train_one(train,ty,sched,seed,'loss_hard')
        print('CAL_TRAINING_ONLY '+json.dumps(dict(rep=rep,method='loss_hard',**diag)),flush=True)
        for lam in LAMBDAS:
            m=base.environment_metrics(model,shared[lam],ey,clean); rows.append(dict(rep=rep,family='shared',lambda_strength=lam,alpha=np.nan,**m))
        for alpha in ALPHAS:
            m=base.environment_metrics(model,nuisance[alpha],ey,clean); rows.append(dict(rep=rep,family='nuisance',lambda_strength=np.nan,alpha=alpha,**m))
        pd.DataFrame(rows).to_csv(outdir/f'dual_transfer_calibration_{start}_{end}.csv',index=False)
    print(json.dumps({'event':'CALIBRATION_COMPLETE','gradnov_used':False}),flush=True)


def confirm_train(start,end,outdir):
    base.CONF_TRAIN_SEEDS=CONF_TRAIN_SEEDS
    base.confirm_train(start,end,outdir)


def confirm_evaluate(start,end,outdir,lam,alpha):
    base.configure_determinism(1); outdir=Path(outdir); _,_,xte,yte=base.load_digits_split(); ey=torch.tensor(yte); clean=torch.tensor(xte).flatten(1)
    shared=shared_envs(xte,CONF_SHARED_SEEDS,lam); nuisance=nuisance_envs(xte,CONF_NUISANCE_SEEDS,alpha); states=outdir/'states'; rows=[]
    for rep in range(start,end):
        for method in ('loss_hard','gradnov'):
            ck=torch.load(states/f'rep{rep:03d}_{method}.pt',map_location='cpu'); model=base.MLP(); model.load_state_dict(ck['state_dict'])
            for family,envs in (('shared',shared),('nuisance',nuisance)):
                metrics=base.environment_metrics(model,envs,ey,clean)
                row=dict(rep=rep,method=method,family=family,lambda_strength=lam if family=='shared' else np.nan,alpha=alpha if family=='nuisance' else np.nan,**metrics)
                rows.append(row); print('CONFIRM_HELDOUT '+json.dumps(row),flush=True)
    pd.DataFrame(rows).to_csv(outdir/f'dual_transfer_heldout_{start}_{end}.csv',index=False)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['calibrate','train','evaluate']); ap.add_argument('--start',type=int,required=True); ap.add_argument('--end',type=int,required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--lambda-strength',type=float); ap.add_argument('--alpha',type=float); a=ap.parse_args()
    if a.mode=='calibrate': calibrate(a.start,a.end,a.output_dir)
    elif a.mode=='train': confirm_train(a.start,a.end,a.output_dir)
    else:
        if a.lambda_strength is None or a.alpha is None: raise ValueError('--lambda-strength and --alpha required')
        confirm_evaluate(a.start,a.end,a.output_dir,a.lambda_strength,a.alpha)

if __name__=='__main__': main()

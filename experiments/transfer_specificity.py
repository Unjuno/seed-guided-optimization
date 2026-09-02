from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch

from common import MLP, configure_determinism, environment_metrics, geometric_environment, head_gradient_directions, load_digits_split, seed_everything, select_hard_gradient_novel, select_loss_hard

K=16; Q=4; BATCH=128; EPOCHS=10; LR=1e-2; WD=1e-3; NOVELTY_WEIGHT=0.6
CAL_TRAIN_SEEDS=np.arange(26000,26064,dtype=int)
CAL_SHARED_SEEDS=np.arange(27000,27040,dtype=int)
CAL_NUISANCE_SEEDS=np.arange(27100,27140,dtype=int)
CONF_TRAIN_SEEDS=np.arange(28000,28064,dtype=int)
CONF_SHARED_SEEDS=np.arange(29000,29080,dtype=int)
CONF_NUISANCE_SEEDS=np.arange(30000,30080,dtype=int)
ALPHAS=tuple(np.round(np.arange(0.10,0.901,0.05),2))


def nuisance_environment(images,seed:int,alpha:float):
    x=np.asarray(images,np.float32).reshape(len(images),64)
    rng=np.random.default_rng(seed*8191+149); perm=rng.permutation(64)
    y=(1-alpha)*x+alpha*x[:,perm]
    return np.clip(y,0,1).reshape(len(images),8,8).astype(np.float32)


def selected_novelty(cos,sel):
    idx=torch.tensor(sel,dtype=torch.long); sub=cos.index_select(0,idx).index_select(1,idx); up=torch.triu_indices(len(sel),len(sel),offset=1)
    return float((1-sub[up[0],up[1]]).mean())


def schedule(n,base,seeds):
    tg=torch.Generator().manual_seed(base+1); er=np.random.default_rng(base+2); out=[]
    for _ in range(EPOCHS):
        for b in torch.randperm(n,generator=tg).split(BATCH): out.append((b,er.choice(len(seeds),K,replace=False).tolist()))
    return out


def train_one(envs,y,sched,base,method):
    seed_everything(base); model=MLP(); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD); ns=[]; cs=[]; ss=[]
    model.train()
    for b,cand in sched:
        xb=torch.cat([envs[e][b] for e in cand]); logits,h=model(xb)
        per=torch.nn.functional.cross_entropy(logits,y[b].repeat(K),reduction='none').reshape(K,-1); losses=per.mean(1)
        dirs=head_gradient_directions(logits,h,y[b],K); cos=dirs@dirs.T
        sel=select_loss_hard(losses,Q) if method=='loss_hard' else select_hard_gradient_novel(losses,cos,q=Q,novelty_weight=NOVELTY_WEIGHT)
        sel=sorted(sel); ns.append(selected_novelty(cos,sel)); cs.append(float(losses.mean().detach())); ss.append(float(losses[sel].mean().detach()))
        opt.zero_grad(set_to_none=True); per[sel].mean().backward(); opt.step()
    return model,dict(selected_pairwise_novelty=float(np.mean(ns)),mean_candidate_loss=float(np.mean(cs)),mean_selected_loss=float(np.mean(ss)))


def geometric_envs(x,seeds): return [torch.tensor(geometric_environment(x,int(s))) for s in seeds]
def nuisance_envs(x,seeds,alpha): return [torch.tensor(nuisance_environment(x,int(s),float(alpha))) for s in seeds]


def calibrate(start,end,outdir):
    configure_determinism(1); outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True); xtr,ytr,xte,yte=load_digits_split(); ty=torch.tensor(ytr); ey=torch.tensor(yte); clean=torch.tensor(xte).flatten(1)
    train=geometric_envs(xtr,CAL_TRAIN_SEEDS); shared=geometric_envs(xte,CAL_SHARED_SEEDS); nuisance={a:nuisance_envs(xte,CAL_NUISANCE_SEEDS,a) for a in ALPHAS}; rows=[]
    for rep in range(start,end):
        base=1_310_000_000+rep*4099; sched=schedule(len(ty),base,CAL_TRAIN_SEEDS); model,diag=train_one(train,ty,sched,base,'loss_hard')
        print('CAL_TRAINING_ONLY '+json.dumps(dict(rep=rep,method='loss_hard',**diag)),flush=True)
        # Evaluation families are constructed/evaluated only after the loss-hard state is sealed in memory.
        m=environment_metrics(model,shared,ey,clean); rows.append(dict(rep=rep,family='shared',alpha=np.nan,**m))
        for alpha in ALPHAS:
            m=environment_metrics(model,nuisance[alpha],ey,clean); rows.append(dict(rep=rep,family='nuisance',alpha=alpha,**m))
        pd.DataFrame(rows).to_csv(outdir/f'transfer_calibration_{start}_{end}.csv',index=False)
    print(json.dumps({'event':'CALIBRATION_COMPLETE','gradnov_used':False}),flush=True)


def confirm_train(start,end,outdir):
    configure_determinism(1); outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True); states=outdir/'states'; states.mkdir(exist_ok=True); xtr,ytr,_,_=load_digits_split(); ty=torch.tensor(ytr); train=geometric_envs(xtr,CONF_TRAIN_SEEDS); rows=[]
    for rep in range(start,end):
        base=1_410_000_000+rep*4099; sched=schedule(len(ty),base,CONF_TRAIN_SEEDS)
        for method in ('loss_hard','gradnov'):
            model,diag=train_one(train,ty,sched,base,method); row=dict(rep=rep,method=method,**diag); rows.append(row)
            torch.save({'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'rep':rep,'method':method},states/f'rep{rep:03d}_{method}.pt')
            print('CONFIRM_TRAINING_ONLY '+json.dumps(row),flush=True)
        pd.DataFrame(rows).to_csv(outdir/f'transfer_diagnostics_{start}_{end}.csv',index=False)
    print(json.dumps({'event':'CONFIRM_TRAINING_ONLY_COMPLETE','heldout_constructed':False}),flush=True)


def confirm_evaluate(start,end,outdir,alpha):
    configure_determinism(1); outdir=Path(outdir); _,_,xte,yte=load_digits_split(); ey=torch.tensor(yte); clean=torch.tensor(xte).flatten(1); shared=geometric_envs(xte,CONF_SHARED_SEEDS); nuisance=nuisance_envs(xte,CONF_NUISANCE_SEEDS,alpha); states=outdir/'states'; rows=[]
    for rep in range(start,end):
        for method in ('loss_hard','gradnov'):
            ck=torch.load(states/f'rep{rep:03d}_{method}.pt',map_location='cpu'); model=MLP(); model.load_state_dict(ck['state_dict'])
            for family,envs in (('shared',shared),('nuisance',nuisance)):
                metrics=environment_metrics(model,envs,ey,clean); row=dict(rep=rep,method=method,family=family,alpha=np.nan if family=='shared' else alpha,**metrics); rows.append(row); print('CONFIRM_HELDOUT '+json.dumps(row),flush=True)
    pd.DataFrame(rows).to_csv(outdir/f'transfer_heldout_{start}_{end}.csv',index=False)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['calibrate','train','evaluate']); ap.add_argument('--start',type=int,required=True); ap.add_argument('--end',type=int,required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--alpha',type=float); a=ap.parse_args()
    if a.mode=='calibrate': calibrate(a.start,a.end,a.output_dir)
    elif a.mode=='train': confirm_train(a.start,a.end,a.output_dir)
    else:
        if a.alpha is None: raise ValueError('--alpha required')
        confirm_evaluate(a.start,a.end,a.output_dir,a.alpha)
if __name__=='__main__': main()

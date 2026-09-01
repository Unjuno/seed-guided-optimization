from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch

from common import MLP, configure_determinism, environment_metrics, geometric_environment, head_gradient_directions, load_digits_split, seed_everything, select_hard_gradient_novel, select_loss_hard

K=16; Q=4; BATCH=128; EPOCHS=10; LR=1e-2; WD=1e-3; NOVELTY_WEIGHT=0.6
TRAIN_SEEDS=np.arange(22000,22064,dtype=int); HELDOUT_SEEDS=np.arange(23000,23080,dtype=int)
ALPHAS=(0.60,0.65,0.70,0.75); BETAS=(0.03,0.06,0.09,0.12,0.15)


def nuisance_environment(images, seed:int, alpha:float, beta:float):
    x=np.asarray(images,np.float32).reshape(len(images),64)
    rng=np.random.default_rng(seed*8191+149)
    perm=rng.permutation(64)
    z=rng.normal(size=64).astype(np.float32); z=(z-z.mean())/(z.std()+1e-8)
    m=(1-alpha)*x+alpha*x[:,perm]
    return np.clip(m+beta*z[None,:],0,1).reshape(len(images),8,8).astype(np.float32)


def novelty(cos,sel):
    idx=torch.tensor(sel,dtype=torch.long); sub=cos.index_select(0,idx).index_select(1,idx); up=torch.triu_indices(len(sel),len(sel),offset=1)
    return float((1-sub[up[0],up[1]]).mean())


def schedule(n,base):
    tg=torch.Generator().manual_seed(base+1); er=np.random.default_rng(base+2); out=[]
    for _ in range(EPOCHS):
        for b in torch.randperm(n,generator=tg).split(BATCH): out.append((b,er.choice(len(TRAIN_SEEDS),K,replace=False).tolist()))
    return out


def make_envs(x,family,alpha=None,beta=None):
    if family=='structured': return [torch.tensor(geometric_environment(x,int(s))) for s in TRAIN_SEEDS]
    return [torch.tensor(nuisance_environment(x,int(s),float(alpha),float(beta))) for s in TRAIN_SEEDS]


def train_one(envs,y,sched,base,method):
    seed_everything(base); model=MLP(); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD); ns=[]; cs=[]; ss=[]
    model.train()
    for b,cand in sched:
        xb=torch.cat([envs[e][b] for e in cand]); logits,h=model(xb)
        per=torch.nn.functional.cross_entropy(logits,y[b].repeat(K),reduction='none').reshape(K,-1); losses=per.mean(1)
        d=head_gradient_directions(logits,h,y[b],K); cos=d@d.T
        sel=select_loss_hard(losses,Q) if method=='loss_hard' else select_hard_gradient_novel(losses,cos,q=Q,novelty_weight=NOVELTY_WEIGHT)
        sel=sorted(sel); ns.append(novelty(cos,sel)); cs.append(float(losses.mean().detach())); ss.append(float(losses[sel].mean().detach()))
        opt.zero_grad(set_to_none=True); per[sel].mean().backward(); opt.step()
    return model,dict(selected_pairwise_novelty=float(np.mean(ns)),mean_candidate_loss=float(np.mean(cs)),mean_selected_loss=float(np.mean(ss)))


def calibrate(start,end,outdir):
    configure_determinism(1); outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True); x,y,_,_=load_digits_split(); ty=torch.tensor(y)
    struct=make_envs(x,'structured'); grid={(a,b):make_envs(x,'nuisance',a,b) for a in ALPHAS for b in BETAS}; rows=[]
    for rep in range(start,end):
        base=910_000_000+rep*4099; sched=schedule(len(ty),base)
        cond=[('structured',np.nan,np.nan,struct)]+[('nuisance',a,b,grid[(a,b)]) for a in ALPHAS for b in BETAS]
        for fam,a,b,envs in cond:
            for method in ('loss_hard','gradnov'):
                _,diag=train_one(envs,ty,sched,base,method); row=dict(rep=rep,family=fam,alpha=a,beta=b,method=method,**diag); rows.append(row); print('CALIBRATION '+json.dumps(row),flush=True)
        pd.DataFrame(rows).to_csv(outdir/f'two_axis_calibration_{start}_{end}.csv',index=False)
    print(json.dumps({'event':'CALIBRATION_COMPLETE','heldout_constructed':False}),flush=True)


def confirm_train(start,end,outdir,alpha,beta):
    configure_determinism(1); outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True); states=outdir/'states'; states.mkdir(exist_ok=True); x,y,_,_=load_digits_split(); ty=torch.tensor(y)
    envs={'structured':make_envs(x,'structured'),'nuisance':make_envs(x,'nuisance',alpha,beta)}; rows=[]
    for rep in range(start,end):
        base=1_010_000_000+rep*4099; sched=schedule(len(ty),base)
        for fam in ('structured','nuisance'):
            for method in ('loss_hard','gradnov'):
                model,diag=train_one(envs[fam],ty,sched,base,method); row=dict(rep=rep,family=fam,alpha=np.nan if fam=='structured' else alpha,beta=np.nan if fam=='structured' else beta,method=method,**diag); rows.append(row)
                torch.save({'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'rep':rep,'family':fam,'method':method,'alpha':alpha,'beta':beta},states/f'rep{rep:03d}_{fam}_{method}.pt')
                print('CONFIRM_TRAINING_ONLY '+json.dumps(row),flush=True)
        pd.DataFrame(rows).to_csv(outdir/f'two_axis_diagnostics_{start}_{end}.csv',index=False)
    print(json.dumps({'event':'CONFIRM_TRAINING_ONLY_COMPLETE','alpha':alpha,'beta':beta,'heldout_constructed':False}),flush=True)


def confirm_evaluate(start,end,outdir,alpha,beta):
    configure_determinism(1); outdir=Path(outdir); _,_,x,y=load_digits_split(); ey=torch.tensor(y); clean=torch.tensor(x).flatten(1); states=outdir/'states'
    held={'structured':[torch.tensor(geometric_environment(x,int(s))) for s in HELDOUT_SEEDS],'nuisance':[torch.tensor(nuisance_environment(x,int(s),alpha,beta)) for s in HELDOUT_SEEDS]}; rows=[]
    for rep in range(start,end):
        for fam in ('structured','nuisance'):
            for method in ('loss_hard','gradnov'):
                ck=torch.load(states/f'rep{rep:03d}_{fam}_{method}.pt',map_location='cpu')
                if float(ck['alpha'])!=float(alpha) or float(ck['beta'])!=float(beta): raise ValueError('sealed parameter mismatch')
                model=MLP(); model.load_state_dict(ck['state_dict']); metrics=environment_metrics(model,held[fam],ey,clean)
                row=dict(rep=rep,family=fam,alpha=np.nan if fam=='structured' else alpha,beta=np.nan if fam=='structured' else beta,method=method,**metrics); rows.append(row); print('CONFIRM_HELDOUT '+json.dumps(row),flush=True)
    pd.DataFrame(rows).to_csv(outdir/f'two_axis_heldout_{start}_{end}.csv',index=False)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['calibrate','train','evaluate']); ap.add_argument('--start',type=int,required=True); ap.add_argument('--end',type=int,required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--alpha',type=float); ap.add_argument('--beta',type=float); a=ap.parse_args()
    if a.mode=='calibrate': calibrate(a.start,a.end,a.output_dir)
    elif a.mode=='train':
        if a.alpha is None or a.beta is None: raise ValueError('--alpha/--beta required')
        confirm_train(a.start,a.end,a.output_dir,a.alpha,a.beta)
    else:
        if a.alpha is None or a.beta is None: raise ValueError('--alpha/--beta required')
        confirm_evaluate(a.start,a.end,a.output_dir,a.alpha,a.beta)
if __name__=='__main__': main()

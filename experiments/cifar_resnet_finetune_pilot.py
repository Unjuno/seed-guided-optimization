from __future__ import annotations

import argparse, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import ttest_rel

from experiments.cifar_resnet_pilot import (
    CIFARResNet20, configure, seed_everything, load_cifar, normalize,
    build_schedule, apply_environment, normalized_env_parameters,
    head_gradient_directions, select_loss, select_gradnov, select_paramnov,
)


def cache_test(xte, seeds):
    return [((apply_environment(xte, s, 900000 + i) * 255).round().to(torch.uint8)) for i, s in enumerate(seeds)]


def evaluate_cached(model, test_cache, yte, xte, batch=256):
    model.eval(); acc=[]
    with torch.no_grad():
        for u8 in test_cache:
            correct=0
            for idx in torch.arange(len(yte)).split(batch):
                x=u8[idx].float()/255
                correct += int((model(normalize(x))[0].argmax(1)==yte[idx]).sum())
            acc.append(correct/len(yte))
        clean=sum(int((model(normalize(xte[idx]))[0].argmax(1)==yte[idx]).sum()) for idx in torch.arange(len(yte)).split(batch))/len(yte)
    a=np.asarray(acc)
    return {'mean_test':float(a.mean()),'sd_test':float(a.std(ddof=1)),'p10_test':float(np.quantile(a,.1)),'min_test':float(a.min()),'clean_test':float(clean)}


def pretrain(base, xtr, ytr, epochs, batch, lr):
    seed_everything(base); model=CIFARResNet20(); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    tg=torch.Generator().manual_seed(base+333); model.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(ytr),generator=tg).split(batch):
            logits,_=model(normalize(xtr[idx])); loss=F.cross_entropy(logits,ytr[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return {k:v.detach().clone() for k,v in model.state_dict().items()}


def cache_schedule(xtr, schedule, train_env):
    out=[]
    for step, idx, cand in schedule:
        seeds=[train_env[e] for e in cand]
        xb=torch.cat([apply_environment(xtr[idx],s,step) for s in seeds],0)
        out.append((step,idx,cand,(xb*255).round().to(torch.uint8)))
    return out


def run(args):
    configure(); xtr,ytr,xte,yte=load_cifar(args.data_root,args.train_per_class,args.test_per_class)
    train_env=[20000+i for i in range(64)]; test_env=[30000+i for i in range(args.test_envs)]
    test_cache=cache_test(xte,test_env); methods=['loss4','random4','paramnov4','gradnov4']; rows=[]; started=time.time()
    for rep in range(args.reps):
        base=81_000_000+rep*7919
        state=pretrain(base,xtr,ytr,args.pretrain_epochs,args.batch_size,args.pretrain_lr)
        base_model=CIFARResNet20(); base_model.load_state_dict(state)
        base_metrics=evaluate_cached(base_model,test_cache,yte,xte)
        schedule=build_schedule(len(ytr),args.finetune_epochs,args.batch_size,64,args.k,base+1000)
        cached=cache_schedule(xtr,schedule,train_env)
        for mi,method in enumerate(methods):
            seed_everything(base); model=CIFARResNet20(); model.load_state_dict(state)
            opt=torch.optim.AdamW(model.parameters(),lr=args.finetune_lr,weight_decay=1e-4)
            rr=np.random.default_rng(base+98765+mi); st=time.time(); model.train()
            for step,idx,cand,u8 in cached:
                xb=normalize(u8.float()/255); logits,h=model(xb)
                per=F.cross_entropy(logits,ytr[idx].repeat(args.k),reduction='none').reshape(args.k,-1)
                loss=per.mean(1); gd=head_gradient_directions(logits,h,ytr[idx],args.k)
                if method=='loss4': chosen=select_loss(loss,args.q)
                elif method=='random4': chosen=rr.choice(args.k,args.q,replace=False).tolist()
                elif method=='paramnov4':
                    seeds=[train_env[e] for e in cand]
                    chosen=select_paramnov(loss,np.stack([normalized_env_parameters(s) for s in seeds]),args.q)
                else: chosen=select_gradnov(loss,gd@gd.T,args.q)
                opt.zero_grad(set_to_none=True); per[chosen].mean().backward(); opt.step()
            row={'rep':rep,'method':method,'train_seconds':time.time()-st,'pretrain_epochs':args.pretrain_epochs,'finetune_epochs':args.finetune_epochs,'base_clean':base_metrics['clean_test'],'base_mean':base_metrics['mean_test'],**evaluate_cached(model,test_cache,yte,xte)}
            rows.append(row); print(json.dumps(row),flush=True)
        pd.DataFrame(rows).to_csv(args.output,index=False); print(f'rep {rep} completed; elapsed={time.time()-started:.1f}s',flush=True)
    df=pd.DataFrame(rows); base=df[df.method=='loss4'].set_index('rep'); summary=[]
    for method in methods[1:]:
        other=df[df.method==method].set_index('rep')
        for metric in ['mean_test','sd_test','p10_test','min_test','clean_test']:
            d=other[metric]-base[metric]; p=float(ttest_rel(other[metric],base[metric]).pvalue) if len(d)>=3 else float('nan')
            summary.append({'comparison':f'{method}-loss4','metric':metric,'n':len(d),'delta':float(d.mean()),'p_unadjusted':p})
    pd.DataFrame(summary).to_csv(str(Path(args.output).with_name(Path(args.output).stem+'_summary.csv')),index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data-root',default='.cache/cifar10'); p.add_argument('--output',default='cifar_resnet_finetune_pilot.csv')
    p.add_argument('--reps',type=int,default=1); p.add_argument('--pretrain-epochs',type=int,default=10); p.add_argument('--finetune-epochs',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=128); p.add_argument('--train-per-class',type=int,default=600); p.add_argument('--test-per-class',type=int,default=300)
    p.add_argument('--test-envs',type=int,default=16); p.add_argument('--k',type=int,default=8); p.add_argument('--q',type=int,default=4)
    p.add_argument('--pretrain-lr',type=float,default=3e-3); p.add_argument('--finetune-lr',type=float,default=1e-3); run(p.parse_args())

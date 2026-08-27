from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10

from cifar_resnet_pilot import (
    CIFARResNet20, build_schedule, configure, normalize, apply_environment,
    seed_everything, select_loss, transform_candidates,
)


def split_train_val(root: str, train_per_class: int, val_per_class: int):
    ds = CIFAR10(root, train=True, download=False)
    y = np.asarray(ds.targets)
    rng = np.random.default_rng(424242)
    tr, va = [], []
    for c in range(10):
        idx = np.flatnonzero(y == c).copy(); rng.shuffle(idx)
        tr += idx[:train_per_class].tolist()
        va += idx[train_per_class:train_per_class+val_per_class].tolist()
    tr = np.asarray(tr); va = np.asarray(va)
    x = torch.from_numpy(np.asarray(ds.data)).permute(0,3,1,2).float()/255.
    yt = torch.tensor(y, dtype=torch.long)
    return x[tr], yt[tr], x[va], yt[va]


def evaluate(model, x, y, seeds, batch=256):
    model.eval(); acc=[]
    with torch.no_grad():
        for s in seeds:
            correct=total=0
            for bi,idx in enumerate(torch.arange(len(y)).split(batch)):
                z=normalize(apply_environment(x[idx], s, 700000+bi))
                pred=model(z)[0].argmax(1); correct += int((pred==y[idx]).sum()); total += len(idx)
            acc.append(correct/total)
        clean=sum(int((model(normalize(x[idx]))[0].argmax(1)==y[idx]).sum()) for idx in torch.arange(len(y)).split(batch))/len(y)
    a=np.asarray(acc)
    return dict(val_mean=float(a.mean()), val_sd=float(a.std(ddof=1)), val_p10=float(np.quantile(a,.1)), val_min=float(a.min()), val_clean=float(clean))


def run(a):
    configure(); xtr,ytr,xv,yv=split_train_val(a.data_root,a.train_per_class,a.val_per_class)
    train_env=[20000+i for i in range(64)]; val_env=[25000+i for i in range(a.val_envs)]
    base=81_000_000
    schedule=build_schedule(len(ytr),a.epochs,a.batch_size,64,a.k,base)
    seed_everything(base); model=CIFARResNet20(); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    st=time.time(); model.train()
    for step,idx,cand in schedule:
        seeds=[train_env[e] for e in cand]; xb=transform_candidates(xtr[idx],seeds,step); logits,_=model(xb)
        per=F.cross_entropy(logits,ytr[idx].repeat(a.k),reduction='none').reshape(a.k,-1)
        chosen=select_loss(per.mean(1),a.q)
        opt.zero_grad(set_to_none=True); per[chosen].mean().backward(); opt.step()
    row=dict(lr=a.lr,weight_decay=a.weight_decay,epochs=a.epochs,train_seconds=time.time()-st,n_train=len(ytr),n_val=len(yv),candidate_k=a.k,backward_q=a.q,**evaluate(model,xv,yv,val_env))
    pd.DataFrame([row]).to_csv(a.output,index=False); print(json.dumps(row),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data-root',default='.cache/cifar10'); p.add_argument('--output',required=True)
    p.add_argument('--lr',type=float,required=True); p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--epochs',type=int,required=True)
    p.add_argument('--train-per-class',type=int,default=600); p.add_argument('--val-per-class',type=int,default=200); p.add_argument('--val-envs',type=int,default=12)
    p.add_argument('--batch-size',type=int,default=128); p.add_argument('--k',type=int,default=8); p.add_argument('--q',type=int,default=4); run(p.parse_args())

from __future__ import annotations

import argparse, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiments.cifar_resnet_pilot import (
    CIFARResNet20, apply_environment, configure, evaluate, head_gradient_directions,
    load_cifar, normalize, seed_everything, select_loss, build_schedule,
)


def run(args):
    configure()
    xtr,ytr,xte,yte=load_cifar(args.data_root,args.train_per_class,args.test_per_class)
    train_env=[20000+i for i in range(64)]
    test_env=[30000+i for i in range(args.test_envs)]
    configs=[
        ('adamw_3e-3','adamw',3e-3),
        ('adamw_1e-2','adamw',1e-2),
        ('sgd_5e-2','sgd',5e-2),
        ('sgd_1e-1','sgd',1e-1),
    ]
    rows=[]
    for rep in range(args.reps):
        base=81_000_000+rep*7919
        schedule=build_schedule(len(ytr),args.epochs,args.batch_size,64,args.k,base)
        for name,opt_name,lr in configs:
            seed_everything(base)
            model=CIFARResNet20()
            if opt_name=='adamw':
                opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
            else:
                opt=torch.optim.SGD(model.parameters(),lr=lr,momentum=.9,weight_decay=5e-4,nesterov=True)
            st=time.time(); model.train()
            for step,idx,cand in schedule:
                seeds=[train_env[e] for e in cand]
                xb=torch.cat([normalize(apply_environment(xtr[idx],s,step)) for s in seeds],0)
                logits,h=model(xb)
                per=F.cross_entropy(logits,ytr[idx].repeat(args.k),reduction='none').reshape(args.k,-1)
                chosen=select_loss(per.mean(1),args.q)
                opt.zero_grad(set_to_none=True); per[chosen].mean().backward(); opt.step()
            metrics=evaluate(model,xte,yte,test_env)
            row={'rep':rep,'config':name,'optimizer':opt_name,'lr':lr,'epochs':args.epochs,'train_seconds':time.time()-st,**metrics}
            rows.append(row); print(json.dumps(row),flush=True)
            pd.DataFrame(rows).to_csv(args.output,index=False)
    print(pd.DataFrame(rows).groupby('config')[['mean_test','p10_test','min_test','clean_test','train_seconds']].mean().to_csv(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data-root',default='.cache/cifar10'); p.add_argument('--output',default='cifar_resnet_calibrate.csv'); p.add_argument('--reps',type=int,default=1); p.add_argument('--epochs',type=int,default=8); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--train-per-class',type=int,default=600); p.add_argument('--test-per-class',type=int,default=300); p.add_argument('--test-envs',type=int,default=16); p.add_argument('--k',type=int,default=8); p.add_argument('--q',type=int,default=4); run(p.parse_args())

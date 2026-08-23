from __future__ import annotations

import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import ttest_rel
from torchvision.datasets import CIFAR10

CIFAR_MEAN = torch.tensor([0.4914,0.4822,0.4465]).view(1,3,1,1)
CIFAR_STD = torch.tensor([0.2470,0.2435,0.2616]).view(1,3,1,1)

def configure():
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)

def seed_everything(seed):
    torch.manual_seed(seed); np.random.seed(seed%(2**32-1)); random.seed(seed)

def stratified_subset(targets, per_class, seed):
    y=np.asarray(targets); rng=np.random.default_rng(seed); out=[]
    for c in range(10):
        idx=np.flatnonzero(y==c); out += rng.choice(idx,size=per_class,replace=False).tolist()
    out=np.asarray(out,dtype=np.int64); rng.shuffle(out); return out

def load_cifar(root, train_per_class, test_per_class):
    tr=CIFAR10(root,train=True,download=True); te=CIFAR10(root,train=False,download=True)
    tri=stratified_subset(tr.targets,train_per_class,314159); tei=stratified_subset(te.targets,test_per_class,271828)
    xtr=torch.from_numpy(np.asarray(tr.data)[tri]).permute(0,3,1,2).float()/255.; ytr=torch.tensor(np.asarray(tr.targets)[tri],dtype=torch.long)
    xte=torch.from_numpy(np.asarray(te.data)[tei]).permute(0,3,1,2).float()/255.; yte=torch.tensor(np.asarray(te.targets)[tei],dtype=torch.long)
    return xtr,ytr,xte,yte

def env_parameters(seed):
    r=np.random.default_rng(seed)
    return np.array([r.uniform(-18,18),r.uniform(-3,3),r.uniform(-3,3),r.uniform(.72,1.28),r.uniform(-.12,.12),r.uniform(0,.075)],dtype=np.float32)

def normalized_env_parameters(seed):
    p=env_parameters(seed).copy(); p[0]/=18.; p[1:3]/=3.; p[3]=(p[3]-1.)/.28; p[4]/=.12; p[5]=(p[5]-.0375)/.0375; return p

def apply_environment(x, seed, step):
    angle,dx,dy,contrast,bright,sigma=map(float,env_parameters(seed)); rad=math.radians(angle); c,s=math.cos(rad),math.sin(rad)
    theta=torch.tensor([[c,-s,2*dx/32.],[s,c,2*dy/32.]],dtype=x.dtype).unsqueeze(0).expand(len(x),-1,-1)
    grid=F.affine_grid(theta,x.shape,align_corners=False); z=F.grid_sample(x,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
    z=(z-.5)*contrast+.5+bright
    if sigma>0:
        g=torch.Generator(device='cpu').manual_seed((seed*1_000_003+step*97+17)%(2**63-1)); z=z+sigma*torch.randn(z.shape,generator=g,dtype=z.dtype)
    return z.clamp(0,1)

def normalize(x): return (x-CIFAR_MEAN)/CIFAR_STD

class BasicBlock(torch.nn.Module):
    def __init__(self,cin,cout,stride=1):
        super().__init__(); self.c1=torch.nn.Conv2d(cin,cout,3,stride=stride,padding=1,bias=False); self.g1=torch.nn.GroupNorm(4,cout)
        self.c2=torch.nn.Conv2d(cout,cout,3,padding=1,bias=False); self.g2=torch.nn.GroupNorm(4,cout)
        self.skip=torch.nn.Identity() if stride==1 and cin==cout else torch.nn.Conv2d(cin,cout,1,stride=stride,bias=False)
    def forward(self,x):
        y=F.relu(self.g1(self.c1(x))); y=self.g2(self.c2(y)); return F.relu(y+self.skip(x))

class CIFARResNet20(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.stem=torch.nn.Sequential(torch.nn.Conv2d(3,16,3,padding=1,bias=False),torch.nn.GroupNorm(4,16),torch.nn.ReLU())
        self.s1=self._stage(16,16,3,1); self.s2=self._stage(16,32,3,2); self.s3=self._stage(32,64,3,2); self.head=torch.nn.Linear(64,10)
    @staticmethod
    def _stage(cin,cout,n,stride): return torch.nn.Sequential(BasicBlock(cin,cout,stride),*[BasicBlock(cout,cout) for _ in range(n-1)])
    def forward(self,x):
        x=self.s3(self.s2(self.s1(self.stem(x)))); h=F.adaptive_avg_pool2d(x,1).flatten(1); return self.head(h),h

def zscore(x): return (x-x.mean())/(x.std(unbiased=False)+1e-8)

def head_gradient_directions(logits,features,y,k):
    b=len(y); logits=logits.detach().reshape(k,b,-1); features=features.detach().reshape(k,b,-1); probs=torch.softmax(logits,-1)
    residual=(probs-F.one_hot(y,10).float()[None])/b; gw=torch.einsum('kbc,kbh->kch',residual,features).reshape(k,-1); gb=residual.sum(1)
    g=torch.cat([gw,gb],1); return g/g.norm(dim=1,keepdim=True).clamp_min(1e-12)

def select_loss(loss,q): return torch.topk(loss,q).indices.tolist()

def select_gradnov(loss,cosine,q,w=.6):
    selected=[int(torch.argmax(loss))]
    while len(selected)<q:
        rem=[i for i in range(len(loss)) if i not in selected]; novelty=zscore((1-cosine[:,selected]).min(1).values); score=zscore(loss.detach())+w*novelty
        selected.append(max(rem,key=lambda i:float(score[i])))
    return selected

def select_paramnov(loss,params,q,w=.6):
    selected=[int(torch.argmax(loss))]; d=torch.tensor(np.sqrt(((params[:,None,:]-params[None,:,:])**2).sum(-1)),dtype=torch.float32)
    while len(selected)<q:
        rem=[i for i in range(len(loss)) if i not in selected]; novelty=zscore(d[:,selected].min(1).values); score=zscore(loss.detach())+w*novelty
        selected.append(max(rem,key=lambda i:float(score[i])))
    return selected

def build_schedule(n,epochs,batch,env_pool,k,base):
    tg=torch.Generator().manual_seed(base+1); er=np.random.default_rng(base+2); out=[]; step=0
    for _ in range(epochs):
        for idx in torch.randperm(n,generator=tg).split(batch): out.append((step,idx,er.choice(env_pool,k,replace=False).tolist())); step+=1
    return out

def transform_candidates(x,seeds,step): return torch.cat([normalize(apply_environment(x,s,step)) for s in seeds],0)

def evaluate(model,xte,yte,seeds,batch_size=256):
    model.eval(); acc=[]
    with torch.no_grad():
        for s in seeds:
            correct=total=0
            for bi,idx in enumerate(torch.arange(len(yte)).split(batch_size)):
                pred=model(normalize(apply_environment(xte[idx],s,900000+bi)))[0].argmax(1); correct+=int((pred==yte[idx]).sum()); total+=len(idx)
            acc.append(correct/total)
        clean=sum(int((model(normalize(xte[idx]))[0].argmax(1)==yte[idx]).sum()) for idx in torch.arange(len(yte)).split(batch_size))/len(yte)
    a=np.asarray(acc); return {'mean_test':float(a.mean()),'sd_test':float(a.std(ddof=1)),'p10_test':float(np.quantile(a,.1)),'min_test':float(a.min()),'clean_test':float(clean)}

def run(args):
    configure(); xtr,ytr,xte,yte=load_cifar(args.data_root,args.train_per_class,args.test_per_class); train_env=[20000+i for i in range(64)]; test_env=[30000+i for i in range(args.test_envs)]
    methods=['loss4','random4','paramnov4','gradnov4']; rows=[]; started=time.time()
    for rep in range(args.reps):
        base=71_000_000+rep*7919; schedule=build_schedule(len(ytr),args.epochs,args.batch_size,64,args.k,base)
        for method in methods:
            seed_everything(base); model=CIFARResNet20(); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4); rr=np.random.default_rng(base+98765); st=time.time(); model.train()
            for step,idx,cand in schedule:
                seeds=[train_env[e] for e in cand]; xb=transform_candidates(xtr[idx],seeds,step); logits,h=model(xb)
                per=F.cross_entropy(logits,ytr[idx].repeat(args.k),reduction='none').reshape(args.k,-1); loss=per.mean(1); gd=head_gradient_directions(logits,h,ytr[idx],args.k)
                if method=='loss4': chosen=select_loss(loss,args.q)
                elif method=='random4': chosen=rr.choice(args.k,args.q,replace=False).tolist()
                elif method=='paramnov4': chosen=select_paramnov(loss,np.stack([normalized_env_parameters(s) for s in seeds]),args.q)
                else: chosen=select_gradnov(loss,gd@gd.T,args.q)
                opt.zero_grad(set_to_none=True); per[chosen].mean().backward(); opt.step()
            row={'rep':rep,'method':method,'train_seconds':time.time()-st,'epochs':args.epochs,'n_train':len(ytr),'n_test':len(yte),'candidate_k':args.k,'backward_q':args.q,**evaluate(model,xte,yte,test_env)}
            rows.append(row); print(json.dumps(row),flush=True)
        pd.DataFrame(rows).to_csv(args.output,index=False); print(f'rep {rep} completed; elapsed={time.time()-started:.1f}s',flush=True)
    df=pd.DataFrame(rows); summary=[]; base_df=df[df.method=='loss4'].set_index('rep')
    for method in ['random4','paramnov4','gradnov4']:
        other=df[df.method==method].set_index('rep')
        for metric in ['mean_test','sd_test','p10_test','min_test','clean_test']:
            d=other[metric]-base_df[metric]; p=float(ttest_rel(other[metric],base_df[metric]).pvalue) if len(d)>=3 else float('nan')
            summary.append({'comparison':f'{method}-loss4','metric':metric,'n':len(d),'delta':float(d.mean()),'p_unadjusted':p})
    sdf=pd.DataFrame(summary); sp=str(Path(args.output).with_name(Path(args.output).stem+'_summary.csv')); sdf.to_csv(sp,index=False); print('SUMMARY'); print(sdf.to_csv(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data-root',default='.cache/cifar10'); p.add_argument('--output',default='cifar_resnet_pilot.csv'); p.add_argument('--reps',type=int,default=3); p.add_argument('--epochs',type=int,default=3); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--train-per-class',type=int,default=600); p.add_argument('--test-per-class',type=int,default=300); p.add_argument('--test-envs',type=int,default=16); p.add_argument('--k',type=int,default=8); p.add_argument('--q',type=int,default=4); p.add_argument('--lr',type=float,default=3e-3); run(p.parse_args())

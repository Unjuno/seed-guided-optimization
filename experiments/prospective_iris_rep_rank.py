from __future__ import annotations
import argparse, random
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
K=16;Q=4;BATCH=32;EPOCHS=35;LR=3e-3;TRAIN_START=91000;TEST_START=92000;AUDIT=[1,9,17,25,33,41,49,57]
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
def zscore(x):
 x=torch.as_tensor(x,dtype=torch.float32);return (x-x.mean())/(x.std(unbiased=False)+1e-8)
def erank(H):
 H=np.asarray(H,np.float64);H=H-H.mean(0,keepdims=True);s=np.linalg.svd(H,compute_uv=False);v=s*s;sm=v.sum();
 if sm<=1e-15:return 1.0
 p=v[v>1e-15]/sm;return float(np.exp(-(p*np.log(p)).sum()))
def trans(X,s,off):
 r=np.random.default_rng(s);sig=r.uniform(.03,.45);mask=r.uniform(0,.22);gain=r.uniform(.78,1.22);shift=r.uniform(-.20,.20);rr=np.random.default_rng(s*13001+off);n=rr.normal(0,sig,X.shape).astype(np.float32);m=(rr.random(X.shape)>=mask).astype(np.float32);return ((X*gain+shift+n)*m).astype(np.float32)
def data():
 X,y=load_iris(return_X_y=True);X=X.astype(np.float32);y=y.astype(np.int64);Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.30,random_state=314159,stratify=y);mu=Xtr.mean(0,keepdims=True);sd=Xtr.std(0,keepdims=True)+1e-6;return ((Xtr-mu)/sd).astype(np.float32),ytr,((Xte-mu)/sd).astype(np.float32),yte
class Net(torch.nn.Module):
 def __init__(self):super().__init__();self.body=torch.nn.Sequential(torch.nn.Linear(4,32),torch.nn.ReLU());self.head=torch.nn.Linear(32,3)
 def forward(self,x):h=self.body(x);return self.head(h),h
def hgr(z,h,y):
 b=len(y);zz=z.detach().reshape(K,b,-1);hh=h.detach().reshape(K,b,-1);p=torch.softmax(zz,-1);oh=torch.nn.functional.one_hot(y,3).float()[None];r=(p-oh)/b;gw=torch.einsum('mbc,mbh->mch',r,hh).reshape(K,-1);gb=r.sum(1);raw=torch.cat([gw,gb],1);u=raw/raw.norm(dim=1,keepdim=True).clamp_min(1e-12);return raw,u
def sel(loss,cos,beta):
 s=[int(torch.argmax(loss))]
 while len(s)<Q:
  rem=[i for i in range(K) if i not in s];nov=zscore((1-cos[:,s]).min(1).values);sc=zscore(loss.detach())+beta*nov;s.append(max(rem,key=lambda i:float(sc[i])))
 return s
def diag(start,end,outdir):
 outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True);(outdir/'states').mkdir(exist_ok=True);Xtr,ytr,_,_=data();ty=torch.tensor(ytr);train=[torch.tensor(trans(Xtr,s,11)) for s in range(TRAIN_START,TRAIN_START+64)];rows=[]
 for rep in range(start,end):
  base=111000000+rep*2029;gen=torch.Generator().manual_seed(base+1);rr=np.random.default_rng(base+2);sched=[]
  for _ in range(EPOCHS):
   for b in torch.randperm(len(ty),generator=gen).split(BATCH):sched.append((b,rr.choice(64,K,replace=False).tolist()))
  for method,beta in [('loss_hard',0.),('gradnov',1.5)]:
   torch.manual_seed(base);np.random.seed(base%(2**32-1));random.seed(base);model=Net();opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-3);G=[]
   for b,c in sched:
    xb=torch.cat([train[e][b] for e in c]);z,h=model(xb);per=torch.nn.functional.cross_entropy(z,ty[b].repeat(K),reduction='none').reshape(K,-1);el=per.mean(1);raw,u=hgr(z,h,ty[b]);cos=u@u.T;idx=sel(el,cos,beta);agg=raw[idx].mean(0);G.append((agg/agg.norm().clamp_min(1e-12)).numpy());opt.zero_grad(set_to_none=True);per[idx].mean().backward();opt.step()
   model.eval();hs=[];ranks=[]
   with torch.no_grad():
    for e in range(64):
     _,h=model(train[e]);ranks.append(erank(h.numpy()));
     if e in AUDIT:hs.append(h.numpy())
   rows.append(dict(rep=rep,method=method,rep_eff_rank=erank(np.concatenate(hs)),grad_eff_rank=erank(np.stack(G)),env_rep_rank_sd=float(np.std(ranks,ddof=1))))
   torch.save({'state_dict':model.state_dict(),'rep':rep,'method':method},outdir/'states'/f'rep{rep:02d}_{method}.pt')
  pd.DataFrame(rows).to_csv(outdir/'diagnostics.csv',index=False);print(rep,flush=True)
 df=pd.DataFrame(rows);w=df.pivot(index='rep',columns='method',values=['rep_eff_rank','grad_eff_rank','env_rep_rank_sd']);d=pd.DataFrame(index=w.index)
 for v in ['rep_eff_rank','grad_eff_rank','env_rep_rank_sd']:d['delta_'+v]=w[(v,'gradnov')]-w[(v,'loss_hard')]
 d.to_csv(outdir/'diagnostic_deltas.csv');print(d.mean().to_string())
def evaluate(outdir):
 outdir=Path(outdir);_,_,Xte,yte=data();ey=torch.tensor(yte);unseen=[torch.tensor(trans(Xte,s,37)) for s in range(TEST_START,TEST_START+80)];clean=torch.tensor(Xte);rows=[]
 for p in sorted((outdir/'states').glob('rep*_*.pt')):
  ck=torch.load(p,map_location='cpu');m=Net();m.load_state_dict(ck['state_dict']);m.eval();
  with torch.no_grad():a=np.array([float((m(x)[0].argmax(1)==ey).float().mean()) for x in unseen]);ca=float((m(clean)[0].argmax(1)==ey).float().mean())
  rows.append(dict(rep=ck['rep'],method=ck['method'],mean_test=a.mean(),sd_test=a.std(ddof=1),p10_test=np.quantile(a,.1),min_test=a.min(),clean_test=ca))
 df=pd.DataFrame(rows);df.to_csv(outdir/'heldout.csv',index=False);w=df.pivot(index='rep',columns='method',values=['mean_test','sd_test','p10_test','min_test','clean_test']);d=pd.DataFrame(index=w.index)
 for v in ['mean_test','sd_test','p10_test','min_test','clean_test']:d['delta_'+v]=w[(v,'gradnov')]-w[(v,'loss_hard')]
 d.to_csv(outdir/'heldout_deltas.csv');print(d.mean().to_string())
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode');ap.add_argument('--start',type=int,default=0);ap.add_argument('--end',type=int,default=20);ap.add_argument('--outdir',default='/mnt/data/prospective_iris');a=ap.parse_args();diag(a.start,a.end,a.outdir) if a.mode=='diagnostic' else evaluate(a.outdir)

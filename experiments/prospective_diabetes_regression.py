from __future__ import annotations
import argparse, random
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
K=16;Q=4;BATCH=64;EPOCHS=20;LR=3e-3;TRAIN_START=101000;TEST_START=102000;AUDIT=[1,9,17,25,33,41,49,57]
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
def zscore(x):
 x=torch.as_tensor(x,dtype=torch.float32);return (x-x.mean())/(x.std(unbiased=False)+1e-8)
def erank(H):
 H=np.asarray(H,np.float64);H=H-H.mean(0,keepdims=True);s=np.linalg.svd(H,compute_uv=False);v=s*s;sm=v.sum();
 if sm<=1e-15:return 1.0
 p=v[v>1e-15]/sm;return float(np.exp(-(p*np.log(p)).sum()))
def trans(X,s,off):
 r=np.random.default_rng(s);sig=r.uniform(.03,.38);mask=r.uniform(0,.18);gain=r.uniform(.80,1.20);shift=r.uniform(-.15,.15);rr=np.random.default_rng(s*15013+off);n=rr.normal(0,sig,X.shape).astype(np.float32);m=(rr.random(X.shape)>=mask).astype(np.float32);return ((X*gain+shift+n)*m).astype(np.float32)
def data():
 X,y=load_diabetes(return_X_y=True);X=X.astype(np.float32);y=y.astype(np.float32);Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.30,random_state=314159);xm=Xtr.mean(0,keepdims=True);xs=Xtr.std(0,keepdims=True)+1e-6;ym=ytr.mean();ys=ytr.std()+1e-6;return ((Xtr-xm)/xs).astype(np.float32),((ytr-ym)/ys).astype(np.float32),((Xte-xm)/xs).astype(np.float32),((yte-ym)/ys).astype(np.float32),ym,ys
class Net(torch.nn.Module):
 def __init__(self):super().__init__();self.body=torch.nn.Sequential(torch.nn.Linear(10,64),torch.nn.ReLU());self.head=torch.nn.Linear(64,1)
 def forward(self,x):h=self.body(x);return self.head(h).squeeze(-1),h
def hgr(pred,h,y):
 b=len(y);pp=pred.detach().reshape(K,b);hh=h.detach().reshape(K,b,-1);res=2*(pp-y[None])/b;gw=torch.einsum('mb,mbh->mh',res,hh);gb=res.sum(1,keepdim=True);raw=torch.cat([gw,gb],1);u=raw/raw.norm(dim=1,keepdim=True).clamp_min(1e-12);return raw,u
def sel(loss,cos,beta):
 s=[int(torch.argmax(loss))]
 while len(s)<Q:
  rem=[i for i in range(K) if i not in s];nov=zscore((1-cos[:,s]).min(1).values);sc=zscore(loss.detach())+beta*nov;s.append(max(rem,key=lambda i:float(sc[i])))
 return s
def diag(start,end,outdir):
 outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True);(outdir/'states').mkdir(exist_ok=True);Xtr,ytr,_,_,_,_=data();ty=torch.tensor(ytr);train=[torch.tensor(trans(Xtr,s,11)) for s in range(TRAIN_START,TRAIN_START+64)];rows=[]
 for rep in range(start,end):
  base=121000000+rep*2029;gen=torch.Generator().manual_seed(base+1);rr=np.random.default_rng(base+2);sched=[]
  for _ in range(EPOCHS):
   for b in torch.randperm(len(ty),generator=gen).split(BATCH):sched.append((b,rr.choice(64,K,replace=False).tolist()))
  for method,beta in [('loss_hard',0.),('gradnov',1.5)]:
   torch.manual_seed(base);np.random.seed(base%(2**32-1));random.seed(base);m=Net();opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-3);G=[]
   for b,c in sched:
    xb=torch.cat([train[e][b] for e in c]);p,h=m(xb);per=(p-ty[b].repeat(K))**2;per=per.reshape(K,-1);el=per.mean(1);raw,u=hgr(p,h,ty[b]);cos=u@u.T;idx=sel(el,cos,beta);agg=raw[idx].mean(0);G.append((agg/agg.norm().clamp_min(1e-12)).numpy());opt.zero_grad(set_to_none=True);per[idx].mean().backward();opt.step()
   m.eval();hs=[]
   with torch.no_grad():
    for e in AUDIT:_,h=m(train[e]);hs.append(h.numpy())
   rows.append(dict(rep=rep,method=method,rep_eff_rank=erank(np.concatenate(hs)),grad_eff_rank=erank(np.stack(G))))
   torch.save({'state_dict':m.state_dict(),'rep':rep,'method':method},outdir/'states'/f'rep{rep:02d}_{method}.pt')
  pd.DataFrame(rows).to_csv(outdir/'diagnostics.csv',index=False);print(rep,flush=True)
 df=pd.DataFrame(rows);w=df.pivot(index='rep',columns='method',values=['rep_eff_rank','grad_eff_rank']);d=pd.DataFrame(index=w.index)
 for v in ['rep_eff_rank','grad_eff_rank']:d['delta_'+v]=w[(v,'gradnov')]-w[(v,'loss_hard')]
 d.to_csv(outdir/'diagnostic_deltas.csv');print(d.mean().to_string())
def evaluate(outdir):
 outdir=Path(outdir);_,_,Xte,yte,_,_=data();ey=torch.tensor(yte);unseen=[torch.tensor(trans(Xte,s,37)) for s in range(TEST_START,TEST_START+80)];clean=torch.tensor(Xte);rows=[]
 for pth in sorted((outdir/'states').glob('rep*_*.pt')):
  ck=torch.load(pth,map_location='cpu');m=Net();m.load_state_dict(ck['state_dict']);m.eval();
  with torch.no_grad():mses=np.array([float(((m(x)[0]-ey)**2).mean()) for x in unseen]);clean_mse=float(((m(clean)[0]-ey)**2).mean())
  rows.append(dict(rep=ck['rep'],method=ck['method'],mean_mse=mses.mean(),sd_mse=mses.std(ddof=1),p90_mse=np.quantile(mses,.9),max_mse=mses.max(),clean_mse=clean_mse))
 df=pd.DataFrame(rows);df.to_csv(outdir/'heldout.csv',index=False);w=df.pivot(index='rep',columns='method',values=['mean_mse','sd_mse','p90_mse','max_mse','clean_mse']);d=pd.DataFrame(index=w.index)
 for v in ['mean_mse','sd_mse','p90_mse','max_mse','clean_mse']:d['benefit_'+v]=w[(v,'loss_hard')]-w[(v,'gradnov')]
 d.to_csv(outdir/'heldout_benefits.csv');print(d.mean().to_string())
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode');ap.add_argument('--start',type=int,default=0);ap.add_argument('--end',type=int,default=20);ap.add_argument('--outdir',default='/mnt/data/prospective_diabetes');a=ap.parse_args();diag(a.start,a.end,a.outdir) if a.mode=='diagnostic' else evaluate(a.outdir)

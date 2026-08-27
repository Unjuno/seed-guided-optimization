from __future__ import annotations
import argparse, random
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

K=16; Q=4; BATCH=64; EPOCHS=35; LR=3e-3; TRAIN_START=81000; TEST_START=82000; AUDIT_ENVS=[1,9,17,25,33,41,49,57]
torch.set_num_threads(1); torch.use_deterministic_algorithms(True)

def zscore(x):
 x=torch.as_tensor(x,dtype=torch.float32); return (x-x.mean())/(x.std(unbiased=False)+1e-8)
def eff_rank(H):
 H=np.asarray(H,np.float64); H=H-H.mean(0,keepdims=True); s=np.linalg.svd(H,compute_uv=False); v=s*s; sm=v.sum()
 if sm<=1e-15:return 1.0
 p=v[v>1e-15]/sm; return float(np.exp(-(p*np.log(p)).sum()))
def transform(Xb,seed,offset):
 r=np.random.default_rng(seed); sig=r.uniform(.04,.50); maskp=r.uniform(0,.28); gain=r.uniform(.72,1.28); shift=r.uniform(-.22,.22)
 rr=np.random.default_rng(seed*11003+offset); n=rr.normal(0,sig,Xb.shape).astype(np.float32); m=(rr.random(Xb.shape)>=maskp).astype(np.float32)
 return ((Xb*gain+shift+n)*m).astype(np.float32)
def data_split():
 X,y=load_wine(return_X_y=True); X=X.astype(np.float32); y=y.astype(np.int64)
 Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.30,random_state=314159,stratify=y)
 mu=Xtr.mean(0,keepdims=True); sd=Xtr.std(0,keepdims=True)+1e-6
 return ((Xtr-mu)/sd).astype(np.float32),ytr,((Xte-mu)/sd).astype(np.float32),yte
class Net(torch.nn.Module):
 def __init__(self):super().__init__(); self.body=torch.nn.Sequential(torch.nn.Linear(13,64),torch.nn.ReLU()); self.head=torch.nn.Linear(64,3)
 def forward(self,x):h=self.body(x); return self.head(h),h
def head_grads(z,h,yb):
 b=len(yb); zz=z.detach().reshape(K,b,-1); hh=h.detach().reshape(K,b,-1); p=torch.softmax(zz,-1); oh=torch.nn.functional.one_hot(yb,3).float()[None]; r=(p-oh)/b
 gw=torch.einsum('mbc,mbh->mch',r,hh).reshape(K,-1); gb=r.sum(1); raw=torch.cat([gw,gb],1); unit=raw/raw.norm(dim=1,keepdim=True).clamp_min(1e-12); return raw,unit
def select(loss,cos,beta):
 sel=[int(torch.argmax(loss))]
 while len(sel)<Q:
  rem=[i for i in range(K) if i not in sel]; nov=zscore((1-cos[:,sel]).min(1).values); score=zscore(loss.detach())+beta*nov; sel.append(max(rem,key=lambda i:float(score[i])))
 return sel
def train_diag(start,end,outdir):
 outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True); (outdir/'states').mkdir(exist_ok=True)
 Xtr,ytr,_,_=data_split(); train=[torch.tensor(transform(Xtr,s,11)) for s in range(TRAIN_START,TRAIN_START+64)]; clean=torch.tensor(Xtr); ty=torch.tensor(ytr); rows=[]
 for rep in range(start,end):
  base=101000000+rep*2029; gen=torch.Generator().manual_seed(base+1); er=np.random.default_rng(base+2); sched=[]
  for _ in range(EPOCHS):
   for b in torch.randperm(len(ty),generator=gen).split(BATCH):sched.append((b,er.choice(64,K,replace=False).tolist()))
  probe=torch.arange(len(ty))
  for method,beta in [('loss_hard',0.),('gradnov',1.5)]:
   torch.manual_seed(base); np.random.seed(base%(2**32-1)); random.seed(base); model=Net(); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-3); G=[]
   for b,cand in sched:
    xb=torch.cat([train[e][b] for e in cand]); z,h=model(xb); per=torch.nn.functional.cross_entropy(z,ty[b].repeat(K),reduction='none').reshape(K,-1); el=per.mean(1); raw,unit=head_grads(z,h,ty[b]); cos=unit@unit.T; idx=select(el,cos,beta); agg=raw[idx].mean(0); G.append((agg/agg.norm().clamp_min(1e-12)).numpy()); opt.zero_grad(set_to_none=True); per[idx].mean().backward(); opt.step()
   model.eval(); hs=[]; env_ranks=[]; env_acc=[]
   with torch.no_grad():
    for e in range(64):
     z,h=model(train[e]); env_acc.append(float((z.argmax(1)==ty).float().mean()));
     if e in AUDIT_ENVS:hs.append(h.numpy())
     env_ranks.append(eff_rank(h.numpy()))
   row=dict(rep=rep,method=method,rep_eff_rank=eff_rank(np.concatenate(hs)),grad_eff_rank=eff_rank(np.stack(G)),env_rep_rank_sd=float(np.std(env_ranks,ddof=1)),env_rep_rank_mean=float(np.mean(env_ranks)),train_acc_p10=float(np.quantile(env_acc,.1)),train_acc_min=float(np.min(env_acc)))
   rows.append(row); torch.save({'state_dict':model.state_dict(),'rep':rep,'method':method},outdir/'states'/f'rep{rep:02d}_{method}.pt')
  pd.DataFrame(rows).to_csv(outdir/'diagnostics.csv',index=False); print('rep',rep,flush=True)
 df=pd.DataFrame(rows); df.to_csv(outdir/'diagnostics.csv',index=False)
 vals=['rep_eff_rank','grad_eff_rank','env_rep_rank_sd','env_rep_rank_mean','train_acc_p10','train_acc_min']; w=df.pivot(index='rep',columns='method',values=vals); d=pd.DataFrame(index=w.index)
 for v in vals:d['delta_'+v]=w[(v,'gradnov')]-w[(v,'loss_hard')]
 d.to_csv(outdir/'diagnostic_deltas.csv'); print(d.mean().to_string())
def evaluate(outdir):
 outdir=Path(outdir); _,_,Xte,yte=data_split(); unseen=[torch.tensor(transform(Xte,s,37)) for s in range(TEST_START,TEST_START+80)]; clean=torch.tensor(Xte); ey=torch.tensor(yte); rows=[]
 for p in sorted((outdir/'states').glob('rep*_*.pt')):
  ck=torch.load(p,map_location='cpu'); model=Net(); model.load_state_dict(ck['state_dict']); model.eval()
  with torch.no_grad():acc=np.array([float((model(x)[0].argmax(1)==ey).float().mean()) for x in unseen]); ca=float((model(clean)[0].argmax(1)==ey).float().mean())
  rows.append(dict(rep=ck['rep'],method=ck['method'],mean_test=float(acc.mean()),sd_test=float(acc.std(ddof=1)),p10_test=float(np.quantile(acc,.1)),min_test=float(acc.min()),clean_test=ca))
 df=pd.DataFrame(rows); df.to_csv(outdir/'heldout.csv',index=False); w=df.pivot(index='rep',columns='method',values=['mean_test','sd_test','p10_test','min_test','clean_test']); d=pd.DataFrame(index=w.index)
 for v in ['mean_test','sd_test','p10_test','min_test','clean_test']:d['delta_'+v]=w[(v,'gradnov')]-w[(v,'loss_hard')]
 d.to_csv(outdir/'heldout_deltas.csv'); print(d.mean().to_string())
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['diagnostic','evaluate']); ap.add_argument('--start',type=int,default=0); ap.add_argument('--end',type=int,default=20); ap.add_argument('--outdir',default='/mnt/data/prospective_wine'); a=ap.parse_args(); train_diag(a.start,a.end,a.outdir) if a.mode=='diagnostic' else evaluate(a.outdir)

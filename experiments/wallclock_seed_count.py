import sys,random,time,math
import numpy as np,pandas as pd,torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
start,end=map(int,sys.argv[1:3]); scale=1.20; BATCH=128; LR=1e-2; STEPS={2:480,4:317,8:186}
X,y=load_digits(return_X_y=True); X=X.astype(np.float32)/16.; y=y.astype(np.int64)
Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=.45,random_state=314159,stratify=y); _,Xte,_,yte=train_test_split(Xtmp,ytmp,test_size=.55,random_state=271828,stratify=ytmp)
def env(Xb,s,off,scale):
 r=np.random.default_rng(s); sig=r.uniform(.02,.32)*scale; mp=np.clip(r.uniform(0,.16)*scale,0,.95); c=1+(r.uniform(.82,1.18)-1)*scale; sh=r.uniform(-.04,.04)*scale; rr=np.random.default_rng(s*1009+off); return np.clip((Xb*c+sh+rr.normal(0,sig,Xb.shape).astype(np.float32))*(rr.random(Xb.shape)>=mp),0,1).astype(np.float32)
tr=[torch.tensor(env(Xtr,s,11,scale)) for s in range(5000,5064)]; te=[torch.tensor(env(Xte,s,37,scale)) for s in range(6000,6080)]; ty=torch.tensor(ytr); ey=torch.tensor(yte); clean=torch.tensor(Xte)
def batches(seed,n=480):
 g=torch.Generator().manual_seed(seed); out=[]
 while len(out)<n: out.extend(list(torch.randperm(len(ty),generator=g).split(BATCH)))
 return out[:n]
def model(): return torch.nn.Sequential(torch.nn.Linear(64,256),torch.nn.ReLU(),torch.nn.Linear(256,256),torch.nn.ReLU(),torch.nn.Linear(256,128),torch.nn.ReLU(),torch.nn.Linear(128,10))
rows=[]; t=time.time()
for rep in range(start,end):
 base=990000+rep*1009; bs=batches(base+1); er=np.random.default_rng(base+2); stream=[]
 for _ in range(30): stream.extend(er.permutation(64).tolist())
 for m,nsteps in STEPS.items():
  torch.manual_seed(base); np.random.seed(base%(2**32-1)); random.seed(base); net=model(); opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=1e-3); pos=0; st=time.time(); lams=[]; samples=0
  for step in range(nsteps):
   b=bs[step]; ids=stream[pos:pos+m]; pos+=m; opt.zero_grad(set_to_none=True); xb=torch.cat([tr[e][b] for e in ids]); yb=ty[b].repeat(m); per=torch.nn.functional.cross_entropy(net(xb),yb,reduction='none').reshape(m,-1); el=per.mean(1); d=el.detach(); cv=d.std(unbiased=True)/(d.mean().abs()+1e-8); lam=cv/(1+cv); top=torch.topk(el,max(1,math.ceil(m/2))).values.mean(); ((1-lam)*el.mean()+lam*top).backward(); opt.step(); lams.append(float(lam)); samples+=len(b)*m
  sec=time.time()-st
  with torch.no_grad(): a=np.array([float((net(z).argmax(1)==ey).float().mean()) for z in te]); ca=float((net(clean).argmax(1)==ey).float().mean())
  rows.append(dict(rep=rep,m=m,n_steps=nsteps,total_env_evals=nsteps*m,train_seconds=sec,sample_env_per_sec=samples/sec,mean_test=a.mean(),sd_test=a.std(ddof=1),p10_test=np.quantile(a,.1),min_test=a.min(),clean_test=ca,lambda_mean=np.mean(lams)))
 print('rep',rep,'done',round(time.time()-t,1),flush=True)
pd.DataFrame(rows).to_csv(f'seed_count_walltime_{start}_{end}.csv',index=False)

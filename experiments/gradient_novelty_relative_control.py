import sys, random, time
import numpy as np, pandas as pd, torch
from sklearn.datasets import make_classification, load_digits
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate, shift, gaussian_filter

task=sys.argv[1]; start,end=map(int,sys.argv[2:4]); K=16; Q=4; BATCH=128
METHODS=['beta0','beta1.5','beta3','relative0.15','relative0.30']; GRID=[0.,.25,.5,1.,1.5,2.,3.,5.]
torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
if task=='synthetic':
 EPOCHS=12; LR=3e-3; NCLASS=4
 X,y=make_classification(n_samples=3200,n_features=40,n_informative=24,n_redundant=8,n_classes=4,n_clusters_per_class=2,class_sep=1.25,flip_y=.02,random_state=20260823); X=X.astype(np.float32); y=y.astype(np.int64)
 Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3,random_state=314159,stratify=y); mu=Xtr.mean(0,keepdims=True); sd=Xtr.std(0,keepdims=True)+1e-6; Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
 def env(Xb,s,off):
  r=np.random.default_rng(s); sig=r.uniform(.03,.30); maskp=r.uniform(0,.12); gain=r.uniform(.85,1.15); sh=r.uniform(-.12,.12); rr=np.random.default_rng(s*2017+off); n=rr.normal(0,sig,Xb.shape).astype(np.float32); m=(rr.random(Xb.shape)>=maskp).astype(np.float32); return ((Xb*gain+sh+n)*m).astype(np.float32)
 train=[torch.tensor(env(Xtr,s,11)) for s in range(7000,7064)]; unseen=[torch.tensor(env(Xte,s,37)) for s in range(8000,8080)]; clean=torch.tensor(Xte); base0=720000
else:
 EPOCHS=10; LR=1e-2; NCLASS=10
 X,y=load_digits(return_X_y=True); X=(X.astype(np.float32)/16.).reshape(-1,8,8); y=y.astype(np.int64); Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=.45,random_state=314159,stratify=y); _,Xte,_,yte=train_test_split(Xtmp,ytmp,test_size=.55,random_state=271828,stratify=ytmp)
 def env(Xb,s,off):
  r=np.random.default_rng(s); ang=r.uniform(-30,30); dx=r.uniform(-1.2,1.2); dy=r.uniform(-1.2,1.2); blur=r.uniform(0,.85); contrast=r.uniform(.78,1.22); bright=r.uniform(-.06,.06); noise=r.uniform(0,.08); out=np.empty_like(Xb); rr=np.random.default_rng(s*4001+off)
  for i,img in enumerate(Xb):
   a=rotate(img,float(ang),reshape=False,order=1,mode='constant',cval=0,prefilter=False); a=shift(a,(float(dy),float(dx)),order=1,mode='constant',cval=0,prefilter=False)
   if blur>1e-6: a=gaussian_filter(a,float(blur),mode='nearest')
   out[i]=np.clip((a-.5)*contrast+.5+bright+rr.normal(0,noise,a.shape),0,1)
  return out.reshape(len(out),-1).astype(np.float32)
 train=[torch.tensor(env(Xtr,s,11)) for s in range(13000,13064)]; unseen=[torch.tensor(env(Xte,s,37)) for s in range(14000,14080)]; clean=torch.tensor(Xte.reshape(len(Xte),-1).astype(np.float32)); Xtr=Xtr.reshape(len(Xtr),-1); base0=27100000

ty=torch.tensor(ytr); ey=torch.tensor(yte); IN=train[0].shape[1]
class Net(torch.nn.Module):
 def __init__(self): super().__init__(); self.body=torch.nn.Sequential(torch.nn.Linear(IN,128),torch.nn.ReLU()); self.head=torch.nn.Linear(128,NCLASS)
 def forward(self,x): h=self.body(x); return self.head(h),h

def gdirs(z,h,yb):
 b=len(yb); zz=z.detach().reshape(K,b,-1); hh=h.detach().reshape(K,b,-1); p=torch.softmax(zz,-1); oh=torch.nn.functional.one_hot(yb,NCLASS).float()[None]; r=(p-oh)/b; gw=torch.einsum('mbc,mbh->mch',r,hh).reshape(K,-1); gb=r.sum(1); g=torch.cat([gw,gb],1); return g/g.norm(dim=1,keepdim=True).clamp_min(1e-12)
def zscore(x): x=torch.as_tensor(x,dtype=torch.float32); return (x-x.mean())/(x.std(unbiased=False)+1e-8)
def select(loss,cos,beta):
 sel=[int(torch.argmax(loss))]
 while len(sel)<Q:
  rem=[i for i in range(K) if i not in sel]; nov=zscore((1-cos[:,sel]).min(1).values); score=zscore(loss.detach())+beta*nov; sel.append(max(rem,key=lambda i:float(score[i])))
 return sel
def pc(cos,idx):
 a=cos[idx][:,idx]; tr=torch.triu_indices(len(idx),len(idx),1); return float(a[tr[0],tr[1]].mean())
def relative_select(loss,cos,rho):
 s0=select(loss,cos,0.); s5=select(loss,cos,5.); c0=pc(cos,s0); c5=pc(cos,s5); target=c5+rho*(c0-c5)
 opts=[]
 for b in GRID:
  s=select(loss,cos,b); c=pc(cos,s); opts.append((abs(c-target),b,s,c))
 _,b,s,c=min(opts,key=lambda x:(x[0],x[1])); return s,b,c,target,c0,c5
rows=[]; t0=time.time()
for rep in range(start,end):
 base=base0+rep*2029; gen=torch.Generator().manual_seed(base+1); er=np.random.default_rng(base+2); sched=[]
 for ep in range(EPOCHS):
  for b in torch.randperm(len(ty),generator=gen).split(BATCH): sched.append((b,er.choice(64,K,replace=False).tolist()))
 for method in METHODS:
  torch.manual_seed(base); np.random.seed(base%(2**32-1)); random.seed(base); model=Net(); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-3); bs=[]; pcs=[]; losses=[]; targets=[]; c0s=[]; c5s=[]; st=time.time()
  for b,cand in sched:
   xb=torch.cat([train[e][b] for e in cand]); z,h=model(xb); per=torch.nn.functional.cross_entropy(z,ty[b].repeat(K),reduction='none').reshape(K,-1); el=per.mean(1); gd=gdirs(z,h,ty[b]); cos=gd@gd.T
   if method.startswith('relative'):
    rho=float(method.replace('relative','')); idx,beta,c,target,c0,c5=relative_select(el,cos,rho); targets.append(target); c0s.append(c0); c5s.append(c5)
   else:
    beta=float(method.replace('beta','')); idx=select(el,cos,beta); c=pc(cos,idx)
   bs.append(beta); pcs.append(c); losses.append(float(el[idx].detach().mean())); opt.zero_grad(set_to_none=True); per[idx].mean().backward(); opt.step()
  model.eval()
  with torch.no_grad(): a=np.array([float((model(x)[0].argmax(1)==ey).float().mean()) for x in unseen]); ca=float((model(clean)[0].argmax(1)==ey).float().mean())
  rows.append(dict(rep=rep,method=method,mean_test=a.mean(),sd_test=a.std(ddof=1),p10_test=np.quantile(a,.1),min_test=a.min(),clean_test=ca,train_seconds=time.time()-st,beta_mean=np.mean(bs),selected_pair_cos=np.mean(pcs),selected_loss=np.mean(losses),target_cos=np.mean(targets) if targets else np.nan,c0=np.mean(c0s) if c0s else np.nan,c5=np.mean(c5s) if c5s else np.nan))
 print(task,'rep',rep,'done',round(time.time()-t0,1),flush=True)
pd.DataFrame(rows).to_csv(f'gradient_novelty_relative_{task}_{start}_{end}.csv',index=False)

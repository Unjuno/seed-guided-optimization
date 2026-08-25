import sys, random, time
import numpy as np, pandas as pd, torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate, shift, gaussian_filter

torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
start,end=map(int,sys.argv[1:3]); BATCH=128; EPOCHS=10; K=16; Q=4; LR=1e-2
METHODS=['beta0','beta1.5','beta3','adaptive0.20','adaptive0.15']
X,y=load_digits(return_X_y=True); X=(X.astype(np.float32)/16.).reshape(-1,8,8); y=y.astype(np.int64)
Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=.45,random_state=314159,stratify=y); _,Xte,_,yte=train_test_split(Xtmp,ytmp,test_size=.55,random_state=271828,stratify=ytmp)

def env_geom(Xb,s):
 r=np.random.default_rng(s); ang=r.uniform(-30,30); dx=r.uniform(-1.2,1.2); dy=r.uniform(-1.2,1.2); blur=r.uniform(0,.85); contrast=r.uniform(.78,1.22); bright=r.uniform(-.06,.06); noise=r.uniform(0,.08)
 out=np.empty_like(Xb); rr=np.random.default_rng(s*4001+17)
 for i,img in enumerate(Xb):
  a=rotate(img,float(ang),reshape=False,order=1,mode='constant',cval=0,prefilter=False); a=shift(a,(float(dy),float(dx)),order=1,mode='constant',cval=0,prefilter=False)
  if blur>1e-6: a=gaussian_filter(a,float(blur),mode='nearest')
  out[i]=np.clip((a-.5)*contrast+.5+bright+rr.normal(0,noise,a.shape),0,1)
 return out.reshape(len(out),-1).astype(np.float32)
train=[torch.tensor(env_geom(Xtr,s)) for s in range(13000,13064)]; unseen=[torch.tensor(env_geom(Xte,s)) for s in range(14000,14080)]; clean=torch.tensor(Xte.reshape(len(Xte),-1).astype(np.float32)); ty=torch.tensor(ytr); ey=torch.tensor(yte)
class Net(torch.nn.Module):
 def __init__(self): super().__init__(); self.body=torch.nn.Sequential(torch.nn.Linear(64,128),torch.nn.ReLU()); self.head=torch.nn.Linear(128,10)
 def forward(self,x): h=self.body(x); return self.head(h),h

def gdirs(z,h,yb):
 b=len(yb); zz=z.detach().reshape(K,b,-1); hh=h.detach().reshape(K,b,-1); p=torch.softmax(zz,-1); oh=torch.nn.functional.one_hot(yb,10).float()[None]; r=(p-oh)/b
 gw=torch.einsum('mbc,mbh->mch',r,hh).reshape(K,-1); gb=r.sum(1); g=torch.cat([gw,gb],1); return g/g.norm(dim=1,keepdim=True).clamp_min(1e-12)
def zscore(x): x=torch.as_tensor(x,dtype=torch.float32); return (x-x.mean())/(x.std(unbiased=False)+1e-8)
def select(loss,cos,beta):
 sel=[int(torch.argmax(loss))]
 while len(sel)<Q:
  rem=[i for i in range(K) if i not in sel]; nov=zscore((1-cos[:,sel]).min(1).values); score=zscore(loss.detach())+beta*nov; sel.append(max(rem,key=lambda i:float(score[i])))
 return sel
def pc(cos,idx):
 a=cos[idx][:,idx]; tr=torch.triu_indices(len(idx),len(idx),1); return float(a[tr[0],tr[1]].mean())
rows=[]; t0=time.time()
for rep in range(start,end):
 base=25100000+rep*2029; tg=torch.Generator().manual_seed(base+1); er=np.random.default_rng(base+2); sched=[]
 for ep in range(EPOCHS):
  for b in torch.randperm(len(ty),generator=tg).split(BATCH): sched.append((b,er.choice(64,K,replace=False).tolist()))
 for method in METHODS:
  torch.manual_seed(base); np.random.seed(base%(2**32-1)); random.seed(base); model=Net(); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-3)
  if method=='beta0': beta=0.0; adaptive=False; target=np.nan
  elif method=='beta1.5': beta=1.5; adaptive=False; target=np.nan
  elif method=='beta3': beta=3.0; adaptive=False; target=np.nan
  else: beta=1.0; adaptive=True; target=float(method.replace('adaptive',''))
  beta_hist=[]; pcs=[]; losses=[]; st=time.time()
  for b,cand in sched:
   xb=torch.cat([train[e][b] for e in cand]); z,h=model(xb); per=torch.nn.functional.cross_entropy(z,ty[b].repeat(K),reduction='none').reshape(K,-1); el=per.mean(1); gd=gdirs(z,h,ty[b]); cos=gd@gd.T
   idx=select(el,cos,beta); c=pc(cos,idx); beta_hist.append(beta); pcs.append(c); losses.append(float(el[idx].detach().mean()))
   opt.zero_grad(set_to_none=True); per[idx].mean().backward(); opt.step()
   if adaptive:
    beta=float(np.clip(beta*np.exp(0.35*(c-target)),0.10,5.0))
  model.eval()
  with torch.no_grad(): a=np.array([float((model(x)[0].argmax(1)==ey).float().mean()) for x in unseen]); ca=float((model(clean)[0].argmax(1)==ey).float().mean())
  rows.append(dict(rep=rep,method=method,mean_test=a.mean(),sd_test=a.std(ddof=1),p10_test=np.quantile(a,.1),min_test=a.min(),clean_test=ca,train_seconds=time.time()-st,beta_mean=np.mean(beta_hist),beta_final=beta,selected_pair_cos=np.mean(pcs),selected_loss=np.mean(losses)))
 print('rep',rep,'done',round(time.time()-t0,1),flush=True)
pd.DataFrame(rows).to_csv(f'gradient_novelty_beta_adaptive_{start}_{end}.csv',index=False)

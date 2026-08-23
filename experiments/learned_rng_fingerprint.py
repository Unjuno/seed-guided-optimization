import sys, random, time
import numpy as np, pandas as pd, torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from scipy.ndimage import rotate, shift, gaussian_filter

torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
start,end=map(int,sys.argv[1:3]); BATCH=128; EPOCHS=10; K=16; PREF=8; Q=4; LR=1e-2
METHODS=['grad16','oracle7_8','raw64_8','learned_pred8','learned_top7_8','random8']

X,y=load_digits(return_X_y=True); X=(X.astype(np.float32)/16.).reshape(-1,8,8); y=y.astype(np.int64)
Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=.45,random_state=314159,stratify=y)
_,Xte,_,yte=train_test_split(Xtmp,ytmp,test_size=.55,random_state=271828,stratify=ytmp)

def params(s):
    r=np.random.default_rng(s)
    return np.array([r.uniform(-30,30),r.uniform(-1.2,1.2),r.uniform(-1.2,1.2),r.uniform(0,.85),r.uniform(.78,1.22),r.uniform(-.06,.06),r.uniform(0,.08)],np.float32)

def env_geom(Xb,s):
    ang,dx,dy,blur,contrast,bright,noise=params(s); out=np.empty_like(Xb); rr=np.random.default_rng(s*4001+17)
    for i,img in enumerate(Xb):
        a=rotate(img,float(ang),reshape=False,order=1,mode='constant',cval=0,prefilter=False)
        a=shift(a,(float(dy),float(dx)),order=1,mode='constant',cval=0,prefilter=False)
        if blur>1e-6: a=gaussian_filter(a,float(blur),mode='nearest')
        out[i]=np.clip((a-.5)*contrast+.5+bright+rr.normal(0,noise,a.shape),0,1)
    return out.reshape(len(out),-1).astype(np.float32)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.body=torch.nn.Sequential(torch.nn.Linear(64,128),torch.nn.ReLU()); self.head=torch.nn.Linear(128,10)
    def forward(self,x): h=self.body(x); return self.head(h),h

def gdir(z,h,yb):
    b=len(yb); p=torch.softmax(z.detach(),-1); oh=torch.nn.functional.one_hot(yb,10).float(); r=(p-oh)/b
    gw=torch.einsum('bc,bh->ch',r,h.detach()).reshape(-1); gb=r.sum(0); g=torch.cat([gw,gb]); return g/g.norm().clamp_min(1e-12)

def gdirs(z,h,yb,M):
    b=len(yb); zz=z.detach().reshape(M,b,-1); hh=h.detach().reshape(M,b,-1)
    p=torch.softmax(zz,-1); oh=torch.nn.functional.one_hot(yb,10).float()[None]; r=(p-oh)/b
    gw=torch.einsum('mbc,mbh->mch',r,hh).reshape(M,-1); gb=r.sum(1); g=torch.cat([gw,gb],1)
    return g/g.norm(dim=1,keepdim=True).clamp_min(1e-12)

def zscore(x):
    x=torch.as_tensor(x,dtype=torch.float32); return (x-x.mean())/(x.std(unbiased=False)+1e-8)

def farthest_subset(cand,features,k):
    A=features[np.array(cand)]; ctr=A.mean(0,keepdims=True); sel=[int(np.argmax(((A-ctr)**2).sum(1)))]
    while len(sel)<k:
        d=((A[:,None,:]-A[np.array(sel)][None,:,:])**2).sum(-1); md=d.min(1); md[np.array(sel)]=-1; sel.append(int(np.argmax(md)))
    return sel

def grad_select(loss,cos,q=4):
    sel=[int(torch.argmax(loss))]
    while len(sel)<q:
        rem=[i for i in range(len(loss)) if i not in sel]; nov=zscore((1-cos[:,sel]).min(1).values); score=zscore(loss.detach())+.6*nov
        sel.append(max(rem,key=lambda i:float(score[i])))
    return sel

CAL_SEEDS=np.arange(12000,12128,dtype=int)
rng=np.random.default_rng(424242); cal_idx=rng.choice(len(ytr),256,replace=False); cal_y=torch.tensor(ytr[cal_idx])
torch.manual_seed(4242); cal_net=Net(); cal_net.eval(); G=[]
for s in CAL_SEEDS:
    xb=torch.tensor(env_geom(Xtr[cal_idx],int(s)))
    with torch.no_grad(): z,h=cal_net(xb)
    G.append(gdir(z,h,cal_y).numpy())
G=np.stack(G)
R64_cal=np.stack([np.random.default_rng(int(s)).random(64) for s in CAL_SEEDS]).astype(np.float32)
mu=R64_cal.mean(0); sd=R64_cal.std(0)+1e-8; Xcal=(R64_cal-mu)/sd
pca=PCA(n_components=16,random_state=0); Y=pca.fit_transform(G); Y=(Y-Y.mean(0))/(Y.std(0)+1e-8)
ridge=RidgeCV(alphas=[.01,.1,1.,10.,100.]).fit(Xcal,Y)
coef=np.asarray(ridge.coef_); relevance=np.sqrt((coef**2).sum(0)); top7=np.argsort(relevance)[-7:][::-1]
print('RIDGE_ALPHA',ridge.alpha_,'TOP7',top7.tolist(),'REL',np.round(relevance[top7],4).tolist(),flush=True)
print('TRUE_RELEVANT_RANKS',[int(np.where(np.argsort(relevance)[::-1]==j)[0][0]+1) for j in range(7)],flush=True)

SEEDS=np.arange(13000,13064,dtype=int)
R64=np.stack([np.random.default_rng(int(s)).random(64) for s in SEEDS]).astype(np.float32); R64z=(R64-mu)/sd
ORACLE=R64z[:,:7]; RAW64=R64z; LEARNED=ridge.predict(R64z).astype(np.float32); TOP7=R64z[:,top7]
def std(A): return (A-A.mean(0))/(A.std(0)+1e-8)
ORACLE,RAW64,LEARNED,TOP7=map(std,[ORACLE,RAW64,LEARNED,TOP7])

train=[torch.tensor(env_geom(Xtr,int(s))) for s in SEEDS]
unseen=[torch.tensor(env_geom(Xte,s)) for s in range(14000,14080)]
clean=torch.tensor(Xte.reshape(len(Xte),-1).astype(np.float32)); ty=torch.tensor(ytr); ey=torch.tensor(yte)

rows=[]; t0=time.time()
for rep in range(start,end):
    base=17100000+rep*2179; tg=torch.Generator().manual_seed(base+1); er=np.random.default_rng(base+2); sched=[]
    for ep in range(EPOCHS):
        for b in torch.randperm(len(ty),generator=tg).split(BATCH): sched.append((b,er.choice(64,K,replace=False).tolist()))
    for mi,method in enumerate(METHODS):
        torch.manual_seed(base); np.random.seed(base%(2**32-1)); random.seed(base); rr=np.random.default_rng(base+9000+mi)
        net=Net(); opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=1e-3); nf=0; st=time.time()
        for b,c16 in sched:
            if method=='grad16': loc=list(range(K))
            elif method=='oracle7_8': loc=farthest_subset(c16,ORACLE,PREF)
            elif method=='raw64_8': loc=farthest_subset(c16,RAW64,PREF)
            elif method=='learned_pred8': loc=farthest_subset(c16,LEARNED,PREF)
            elif method=='learned_top7_8': loc=farthest_subset(c16,TOP7,PREF)
            else: loc=sorted(rr.choice(K,PREF,replace=False).tolist())
            cand=[c16[i] for i in loc]; M=len(cand); nf+=M
            xb=torch.cat([train[e][b] for e in cand]); z,h=net(xb)
            per=torch.nn.functional.cross_entropy(z,ty[b].repeat(M),reduction='none').reshape(M,-1); el=per.mean(1); gd=gdirs(z,h,ty[b],M); idx=grad_select(el,gd@gd.T,Q)
            opt.zero_grad(set_to_none=True); per[idx].mean().backward(); opt.step()
        net.eval(); sec=time.time()-st
        with torch.no_grad():
            a=np.array([float((net(x)[0].argmax(1)==ey).float().mean()) for x in unseen]); ca=float((net(clean)[0].argmax(1)==ey).float().mean())
        rows.append(dict(rep=rep,method=method,train_seconds=sec,env_forwards=nf,mean_test=a.mean(),sd_test=a.std(ddof=1),p10_test=np.quantile(a,.1),min_test=a.min(),clean_test=ca))
    print('rep',rep,'done',round(time.time()-t0,1),flush=True)
pd.DataFrame(rows).to_csv(f'learned_rng_fingerprint_{start}_{end}.csv',index=False)

import sys, random, time
import numpy as np, pandas as pd, torch
from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate, shift, gaussian_filter

torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
K=16; Q=4

def zscore(x):
    x=torch.as_tensor(x,dtype=torch.float32)
    return (x-x.mean())/(x.std(unbiased=False)+1e-8)

def eff_rank_from_rows(G):
    G=np.asarray(G,np.float64)
    if len(G)<2: return np.nan
    G=G-G.mean(0,keepdims=True)
    gram=G@G.T
    vals=np.linalg.eigvalsh(gram)
    vals=np.clip(vals,0,None)
    s=vals.sum()
    if s<=1e-15: return 1.0
    p=vals[vals>1e-15]/s
    return float(np.exp(-(p*np.log(p)).sum()))

def eff_rank_features(H):
    H=np.asarray(H,np.float64)
    H=H-H.mean(0,keepdims=True)
    s=np.linalg.svd(H,compute_uv=False)
    vals=s*s
    sm=vals.sum()
    if sm<=1e-15: return 1.0
    p=vals[vals>1e-15]/sm
    return float(np.exp(-(p*np.log(p)).sum()))

def select(loss,cos,beta):
    selected=[int(torch.argmax(loss))]
    while len(selected)<Q:
        rem=[i for i in range(K) if i not in selected]
        nov=zscore((1-cos[:,selected]).min(1).values)
        score=zscore(loss.detach())+beta*nov
        selected.append(max(rem,key=lambda i:float(score[i])))
    return selected

def head_grads(logits,h,yb,nclass):
    b=len(yb); zz=logits.detach().reshape(K,b,-1); hh=h.detach().reshape(K,b,-1)
    p=torch.softmax(zz,-1); oh=torch.nn.functional.one_hot(yb,nclass).float()[None]
    r=(p-oh)/b
    gw=torch.einsum('mbc,mbh->mch',r,hh).reshape(K,-1); gb=r.sum(1)
    raw=torch.cat([gw,gb],1)
    unit=raw/raw.norm(dim=1,keepdim=True).clamp_min(1e-12)
    return raw,unit

def setup_digits():
    X,y=load_digits(return_X_y=True); X=(X.astype(np.float32)/16.).reshape(-1,8,8); y=y.astype(np.int64)
    Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=.45,random_state=314159,stratify=y)
    _,Xte,_,yte=train_test_split(Xtmp,ytmp,test_size=.55,random_state=271828,stratify=ytmp)
    def env_geom(Xb,s):
        r=np.random.default_rng(s); ang=r.uniform(-30,30); dx=r.uniform(-1.2,1.2); dy=r.uniform(-1.2,1.2); blur=r.uniform(0,.85); contrast=r.uniform(.78,1.22); bright=r.uniform(-.06,.06); noise=r.uniform(0,.08)
        rr=np.random.default_rng(s*4001+17)
        a=rotate(Xb,float(ang),axes=(1,2),reshape=False,order=1,mode='constant',cval=0,prefilter=False)
        a=shift(a,(0,float(dy),float(dx)),order=1,mode='constant',cval=0,prefilter=False)
        if blur>1e-6: a=gaussian_filter(a,(0,float(blur),float(blur)),mode='nearest')
        a=(a-.5)*contrast+.5+bright+rr.normal(0,noise,a.shape)
        return np.clip(a,0,1).reshape(len(a),-1).astype(np.float32)
    train=[torch.tensor(env_geom(Xtr,s)) for s in range(13000,13064)]
    unseen=[torch.tensor(env_geom(Xte,s)) for s in range(14000,14080)]
    cleantr=torch.tensor(Xtr.reshape(len(Xtr),-1).astype(np.float32)); cleante=torch.tensor(Xte.reshape(len(Xte),-1).astype(np.float32))
    ty=torch.tensor(ytr); ey=torch.tensor(yte)
    class Net(torch.nn.Module):
        def __init__(self): super().__init__(); self.body=torch.nn.Sequential(torch.nn.Linear(64,128),torch.nn.ReLU()); self.head=torch.nn.Linear(128,10)
        def forward(self,x): h=self.body(x); return self.head(h),h
    return dict(name='digits',train=train,unseen=unseen,cleantr=cleantr,cleante=cleante,ty=ty,ey=ey,Net=Net,nclass=10,batch=128,epochs=10,lr=1e-2,audit_env=[1,9,17,25,33,41,49,57])

def setup_breast(regime='high'):
    X,y=load_breast_cancer(return_X_y=True); X=X.astype(np.float32); y=y.astype(np.int64)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.30,random_state=314159,stratify=y)
    mu=Xtr.mean(0,keepdims=True); sd=Xtr.std(0,keepdims=True)+1e-6
    Xtr=((Xtr-mu)/sd).astype(np.float32); Xte=((Xte-mu)/sd).astype(np.float32)
    def env(Xb,seed,offset):
        r=np.random.default_rng(seed)
        if regime=='low': sigma=r.uniform(.02,.15); mask_p=r.uniform(0,.08); gain=r.uniform(.90,1.10); sh=r.uniform(-.08,.08)
        else: sigma=r.uniform(.05,.55); mask_p=r.uniform(0,.25); gain=r.uniform(.72,1.28); sh=r.uniform(-.25,.25)
        rr=np.random.default_rng(seed*2017+offset); noise=rr.normal(0,sigma,Xb.shape).astype(np.float32); mask=(rr.random(Xb.shape)>=mask_p).astype(np.float32)
        return ((Xb*gain+sh+noise)*mask).astype(np.float32)
    ts=21000 if regime=='low' else 23000; us=22000 if regime=='low' else 24000
    train=[torch.tensor(env(Xtr,s,11)) for s in range(ts,ts+64)]
    unseen=[torch.tensor(env(Xte,s,37)) for s in range(us,us+80)]
    cleantr=torch.tensor(Xtr); cleante=torch.tensor(Xte); ty=torch.tensor(ytr); ey=torch.tensor(yte)
    class Net(torch.nn.Module):
        def __init__(self): super().__init__(); self.body=torch.nn.Sequential(torch.nn.Linear(30,64),torch.nn.ReLU()); self.head=torch.nn.Linear(64,2)
        def forward(self,x): h=self.body(x); return self.head(h),h
    return dict(name='breast_'+regime,train=train,unseen=unseen,cleantr=cleantr,cleante=cleante,ty=ty,ey=ey,Net=Net,nclass=2,batch=64,epochs=20,lr=3e-3,audit_env=[1,9,17,25,33,41,49,57])

def setup_synth():
    from sklearn.datasets import make_classification
    X,y=make_classification(n_samples=3200,n_features=40,n_informative=24,n_redundant=8,n_classes=4,n_clusters_per_class=2,class_sep=1.25,flip_y=.02,random_state=20260823)
    X=X.astype(np.float32); y=y.astype(np.int64)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3,random_state=314159,stratify=y)
    mu=Xtr.mean(0,keepdims=True); sd=Xtr.std(0,keepdims=True)+1e-6; Xtr=((Xtr-mu)/sd).astype(np.float32); Xte=((Xte-mu)/sd).astype(np.float32)
    def env(Xb,s,off):
        r=np.random.default_rng(s); sig=r.uniform(.03,.30); maskp=r.uniform(0,.12); gain=r.uniform(.85,1.15); sh=r.uniform(-.12,.12)
        rr=np.random.default_rng(s*2017+off); n=rr.normal(0,sig,Xb.shape).astype(np.float32); m=(rr.random(Xb.shape)>=maskp).astype(np.float32)
        return ((Xb*gain+sh+n)*m).astype(np.float32)
    train=[torch.tensor(env(Xtr,s,11)) for s in range(7000,7064)]
    unseen=[torch.tensor(env(Xte,s,37)) for s in range(8000,8080)]
    cleantr=torch.tensor(Xtr); cleante=torch.tensor(Xte); ty=torch.tensor(ytr); ey=torch.tensor(yte)
    class Net(torch.nn.Module):
        def __init__(self): super().__init__(); self.body=torch.nn.Sequential(torch.nn.Linear(40,128),torch.nn.ReLU()); self.head=torch.nn.Linear(128,4)
        def forward(self,x): h=self.body(x); return self.head(h),h
    return dict(name='synthetic',train=train,unseen=unseen,cleantr=cleantr,cleante=cleante,ty=ty,ey=ey,Net=Net,nclass=4,batch=128,epochs=12,lr=3e-3,audit_env=[1,9,17,25,33,41,49,57])

def audit(model,cfg,probe_idx):
    model.eval(); hs=[]; margins=[]; inv=[]
    with torch.no_grad():
        zc,hc=model(cfg['cleantr'][probe_idx]); hcn=hc/hc.norm(dim=1,keepdim=True).clamp_min(1e-12)
        for e in cfg['audit_env']:
            z,h=model(cfg['train'][e][probe_idx])
            hn=h/h.norm(dim=1,keepdim=True).clamp_min(1e-12)
            inv.append((hn*hcn).sum(1).cpu().numpy())
            yy=cfg['ty'][probe_idx]
            true=z.gather(1,yy[:,None]).squeeze(1)
            z2=z.clone(); z2[torch.arange(len(yy)),yy]=-1e9
            mar=true-z2.max(1).values
            margins.append(mar.cpu().numpy()); hs.append(h.cpu().numpy())
    H=np.concatenate(hs,0); M=np.concatenate(margins); I=np.concatenate(inv)
    return dict(rep_eff_rank=eff_rank_features(H),margin_mean=float(M.mean()),margin_p10=float(np.quantile(M,.1)),feature_invariance=float(I.mean()),feature_invariance_p10=float(np.quantile(I,.1)))

def run_task(cfg,start,end,out):
    rows=[]; t0=time.time()
    for rep in range(start,end):
        base=(61000000 if cfg['name']=='digits' else 71000000)+rep*2029
        gen=torch.Generator().manual_seed(base+1); er=np.random.default_rng(base+2); sched=[]
        for ep in range(cfg['epochs']):
            for b in torch.randperm(len(cfg['ty']),generator=gen).split(cfg['batch']): sched.append((ep,b,er.choice(64,K,replace=False).tolist()))
        prng=np.random.default_rng(base+3); probe_idx=torch.tensor(prng.choice(len(cfg['ty']),min(192,len(cfg['ty'])),replace=False))
        for method,beta in [('loss_hard',0.0),('gradnov',1.5)]:
            torch.manual_seed(base); np.random.seed(base%(2**32-1)); random.seed(base)
            model=cfg['Net'](); init=torch.cat([p.detach().flatten() for p in model.parameters()]).clone(); opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=1e-3)
            upd_dirs=[]; upd_norms=[]; last_ep=-1
            for ep,b,cand in sched:
                model.train(); opt.zero_grad(set_to_none=True)
                xb=torch.cat([cfg['train'][e][b] for e in cand]); z,h=model(xb)
                per=torch.nn.functional.cross_entropy(z,cfg['ty'][b].repeat(K),reduction='none').reshape(K,-1); el=per.mean(1)
                raw,unit=head_grads(z,h,cfg['ty'][b],cfg['nclass']); cos=unit@unit.T; idx=select(el,cos,beta)
                agg=raw[idx].mean(0); upd_norms.append(float(agg.norm())); upd_dirs.append((agg/agg.norm().clamp_min(1e-12)).cpu().numpy())
                per[idx].mean().backward(); opt.step()
                if ep!=last_ep and ep in {0,max(0,cfg['epochs']//2-1),cfg['epochs']-1}: last_ep=ep
            final=torch.cat([p.detach().flatten() for p in model.parameters()])
            A=audit(model,cfg,probe_idx)
            G=np.stack(upd_dirs)
            consec=float(np.mean(np.sum(G[1:]*G[:-1],axis=1))) if len(G)>1 else np.nan
            path_len=float(np.sum(upd_norms))*cfg['lr']
            disp=float(torch.norm(final-init))
            model.eval()
            with torch.no_grad():
                acc=np.array([float((model(x)[0].argmax(1)==cfg['ey']).float().mean()) for x in cfg['unseen']])
                clean=float((model(cfg['cleante'])[0].argmax(1)==cfg['ey']).float().mean())
            row=dict(task=cfg['name'],rep=rep,method=method,mean_test=float(acc.mean()),sd_test=float(acc.std(ddof=1)),p10_test=float(np.quantile(acc,.1)),min_test=float(acc.min()),clean_test=clean,grad_eff_rank=eff_rank_from_rows(G),grad_eff_rank_frac=eff_rank_from_rows(G)/len(G),consecutive_grad_cos=consec,head_path_len=path_len,param_displacement=disp,path_disp_ratio=path_len/(disp+1e-12),**A)
            rows.append(row)
        pd.DataFrame(rows).to_csv(out,index=False)
        print(cfg['name'],'rep',rep,'done',round(time.time()-t0,1),flush=True)
    pd.DataFrame(rows).to_csv(out,index=False)

if __name__=='__main__':
    task=sys.argv[1]; start=int(sys.argv[2]); end=int(sys.argv[3]); out=sys.argv[4]
    cfg=setup_digits() if task=='digits' else (setup_synth() if task=='synthetic' else setup_breast('low' if task=='breast_low' else 'high')); run_task(cfg,start,end,out)

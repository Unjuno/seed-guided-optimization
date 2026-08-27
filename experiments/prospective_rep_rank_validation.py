from __future__ import annotations
import argparse, random, math
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter

K=16; Q=4; BATCH=128; EPOCHS=10; LR=1e-2
TRAIN_START=51000; TEST_START=52000
AUDIT_ENVS=[1,9,17,25,33,41,49,57]
torch.set_num_threads(1); torch.use_deterministic_algorithms(True)

def zscore(x):
    x=torch.as_tensor(x,dtype=torch.float32)
    return (x-x.mean())/(x.std(unbiased=False)+1e-8)

def eff_rank_features(H):
    H=np.asarray(H,np.float64); H=H-H.mean(0,keepdims=True)
    s=np.linalg.svd(H,compute_uv=False); vals=s*s; sm=vals.sum()
    if sm<=1e-15:return 1.0
    p=vals[vals>1e-15]/sm
    return float(np.exp(-(p*np.log(p)).sum()))

def eff_rank_rows(G):
    G=np.asarray(G,np.float64); G=G-G.mean(0,keepdims=True)
    s=np.linalg.svd(G,compute_uv=False); vals=s*s; sm=vals.sum()
    if sm<=1e-15:return 1.0
    p=vals[vals>1e-15]/sm
    return float(np.exp(-(p*np.log(p)).sum()))

def transform_photometric(Xb, seed, offset):
    r=np.random.default_rng(seed)
    gamma=r.uniform(.65,1.45)
    amp=r.uniform(0,.28)
    freq=r.uniform(.55,2.25)
    angle=r.uniform(0,2*np.pi)
    phase=r.uniform(0,2*np.pi)
    contrast=r.uniform(.75,1.25)
    bright=r.uniform(-.10,.10)
    blur=r.uniform(0,.55)
    occ_w=int(r.integers(0,4)); occ_h=int(r.integers(0,4))
    occ_x=int(r.integers(0,8)); occ_y=int(r.integers(0,8))
    noise=r.uniform(0,.07)
    a=np.clip(Xb,0,1).astype(np.float32)**gamma
    yy,xx=np.mgrid[0:8,0:8].astype(np.float32)
    coord=(np.cos(angle)*xx+np.sin(angle)*yy)/8.0
    illum=1.0+amp*np.sin(2*np.pi*freq*coord+phase)
    a=a*illum[None]
    a=(a-.5)*contrast+.5+bright
    if blur>1e-6:a=gaussian_filter(a,(0,float(blur),float(blur)),mode='nearest')
    if occ_w>0 and occ_h>0:
        x0=max(0,min(7,occ_x-occ_w//2)); x1=min(8,x0+occ_w)
        y0=max(0,min(7,occ_y-occ_h//2)); y1=min(8,y0+occ_h)
        a[:,y0:y1,x0:x1]*=r.uniform(0,.35)
    rr=np.random.default_rng(seed*5003+offset)
    a=a+rr.normal(0,noise,a.shape).astype(np.float32)
    return np.clip(a,0,1).reshape(len(a),-1).astype(np.float32)

def data_split():
    X,y=load_digits(return_X_y=True); X=(X.astype(np.float32)/16.).reshape(-1,8,8); y=y.astype(np.int64)
    Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=.45,random_state=314159,stratify=y)
    _,Xte,_,yte=train_test_split(Xtmp,ytmp,test_size=.55,random_state=271828,stratify=ytmp)
    return Xtr,ytr,Xte,yte

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.body=torch.nn.Sequential(torch.nn.Linear(64,128),torch.nn.ReLU()); self.head=torch.nn.Linear(128,10)
    def forward(self,x): h=self.body(x); return self.head(h),h

def head_grads(logits,h,yb):
    b=len(yb); zz=logits.detach().reshape(K,b,-1); hh=h.detach().reshape(K,b,-1)
    p=torch.softmax(zz,-1); oh=torch.nn.functional.one_hot(yb,10).float()[None]
    rr=(p-oh)/b
    gw=torch.einsum('mbc,mbh->mch',rr,hh).reshape(K,-1); gb=rr.sum(1)
    raw=torch.cat([gw,gb],1); unit=raw/raw.norm(dim=1,keepdim=True).clamp_min(1e-12)
    return raw,unit

def select(loss,cos,beta):
    sel=[int(torch.argmax(loss))]
    while len(sel)<Q:
        rem=[i for i in range(K) if i not in sel]
        nov=zscore((1-cos[:,sel]).min(1).values)
        score=zscore(loss.detach())+beta*nov
        sel.append(max(rem,key=lambda i:float(score[i])))
    return sel

def audit_rep(model, train, cleantr, ty, probe_idx):
    model.eval(); hs=[]; margins=[]; inv=[]
    with torch.no_grad():
        zc,hc=model(cleantr[probe_idx]); hcn=hc/hc.norm(dim=1,keepdim=True).clamp_min(1e-12)
        for e in AUDIT_ENVS:
            z,h=model(train[e][probe_idx]); hn=h/h.norm(dim=1,keepdim=True).clamp_min(1e-12)
            inv.append((hn*hcn).sum(1).cpu().numpy())
            yy=ty[probe_idx]; true=z.gather(1,yy[:,None]).squeeze(1); z2=z.clone(); z2[torch.arange(len(yy)),yy]=-1e9
            margins.append((true-z2.max(1).values).cpu().numpy()); hs.append(h.cpu().numpy())
    H=np.concatenate(hs); M=np.concatenate(margins); I=np.concatenate(inv)
    return dict(rep_eff_rank=eff_rank_features(H), margin_mean=float(M.mean()), margin_p10=float(np.quantile(M,.1)), feature_invariance=float(I.mean()))

def train_diagnostic(start,end,outdir):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True); (outdir/'states').mkdir(exist_ok=True)
    Xtr,ytr,_,_=data_split(); train=[torch.tensor(transform_photometric(Xtr,s,11)) for s in range(TRAIN_START,TRAIN_START+64)]
    cleantr=torch.tensor(Xtr.reshape(len(Xtr),-1).astype(np.float32)); ty=torch.tensor(ytr)
    rows=[]
    for rep in range(start,end):
        base=91000000+rep*2029; gen=torch.Generator().manual_seed(base+1); er=np.random.default_rng(base+2); sched=[]
        for _ in range(EPOCHS):
            for b in torch.randperm(len(ty),generator=gen).split(BATCH): sched.append((b,er.choice(64,K,replace=False).tolist()))
        prng=np.random.default_rng(base+3); probe_idx=torch.tensor(prng.choice(len(ty),min(192,len(ty)),replace=False))
        for method,beta in [('loss_hard',0.0),('gradnov',1.5)]:
            torch.manual_seed(base); np.random.seed(base%(2**32-1)); random.seed(base)
            model=Net(); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-3); G=[]
            for b,cand in sched:
                xb=torch.cat([train[e][b] for e in cand]); z,h=model(xb)
                per=torch.nn.functional.cross_entropy(z,ty[b].repeat(K),reduction='none').reshape(K,-1); el=per.mean(1)
                raw,unit=head_grads(z,h,ty[b]); cos=unit@unit.T; idx=select(el,cos,beta)
                agg=raw[idx].mean(0); G.append((agg/agg.norm().clamp_min(1e-12)).cpu().numpy())
                opt.zero_grad(set_to_none=True); per[idx].mean().backward(); opt.step()
            A=audit_rep(model,train,cleantr,ty,probe_idx)
            row=dict(rep=rep,method=method,grad_eff_rank=eff_rank_rows(np.stack(G)),**A)
            rows.append(row)
            torch.save({'state_dict':model.state_dict(),'rep':rep,'method':method},outdir/'states'/f'rep{rep:02d}_{method}.pt')
        pd.DataFrame(rows).to_csv(outdir/'diagnostics.csv',index=False)
        print('diagnostic rep',rep,'done',flush=True)
    df=pd.DataFrame(rows); df.to_csv(outdir/'diagnostics.csv',index=False)
    wide=df.pivot(index='rep',columns='method',values=['rep_eff_rank','grad_eff_rank','margin_mean','feature_invariance'])
    deltas=pd.DataFrame(index=wide.index)
    for m in ['rep_eff_rank','grad_eff_rank','margin_mean','feature_invariance']:
        deltas['delta_'+m]=wide[(m,'gradnov')]-wide[(m,'loss_hard')]
    deltas.to_csv(outdir/'diagnostic_deltas.csv')
    print(deltas.mean().to_string())

def evaluate(outdir):
    outdir=Path(outdir); _,_,Xte,yte=data_split(); unseen=[torch.tensor(transform_photometric(Xte,s,37)) for s in range(TEST_START,TEST_START+80)]
    cleante=torch.tensor(Xte.reshape(len(Xte),-1).astype(np.float32)); ey=torch.tensor(yte)
    rows=[]
    for p in sorted((outdir/'states').glob('rep*_*.pt')):
        ck=torch.load(p,map_location='cpu'); model=Net(); model.load_state_dict(ck['state_dict']); model.eval()
        with torch.no_grad():
            acc=np.array([float((model(x)[0].argmax(1)==ey).float().mean()) for x in unseen]); clean=float((model(cleante)[0].argmax(1)==ey).float().mean())
        rows.append(dict(rep=ck['rep'],method=ck['method'],mean_test=float(acc.mean()),sd_test=float(acc.std(ddof=1)),p10_test=float(np.quantile(acc,.1)),min_test=float(acc.min()),clean_test=clean))
    df=pd.DataFrame(rows); df.to_csv(outdir/'heldout.csv',index=False)
    w=df.pivot(index='rep',columns='method',values=['mean_test','sd_test','p10_test','min_test','clean_test']); d=pd.DataFrame(index=w.index)
    for m in ['mean_test','sd_test','p10_test','min_test','clean_test']: d['delta_'+m]=w[(m,'gradnov')]-w[(m,'loss_hard')]
    d.to_csv(outdir/'heldout_deltas.csv')
    print(d.mean().to_string())

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['diagnostic','evaluate']); ap.add_argument('--start',type=int,default=0); ap.add_argument('--end',type=int,default=20); ap.add_argument('--outdir',default='/mnt/data/prospective_photometric')
    a=ap.parse_args(); train_diagnostic(a.start,a.end,a.outdir) if a.mode=='diagnostic' else evaluate(a.outdir)

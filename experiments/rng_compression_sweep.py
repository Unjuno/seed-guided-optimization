from __future__ import annotations
import argparse, time
import numpy as np, pandas as pd, torch
from common import MLP, configure_determinism, environment_metrics, farthest_subset, geometric_environment, head_gradient_directions, load_digits_split, rng_fingerprint, seed_everything, select_hard_gradient_novel

def main():
 p=argparse.ArgumentParser(); p.add_argument('--start',type=int,default=0); p.add_argument('--end',type=int,default=20); p.add_argument('--output',default='rng_compression_sweep.csv'); a=p.parse_args()
 configure_determinism(1); B,E,K,Q,LR=128,10,16,4,1e-2; prefilters=[16,12,8,6,4]
 xtr,ytr,xte,yte=load_digits_split(); seeds=np.arange(13000,13064,dtype=int); train=[torch.tensor(geometric_environment(xtr,int(s))) for s in seeds]; unseen=[torch.tensor(geometric_environment(xte,s)) for s in range(14000,14080)]; clean=torch.tensor(xte).flatten(1); ty,ey=torch.tensor(ytr),torch.tensor(yte)
 fp=np.stack([rng_fingerprint(int(s),7) for s in seeds]); fp=(fp-fp.mean(0))/(fp.std(0)+1e-8); rows=[]
 for rep in range(a.start,a.end):
  base=16100000+rep*2131; tg=torch.Generator().manual_seed(base+1); er=np.random.default_rng(base+2); sched=[]
  for _ in range(E):
   for b in torch.randperm(len(ty),generator=tg).split(B): sched.append((b,er.choice(64,K,replace=False).tolist()))
  for m in prefilters:
   seed_everything(base); model=MLP(); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-3); st=time.time(); nf=0
   for b,c16 in sched:
    local=list(range(K)) if m==K else farthest_subset(fp[np.array(c16)],m); cand=[c16[i] for i in local]; nf+=m; xb=torch.cat([train[e][b] for e in cand]); logits,h=model(xb); per=torch.nn.functional.cross_entropy(logits,ty[b].repeat(m),reduction='none').reshape(m,-1); loss=per.mean(1); gd=head_gradient_directions(logits,h,ty[b],m); idx=select_hard_gradient_novel(loss,gd@gd.T,Q); opt.zero_grad(set_to_none=True); per[idx].mean().backward(); opt.step()
   rows.append({'rep':rep,'prefilter':m,'env_forwards':nf,'train_seconds':time.time()-st,**environment_metrics(model,unseen,ey,clean)})
 pd.DataFrame(rows).to_csv(a.output,index=False)
if __name__=='__main__': main()

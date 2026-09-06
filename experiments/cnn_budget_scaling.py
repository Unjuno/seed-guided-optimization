"""Issue #70: SmallCNN finite-budget Q-scaling replication."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, torch
import cnn_regime_interaction as cr
import fixed_dose_response as fd
import transfer_specificity as base
from common import SmallCNN, head_gradient_directions, select_loss_hard, select_hard_gradient_novel, seed_everything

REPS=tuple(range(1500,1530)); METHODS=("loss_hard","gradnov"); QS=(2,4,8,12,16)
K=16; BATCH=128; EPOCHS=10; LR=5e-3; WD=1e-3; NOVELTY_WEIGHT=.6
TRAIN_SEEDS=tuple(range(48000,48064)); HELDOUT_SEEDS=tuple(range(49000,49080))
PROTOCOL={"issue":70,"reps":REPS,"methods":METHODS,"q":QS,"K":K,"batch":BATCH,"epochs":EPOCHS,
          "lr":LR,"wd":WD,"novelty_weight":NOVELTY_WEIGHT,"train_seeds":TRAIN_SEEDS,"heldout_seeds":HELDOUT_SEEDS,
          "base_seed_offset":1910000000,"base_seed_stride":4099,"evaluation":"canonical_nontraining_union_809"}
PROTOCOL_HASH=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()

def state_digest(model):
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        a=t.detach().cpu().contiguous().numpy(); h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

def schedule(n,seed):
    tg=torch.Generator().manual_seed(seed+1); er=np.random.default_rng(seed+2); out=[]
    for _ in range(EPOCHS):
        for b in torch.randperm(n,generator=tg).split(BATCH): out.append((b,er.choice(64,K,replace=False).tolist()))
    return out

def pairnov(cos,sel):
    idx=torch.tensor(sel,dtype=torch.long); sub=cos.index_select(0,idx).index_select(1,idx); up=torch.triu_indices(len(sel),len(sel),offset=1)
    return float((1-sub[up[0],up[1]]).mean()) if up.shape[1] else 0.0

def check_range(start,end):
    if start>=end or not set(range(start,end)).issubset(REPS): raise ValueError("unregistered range")

def train_one(envs,y,sched,seed,method,q):
    seed_everything(seed); model=SmallCNN(); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD); ns=[]; cs=[]; ss=[]
    model.train()
    for b,cand in sched:
        xb=torch.cat([envs[e][b] for e in cand]); logits,h=model(xb)
        per=torch.nn.functional.cross_entropy(logits,y[b].repeat(K),reduction="none").reshape(K,-1); losses=per.mean(1)
        dirs=head_gradient_directions(logits,h,y[b],K); cos=dirs@dirs.T
        sel=select_loss_hard(losses,q) if method=="loss_hard" else select_hard_gradient_novel(losses,cos,q=q,novelty_weight=NOVELTY_WEIGHT)
        sel=sorted(sel)
        ns.append(pairnov(cos,sel)); cs.append(float(losses.mean().detach())); ss.append(float(losses[sel].mean().detach()))
        opt.zero_grad(set_to_none=True); per[sel].mean().backward(); opt.step()
    return model,{"selected_pairwise_novelty":float(np.mean(ns)),"mean_candidate_loss":float(np.mean(cs)),"mean_selected_loss":float(np.mean(ss)),"state_digest":state_digest(model)}

def train(start,end,out):
    check_range(start,end); base.configure_determinism(1); x,y,_,_,_,_,_,_=cr.data_split(); ty=torch.tensor(y)
    out.mkdir(parents=True,exist_ok=True); states=out/"states"; states.mkdir(exist_ok=True)
    if list(states.glob("*.pt")): raise ValueError("refuse overwrite")
    envs=base.geometric_envs(x,TRAIN_SEEDS); rows=[]; hashes={}
    for rep in range(start,end):
        seed=1910000000+4099*rep; sched=schedule(len(ty),seed)
        for q in QS:
            for method in METHODS:
                model,d=train_one(envs,ty,sched,seed,method,q); p=states/f"rep{rep}_q{q}_{method}.pt"
                torch.save({"state_dict":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},"rep":rep,"q":q,"method":method,"protocol_hash":PROTOCOL_HASH},p)
                hashes[p.name]=fd.sha256(p); rows.append({"rep":rep,"q":q,"method":method,"candidate_k":K,"coverage_ratio":q/K,**d})
        print(f"TRAINED rep={rep}; heldout not constructed",flush=True)
    pd.DataFrame(rows).to_csv(out/f"cnn_budget_training_{start}_{end}.csv",index=False)
    sources={n:fd.sha256(Path(__file__).parent/n) for n in ("common.py","cnn_regime_interaction.py","cnn_budget_scaling.py")}
    manifest={"protocol":PROTOCOL,"protocol_hash":PROTOCOL_HASH,"start":start,"end":end,"checkpoints":hashes,
              "source_hashes":sources,"split_hashes":cr.split_hashes(),"runtime":fd.runtime(),"n_train":988,"n_eval":809}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)); print("SEALED_TRAINING_COMPLETE "+PROTOCOL_HASH,flush=True)

def evaluate(start,end,out):
    check_range(start,end); base.configure_determinism(1); m=json.loads((out/"manifest.json").read_text())
    if (m["start"],m["end"],m["protocol_hash"])!=(start,end,PROTOCOL_HASH) or m["split_hashes"]!=cr.split_hashes(): raise ValueError("manifest/split")
    for n,h in m["source_hashes"].items():
        if fd.sha256(Path(__file__).parent/n)!=h: raise ValueError("source changed")
    for n,h in m["checkpoints"].items():
        if fd.sha256(out/"states"/n)!=h: raise ValueError("checkpoint changed")
    _,_,x,y,_,_,_,_=cr.data_split(); labels=torch.tensor(y); clean=torch.tensor(x); held=[torch.tensor(base.geometric_environment(x,int(s))) for s in HELDOUT_SEEDS]
    rows=[]; perenv=[]
    for rep in range(start,end):
        for q in QS:
            for method in METHODS:
                p=out/"states"/f"rep{rep}_q{q}_{method}.pt"; ck=torch.load(p,map_location="cpu",weights_only=True)
                if (ck["rep"],ck["q"],ck["method"],ck["protocol_hash"])!=(rep,q,method,PROTOCOL_HASH): raise ValueError("checkpoint metadata")
                model=SmallCNN(); model.load_state_dict(ck["state_dict"]); model.eval(); sd=state_digest(model); acc=[]
                with torch.no_grad():
                    clean_acc=float((model(clean)[0].argmax(1)==labels).float().mean())
                    for s,xx in zip(HELDOUT_SEEDS,held):
                        pred=model(xx)[0].argmax(1); a=float((pred==labels).float().mean()); acc.append(a); perenv.append({"rep":rep,"q":q,"method":method,"env_seed":s,"accuracy":a})
                a=np.asarray(acc,np.float64); rows.append({"rep":rep,"q":q,"method":method,"candidate_k":K,"coverage_ratio":q/K,
                    "mean_test":float(a.mean()),"sd_test":float(a.std(ddof=1)),"p10_test":float(np.quantile(a,.1)),"min_test":float(a.min()),"clean_test":clean_acc,"state_digest":sd})
    for n,h in m["checkpoints"].items():
        if fd.sha256(out/"states"/n)!=h: raise ValueError("checkpoint mutated")
    pd.DataFrame(rows).to_csv(out/f"cnn_budget_heldout_{start}_{end}.csv",index=False); pd.DataFrame(perenv).to_csv(out/f"cnn_budget_environment_{start}_{end}.csv.gz",index=False,compression="gzip")
    print(f"EVALUATION_COMPLETE {start}:{end}; n_eval=809",flush=True)

def selftest():
    assert cr.data_split()[2].shape[0]==809
    x=torch.eye(16); cos=x@x.T
    for q in QS:
        losses=torch.arange(16,dtype=torch.float32); a=sorted(select_loss_hard(losses,q)); b=sorted(select_hard_gradient_novel(losses,cos,q=q,novelty_weight=.6)); assert len(a)==q and len(b)==q and len(set(b))==q
    a=sorted(select_loss_hard(torch.arange(16,dtype=torch.float32),16)); b=sorted(select_hard_gradient_novel(torch.arange(16,dtype=torch.float32),cos,q=16,novelty_weight=.6)); assert a==b==list(range(16))
    print("SELFTEST PASS",PROTOCOL_HASH)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("mode",choices=("selftest","train","evaluate")); ap.add_argument("--start",type=int); ap.add_argument("--end",type=int); ap.add_argument("--output-dir") ; a=ap.parse_args()
    if a.mode=="selftest": selftest(); return
    if a.start is None or a.end is None or a.output_dir is None: raise ValueError("range/output required")
    (train if a.mode=="train" else evaluate)(a.start,a.end,Path(a.output_dir))
if __name__=="__main__": main()

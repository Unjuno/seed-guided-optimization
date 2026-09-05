"""Issue #67: SmallCNN cross-architecture replication of full-strength regime interaction."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import fixed_dose_response as fd
import transfer_specificity as base
from common import SmallCNN, head_gradient_directions, select_loss_hard, select_hard_gradient_novel, seed_everything

REPS=tuple(range(1400,1430)); METHODS=("loss_hard","gradnov")
DOSES=(0.0,0.10,0.50,1.0); TRAIN_SEEDS=tuple(range(46000,46064)); EVAL_SEEDS=tuple(range(47000,47080))
K=16; Q=4; BATCH=128; EPOCHS=10; LR=5e-3; WD=1e-3; NOVELTY_WEIGHT=.6
PROTOCOL={"issue":67,"reps":REPS,"methods":METHODS,"doses":DOSES,"train_seeds":TRAIN_SEEDS,
          "eval_seeds":EVAL_SEEDS,"K":K,"Q":Q,"batch":BATCH,"epochs":EPOCHS,"lr":LR,"wd":WD,
          "novelty_weight":NOVELTY_WEIGHT,"base_seed_offset":1810000000,"base_seed_stride":4099,
          "evaluation":"canonical_nontraining_union_809","difficulty_proximity_tolerance":.03}
PROTOCOL_HASH=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()

def arrhash(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def data_split():
    x,y=load_digits(return_X_y=True); x=(x.astype(np.float32)/16).reshape(-1,8,8); y=y.astype(np.int64)
    idx=np.arange(len(y),dtype=np.int64)
    tr,tmp=train_test_split(idx,test_size=.45,random_state=314159,stratify=y)
    reserve,usual=train_test_split(tmp,test_size=.55,random_state=271828,stratify=y[tmp])
    if (len(tr),len(tmp),len(reserve),len(usual))!=(988,809,364,445): raise ValueError("split counts changed")
    bx,by,bxt,byt=base.load_digits_split()
    if not (np.array_equal(x[tr],bx) and np.array_equal(y[tr],by) and np.array_equal(x[usual],bxt) and np.array_equal(y[usual],byt)):
        raise ValueError("canonical split mismatch")
    return x[tr],y[tr],x[tmp],y[tmp],tr,tmp,reserve,usual

def split_hashes():
    xtr,ytr,xe,ye,tr,tmp,reserve,usual=data_split()
    return {"train_index":arrhash(tr),"nontrain_index":arrhash(tmp),"reserve_index":arrhash(reserve),"usual_index":arrhash(usual),
            "eval_images":arrhash(xe),"eval_labels":arrhash(ye)}

def schedule(n,seed):
    tg=torch.Generator().manual_seed(seed+1); er=np.random.default_rng(seed+2); out=[]
    for _ in range(EPOCHS):
        for b in torch.randperm(n,generator=tg).split(BATCH): out.append((b,er.choice(64,K,replace=False).tolist()))
    return out

def selected_novelty(cos,sel):
    idx=torch.tensor(sel,dtype=torch.long); sub=cos.index_select(0,idx).index_select(1,idx); up=torch.triu_indices(len(sel),len(sel),offset=1)
    return float((1-sub[up[0],up[1]]).mean())

def train_one(envs,y,sched,seed,method):
    seed_everything(seed); model=SmallCNN(); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD); ns=[]; cs=[]; ss=[]
    model.train()
    for b,cand in sched:
        xb=torch.cat([envs[e][b] for e in cand]); logits,h=model(xb)
        per=torch.nn.functional.cross_entropy(logits,y[b].repeat(K),reduction="none").reshape(K,-1); losses=per.mean(1)
        dirs=head_gradient_directions(logits,h,y[b],K); cos=dirs@dirs.T
        sel=select_loss_hard(losses,Q) if method=="loss_hard" else select_hard_gradient_novel(losses,cos,q=Q,novelty_weight=NOVELTY_WEIGHT)
        ns.append(selected_novelty(cos,sel)); cs.append(float(losses.mean().detach())); ss.append(float(losses[sel].mean().detach()))
        opt.zero_grad(set_to_none=True); per[sel].mean().backward(); opt.step()
    return model,{"selected_pairwise_novelty":float(np.mean(ns)),"mean_candidate_loss":float(np.mean(cs)),"mean_selected_loss":float(np.mean(ss))}

def check_range(start,end):
    if start>=end or not set(range(start,end)).issubset(REPS): raise ValueError("unregistered range")

def train(start,end,out):
    check_range(start,end); base.configure_determinism(1); sh=split_hashes(); x,y,_,_,_,_,_,_=data_split(); ty=torch.tensor(y)
    out.mkdir(parents=True,exist_ok=True); states=out/"states"; states.mkdir(exist_ok=True)
    if list(states.glob("*.pt")): raise ValueError("state overwrite refused")
    envs=base.geometric_envs(x,TRAIN_SEEDS); diag=[]; hashes={}
    for rep in range(start,end):
        seed=1810000000+4099*rep; sched=schedule(len(ty),seed)
        for method in METHODS:
            model,d=train_one(envs,ty,sched,seed,method); p=states/f"rep{rep}_{method}.pt"
            torch.save({"state_dict":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},"rep":rep,"method":method,"protocol_hash":PROTOCOL_HASH},p)
            hashes[p.name]=fd.sha256(p); diag.append({"rep":rep,"method":method,**d})
        print(f"TRAINED {rep}; CNN evaluation not constructed",flush=True)
    pd.DataFrame(diag).to_csv(out/f"cnn_regime_training_{start}_{end}.csv",index=False)
    sources={n:fd.sha256(Path(__file__).parent/n) for n in ("common.py","fixed_dose_response.py","cnn_regime_interaction.py")}
    manifest={"protocol":PROTOCOL,"protocol_hash":PROTOCOL_HASH,"start":start,"end":end,"checkpoints":hashes,
              "source_hashes":sources,"split_hashes":sh,"runtime":fd.runtime(),"n_train":988,"n_eval":809}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)); print("SEALED_TRAINING_COMPLETE "+PROTOCOL_HASH,flush=True)

def evaluate(start,end,out):
    check_range(start,end); base.configure_determinism(1); m=json.loads((out/"manifest.json").read_text())
    if (m["start"],m["end"],m["protocol_hash"])!=(start,end,PROTOCOL_HASH) or m["split_hashes"]!=split_hashes(): raise ValueError("manifest/split mismatch")
    expected={f"rep{r}_{method}.pt" for r in range(start,end) for method in METHODS}
    if set(m["checkpoints"])!=expected: raise ValueError("checkpoint set")
    for n,h in m["source_hashes"].items():
        if fd.sha256(Path(__file__).parent/n)!=h: raise ValueError("source changed after seal")
    for n,h in m["checkpoints"].items():
        if fd.sha256(out/"states"/n)!=h: raise ValueError("checkpoint changed")
    _,_,x,y,_,_,_,_=data_split(); labels=torch.tensor(y); clean=torch.tensor(x)
    geo=[base.geometric_environment(x,int(s)) for s in EVAL_SEEDS]
    envs={v:[(s,torch.tensor(fd.blend(x,g,v))) for s,g in zip(EVAL_SEEDS,geo)] for v in DOSES if v>0}; envs[0.0]=[(-1,clean)]
    inputs=[]
    for v,items in envs.items():
        for s,xx in items:
            d=xx.numpy().astype(np.float64)-x.astype(np.float64); inputs.append({"strength":v,"env_seed":s,"rms":float(np.sqrt(np.mean(d*d))),"linf":float(np.max(np.abs(d)))})
    pd.DataFrame(inputs).to_csv(out/f"cnn_regime_inputs_{start}_{end}.csv",index=False)
    rows=[]; perenv=[]
    for rep in range(start,end):
        for method in METHODS:
            p=out/"states"/f"rep{rep}_{method}.pt"; ck=torch.load(p,map_location="cpu",weights_only=True)
            if (ck["rep"],ck["method"],ck["protocol_hash"])!=(rep,method,PROTOCOL_HASH): raise ValueError("checkpoint metadata")
            model=SmallCNN(); model.load_state_dict(ck["state_dict"]); model.eval()
            with torch.no_grad():
                clean_pred=model(clean)[0].argmax(1)
                for v in DOSES:
                    block=[]
                    for s,xx in envs[v]:
                        logits=model(xx)[0]; pred=logits.argmax(1); q={"rep":rep,"method":method,"strength":v,"env_seed":s,
                            "accuracy":float((pred==labels).float().mean()),"nll":float(torch.nn.functional.cross_entropy(logits,labels)),
                            "prediction_disagreement":float((pred!=clean_pred).float().mean())}; block.append(q); perenv.append(q)
                    a=np.array([q["accuracy"] for q in block],np.float64); rows.append({"rep":rep,"method":method,"strength":v,
                        "mean_accuracy":float(a.mean()),"mean_nll":float(np.mean([q["nll"] for q in block])),
                        "disagreement":float(np.mean([q["prediction_disagreement"] for q in block])),"p10":float(np.quantile(a,.1)),
                        "minimum":float(a.min()),"n_env":len(block),"n_examples":809,"checkpoint_sha256":m["checkpoints"][p.name]})
    for n,h in m["checkpoints"].items():
        if fd.sha256(out/"states"/n)!=h: raise ValueError("checkpoint mutated")
    pd.DataFrame(rows).to_csv(out/f"cnn_regime_metrics_{start}_{end}.csv",index=False)
    pd.DataFrame(perenv).to_csv(out/f"cnn_regime_environment_{start}_{end}.csv.gz",index=False,compression="gzip")
    print(f"EVALUATION_COMPLETE {start}:{end}; n_eval=809",flush=True)

def analyze(df,reps=REPS):
    key=["rep","method","strength"]; exp={(r,m,v) for r in reps for m in METHODS for v in DOSES}
    if len(df)!=len(exp) or df.duplicated(key).any() or set(df[key].itertuples(index=False,name=None))!=exp: raise ValueError("grid invalid")
    if not np.isfinite(df[["mean_accuracy","mean_nll","disagreement","p10","minimum"]].to_numpy()).all(): raise ValueError("nonfinite")
    if not df.n_examples.eq(809).all() or not df.checkpoint_sha256.str.fullmatch("[0-9a-f]{64}",na=False).all(): raise ValueError("metadata")
    if not np.array_equal(df.n_env.to_numpy(),np.where(df.strength.eq(0),1,80)): raise ValueError("n_env")
    if df.groupby(["rep","method"]).checkpoint_sha256.nunique().ne(1).any(): raise ValueError("states differ by dose")
    z=df.set_index(key); paired=[]
    for r in reps:
        b={}
        for name,v in (("clean",0.0),("weak",.1),("mid",.5),("full",1.0)):
            b[name]=float(z.loc[(r,"gradnov",v),"mean_accuracy"]-z.loc[(r,"loss_hard",v),"mean_accuracy"])
        paired.append({"rep":r,"clean_benefit":b["clean"],"weak_benefit":b["weak"],"mid_benefit":b["mid"],"full_benefit":b["full"],
                       "clean_interaction":b["full"]-b["clean"],"weak_interaction":b["full"]-b["weak"]})
    paired=pd.DataFrame(paired); full=fd.estimate(paired.full_benefit); ci=fd.estimate(paired.clean_interaction); wi=fd.estimate(paired.weak_interaction)
    fp=full["mean"]>0 and full["p_one_sided"]<.05; ip=ci["mean"]>0 and ci["p_one_sided"]<.05
    decision=("CROSS-ARCH DOSE INTERACTION REPLICATES" if fp and ip else "CNN FULL EFFECT REPLICATES / CLEAN INTERACTION DOES NOT" if fp else "CNN FULL EFFECT DOES NOT REPLICATE")
    lh=df[df.method=="loss_hard"].groupby("strength").mean_accuracy.mean(); # unused scalar guard below
    lhweak=float(df[(df.method=="loss_hard")&(df.strength==.1)].mean_accuracy.mean()); lhfull=float(df[(df.method=="loss_hard")&(df.strength==1.)].mean_accuracy.mean())
    proximity=abs(lhfull-lhweak); secondary=bool(wi["mean"]>0 and wi["p_one_sided"]<.05 and proximity<=.03)
    summary=[]
    for v in DOSES:
        p=df[df.strength==v].pivot(index="rep",columns="method",values="mean_accuracy"); summary.append({"strength":v,"loss_hard_mean":float(p.loss_hard.mean()),"gradnov_mean":float(p.gradnov.mean()),**fd.estimate(p.gradnov-p.loss_hard)})
    dec={"decision":decision,"full_pass":fp,"clean_interaction_pass":ip,"secondary_difficulty_proximate_support":secondary,
         "loss_hard_weak_mean":lhweak,"loss_hard_full_mean":lhfull,"difficulty_proximity":proximity,"difficulty_tolerance":.03,
         "positive_full_pairs":int((paired.full_benefit>0).sum()),"positive_clean_interaction_pairs":int((paired.clean_interaction>0).sum()),
         "positive_weak_interaction_pairs":int((paired.weak_interaction>0).sum()),"protocol_hash":PROTOCOL_HASH,
         **{"full_"+k:v for k,v in full.items()},**{"clean_interaction_"+k:v for k,v in ci.items()},**{"weak_interaction_"+k:v for k,v in wi.items()}}
    return paired,pd.DataFrame(summary),pd.DataFrame([dec])

def summarize(root,out):
    paths=sorted(root.rglob("cnn_regime_metrics_*.csv"));
    if not paths: raise FileNotFoundError("no metrics")
    df=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True); paired,summary,decision=analyze(df)
    covered=[]; src=[]; splits=[]
    for path in sorted(root.rglob("manifest.json")):
        m=json.loads(path.read_text());
        if m["protocol_hash"]!=PROTOCOL_HASH: raise ValueError("manifest protocol")
        covered.extend(range(m["start"],m["end"])); src.append(m["source_hashes"]); splits.append(m["split_hashes"])
        for n,h in m["checkpoints"].items():
            if fd.sha256(path.parent/"states"/n)!=h: raise ValueError("checkpoint hash")
        for row in df[df.rep.between(m["start"],m["end"]-1)].itertuples():
            if m["checkpoints"][f"rep{row.rep}_{row.method}.pt"]!=row.checkpoint_sha256: raise ValueError("metric hash")
    if sorted(covered)!=list(REPS) or any(x!=src[0] for x in src) or any(x!=splits[0] for x in splits): raise ValueError("manifest coverage")
    if splits[0]!=split_hashes(): raise ValueError("split changed")
    out.mkdir(parents=True,exist_ok=True); df.to_csv(out/"cnn_regime_metrics30.csv",index=False); paired.to_csv(out/"cnn_regime_paired30.csv",index=False)
    summary.to_csv(out/"cnn_regime_summary30.csv",index=False); decision.to_csv(out/"cnn_regime_decision30.csv",index=False); print(decision.to_string(index=False),flush=True)

def selftest():
    sh=split_hashes(); assert len(sh)==6
    reps=tuple(range(4)); rows=[]
    for r in reps:
        for m in METHODS:
            for v in DOSES:
                gain=(.01+.03*v) if m=="gradnov" else 0
                rows.append({"rep":r,"method":m,"strength":v,"mean_accuracy":.5+gain,"mean_nll":1.,"disagreement":0.,"p10":.5,"minimum":.4,
                             "n_env":1 if v==0 else 80,"n_examples":809,"checkpoint_sha256":hashlib.sha256(f"{r}_{m}".encode()).hexdigest()})
    d=analyze(pd.DataFrame(rows),reps)[2].iloc[0]
    if d.decision!="CROSS-ARCH DOSE INTERACTION REPLICATES" or not bool(d.secondary_difficulty_proximate_support): raise AssertionError("pass branch")
    print("SELFTEST PASS",flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("mode",choices=["selftest","train","evaluate","summarize"]); ap.add_argument("--start",type=int); ap.add_argument("--end",type=int); ap.add_argument("--output-dir"); ap.add_argument("--input-dir"); a=ap.parse_args()
    if a.mode=="selftest": selftest()
    elif a.mode=="train": train(a.start,a.end,Path(a.output_dir))
    elif a.mode=="evaluate": evaluate(a.start,a.end,Path(a.output_dir))
    else: summarize(Path(a.input_dir),Path(a.output_dir))
if __name__=="__main__": main()

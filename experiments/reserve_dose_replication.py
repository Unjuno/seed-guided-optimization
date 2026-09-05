"""Issue #64: reserve-image replication of Issue #61 full-vs-clean interaction."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import fixed_dose_response as fd
import transfer_specificity as base

REPS=tuple(range(1300,1330)); METHODS=("loss_hard","gradnov")
DOSES=(0.0,0.10,0.50,1.0); CONDITIONS=tuple(("shared",v) for v in DOSES)
TRAIN_SEEDS=tuple(range(44000,44064)); EVAL_SEEDS=tuple(range(45000,45080))
PROTOCOL={"issue":64,"reps":REPS,"methods":METHODS,"doses":DOSES,
          "train_seeds":TRAIN_SEEDS,"eval_seeds":EVAL_SEEDS,
          "base_seed_offset":1710000000,"base_seed_stride":4099,
          "reserve_split":[.45,314159,.55,271828,"first_return",364]}
PROTOCOL_HASH=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()

def arrhash(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def reserve_split():
    x,y=load_digits(return_X_y=True); x=(x.astype(np.float32)/16).reshape(-1,8,8); y=y.astype(np.int64)
    idx=np.arange(len(y),dtype=np.int64)
    tr,tmp=train_test_split(idx,test_size=.45,random_state=314159,stratify=y)
    reserve,usual=train_test_split(tmp,test_size=.55,random_state=271828,stratify=y[tmp])
    if (len(tr),len(reserve),len(usual))!=(988,364,445): raise ValueError("split counts changed")
    if set(tr)&set(reserve) or set(tr)&set(usual) or set(reserve)&set(usual): raise ValueError("split overlap")
    return x[tr],y[tr],x[reserve],y[reserve],x[usual],y[usual],tr,reserve,usual

def split_hashes():
    xtr,ytr,xr,yr,xu,yu,tr,reserve,usual=reserve_split()
    bxtr,bytr,bxte,byte=base.load_digits_split()
    if not (np.array_equal(xtr,bxtr) and np.array_equal(ytr,bytr) and np.array_equal(xu,bxte) and np.array_equal(yu,byte)):
        raise ValueError("canonical split reconstruction mismatch")
    return {"train_index":arrhash(tr),"reserve_index":arrhash(reserve),"usual_test_index":arrhash(usual),
            "reserve_images":arrhash(xr),"reserve_labels":arrhash(yr)}

def check_range(start,end):
    if start>=end or not set(range(start,end)).issubset(REPS): raise ValueError("unregistered rep range")

def train(start,end,out):
    check_range(start,end); base.configure_determinism(1)
    if (base.K,base.Q,base.BATCH,base.EPOCHS,base.LR,base.WD,base.NOVELTY_WEIGHT)!=(16,4,128,10,.01,.001,.6):
        raise ValueError("base protocol changed")
    sh=split_hashes(); x,y,_,_,_,_,_,_,_=reserve_split(); ty=torch.tensor(y)
    out.mkdir(parents=True,exist_ok=True); states=out/"states"; states.mkdir(exist_ok=True)
    if list(states.glob("*.pt")): raise ValueError("state overwrite refused")
    envs=base.geometric_envs(x,TRAIN_SEEDS); diag=[]; hashes={}
    for rep in range(start,end):
        seed=1710000000+4099*rep; sched=base.schedule(len(ty),seed,TRAIN_SEEDS)
        for method in METHODS:
            model,d=base.train_one(envs,ty,sched,seed,method); p=states/f"rep{rep}_{method}.pt"
            torch.save({"state_dict":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        "rep":rep,"method":method,"protocol_hash":PROTOCOL_HASH},p)
            hashes[p.name]=fd.sha256(p); diag.append({"rep":rep,"method":method,**d})
        print(f"TRAINED {rep}; reserve evaluation not constructed",flush=True)
    pd.DataFrame(diag).to_csv(out/f"reserve_dose_training_{start}_{end}.csv",index=False)
    sources={n:fd.sha256(Path(__file__).parent/n) for n in
             ("common.py","transfer_specificity.py","fixed_dose_response.py","reserve_dose_replication.py")}
    manifest={"protocol":PROTOCOL,"protocol_hash":PROTOCOL_HASH,"start":start,"end":end,
              "checkpoints":hashes,"source_hashes":sources,"split_hashes":sh,
              "runtime":fd.runtime(),"n_train":988,"n_reserve":364}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print("SEALED_TRAINING_COMPLETE "+PROTOCOL_HASH,flush=True)

def evaluate(start,end,out):
    check_range(start,end); base.configure_determinism(1); m=json.loads((out/"manifest.json").read_text())
    if (m["start"],m["end"],m["protocol_hash"])!=(start,end,PROTOCOL_HASH): raise ValueError("manifest mismatch")
    if m["split_hashes"]!=split_hashes(): raise ValueError("split hash mismatch")
    expected={f"rep{r}_{method}.pt" for r in range(start,end) for method in METHODS}
    if set(m["checkpoints"])!=expected: raise ValueError("checkpoint set mismatch")
    for n,h in m["source_hashes"].items():
        if fd.sha256(Path(__file__).parent/n)!=h: raise ValueError("source changed after seal")
    for n,h in m["checkpoints"].items():
        if fd.sha256(out/"states"/n)!=h: raise ValueError("checkpoint changed after seal")
    _,_,x,y,_,_,_,_,_=reserve_split(); labels=torch.tensor(y); clean=torch.tensor(x)
    geo=[base.geometric_environment(x,int(s)) for s in EVAL_SEEDS]
    envs={v:[(s,torch.tensor(fd.blend(x,g,v))) for s,g in zip(EVAL_SEEDS,geo)] for v in DOSES if v>0}
    envs[0.0]=[(-1,clean)]
    inputs=[]
    for v,items in envs.items():
        for s,xx in items:
            d=xx.numpy().astype(np.float64)-x.astype(np.float64)
            inputs.append({"strength":v,"env_seed":s,"rms":float(np.sqrt(np.mean(d*d))),"linf":float(np.max(np.abs(d)))})
    pd.DataFrame(inputs).to_csv(out/f"reserve_dose_inputs_{start}_{end}.csv",index=False)
    rows=[]; perenv=[]
    for rep in range(start,end):
        for method in METHODS:
            p=out/"states"/f"rep{rep}_{method}.pt"; ck=torch.load(p,map_location="cpu",weights_only=True)
            if (ck["rep"],ck["method"],ck["protocol_hash"])!=(rep,method,PROTOCOL_HASH): raise ValueError("checkpoint metadata")
            model=base.MLP(); model.load_state_dict(ck["state_dict"]); model.eval()
            with torch.no_grad():
                clean_pred=model(clean)[0].argmax(1)
                for v in DOSES:
                    block=[]
                    for s,xx in envs[v]:
                        logits=model(xx)[0]; pred=logits.argmax(1)
                        q={"rep":rep,"method":method,"strength":v,"env_seed":s,
                           "accuracy":float((pred==labels).float().mean()),
                           "nll":float(torch.nn.functional.cross_entropy(logits,labels)),
                           "prediction_disagreement":float((pred!=clean_pred).float().mean())}
                        block.append(q); perenv.append(q)
                    a=np.array([q["accuracy"] for q in block],np.float64)
                    rows.append({"rep":rep,"method":method,"family":"shared","strength":v,
                                 "mean_accuracy":float(a.mean()),"mean_nll":float(np.mean([q["nll"] for q in block])),
                                 "disagreement":float(np.mean([q["prediction_disagreement"] for q in block])),
                                 "p10":float(np.quantile(a,.1)),"minimum":float(a.min()),"n_env":len(block),
                                 "n_examples":364,"checkpoint_sha256":m["checkpoints"][p.name]})
    for n,h in m["checkpoints"].items():
        if fd.sha256(out/"states"/n)!=h: raise ValueError("checkpoint mutated during evaluation")
    pd.DataFrame(rows).to_csv(out/f"reserve_dose_metrics_{start}_{end}.csv",index=False)
    pd.DataFrame(perenv).to_csv(out/f"reserve_dose_environment_{start}_{end}.csv.gz",index=False,compression="gzip")
    print(f"EVALUATION_COMPLETE {start}:{end}; n_reserve=364",flush=True)

def analyze(df,reps=REPS):
    key=["rep","method","family","strength"]; exp={(r,m,"shared",v) for r in reps for m in METHODS for v in DOSES}
    if len(df)!=len(exp) or df.duplicated(key).any() or set(df[key].itertuples(index=False,name=None))!=exp:
        raise ValueError("metric grid invalid")
    if not np.isfinite(df[["mean_accuracy","mean_nll","disagreement","p10","minimum"]].to_numpy()).all(): raise ValueError("nonfinite")
    if not df.n_examples.eq(364).all() or not df.checkpoint_sha256.str.fullmatch("[0-9a-f]{64}",na=False).all(): raise ValueError("metadata invalid")
    if not np.array_equal(df.n_env.to_numpy(),np.where(df.strength.eq(0),1,80)): raise ValueError("n_env invalid")
    if df.groupby(["rep","method"]).checkpoint_sha256.nunique().ne(1).any(): raise ValueError("states differ across doses")
    z=df.set_index(key); paired=[]
    for r in reps:
        b={}
        for name,v in (("clean",0.0),("mid",.5),("full",1.0)):
            b[name]=float(z.loc[(r,"gradnov","shared",v),"mean_accuracy"]-z.loc[(r,"loss_hard","shared",v),"mean_accuracy"])
        paired.append({"rep":r,"clean_benefit":b["clean"],"mid_benefit":b["mid"],"full_benefit":b["full"],
                       "interaction":b["full"]-b["clean"]})
    paired=pd.DataFrame(paired); full=fd.estimate(paired.full_benefit); inter=fd.estimate(paired.interaction)
    fp=full["mean"]>0 and full["p_one_sided"]<.05; ip=inter["mean"]>0 and inter["p_one_sided"]<.05
    decision=("DOSE-DEPENDENT BENEFIT REPLICATES ON RESERVE IMAGES" if fp and ip else
              "FULL EFFECT REPLICATES / INTERACTION DOES NOT REPLICATE" if fp else "NO FULL-STRENGTH REPLICATION")
    summary=[]
    for v in DOSES:
        p=df[df.strength==v].pivot(index="rep",columns="method",values="mean_accuracy")
        summary.append({"strength":v,"loss_hard_mean":float(p.loss_hard.mean()),"gradnov_mean":float(p.gradnov.mean()),**fd.estimate(p.gradnov-p.loss_hard)})
    dec={"decision":decision,"full_pass":fp,"interaction_pass":ip,"protocol_hash":PROTOCOL_HASH,
         "positive_full_pairs":int((paired.full_benefit>0).sum()),"positive_interaction_pairs":int((paired.interaction>0).sum()),
         **{"full_"+k:v for k,v in full.items()},**{"interaction_"+k:v for k,v in inter.items()}}
    return paired,pd.DataFrame(summary),pd.DataFrame([dec])

def summarize(root,out):
    paths=sorted(root.rglob("reserve_dose_metrics_*.csv"))
    if not paths: raise FileNotFoundError("no metrics")
    df=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True); paired,summary,decision=analyze(df)
    covered=[]; src=[]; splits=[]
    for path in sorted(root.rglob("manifest.json")):
        m=json.loads(path.read_text())
        if m["protocol_hash"]!=PROTOCOL_HASH: raise ValueError("manifest protocol")
        covered.extend(range(m["start"],m["end"])); src.append(m["source_hashes"]); splits.append(m["split_hashes"])
        for n,h in m["checkpoints"].items():
            if fd.sha256(path.parent/"states"/n)!=h: raise ValueError("aggregate checkpoint hash")
        for row in df[df.rep.between(m["start"],m["end"]-1)].itertuples():
            if m["checkpoints"][f"rep{row.rep}_{row.method}.pt"]!=row.checkpoint_sha256: raise ValueError("metric hash")
    if sorted(covered)!=list(REPS) or any(x!=src[0] for x in src) or any(x!=splits[0] for x in splits): raise ValueError("manifest coverage")
    if splits[0]!=split_hashes(): raise ValueError("reserve split changed")
    out.mkdir(parents=True,exist_ok=True); df.to_csv(out/"reserve_dose_metrics30.csv",index=False)
    paired.to_csv(out/"reserve_dose_paired30.csv",index=False); summary.to_csv(out/"reserve_dose_summary30.csv",index=False)
    decision.to_csv(out/"reserve_dose_decision30.csv",index=False); print(decision.to_string(index=False),flush=True)

def selftest():
    if len(split_hashes()["reserve_index"])!=64: raise AssertionError("split selftest")
    rng=np.random.default_rng(7); x=rng.random((4,8,8),dtype=np.float32); y=rng.random((4,8,8),dtype=np.float32)
    assert np.array_equal(fd.blend(x,y,0),x) and np.array_equal(fd.blend(x,y,1),y)
    reps=tuple(range(4)); rows=[]
    for r in reps:
        for m in METHODS:
            for v in DOSES:
                gain=(.01+.03*v) if m=="gradnov" else 0
                rows.append({"rep":r,"method":m,"family":"shared","strength":v,"mean_accuracy":.5+gain,
                             "mean_nll":1.,"disagreement":0.,"checkpoint_sha256":hashlib.sha256(f"{r}_{m}".encode()).hexdigest(),
                             "p10":.5,"minimum":.4,"n_env":1 if v==0 else 80,"n_examples":364})
    if analyze(pd.DataFrame(rows),reps)[2].loc[0,"decision"]!="DOSE-DEPENDENT BENEFIT REPLICATES ON RESERVE IMAGES": raise AssertionError("pass branch")
    for q in rows:
        if q["method"]=="gradnov": q["mean_accuracy"]=.52
    if analyze(pd.DataFrame(rows),reps)[2].loc[0,"decision"]!="FULL EFFECT REPLICATES / INTERACTION DOES NOT REPLICATE": raise AssertionError("full-only branch")
    print("SELFTEST PASS",flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("mode",choices=["selftest","train","evaluate","summarize"])
    ap.add_argument("--start",type=int); ap.add_argument("--end",type=int); ap.add_argument("--output-dir"); ap.add_argument("--input-dir"); a=ap.parse_args()
    if a.mode=="selftest": selftest()
    elif a.mode=="train": train(a.start,a.end,Path(a.output_dir))
    elif a.mode=="evaluate": evaluate(a.start,a.end,Path(a.output_dir))
    else: summarize(Path(a.input_dir),Path(a.output_dir))
if __name__=="__main__": main()

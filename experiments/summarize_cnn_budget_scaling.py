"""Frozen Issue #70 aggregation and decision."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

REPS=tuple(range(1500,1530)); METHODS=("loss_hard","gradnov"); QS=(2,4,8,12,16)
TRAIN_METRICS=("selected_pairwise_novelty","mean_candidate_loss","mean_selected_loss")
HELD_METRICS=("mean_test","sd_test","p10_test","min_test","clean_test")

def load(root,prefix):
    f=sorted(root.rglob(prefix+"*.csv"));
    if not f: raise FileNotFoundError(prefix)
    return pd.concat([pd.read_csv(x) for x in f],ignore_index=True)

def grid(df,label):
    exp={(r,q,m) for r in REPS for q in QS for m in METHODS}; obs=set(zip(df.rep.astype(int),df.q.astype(int),df.method.astype(str)))
    if len(df)!=len(exp) or obs!=exp or df.duplicated(["rep","q","method"]).any(): raise ValueError(label+" grid")
    cols=[*TRAIN_METRICS] if label=="training" else [*HELD_METRICS]
    if not np.isfinite(df[cols].to_numpy(float)).all(): raise ValueError(label+" nonfinite")
    if not df.state_digest.str.fullmatch("[0-9a-f]{64}",na=False).all(): raise ValueError(label+" state digest")

def estimate(x):
    x=np.asarray(x,np.float64); mean=float(x.mean()); se=float(stats.sem(x)); k=float(stats.t.ppf(.975,len(x)-1)); p=float(stats.ttest_1samp(x,0,alternative="greater").pvalue) if se else (0.0 if mean>0 else .5 if mean==0 else 1.0)
    return {"n":len(x),"mean":mean,"se":se,"ci95_low":mean-k*se,"ci95_high":mean+k*se,"p_one_sided":p}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input-dir",required=True); ap.add_argument("--output-dir",required=True); a=ap.parse_args(); root=Path(a.input_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    tr=load(root,"cnn_budget_training_"); he=load(root,"cnn_budget_heldout_"); grid(tr,"training"); grid(he,"heldout")
    if set(tr.state_digest)!=set(he.state_digest): raise ValueError("training/heldout digest sets")
    keys=["rep","q","method","candidate_k","coverage_ratio","state_digest"]; allr=tr.merge(he,on=keys,validate="one_to_one").sort_values(["rep","q","method"]).reset_index(drop=True)
    if len(allr)!=300: raise ValueError("merge rows")
    mism=[]; fields=("candidate_k","coverage_ratio","state_digest",*TRAIN_METRICS,*HELD_METRICS)
    for r in REPS:
        p=allr[(allr.rep==r)&(allr.q==16)].set_index("method")
        for f in fields:
            x=p.loc["loss_hard",f]; y=p.loc["gradnov",f]
            if x!=y: mism.append({"rep":r,"field":f,"loss_hard":x,"gradnov":y})
    identity=len(mism)==0
    deltas=[]
    for r in REPS:
        for q in QS:
            p=allr[(allr.rep==r)&(allr.q==q)].set_index("method"); row={"rep":r,"q":q,"coverage_ratio":q/16}
            for f in (*TRAIN_METRICS,*HELD_METRICS): row["delta_"+f]=float(p.loc["gradnov",f]-p.loc["loss_hard",f])
            deltas.append(row)
    de=pd.DataFrame(deltas); att=[]
    for r in REPS:
        p=de[de.rep==r].set_index("q"); low=float(p.loc[[2,4],"delta_mean_test"].mean()); high=float(p.loc[[12,16],"delta_mean_test"].mean())
        att.append({"rep":r,"benefit_low":low,"benefit_high":high,"benefit_attenuation":low-high,
                    "slope_vs_coverage":float(np.polyfit(p.loc[list(QS),"coverage_ratio"],p.loc[list(QS),"delta_mean_test"],1)[0])})
    att=pd.DataFrame(att); est=estimate(att.benefit_attenuation); coverage=bool(est["mean"]>0 and est["p_one_sided"]<.05)
    decision="INVALID / Q16 IDENTITY FAILURE" if not identity else "CNN FINITE-BUDGET COVERAGE REPLICATES" if coverage else "CNN COVERAGE THEORY FAIL"
    qrows=[]
    for q in QS:
        p=de[de.q==q]; row={"q":q,"coverage_ratio":q/16,"n":len(p)}
        for f in ("delta_mean_test","delta_clean_test","delta_p10_test","delta_min_test","delta_sd_test","delta_selected_pairwise_novelty","delta_mean_selected_loss","delta_mean_candidate_loss"):
            e=estimate(p[f]); row.update({"mean_"+f:e["mean"],"se_"+f:e["se"],"p_one_sided_"+f:e["p_one_sided"]})
        row["positive_mean_pairs"]=int((p.delta_mean_test>0).sum()); qrows.append(row)
    qs=pd.DataFrame(qrows); dec=pd.DataFrame([{"decision":decision,"n":30,"q16_identity_pass":identity,"q16_mismatch_count":len(mism),"coverage_pass":coverage,
        "mean_benefit_low":float(att.benefit_low.mean()),"mean_benefit_high":float(att.benefit_high.mean()),
        "mean_benefit_attenuation":est["mean"],"benefit_attenuation_se":est["se"],"benefit_attenuation_ci95_low":est["ci95_low"],"benefit_attenuation_ci95_high":est["ci95_high"],"benefit_attenuation_p_one_sided":est["p_one_sided"]}])
    allr.to_csv(out/"cnn_budget_all30.csv",index=False); de.to_csv(out/"cnn_budget_deltas30.csv",index=False); att.to_csv(out/"cnn_budget_attenuation30.csv",index=False); qs.to_csv(out/"cnn_budget_q_summary30.csv",index=False); dec.to_csv(out/"cnn_budget_decision30.csv",index=False); pd.DataFrame(mism,columns=["rep","field","loss_hard","gradnov"]).to_csv(out/"cnn_budget_q16_mismatches.csv",index=False)
    print(dec.to_string(index=False),flush=True); print("\nQ summary\n"+qs.to_string(index=False),flush=True)
if __name__=="__main__": main()

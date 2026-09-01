from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

EXPECTED_REPS=tuple(range(600,630))
FAMILIES=("structured","highdim")
METHODS=("loss_hard","gradnov")
HELDOUT=("mean_test","sd_test","p10_test","min_test","clean_test")


def load_parts(root:Path,prefix:str)->pd.DataFrame:
    files=sorted(root.rglob(prefix+"*.csv"))
    if not files: raise FileNotFoundError(prefix)
    return pd.concat([pd.read_csv(p) for p in files],ignore_index=True)


def validate(df:pd.DataFrame,label:str)->None:
    exp={(r,f,m) for r in EXPECTED_REPS for f in FAMILIES for m in METHODS}
    obs=set(zip(df.rep.astype(int),df.family.astype(str),df.method.astype(str)))
    if len(df)!=len(exp) or obs!=exp: raise ValueError(f"{label} grid mismatch")
    if df.duplicated(["rep","family","method"]).any(): raise ValueError(f"{label} duplicates")


def one_sided(x:np.ndarray)->tuple[float,float]:
    x=np.asarray(x,np.float64)
    if np.all(x==x[0]):
        if x[0]>0:return float("inf"),0.0
        if x[0]<0:return float("-inf"),1.0
        return 0.0,0.5
    r=stats.ttest_1samp(x,0.0); t=float(r.statistic); p2=float(r.pvalue)
    return t,p2/2 if t>0 else 1-p2/2


def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument("--input-dir",required=True); ap.add_argument("--output-dir",required=True); ap.add_argument("--sealed-severity",type=float,required=True); a=ap.parse_args()
    root=Path(a.input_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    diag=load_parts(root,"highdim_diagnostics_"); held=load_parts(root,"highdim_heldout_")
    validate(diag,"diagnostics"); validate(held,"heldout")
    sev=diag.loc[diag.family=="highdim","severity"].dropna().to_numpy(float)
    hsev=held.loc[held.family=="highdim","severity"].dropna().to_numpy(float)
    if len(sev)==0 or not np.all(sev==a.sealed_severity): raise ValueError("training severity mismatch")
    if len(hsev)==0 or not np.all(hsev==a.sealed_severity): raise ValueError("heldout severity mismatch")
    train=[]; family_delta=[]; conv=[]
    for rep in EXPECTED_REPS:
        rt=diag[diag.rep==rep]; rh=held[held.rep==rep]; benefits={}
        for fam in FAMILIES:
            t=rt[rt.family==fam].set_index("method"); h=rh[rh.family==fam].set_index("method")
            ng=float(t.loc["gradnov","selected_pairwise_novelty"]-t.loc["loss_hard","selected_pairwise_novelty"])
            cl=float(t["mean_candidate_loss"].mean())
            sl=float(t["mean_selected_loss"].mean())
            train.append({"rep":rep,"family":fam,"novelty_gain":ng,"candidate_loss_level":cl,"selected_loss_level":sl})
            row={"rep":rep,"family":fam}
            for metric in HELDOUT: row["delta_"+metric]=float(h.loc["gradnov",metric]-h.loc["loss_hard",metric])
            family_delta.append(row); benefits[fam]=row["delta_mean_test"]
        conv.append({"rep":rep,"structured_benefit":benefits["structured"],"highdim_benefit":benefits["highdim"],"conversion_contrast":benefits["structured"]-benefits["highdim"]})
    train=pd.DataFrame(train); family_delta=pd.DataFrame(family_delta); conv=pd.DataFrame(conv)
    s=train[train.family=="structured"]; u=train[train.family=="highdim"]
    ns=float(s.novelty_gain.mean()); nu=float(u.novelty_gain.mean()); ls=float(s.candidate_loss_level.mean()); lu=float(u.candidate_loss_level.mean())
    nm=abs(ns-nu); lm=abs(ls-lu); match=bool(nm<=0.03 and lm<=0.10)
    x=conv.conversion_contrast.to_numpy(float); t,p=one_sided(x); cp=bool(x.mean()>0 and p<0.05); se=float(x.std(ddof=1)/np.sqrt(len(x)))
    if not match: decision="CONFIRMATORY MATCH FAILURE / INCONCLUSIVE"
    elif cp: decision="REUSABILITY-CONVERSION PASS"
    else: decision="NO CONVERSION SEPARATION"
    dec=pd.DataFrame([{
        "n":len(x),"sealed_severity":a.sealed_severity,
        "mean_structured_novelty_gain":ns,"mean_highdim_novelty_gain":nu,"novelty_mismatch":nm,"novelty_tolerance":0.03,
        "mean_structured_candidate_loss":ls,"mean_highdim_candidate_loss":lu,"candidate_loss_mismatch":lm,"candidate_loss_tolerance":0.10,"match_pass":match,
        "mean_structured_benefit":float(conv.structured_benefit.mean()),"mean_highdim_benefit":float(conv.highdim_benefit.mean()),"mean_conversion_contrast":float(x.mean()),"conversion_se":se,"conversion_t":t,"conversion_p_one_sided":p,"conversion_pass":cp,"decision":decision
    }])
    train.to_csv(out/"highdim_confirm_training30.csv",index=False); family_delta.to_csv(out/"highdim_family_deltas30.csv",index=False); conv.to_csv(out/"highdim_conversion30.csv",index=False); dec.to_csv(out/"highdim_decision30.csv",index=False)
    print(dec.to_string(index=False),flush=True)


if __name__=="__main__": main()

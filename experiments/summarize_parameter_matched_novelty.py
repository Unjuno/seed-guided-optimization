"""Frozen Issue #83 Stage-B decision."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

REPS=tuple(range(1800,1830)); ARMS=("maxnov","minnov"); CALIPER=.05; MARGIN=.05
STAGE_A_PROTOCOL_HASH="444ed1dd41a200f03e883b0e86583b92e86f4f182226f4f6fac3f7fed3a5b9d3"
PROTOCOL={"issue":83,"stage":"B","reps":REPS,"arms":ARMS,"K":16,"Q":4,"batch":128,"epochs":10,"lr":5e-3,"wd":1e-3,
          "train_seeds":tuple(range(55000,55064)),"heldout_seeds":tuple(range(56000,56080)),"selected_caliper":CALIPER,
          "hardness_equivalence_margin":MARGIN,"parameter_equivalence_margin":MARGIN,"stage_a_protocol_hash":STAGE_A_PROTOCOL_HASH,
          "base_seed_offset":2210000000,"base_seed_stride":4099,"feasible_ranks":tuple(range(2,11)),"evaluation":"canonical_nontraining_union_809"}
PROTOCOL_HASH=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def est(x,level=.95):
    x=np.asarray(x,np.float64); n=len(x); m=float(x.mean()); se=float(x.std(ddof=1)/np.sqrt(n))
    if se==0: p=0. if m>0 else (.5 if m==0 else 1.); lo=hi=m
    else:
        p=float(stats.t.sf(m/se,n-1)); crit=float(stats.t.ppf(.5+level/2,n-1)); lo=m-crit*se; hi=m+crit*se
    return {"mean":m,"se":se,"ci_low":float(lo),"ci_high":float(hi),"p":p,"positive":int((x>0).sum())}

def tost(x,margin=MARGIN):
    x=np.asarray(x,np.float64); n=len(x); m=float(x.mean()); se=float(x.std(ddof=1)/np.sqrt(n)); crit=float(stats.t.ppf(.95,n-1)); lo=m-crit*se; hi=m+crit*se
    if se==0: pl=0. if m>-margin else 1.; pu=0. if m<margin else 1.
    else: pl=float(stats.t.sf((m+margin)/se,n-1)); pu=float(stats.t.cdf((m-margin)/se,n-1))
    return {"mean":m,"se":se,"ci90_low":float(lo),"ci90_high":float(hi),"p_lower":pl,"p_upper":pu,"pass":bool(lo>-margin and hi<margin and pl<.05 and pu<.05)}

def load(root,pat):
    ps=sorted(root.rglob(pat))
    if not ps: raise FileNotFoundError(pat)
    return pd.concat([pd.read_csv(p) for p in ps],ignore_index=True)

def validate(root):
    aps=sorted(root.rglob("arm_manifest.json"))
    if len(aps)!=6: raise ValueError(f"expected6 manifests got{len(aps)}")
    covered=[]; src0=None; split0=None
    for ap in aps:
        shard=ap.parent; arm=json.loads(ap.read_text()); rp=shard/"reference_manifest.json"; ref=json.loads(rp.read_text())
        if arm["protocol_hash"]!=PROTOCOL_HASH or ref["protocol_hash"]!=PROTOCOL_HASH: raise ValueError("protocol")
        if (arm["start"],arm["end"])!=(ref["start"],ref["end"]): raise ValueError("range")
        covered+=list(range(arm["start"],arm["end"]))
        if arm["source_hashes"]!=ref["source_hashes"] or arm["split_hashes"]!=ref["split_hashes"]: raise ValueError("source/split")
        if src0 is None: src0=ref["source_hashes"]; split0=ref["split_hashes"]
        elif ref["source_hashes"]!=src0 or ref["split_hashes"]!=split0: raise ValueError("cross-shard source/split")
        for n,d in ref["schedule_hashes"].items():
            if sha(shard/"schedules"/n)!=d: raise ValueError("schedule hash")
        for n,d in arm["checkpoint_hashes"].items():
            if sha(shard/"states"/n)!=d: raise ValueError("checkpoint hash")
        for n,d in ref["source_hashes"].items():
            if sha(shard/"source"/n)!=d: raise ValueError(f"source hash {n}")
    if sorted(covered)!=list(REPS): raise ValueError("rep coverage")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input-dir",required=True); ap.add_argument("--output-dir",required=True); a=ap.parse_args(); root=Path(a.input_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    validate(root); steps=load(root,"parameter_matched_reference_steps_*.csv.gz"); tr=load(root,"parameter_matched_arm_training_*.csv"); held=load(root,"parameter_matched_heldout_*.csv")
    if set(steps.rep.astype(int))!=set(REPS): raise ValueError("step reps")
    cnt=steps.groupby("rep").size()
    if cnt.nunique()!=1 or int(cnt.iloc[0])!=80: raise ValueError("step counts")
    cols=["gradient_gap","hardness_diff","parameter_z_diff","parameter_abs_z_diff","raw_parameter_diversity_diff","raw_selected_loss_diff","schedule_overlap"]
    if not np.isfinite(steps[cols].to_numpy(float)).all(): raise ValueError("step nonfinite")
    if (steps.gradient_gap< -1e-12).any(): raise ValueError("gradient ordering")
    if (steps.hardness_diff.abs()>CALIPER+1e-10).any() or (steps.parameter_abs_z_diff>CALIPER+1e-10).any(): raise ValueError("caliper violation")
    exp={(r,m) for r in REPS for m in ARMS}
    if len(tr)!=60 or set(zip(tr.rep.astype(int),tr.arm.astype(str)))!=exp or tr.duplicated(["rep","arm"]).any(): raise ValueError("training grid")
    if len(held)!=60 or set(zip(held.rep.astype(int),held.arm.astype(str)))!=exp or held.duplicated(["rep","arm"]).any(): raise ValueError("heldout grid")
    tdig=tr.set_index(["rep","arm"]).state_digest.sort_index(); hdig=held.set_index(["rep","arm"]).state_digest.sort_index()
    if not tdig.equals(hdig): raise ValueError("state digest mismatch")
    if not np.isfinite(held[["mean_test","sd_test","p10_test","min_test","clean_test"]].to_numpy(float)).all(): raise ValueError("heldout nonfinite")
    ref=steps.groupby("rep",as_index=False).agg(gradient_gap=("gradient_gap","mean"),hardness_diff=("hardness_diff","mean"),parameter_z_diff=("parameter_z_diff","mean"),
        mean_abs_parameter_z_diff=("parameter_abs_z_diff","mean"),raw_parameter_diversity_diff=("raw_parameter_diversity_diff","mean"),raw_selected_loss_diff=("raw_selected_loss_diff","mean"),schedule_overlap=("schedule_overlap","mean"))
    piv=held.pivot(index="rep",columns="arm"); paired=ref.set_index("rep")
    for metric in ("mean_test","clean_test","p10_test","min_test"):
        paired[("heldout_benefit" if metric=="mean_test" else metric.replace("_test","")+"_benefit")]=piv[metric]["maxnov"]-piv[metric]["minnov"]
    paired=paired.reset_index(); ng=est(paired.gradient_gap); h=tost(paired.hardness_diff); p=tost(paired.parameter_z_diff); perf=est(paired.heldout_benefit)
    ngpass=ng["mean"]>0 and ng["p"]<.05; perfpass=perf["mean"]>0 and perf["p"]<.05
    if not ngpass: dec="GRADIENT MANIPULATION FAIL"
    elif not h["pass"]: dec="HARDNESS EQUIVALENCE FAIL / CONFOUNDED"
    elif not p["pass"]: dec="PARAMETER-DIVERSITY EQUIVALENCE FAIL / CONFOUNDED"
    elif perfpass: dec="PARAMETER-MATCHED GRADIENT NONREDUNDANCY SUPPORT"
    else: dec="GRADIENT-SPECIFIC DIRECT TEST FAIL"
    row={"decision":dec,"n":30,"gradient_manipulation_pass":ngpass,"hardness_equivalence_pass":h["pass"],"parameter_equivalence_pass":p["pass"],"performance_pass":perfpass,
         "mean_gradient_gap":ng["mean"],"gradient_gap_se":ng["se"],"gradient_gap_ci95_low":ng["ci_low"],"gradient_gap_ci95_high":ng["ci_high"],"gradient_gap_p_one_sided":ng["p"],
         "mean_hardness_diff":h["mean"],"hardness_se":h["se"],"hardness_ci90_low":h["ci90_low"],"hardness_ci90_high":h["ci90_high"],"hardness_tost_p_lower":h["p_lower"],"hardness_tost_p_upper":h["p_upper"],
         "mean_parameter_z_diff":p["mean"],"parameter_z_se":p["se"],"parameter_z_ci90_low":p["ci90_low"],"parameter_z_ci90_high":p["ci90_high"],"parameter_tost_p_lower":p["p_lower"],"parameter_tost_p_upper":p["p_upper"],
         "mean_heldout_benefit":perf["mean"],"heldout_benefit_se":perf["se"],"heldout_benefit_ci95_low":perf["ci_low"],"heldout_benefit_ci95_high":perf["ci_high"],"heldout_benefit_p_one_sided":perf["p"],"positive_heldout_pairs":perf["positive"],
         "mean_abs_parameter_z_diff":float(paired.mean_abs_parameter_z_diff.mean()),"mean_raw_parameter_diversity_diff":float(paired.raw_parameter_diversity_diff.mean()),"mean_raw_selected_loss_diff":float(paired.raw_selected_loss_diff.mean()),"mean_schedule_overlap":float(paired.schedule_overlap.mean()),
         "mean_clean_benefit_secondary":float(paired.clean_benefit.mean()),"mean_p10_benefit_secondary":float(paired.p10_benefit.mean()),"mean_min_benefit_secondary":float(paired.min_benefit.mean()),"protocol_hash":PROTOCOL_HASH}
    paired.to_csv(out/"parameter_matched_paired30.csv",index=False); ref.to_csv(out/"parameter_matched_reference_summary30.csv",index=False); pd.DataFrame([row]).to_csv(out/"parameter_matched_decision30.csv",index=False); print(pd.DataFrame([row]).to_string(index=False),flush=True)
if __name__=="__main__": main()

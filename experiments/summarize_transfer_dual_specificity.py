from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

EXPECTED_REPS=tuple(range(1100,1130)); METHODS=('loss_hard','gradnov'); FAMILIES=('shared','nuisance'); METRICS=('mean_test','sd_test','p10_test','min_test','clean_test')


def load_parts(root:Path,prefix:str)->pd.DataFrame:
    files=sorted(root.rglob(prefix+'*.csv'))
    if not files: raise FileNotFoundError(prefix)
    return pd.concat([pd.read_csv(p) for p in files],ignore_index=True)


def validate_diag(df:pd.DataFrame)->None:
    exp={(r,m) for r in EXPECTED_REPS for m in METHODS}; obs=set(zip(df.rep.astype(int),df.method.astype(str)))
    if len(df)!=len(exp) or obs!=exp: raise ValueError('diagnostic grid mismatch')
    if df.duplicated(['rep','method']).any(): raise ValueError('diagnostic duplicates')


def validate_held(df:pd.DataFrame)->None:
    exp={(r,m,f) for r in EXPECTED_REPS for m in METHODS for f in FAMILIES}; obs=set(zip(df.rep.astype(int),df.method.astype(str),df.family.astype(str)))
    if len(df)!=len(exp) or obs!=exp: raise ValueError('heldout grid mismatch')
    if df.duplicated(['rep','method','family']).any(): raise ValueError('heldout duplicates')


def one_sided(x):
    x=np.asarray(x,np.float64)
    if np.all(x==x[0]):
        if x[0]>0:return float('inf'),0.0
        if x[0]<0:return float('-inf'),1.0
        return 0.0,0.5
    r=stats.ttest_1samp(x,0.0); t=float(r.statistic); p2=float(r.pvalue); return t,p2/2 if t>0 else 1-p2/2


def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--sealed-lambda',type=float,required=True); ap.add_argument('--sealed-alpha',type=float,required=True); a=ap.parse_args()
    root=Path(a.input_dir); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    diag=load_parts(root,'transfer_diagnostics_'); held=load_parts(root,'dual_transfer_heldout_'); validate_diag(diag); validate_held(held)
    sv=held[held.family=='shared'].lambda_strength.dropna().to_numpy(float); nv=held[held.family=='nuisance'].alpha.dropna().to_numpy(float)
    if len(sv)==0 or not np.all(np.isclose(sv,a.sealed_lambda)): raise ValueError('sealed lambda mismatch')
    if len(nv)==0 or not np.all(np.isclose(nv,a.sealed_alpha)): raise ValueError('sealed alpha mismatch')
    loss=held[held.method=='loss_hard']; shared_loss=float(loss[loss.family=='shared'].mean_test.mean()); nuisance_loss=float(loss[loss.family=='nuisance'].mean_test.mean()); difficulty_mismatch=abs(shared_loss-nuisance_loss); difficulty_pass=bool(difficulty_mismatch<=0.02)
    deltas=[]; contrasts=[]
    for rep in EXPECTED_REPS:
        part=held[held.rep==rep]; benefits={}
        for fam in FAMILIES:
            p=part[part.family==fam].set_index('method'); row={'rep':rep,'family':fam}
            for metric in METRICS: row['delta_'+metric]=float(p.loc['gradnov',metric]-p.loc['loss_hard',metric])
            deltas.append(row); benefits[fam]=row['delta_mean_test']
        contrasts.append({'rep':rep,'shared_benefit':benefits['shared'],'nuisance_benefit':benefits['nuisance'],'specificity_contrast':benefits['shared']-benefits['nuisance']})
    deltas=pd.DataFrame(deltas); contrasts=pd.DataFrame(contrasts); bs=contrasts.shared_benefit.to_numpy(float); sp=contrasts.specificity_contrast.to_numpy(float)
    ts,ps=one_sided(bs); tt,pt=one_sided(sp); shared_pass=bool(bs.mean()>0 and ps<0.05); specificity_pass=bool(sp.mean()>0 and pt<0.05)
    if not difficulty_pass: decision='CONFIRMATORY DIFFICULTY MATCH FAILURE / INCONCLUSIVE'
    elif not shared_pass: decision='NO SHARED REPLICATION'
    elif specificity_pass: decision='TRANSFER-SPECIFICITY PASS'
    else: decision='SHARED EFFECT ONLY / NO SPECIFICITY'
    dec=pd.DataFrame([{'n':len(EXPECTED_REPS),'sealed_lambda':a.sealed_lambda,'sealed_alpha':a.sealed_alpha,'loss_hard_shared_mean_accuracy':shared_loss,'loss_hard_nuisance_mean_accuracy':nuisance_loss,'confirmatory_difficulty_mismatch':difficulty_mismatch,'confirmatory_difficulty_tolerance':0.02,'confirmatory_difficulty_pass':difficulty_pass,'mean_shared_benefit':float(bs.mean()),'shared_benefit_se':float(bs.std(ddof=1)/np.sqrt(len(bs))),'shared_t':ts,'shared_p_one_sided':ps,'shared_effect_pass':shared_pass,'mean_nuisance_benefit':float(contrasts.nuisance_benefit.mean()),'mean_specificity_contrast':float(sp.mean()),'specificity_se':float(sp.std(ddof=1)/np.sqrt(len(sp))),'specificity_t':tt,'specificity_p_one_sided':pt,'specificity_pass':specificity_pass,'decision':decision}])
    diag.to_csv(out/'dual_transfer_training_diagnostics30.csv',index=False); deltas.to_csv(out/'dual_transfer_family_deltas30.csv',index=False); contrasts.to_csv(out/'dual_transfer_specificity30.csv',index=False); dec.to_csv(out/'dual_transfer_decision30.csv',index=False); print(dec.to_string(index=False),flush=True)

if __name__=='__main__': main()

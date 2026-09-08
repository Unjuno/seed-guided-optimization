"""Read-only independent audit of Issue91's consolidated Actions evidence.
Usage: python check_overlap_translation_evidence.py --evidence-dir EXTRACTED --output-dir AUDIT
No training, selection tuning, dataset download or raw-image inference is performed.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from scipy import stats

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return pd.read_csv(p,float_precision='round_trip')
def digest(state):
 h=hashlib.sha256()
 for n,t in sorted(state.items()):
  a=t.detach().cpu().contiguous().numpy()
  for b in (n.encode(),str(a.dtype).encode(),str(a.shape).encode(),a.tobytes()):h.update(b)
 return h.hexdigest()
def est(x):
 x=np.asarray(x,float);n=len(x);assert n==30 and np.isfinite(x).all()
 se=float(np.std(x,ddof=1)/np.sqrt(n));mean=float(np.mean(x));k=stats.t.ppf(.975,n-1)
 p=float(stats.ttest_1samp(x,0,alternative='greater').pvalue) if se else (0. if mean>0 else .5 if mean==0 else 1.)
 return dict(mean=mean,se=se,ci95_low=mean-k*se,ci95_high=mean+k*se,p_one_sided=p,positive_pairs=int((x>0).sum()))
def tost(x,margin):
 x=np.asarray(x,float);s=est(x);se=s['se'];mean=s['mean'];k=stats.t.ppf(.95,29)
 pl=stats.t.sf((mean+margin)/se,29) if se else float(mean<=-margin)
 pu=stats.t.cdf((mean-margin)/se,29) if se else float(mean>=margin)
 return bool(pl<.05 and pu<.05),dict(mean=mean,se=se,ci90_low=mean-k*se,ci90_high=mean+k*se,p_lower=float(pl),p_upper=float(pu),margin=margin)
def run(root,out):
 all_steps=[];all_held=[];all_env=[];checks=[];checked_states=0;checked_pairs=0;max_diag=0.;max_env=0.;coverage=[];source_sets=[];split_sets=[]
 for rp in sorted((root/'downloaded').rglob('reference_manifest.json')):
  d=rp.parent;r=json.loads(rp.read_text());a=json.loads((d/'arm_manifest.json').read_text());source_sets.append(r['source_hashes']);split_sets.append(r['split_hashes'])
  assert a['reference_manifest_sha256']==sha(rp) and a['protocol_hash']==r['protocol_hash']
  for m in (r,a):
   for n,h in m['files'].items():assert sha(d/n)==h
   for n,h in m['source_hashes'].items():assert sha(d/'source'/n)==h
  steps=load(next(d.glob('translation_reference_*.csv.gz')));sub=load(next(d.glob('translation_subsets_*.csv.gz')));held=load(next(d.glob('spatial_heldout_*.csv')));env=load(next(d.glob('spatial_environment_*.csv.gz')))
  assert not steps.duplicated(['rep','step','family']).any() and not sub.duplicated(['rep','step','subset_id']).any()
  for (rep,step),sv in sub.groupby(['rep','step'],sort=False):
   sv=sv.sort_values('subset_id');assert len(sv)==84 and set(sv.subset_id)==set(range(84))
   ar=sv[['gradient_novelty','z_hard','z_param','translation_score']].to_numpy();assert np.isfinite(ar).all()
   i,j=np.triu_indices(84,1);dif=ar[i]-ar[j];hp=(abs(dif[:,1])<=.05+1e-12)&(abs(dif[:,2])<=.05+1e-12)
   rows=steps[(steps.rep==rep)&(steps.step==step)].set_index('family');assert set(rows.index)=={'u','m'}
   pl=json.loads((d/'schedules'/f'rep{rep}.json').read_text())
   locals_=[tuple(map(int,str(x).split(';'))) for x in sv.local];envs=[tuple(map(int,str(x).split(';'))) for x in sv.envs]
   for f in ('u','m'):
    valid=np.flatnonzero(hp&((abs(dif[:,3])<=.2+1e-12) if f=='m' else True));v=rows.loc[f]
    if not len(valid):
     assert f=='m' and bool(v.fallback) and v.high_envs==v.low_envs
     hi=lo=envs.index(tuple(map(int,v.high_envs.split(';'))));assert ar[hi,1]>=ar[:,1].max()-1e-6
    else:
     k=min(valid,key=lambda z:(-abs(dif[z,0]),abs(dif[z,2]),abs(dif[z,1]),locals_[i[z]],locals_[j[z]]))
     hi,lo=(int(i[k]),int(j[k])) if ar[i[k],0]>=ar[j[k],0] else (int(j[k]),int(i[k]))
     assert not bool(v.fallback)
    assert tuple(pl[f+'_high'][step])==envs[hi] and tuple(pl[f+'_low'][step])==envs[lo]
    observed=v[['gradient_gap','hardness_diff','parameter_z_diff','translation_gap']].to_numpy(dtype=float)
    max_diag=max(max_diag,float(abs(observed-(ar[hi]-ar[lo])).max()));assert max_diag<1e-12
    checked_pairs+=1
  hz=held.set_index(['rep','arm'])
  for p in (d/'states').glob('*.pt'):
   ck=torch.load(p,map_location='cpu',weights_only=True);assert ck['protocol_hash']==r['protocol_hash'];assert ck['schedule_sha256']==sha(d/'schedules'/f"rep{ck['rep']}.json")
   assert digest(ck['state_dict'])==hz.loc[(ck['rep'],ck['arm']),'state_digest'];checked_states+=1
  for key,b in env.groupby(['rep','arm']):
   assert len(b)==80 and set(b.env_seed)==set(range(64000,64080));x=b.sort_values('env_seed').accuracy.to_numpy()
   vals=np.array([x.mean(),x.std(ddof=1),np.quantile(x,.1),x.min()]);want=hz.loc[key,['mean_test','sd_test','p10_test','min_test']].to_numpy(dtype=float)
   max_env=max(max_env,float(abs(vals-want).max()));assert max_env<1e-12
  all_steps.append(steps);all_held.append(held);all_env.append(env);coverage.extend(range(r['start'],r['end']))
  checks.append({'start':r['start'],'reference_runtime':r['runtime'],'arm_runtime':a['runtime'],'evaluation_runtime':json.loads((d/'evaluation_runtime.json').read_text())})
 assert sorted(coverage)==list(range(2200,2230)) and checked_states==150 and checked_pairs==4800
 assert all(s==source_sets[0] for s in source_sets) and all(s==split_sets[0] for s in split_sets)
 steps=pd.concat(all_steps);held=pd.concat(all_held);en=pd.concat(all_env);assert len(steps)==4800 and len(held)==150 and len(en)==12000
 z=held.pivot(index='rep',columns='arm',values='mean_test');ref=steps.groupby(['rep','family']).mean(numeric_only=True);paired=pd.DataFrame({'rep':sorted(z.index)})
 summary=[];balance=[];ok=True
 for f in ('u','m'):
  paired[f+'_benefit']=(z[f+'_high']-z[f+'_low']).to_numpy();summary.append({'endpoint':f+'_benefit',**est(paired[f+'_benefit'])})
  rr=ref.xs(f,level='family').sort_index()
  for col in ['gradient_gap','hardness_diff','parameter_z_diff','translation_gap']:paired[f+'_'+col]=rr[col].to_numpy()
  ok &= rr.gradient_gap.mean()>=.05 and est(rr.gradient_gap)['p_one_sided']<.05
  for col in ['hardness_diff','parameter_z_diff']:
   passed,b=tost(rr[col],.05);ok &= passed;balance.append({'endpoint':f+'_'+col,'pass':passed,**b})
  for arm in [f+'_high',f+'_low']:summary.append({'endpoint':arm+'_vs_reference',**est(z[arm]-z.reference)})
 passed,b=tost(paired.m_translation_gap,.2);ok &= passed;balance.append({'endpoint':'m_translation_gap','pass':passed,**b})
 mp=est(paired.m_benefit);up=est(paired.u_benefit);m=mp['mean']>0 and mp['p_one_sided']<.05;u=up['mean']>0 and up['p_one_sided']<.05
 support=1-steps[steps.family=='m'].groupby('rep').fallback.mean();assert support.mean()>=.90 and support.min()>=.80
 expected='MATCH OR MANIPULATION FAILURE' if not ok else 'OVERLAP-RESTRICTED TRANSLATION-BALANCED NOVELTY SUPPORT' if m and u else 'UNMATCHED REPLICATION ONLY / MATCHED EFFECT NOT SUPPORTED' if u else 'MATCHED EFFECT ONLY / UNMATCHED CONTROL NOT REPLICATED' if m else 'NO PERFORMANCE REPLICATION'
 published=load(root/'aggregate/translation_matched_decision30.csv').iloc[0];assert published.decision==expected
 p=load(root/'aggregate/translation_matched_paired30.csv').set_index('rep')
 for f in ['u','m']:np.testing.assert_allclose(p[f+'_mean_test_benefit'],paired[f+'_benefit'],atol=1e-14,rtol=0)
 out.mkdir(parents=True,exist_ok=True);pd.DataFrame(summary).to_csv(out/'independent_summary.csv',index=False);pd.DataFrame(balance).to_csv(out/'independent_balance.csv',index=False)
 pd.DataFrame(paired).to_csv(out/'independent_paired.csv',index=False)
 metadata={'decision':expected,'states_checked':checked_states,'subset_pairs_reconstructed':checked_pairs,'reference_subset_rows':201600,'environment_rows':12000,
           'max_diagnostic_reconstruction_error':max_diag,'max_environment_reaggregation_error':max_env,'pooled_support':float(support.mean()),'min_replicate_support':float(support.min()),'runtime':checks}
 (out/'independent_audit.json').write_text(json.dumps(metadata,indent=2))
 print(pd.DataFrame(summary).to_string(index=False));print(json.dumps({k:v for k,v in metadata.items() if k!='runtime'}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--evidence-dir',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();run(Path(a.evidence_dir),Path(a.output_dir))

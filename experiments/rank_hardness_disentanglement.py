"""Issues112/113 Stage A: disentangle candidate rank from selected hardness.

Two independent fresh training-only calibrations:
- mode=rank: maximize mean rank gap while matching selected hardness H;
- mode=hardness: maximize hardness gap while matching mean rank R.
No intervention arms or heldout environments are constructed here.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, torch
import corrected_centered_span_calibration as cc
import rank_allocation_calibration as rc
import parameter_matched_novelty_confirmatory as pm
import translation_matched_calibration as tc
import cnn_regime_interaction as cr
import fixed_dose_response as fd
import transfer_specificity as base
from common import SmallCNN, head_gradient_directions, seed_everything

K=16; Q=4; STEPS=80; SLACK=1e-12; MAX_ID=1e-10
L=.05; P=.05; T=.20; G=.05; S=.10; O=.15
CONFIG={
 'rank': {
   'issue':112,'reps':tuple(range(3700,3710)),'seeds':tuple(range(93000,93064)),'offset':4110000000,
   'grid':(.02,.05,.10,.20),'grid_name':'hardness_caliper','min_pool':.90,'min_rep':.80,'min_gap':1.00,'min_rep_gap':.50,
   'pass_label':'HARDNESS-MATCHED RANK CALIBRATION PASS','fail_label':'HARDNESS-MATCHED RANK CALIBRATION FAIL'},
 'hardness': {
   'issue':113,'reps':tuple(range(3900,3910)),'seeds':tuple(range(97000,97064)),'offset':4310000000,
   'grid':(0.0,.25,.50,.75),'grid_name':'rank_caliper','min_pool':.90,'min_rep':.80,'min_gap':.20,'min_rep_gap':.10,
   'pass_label':'RANK-MATCHED HARDNESS CALIBRATION PASS','fail_label':'RANK-MATCHED HARDNESS CALIBRATION FAIL'},
}
SOURCES=(
 'common.py','centered_span_calibration.py','corrected_centered_span_calibration.py','rank_allocation_calibration.py',
 'parameter_matched_novelty_calibration.py','parameter_matched_novelty_confirmatory.py','translation_matched_calibration.py',
 'cnn_regime_interaction.py','fixed_dose_response.py','transfer_specificity.py','rank_hardness_disentanglement.py')

def protocol(mode):
 c=CONFIG[mode]
 p={'issue':c['issue'],'stage':'A','mode':mode,'reps':c['reps'],'train_seeds':c['seeds'],'offset':c['offset'],'stride':4099,
    'K':K,'Q':Q,'steps':STEPS,'epochs':10,'batch':128,'lr':.005,'wd':.001,
    'subset_family':'loss-rank1 plus any3 ranks2-16','subset_count':455,
    'grid':c['grid'],'grid_name':c['grid_name'],'loss_variance_z_caliper':L,'p_caliper':P,'t_caliper':T,
    'g_caliper':G,'span_caliper':S,'opposition_caliper':O,'min_pooled_support':c['min_pool'],
    'min_rep_support':c['min_rep'],'minimum_target_gap':c['min_gap'],'minimum_rep_target_gap':c['min_rep_gap'],
    'max_general_identity_error':MAX_ID,'threads':1}
 return p

def ph(mode): return hashlib.sha256(json.dumps(protocol(mode),sort_keys=True).encode()).hexdigest()
def source_hashes():
 root=Path(__file__).parent; return {n:fd.sha256(root/n) for n in SOURCES}

def common_keep(d):
 return ((np.abs(d[:,2])<=L+SLACK)&(np.abs(d[:,3])<=P+SLACK)&(np.abs(d[:,4])<=T+SLACK)&
         (np.abs(d[:,5])<=G+SLACK)&(np.abs(d[:,6])<=S+SLACK)&(np.abs(d[:,7])<=O+SLACK))

def choose(rows,mode,grid_value):
 # columns: R,H,L,P,T,G,S,O
 a=np.asarray([[r[f] for f in ('mean_rank','z_hard','z_loss_variance','z_param','translation_score','gradient_novelty','centered_erank','opposition_score')] for r in rows],np.float64)
 ii,jj=np.triu_indices(len(rows),1); d=a[ii]-a[jj]; keep=common_keep(d)
 if mode=='rank':
   keep &= np.abs(d[:,1])<=grid_value+SLACK; target=np.abs(d[:,0])
   loc=np.flatnonzero(keep)
   if not len(loc): return None
   z=min(loc,key=lambda k:(-target[k],abs(d[k,1]),abs(d[k,7]),abs(d[k,6]),abs(d[k,5]),abs(d[k,4]),abs(d[k,3]),abs(d[k,2]),tuple(rows[ii[k]]['local']),tuple(rows[jj[k]]['local'])))
   i,j=int(ii[z]),int(jj[z]); hi,lo=(i,j) if a[i,0]>=a[j,0] else (j,i)
   target_gap=float(a[hi,0]-a[lo,0])
   orientation='high_rank_minus_low_rank'
 else:
   keep &= np.abs(d[:,0])<=grid_value+SLACK; target=np.abs(d[:,1])
   loc=np.flatnonzero(keep)
   if not len(loc): return None
   z=min(loc,key=lambda k:(-target[k],abs(d[k,0]),abs(d[k,7]),abs(d[k,6]),abs(d[k,5]),abs(d[k,4]),abs(d[k,3]),abs(d[k,2]),tuple(rows[ii[k]]['local']),tuple(rows[jj[k]]['local'])))
   i,j=int(ii[z]),int(jj[z]); low,high=(i,j) if a[i,1]<=a[j,1] else (j,i)
   hi,lo=low,high  # arm orientation: preferred low-H first, comparator high-H second
   target_gap=float(a[high,1]-a[low,1])
   orientation='low_hardness_minus_high_hardness'
 return {
   'preferred':hi,'comparison':lo,'eligible_pairs':int(len(loc)),'target_gap':target_gap,'orientation':orientation,
   'rank_diff':float(a[hi,0]-a[lo,0]),'hardness_diff':float(a[hi,1]-a[lo,1]),
   'loss_variance_z_diff':float(a[hi,2]-a[lo,2]),'parameter_z_diff':float(a[hi,3]-a[lo,3]),
   'translation_diff':float(a[hi,4]-a[lo,4]),'gradient_novelty_diff':float(a[hi,5]-a[lo,5]),
   'span_diff':float(a[hi,6]-a[lo,6]),'opposition_diff':float(a[hi,7]-a[lo,7]),
   'preferred_local':';'.join(map(str,rows[hi]['local'])),'comparison_local':';'.join(map(str,rows[lo]['local'])),
   'preferred_ranks':';'.join(map(str,rows[hi]['ranks'])),'comparison_ranks':';'.join(map(str,rows[lo]['ranks']))}

def check_range(mode,start,end):
 reps=CONFIG[mode]['reps']
 if start is None or end is None or start>=end or not set(range(start,end))<=set(reps): raise ValueError('unregistered range')

def run(mode,start,end,out):
 c=CONFIG[mode]; check_range(mode,start,end);base.configure_determinism(1);out.mkdir(parents=True,exist_ok=True)
 x,y,*_=cr.data_split();ty=torch.tensor(y);envs=base.geometric_envs(x,c['seeds']);params=tc.parameters(c['seeds']);pair_rows=[];subset_rows=[];max_id=0.
 for rep in range(start,end):
   seed=c['offset']+4099*rep;sched=pm.build_schedule(len(ty),seed);seed_everything(seed);model=SmallCNN();model.train();opt=torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=.001)
   if len(sched)!=STEPS: raise ValueError('steps')
   for step,(b,cand) in enumerate(sched):
     logits,h=model(torch.cat([envs[e][b] for e in cand]));per=torch.nn.functional.cross_entropy(logits,ty[b].repeat(K),reduction='none').reshape(K,-1);losses=per.mean(1);gd=head_gradient_directions(logits,h,ty[b],K)
     rows=rc.enrich(cc.expanded_subsets(losses,gd,cand,params),gd);step_id=max(abs(float(r['general_identity_error'])) for r in rows);max_id=max(max_id,step_id)
     for grid_value in c['grid']:
       q=choose(rows,mode,grid_value);pair_rows.append({'rep':rep,'step':step,c['grid_name']:grid_value,'feasible':q is not None,'step_max_general_identity_error':step_id,**(q or {})})
     for n,r in enumerate(rows): subset_rows.append({'rep':rep,'step':step,'subset_id':n,'local':';'.join(map(str,r['local'])),'ranks':';'.join(map(str,r['ranks'])),'envs':';'.join(map(str,r['envs'])),**{f:r[f] for f in ('mean_rank','z_hard','z_loss_variance','z_param','translation_score','gradient_novelty','centered_erank','opposition_score','general_identity_error')}})
     sel=sorted(sorted(range(K),key=lambda j:(-float(losses[j].detach()),j))[:Q]);opt.zero_grad(set_to_none=True);per[sel].mean().backward();opt.step()
   print(f'RANK_HARDNESS_REFERENCE_COMPLETE mode={mode} rep={rep}; no arms, no heldout',flush=True)
 p=out/f'{mode}_pairs_{start}_{end}.csv';s=out/f'{mode}_subsets_{start}_{end}.csv.gz';pd.DataFrame(pair_rows).to_csv(p,index=False);pd.DataFrame(subset_rows).to_csv(s,index=False,compression='gzip')
 (out/'manifest.json').write_text(json.dumps({'protocol':protocol(mode),'protocol_hash':ph(mode),'mode':mode,'start':start,'end':end,'source_hashes':source_hashes(),'split_hashes':cr.split_hashes(),'files':{q.name:fd.sha256(q) for q in (p,s)},'runtime':fd.runtime(),'maximum_general_identity_error':max_id,'heldout_constructed':False,'intervention_arms_trained':0},indent=2,allow_nan=False))

def summarize(mode,root,out):
 c=CONFIG[mode];frames=[];covered=[];max_id=0.
 for mp in sorted(root.rglob('manifest.json')):
   m=json.loads(mp.read_text())
   if m['mode']!=mode or m['protocol_hash']!=ph(mode) or m['source_hashes']!=source_hashes() or m['split_hashes']!=cr.split_hashes(): raise ValueError('provenance')
   if m['heldout_constructed'] or m['intervention_arms_trained']!=0: raise ValueError('boundary')
   for n,d in m['files'].items():
     if fd.sha256(mp.parent/n)!=d: raise ValueError('hash')
   covered.extend(range(m['start'],m['end']));max_id=max(max_id,abs(float(m['maximum_general_identity_error'])));frames.append(pd.read_csv(mp.parent/f"{mode}_pairs_{m['start']}_{m['end']}.csv",float_precision='round_trip'))
 if sorted(covered)!=list(c['reps']): raise ValueError('coverage')
 df=pd.concat(frames,ignore_index=True); key=c['grid_name'];exp={(r,t,g) for r in c['reps'] for t in range(STEPS) for g in c['grid']}
 if len(df)!=len(exp) or df.duplicated(['rep','step',key]).any() or set(df[['rep','step',key]].itertuples(index=False,name=None))!=exp: raise ValueError('grid')
 results=[];selected=None;identity=max_id<=MAX_ID
 for gval in c['grid']:
   v=df[np.isclose(df[key],gval)];f=v.feasible.astype(bool);q=v[f]
   if len(q):
     if (q.loss_variance_z_diff.abs()>L+SLACK).any() or (q.parameter_z_diff.abs()>P+SLACK).any() or (q.translation_diff.abs()>T+SLACK).any() or (q.gradient_novelty_diff.abs()>G+SLACK).any() or (q.span_diff.abs()>S+SLACK).any() or (q.opposition_diff.abs()>O+SLACK).any(): raise ValueError('common caliper')
     if mode=='rank' and (q.hardness_diff.abs()>gval+SLACK).any(): raise ValueError('H caliper')
     if mode=='hardness' and (q.rank_diff.abs()>gval+SLACK).any(): raise ValueError('R caliper')
   sr=v.assign(ok=f).groupby('rep').ok.mean().reindex(c['reps'],fill_value=0.);tg=q.groupby('rep').target_gap.mean().reindex(c['reps'],fill_value=0.)
   pooled=float(f.mean());mins=float(sr.min());mg=float(q.target_gap.mean()) if len(q) else 0.;mrg=float(tg.min());passed=identity and pooled>=c['min_pool'] and mins>=c['min_rep'] and mg>=c['min_gap'] and mrg>=c['min_rep_gap']
   results.append({key:gval,'supported_steps':int(f.sum()),'total_steps':len(v),'pooled_support':pooled,'minimum_replicate_support':mins,'mean_target_gap_supported':mg,'minimum_replicate_mean_target_gap_supported':mrg,
      'mean_abs_rank_diff_supported':float(q.rank_diff.abs().mean()) if len(q) else None,'mean_abs_hardness_diff_supported':float(q.hardness_diff.abs().mean()) if len(q) else None,
      'mean_abs_loss_variance_z_diff_supported':float(q.loss_variance_z_diff.abs().mean()) if len(q) else None,'mean_abs_parameter_z_diff_supported':float(q.parameter_z_diff.abs().mean()) if len(q) else None,
      'mean_abs_translation_diff_supported':float(q.translation_diff.abs().mean()) if len(q) else None,'mean_abs_gradient_novelty_diff_supported':float(q.gradient_novelty_diff.abs().mean()) if len(q) else None,
      'mean_abs_span_diff_supported':float(q.span_diff.abs().mean()) if len(q) else None,'mean_abs_opposition_diff_supported':float(q.opposition_diff.abs().mean()) if len(q) else None,
      'maximum_general_identity_error':max_id,'identity_pass':identity,'pass':bool(passed)})
   if passed and selected is None: selected=gval
 decision={'decision':c['pass_label'] if selected is not None else c['fail_label'],'selected_caliper':selected,'mode':mode,'protocol_hash':ph(mode),'maximum_general_identity_error':max_id,'heldout_constructed':False,'intervention_arms_trained':0}
 out.mkdir(parents=True,exist_ok=True);pd.DataFrame(results).to_csv(out/f'{mode}_calibration_summary.csv',index=False);(out/f'{mode}_calibration_decision.json').write_text(json.dumps(decision,indent=2,allow_nan=False));print(json.dumps(decision),flush=True);print(pd.DataFrame(results).to_string(index=False),flush=True)

def selftest():
 rows=[{'local':(0,1,2,3),'ranks':(1,2,3,4),'mean_rank':2.5,'z_hard':.2,'z_loss_variance':0.,'z_param':0.,'translation_score':0.,'gradient_novelty':.5,'centered_erank':2.,'opposition_score':.8},
       {'local':(0,4,5,6),'ranks':(1,5,6,7),'mean_rank':4.75,'z_hard':.21,'z_loss_variance':.01,'z_param':.01,'translation_score':.01,'gradient_novelty':.51,'centered_erank':2.02,'opposition_score':.82},
       {'local':(0,7,8,9),'ranks':(1,8,9,10),'mean_rank':7.,'z_hard':-.1,'z_loss_variance':.01,'z_param':.01,'translation_score':.01,'gradient_novelty':.51,'centered_erank':2.02,'opposition_score':.82}]
 q=choose(rows,'rank',.02);assert q and q['target_gap']>2.; h=choose(rows,'hardness',.0);assert h and h['target_gap']>=0.; print('RANK_HARDNESS_SELFTEST_PASS',ph('rank'),ph('hardness'))

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('selftest','run','summarize'));p.add_argument('--experiment',choices=('rank','hardness'));p.add_argument('--start',type=int);p.add_argument('--end',type=int);p.add_argument('--output-dir');p.add_argument('--input-dir');a=p.parse_args()
 if a.mode=='selftest': selftest(); return
 if a.experiment is None: raise ValueError('experiment required')
 if a.mode=='run': run(a.experiment,a.start,a.end,Path(a.output_dir))
 else: summarize(a.experiment,Path(a.input_dir),Path(a.output_dir))
if __name__=='__main__':main()

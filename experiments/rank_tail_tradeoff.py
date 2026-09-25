"""Issue115: fresh n=60 mean-versus-tail replication of matched rank relaxation.

Reuses the frozen Issue104 rank-relaxation pair rule. Global barriers remain:
all reference schedules -> all intervention/reference states -> heldout construction.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, torch

import rank_allocation_confirmation as base
import centered_span_confirmation as core
import parameter_matched_novelty_confirmatory as pm

REPS=tuple(range(4100,4160)); TRAIN_SEEDS=tuple(range(101000,101064)); HELDOUT_SEEDS=tuple(range(102000,102080)); OFFSET=4510000000
PROTOCOL={
    'issue':115,'stage':'fresh_mean_tail_replication','reps':REPS,'arms':('high','low'),'baseline':'reference_loss_hard',
    'train_seeds':TRAIN_SEEDS,'heldout_seeds':HELDOUT_SEEDS,'base_seed_offset':OFFSET,'base_seed_stride':4099,
    'K':16,'Q':4,'steps':80,'epochs':10,'batch':128,'lr':.005,'wd':.001,
    'subset_family':'loss-rank1 plus any3 ranks2-16','subset_count':455,
    'loss_variance_equivalence_margin':.05,'physical_diversity_equivalence_margin':.05,
    'translation_equivalence_margin':.20,'gradient_novelty_equivalence_margin':.05,
    'centered_span_equivalence_margin':.10,'opposition_equivalence_margin':.15,
    'minimum_rank_manipulation':1.50,'minimum_pooled_support':.90,'minimum_replicate_support':.80,
    'max_general_identity_error':1e-10,'fallback':'identical_reference_loss_hard_top4',
    'pair_rule':'Issue104 unchanged','stage_a_protocol_hash':'eae9daf7edfc5ba979725fc67dc4ac0207aeca8e6a80cb69ddf66776d739886b',
    'primary_endpoints':('heldout_mean_high_minus_low','heldout_min_high_minus_low'),'n':60,'threads':1,
}
PH=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()
SOURCES=(
    'common.py','centered_span_calibration.py','corrected_centered_span_calibration.py','centered_span_confirmation.py',
    'rank_allocation_calibration.py','rank_allocation_confirmation.py','parameter_matched_novelty_calibration.py',
    'parameter_matched_novelty_confirmatory.py','translation_matched_calibration.py','cnn_regime_interaction.py',
    'fixed_dose_response.py','transfer_specificity.py','rank_tail_tradeoff.py',
)

def configure():
    base.REPS=REPS; base.TRAIN_SEEDS=TRAIN_SEEDS; base.HELDOUT_SEEDS=HELDOUT_SEEDS; base.OFFSET=OFFSET
    base.PROTOCOL=PROTOCOL; base.PH=PH; base.SOURCES=SOURCES
    base.STAGE_A_PROTOCOL_HASH=PROTOCOL['stage_a_protocol_hash']
    base.configure()

def check_range(start,end):
    if start is None or end is None or start>=end or not set(range(start,end))<=set(REPS): raise ValueError('unregistered Issue115 range')

def estimate_negative(x):
    e=core.estimate(-np.asarray(x,dtype=np.float64));
    return {'n':e['n'],'mean_original':float(np.mean(np.asarray(x,dtype=np.float64))),'se':e['se'],'ci95_low_original':-e['ci95_high'],'ci95_high_original':-e['ci95_low'],'p_one_sided_negative':e['p_one_sided'],'negative_pairs':int((np.asarray(x,dtype=np.float64)<0).sum())}

def summarize(root,out,global_seal):
    configure(); core.validate_global_seal(global_seal,'arms'); covered=[];state_count=0;state_digests={}
    for mp in sorted(root.rglob('evaluation_manifest.json')):
        m=json.loads(mp.read_text());start,end=int(m['start']),int(m['end']);check_range(start,end);core.validate_evaluation_shard(mp.parent,start,end);covered.extend(range(start,end))
        for rep in range(start,end):
            for arm in core.ALL_ARMS:
                p=mp.parent/'states'/f'rep{rep}_{arm}.pt';ck=torch.load(p,map_location='cpu',weights_only=True)
                if (ck['rep'],ck['arm'],ck['protocol_hash'])!=(rep,arm,PH): raise ValueError('summary state metadata')
                state_digests[(rep,arm)]=pm.state_digest(ck['state_dict']);state_count+=1
    expected_states=len(REPS)*len(core.ALL_ARMS)
    if sorted(covered)!=list(REPS) or state_count!=expected_states: raise ValueError('global evaluation coverage/state count')
    steps=core.load_frames(root,'rank_allocation_reference_*.csv.gz');held=core.load_frames(root,'centered_span_heldout_*.csv');env=core.load_frames(root,'centered_span_environment_*.csv.gz')
    exp_steps={(r,t) for r in REPS for t in range(base.STEPS)};exp_held={(r,a) for r in REPS for a in core.ALL_ARMS};exp_env={(r,a,s) for r in REPS for a in core.ALL_ARMS for s in HELDOUT_SEEDS}
    if len(steps)!=len(exp_steps) or steps.duplicated(['rep','step']).any() or set(steps[['rep','step']].itertuples(index=False,name=None))!=exp_steps: raise ValueError('step grid')
    if len(held)!=len(exp_held) or held.duplicated(['rep','arm']).any() or set(held[['rep','arm']].itertuples(index=False,name=None))!=exp_held: raise ValueError('held grid')
    if len(env)!=len(exp_env) or env.duplicated(['rep','arm','env_seed']).any() or set(env[['rep','arm','env_seed']].itertuples(index=False,name=None))!=exp_env: raise ValueError('environment grid')
    numeric=['rank_gap','hardness_diff','loss_variance_z_diff','parameter_z_diff','translation_diff','gradient_novelty_diff','span_diff','opposition_diff','step_max_general_identity_error']
    if not np.isfinite(steps[numeric].to_numpy(float)).all() or not np.isfinite(held[['mean_test','sd_test','p10_test','min_test','clean_test']].to_numpy(float)).all() or not np.isfinite(env.accuracy.to_numpy(float)).all(): raise ValueError('nonfinite')
    if float(steps.step_max_general_identity_error.abs().max())>base.MAX_ID: raise ValueError('identity')
    supported=steps.supported.astype(bool);live=steps[supported]
    if len(live) and ((live.loss_variance_z_diff.abs()>base.L_MARGIN+1e-12).any() or (live.parameter_z_diff.abs()>base.P_MARGIN+1e-12).any() or (live.translation_diff.abs()>base.T_MARGIN+1e-12).any() or (live.gradient_novelty_diff.abs()>base.G_MARGIN+1e-12).any() or (live.span_diff.abs()>base.S_MARGIN+1e-12).any() or (live.opposition_diff.abs()>base.O_MARGIN+1e-12).any()): raise ValueError('caliper')
    fallback=steps[~supported];zero=['rank_gap','hardness_diff','loss_variance_z_diff','parameter_z_diff','translation_diff','gradient_novelty_diff','span_diff','opposition_diff']
    if len(fallback) and (not fallback.high_envs.eq(fallback.low_envs).all() or not fallback[zero].eq(0).all().all()): raise ValueError('fallback asymmetry')
    support_by=steps.assign(sb=supported).groupby('rep').sb.mean().reindex(REPS);pooled=float(supported.mean());minimum=float(support_by.min())
    if pooled<base.MIN_POOLED_SUPPORT or minimum<base.MIN_REP_SUPPORT: raise ValueError('support gate')
    hz=held.set_index(['rep','arm'])
    for key,d in state_digests.items():
        if hz.loc[key,'state_digest']!=d: raise ValueError('state digest mismatch')
    maxerr=0.
    for (rep,arm),block in env.groupby(['rep','arm']):
        v=block.sort_values('env_seed').accuracy.to_numpy(np.float64);obs=np.array([v.mean(),v.std(ddof=1),np.quantile(v,.1),v.min()]);reported=hz.loc[(rep,arm),['mean_test','sd_test','p10_test','min_test']].to_numpy(float);maxerr=max(maxerr,float(np.max(np.abs(obs-reported))))
    if maxerr>1e-12: raise ValueError('environment reaggregation')
    ref=steps.groupby('rep').mean(numeric_only=True).reindex(REPS);paired=[]
    for rep in REPS:
        row={'rep':rep,'support_fraction':float(support_by.loc[rep]),**{f:float(ref.loc[rep,f]) for f in ('rank_gap','hardness_diff','loss_variance_z_diff','parameter_z_diff','translation_diff','gradient_novelty_diff','span_diff','opposition_diff')}}
        for met in ('mean_test','sd_test','p10_test','min_test','clean_test'):
            row[met+'_benefit']=float(hz.loc[(rep,'high'),met]-hz.loc[(rep,'low'),met]);row['high_'+met+'_vs_reference']=float(hz.loc[(rep,'high'),met]-hz.loc[(rep,'reference'),met]);row['low_'+met+'_vs_reference']=float(hz.loc[(rep,'low'),met]-hz.loc[(rep,'reference'),met])
        paired.append(row)
    paired=pd.DataFrame(paired);balances=[];gates=True
    for col,margin in (('loss_variance_z_diff',base.L_MARGIN),('parameter_z_diff',base.P_MARGIN),('translation_diff',base.T_MARGIN),('gradient_novelty_diff',base.G_MARGIN),('span_diff',base.S_MARGIN),('opposition_diff',base.O_MARGIN)):
        q=core.equivalent(paired[col],margin);balances.append({'endpoint':col,**q});gates &= bool(q['pass'])
    rank=core.estimate(paired.rank_gap);rank_pass=rank['mean']>=base.RANK_MIN and rank['p_one_sided']<.05;gates &= rank_pass
    mean_eff=core.estimate(paired.mean_test_benefit);tail_eff=estimate_negative(paired.min_test_benefit)
    mean_pass=mean_eff['mean']>0 and mean_eff['p_one_sided']<.05;tail_pass=tail_eff['mean_original']<0 and tail_eff['p_one_sided_negative']<.05
    if not gates: decision='MEAN-TAIL MATCH OR MANIPULATION FAILURE'
    elif mean_pass and tail_pass: decision='MEAN-TAIL TRADEOFF SUPPORT'
    elif mean_pass: decision='MEAN BENEFIT REPLICATED / TAIL DEGRADATION NOT CONFIRMED'
    elif tail_pass: decision='TAIL DEGRADATION ONLY / MEAN BENEFIT NOT REPLICATED'
    else: decision='NO MEAN-TAIL TRADEOFF REPLICATION'
    summary=[]
    for col in paired.columns:
        if col!='rep':summary.append({'endpoint':col,**core.estimate(paired[col])})
    d={'decision':decision,'n':len(REPS),'all_balance_manipulation_gates_pass':bool(gates),'rank_manipulation_pass':bool(rank_pass),'mean_benefit_pass':bool(mean_pass),'tail_degradation_pass':bool(tail_pass),'pooled_support':pooled,'minimum_replicate_support':minimum,'fallback_steps':int((~supported).sum()),'states_checked':state_count,'environment_rows':len(env),'max_environment_reaggregation_error':maxerr,'maximum_general_identity_error':float(steps.step_max_general_identity_error.abs().max()),'protocol_hash':PH}
    out.mkdir(parents=True,exist_ok=True);paired.to_csv(out/'rank_tail_primary60.csv',index=False);pd.DataFrame(summary).to_csv(out/'rank_tail_summary60.csv',index=False);pd.DataFrame(balances).to_csv(out/'rank_tail_balance60.csv',index=False);pd.DataFrame([d]).to_csv(out/'rank_tail_decision60.csv',index=False)
    print(json.dumps(d),flush=True);print('MEAN_PRIMARY',json.dumps(mean_eff),flush=True);print('TAIL_PRIMARY',json.dumps(tail_eff),flush=True);print('RANK',json.dumps(rank),flush=True);print(pd.DataFrame(balances).to_string(index=False),flush=True)

def selftest():
    configure();assert len(REPS)==60 and REPS[0]==4100 and REPS[-1]==4159;z=np.full(60,-.02);e=estimate_negative(z);assert e['mean_original']<0 and e['p_one_sided_negative']<.05;print('RANK_TAIL_SELFTEST_PASS',PH)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('selftest','reference','reference-seal','train','arms-seal','evaluate','summarize'));p.add_argument('--start',type=int);p.add_argument('--end',type=int);p.add_argument('--input-dir');p.add_argument('--output-dir');p.add_argument('--global-seal');a=p.parse_args();configure()
    if a.mode=='selftest':selftest();return
    if a.output_dir is None:raise ValueError('output-dir required')
    out=Path(a.output_dir)
    if a.mode=='reference':check_range(a.start,a.end);base.reference(a.start,a.end,out)
    elif a.mode=='reference-seal':base.reference_seal(Path(a.input_dir),out)
    elif a.mode=='train':check_range(a.start,a.end);core.train(a.start,a.end,out,Path(a.global_seal))
    elif a.mode=='arms-seal':core.arms_seal(Path(a.input_dir),out)
    elif a.mode=='evaluate':check_range(a.start,a.end);core.evaluate(a.start,a.end,out,Path(a.global_seal))
    else:summarize(Path(a.input_dir),out,Path(a.global_seal))
if __name__=='__main__':main()

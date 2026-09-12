"""Issue104 Stage B: confirm loss-rank allocation at matched gradient geometry.

Global barriers:
1. seal all 30 reference schedules before any intervention training;
2. seal all 60 intervention states plus 30 reference states;
3. only then construct fresh heldout environments.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, torch

import centered_span_confirmation as core
import rank_allocation_calibration as cal
import corrected_centered_span_calibration as cc
import parameter_matched_novelty_confirmatory as pm
import translation_matched_calibration as tc
import cnn_regime_interaction as cr
import fixed_dose_response as fd
import transfer_specificity as base
from common import SmallCNN, head_gradient_directions, seed_everything

REPS=tuple(range(3200,3230)); TRAIN_SEEDS=tuple(range(83000,83064)); HELDOUT_SEEDS=tuple(range(84000,84080))
OFFSET=3610000000; K=16; Q=4; STEPS=80
L_MARGIN=.05; P_MARGIN=.05; T_MARGIN=.20; G_MARGIN=.05; S_MARGIN=.10; O_MARGIN=.15
RANK_MIN=1.50; MIN_POOLED_SUPPORT=.90; MIN_REP_SUPPORT=.80; MAX_ID=1e-10
STAGE_A_PROTOCOL_HASH='eae9daf7edfc5ba979725fc67dc4ac0207aeca8e6a80cb69ddf66776d739886b'
PROTOCOL={
    'issue':104,'stage':'B','reps':REPS,'arms':('high','low'),'baseline':'reference_loss_hard',
    'train_seeds':TRAIN_SEEDS,'heldout_seeds':HELDOUT_SEEDS,'base_seed_offset':OFFSET,'base_seed_stride':4099,
    'K':K,'Q':Q,'steps':STEPS,'epochs':10,'batch':128,'lr':.005,'wd':.001,
    'subset_family':'loss-rank1 plus any3 ranks2-16','subset_count':455,
    'loss_variance_equivalence_margin':L_MARGIN,'physical_diversity_equivalence_margin':P_MARGIN,
    'translation_equivalence_margin':T_MARGIN,'gradient_novelty_equivalence_margin':G_MARGIN,
    'centered_span_equivalence_margin':S_MARGIN,'opposition_equivalence_margin':O_MARGIN,
    'minimum_rank_manipulation':RANK_MIN,'minimum_pooled_support':MIN_POOLED_SUPPORT,
    'minimum_replicate_support':MIN_REP_SUPPORT,'max_general_identity_error':MAX_ID,
    'fallback':'identical_reference_loss_hard_top4','stage_a_protocol_hash':STAGE_A_PROTOCOL_HASH,'threads':1,
}
PH=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()
SOURCES=(
    'common.py','centered_span_calibration.py','corrected_centered_span_calibration.py','centered_span_confirmation.py',
    'rank_allocation_calibration.py','parameter_matched_novelty_calibration.py','parameter_matched_novelty_confirmatory.py',
    'translation_matched_calibration.py','cnn_regime_interaction.py','fixed_dose_response.py','transfer_specificity.py',
    'rank_allocation_confirmation.py',
)

def configure():
    core.REPS=REPS; core.TRAIN_SEEDS=TRAIN_SEEDS; core.HELDOUT_SEEDS=HELDOUT_SEEDS; core.OFFSET=OFFSET
    core.K=K; core.Q=Q; core.STEPS=STEPS; core.SELECTED_G_CALIPER=G_MARGIN
    core.MIN_POOLED_SUPPORT=MIN_POOLED_SUPPORT; core.MIN_REP_SUPPORT=MIN_REP_SUPPORT
    core.MAX_GENERAL_IDENTITY_ERROR=MAX_ID; core.STAGE_A_PROTOCOL_HASH=STAGE_A_PROTOCOL_HASH
    core.PROTOCOL=PROTOCOL; core.PH=PH; core.SOURCES=SOURCES

def source_hashes():
    configure(); return core.source_hashes()

def write_json(path,obj):
    path.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False))

def check_range(start,end):
    if start is None or end is None or start>=end or not set(range(start,end))<=set(REPS): raise ValueError('unregistered Stage-B range')

def reference(start,end,out):
    configure(); check_range(start,end); base.configure_determinism(1)
    out.mkdir(parents=True,exist_ok=True); (out/'schedules').mkdir(exist_ok=True); (out/'states').mkdir(exist_ok=True)
    if (out/'reference_manifest.json').exists(): raise ValueError('reference overwrite refused')
    x,y,*_=cr.data_split(); ty=torch.tensor(y); envs=base.geometric_envs(x,TRAIN_SEEDS); params=tc.parameters(TRAIN_SEEDS)
    step_rows=[]; subset_rows=[]; files={}; schedule_hashes={}; max_id=0.
    for rep in range(start,end):
        seed=OFFSET+4099*rep; sched=pm.build_schedule(len(ty),seed)
        if len(sched)!=STEPS: raise ValueError('step count')
        seed_everything(seed); model=SmallCNN(); model.train(); opt=torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=.001)
        selections={'high':[],'low':[]}; fallbacks=[]
        for step,(b,cand) in enumerate(sched):
            logits,h=model(torch.cat([envs[e][b] for e in cand])); per=torch.nn.functional.cross_entropy(logits,ty[b].repeat(K),reduction='none').reshape(K,-1)
            losses=per.mean(1); gd=head_gradient_directions(logits,h,ty[b],K)
            subsets=cal.enrich(cc.expanded_subsets(losses,gd,cand,params),gd)
            step_id=max(abs(float(s['general_identity_error'])) for s in subsets); max_id=max(max_id,step_id)
            for idx,s in enumerate(subsets):
                subset_rows.append({'rep':rep,'step':step,'subset_id':idx,'local':';'.join(map(str,s['local'])),'ranks':';'.join(map(str,s['ranks'])),'envs':';'.join(map(str,s['envs'])),
                    **{f:s[f] for f in ('mean_rank','z_hard','z_loss_variance','z_param','translation_score','gradient_novelty','centered_erank','opposition_score','general_identity_error')}})
            pair=cal.choose_pair(subsets)
            top4_local=sorted(sorted(range(K),key=lambda j:(-float(losses[j].detach()),j))[:Q]); top4_envs=[int(cand[j]) for j in top4_local]
            if pair is None:
                hi_envs=list(top4_envs); lo_envs=list(top4_envs); fallback=True
                rec={'rank_gap':0.,'hardness_diff':0.,'loss_variance_z_diff':0.,'parameter_z_diff':0.,'translation_diff':0.,'gradient_novelty_diff':0.,'span_diff':0.,'opposition_diff':0.,'eligible_pairs':0,'high_ranks':'1;2;3;4','low_ranks':'1;2;3;4'}
            else:
                hi=subsets[pair['high']]; lo=subsets[pair['low']]; hi_envs=list(map(int,hi['envs'])); lo_envs=list(map(int,lo['envs'])); fallback=False; rec=dict(pair)
            selections['high'].append(hi_envs); selections['low'].append(lo_envs); fallbacks.append(fallback)
            step_rows.append({'rep':rep,'step':step,'supported':not fallback,'fallback':fallback,'high_envs':';'.join(map(str,hi_envs)),'low_envs':';'.join(map(str,lo_envs)),
                'overlap':len(set(hi_envs)&set(lo_envs))/Q,'step_max_general_identity_error':step_id,**{k:v for k,v in rec.items() if k not in ('high','low')}})
            opt.zero_grad(set_to_none=True); per[top4_local].mean().backward(); opt.step()
        payload={'rep':rep,'protocol_hash':PH,'base_schedule_digest':pm.schedule_digest(sched),'selected_g_caliper':G_MARGIN,'fallback':fallbacks,**selections}
        sp=out/'schedules'/f'rep{rep}.json'; write_json(sp,payload); schedule_hashes[sp.name]=fd.sha256(sp); files[str(sp.relative_to(out))]=fd.sha256(sp)
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; rp=core.save_state(out,rep,'reference',state,fd.sha256(sp)); files[str(rp.relative_to(out))]=fd.sha256(rp)
        print(f'RANK_REFERENCE_SEALED rep={rep}; arms=0 heldout=0',flush=True)
    spath=out/f'rank_allocation_reference_{start}_{end}.csv.gz'; upath=out/f'rank_allocation_subsets_{start}_{end}.csv.gz'
    pd.DataFrame(step_rows).to_csv(spath,index=False,compression='gzip'); pd.DataFrame(subset_rows).to_csv(upath,index=False,compression='gzip'); files[spath.name]=fd.sha256(spath);files[upath.name]=fd.sha256(upath)
    write_json(out/'reference_manifest.json',{'protocol':PROTOCOL,'protocol_hash':PH,'start':start,'end':end,'files':files,'schedule_hashes':schedule_hashes,'source_hashes':source_hashes(),'split_hashes':cr.split_hashes(),'runtime':fd.runtime(),'maximum_general_identity_error':max_id,'heldout_constructed':False,'intervention_arms_trained':0})

def reference_seal(root,out):
    configure(); covered=[]; shards=[]; max_id=0.
    for mp in sorted(root.rglob('reference_manifest.json')):
        m=json.loads(mp.read_text()); start,end=int(m['start']),int(m['end']); check_range(start,end); core.validate_reference_shard(mp.parent,start,end); covered.extend(range(start,end)); max_id=max(max_id,abs(float(m['maximum_general_identity_error']))); shards.append({'start':start,'end':end,'manifest_sha256':fd.sha256(mp)})
    if sorted(covered)!=list(REPS): raise ValueError('global reference coverage')
    steps=core.load_frames(root,'rank_allocation_reference_*.csv.gz'); expected={(r,t) for r in REPS for t in range(STEPS)}
    if len(steps)!=len(expected) or steps.duplicated(['rep','step']).any() or set(steps[['rep','step']].itertuples(index=False,name=None))!=expected: raise ValueError('reference step grid')
    supported=steps.supported.astype(bool)
    if not (supported==~steps.fallback.astype(bool)).all(): raise ValueError('support/fallback mismatch')
    live=steps[supported]
    if len(live) and ((live.loss_variance_z_diff.abs()>L_MARGIN+1e-12).any() or (live.parameter_z_diff.abs()>P_MARGIN+1e-12).any() or (live.translation_diff.abs()>T_MARGIN+1e-12).any() or (live.gradient_novelty_diff.abs()>G_MARGIN+1e-12).any() or (live.span_diff.abs()>S_MARGIN+1e-12).any() or (live.opposition_diff.abs()>O_MARGIN+1e-12).any()): raise ValueError('supported-step caliper violation')
    fallback=steps[~supported]; zeros=['rank_gap','hardness_diff','loss_variance_z_diff','parameter_z_diff','translation_diff','gradient_novelty_diff','span_diff','opposition_diff']
    if len(fallback) and (not fallback.high_envs.eq(fallback.low_envs).all() or not fallback[zeros].eq(0).all().all()): raise ValueError('fallback asymmetry')
    support_by=steps.assign(sb=supported).groupby('rep').sb.mean().reindex(REPS); pooled=float(supported.mean()); minimum=float(support_by.min())
    if pooled<MIN_POOLED_SUPPORT or minimum<MIN_REP_SUPPORT: raise RuntimeError(f'RANK-ALLOCATION OVERLAP SUPPORT FAILURE pooled={pooled} minimum={minimum}')
    if max_id>MAX_ID: raise ValueError('general identity')
    out.mkdir(parents=True,exist_ok=True); write_json(out/'reference_seal.json',{'protocol_hash':PH,'stage':'reference','reps':list(REPS),'pooled_support':pooled,'minimum_replicate_support':minimum,'supported_steps':int(supported.sum()),'fallback_steps':int((~supported).sum()),'maximum_general_identity_error':max_id,'shards':sorted(shards,key=lambda z:z['start'])}); print(f'GLOBAL_RANK_REFERENCE_SEAL support={pooled:.6f} min={minimum:.6f}',flush=True)

def summarize(root,out,global_seal):
    configure(); core.validate_global_seal(global_seal,'arms'); covered=[]; state_count=0; state_digests={}
    for mp in sorted(root.rglob('evaluation_manifest.json')):
        m=json.loads(mp.read_text()); start,end=int(m['start']),int(m['end']); check_range(start,end); core.validate_evaluation_shard(mp.parent,start,end); covered.extend(range(start,end))
        for rep in range(start,end):
            for arm in core.ALL_ARMS:
                p=mp.parent/'states'/f'rep{rep}_{arm}.pt'; ck=torch.load(p,map_location='cpu',weights_only=True)
                if (ck['rep'],ck['arm'],ck['protocol_hash'])!=(rep,arm,PH): raise ValueError('summary state metadata')
                state_digests[(rep,arm)]=pm.state_digest(ck['state_dict']); state_count+=1
    if sorted(covered)!=list(REPS) or state_count!=90: raise ValueError('summary global coverage')
    steps=core.load_frames(root,'rank_allocation_reference_*.csv.gz'); held=core.load_frames(root,'centered_span_heldout_*.csv'); env=core.load_frames(root,'centered_span_environment_*.csv.gz')
    expected_steps={(r,t) for r in REPS for t in range(STEPS)}; expected_held={(r,a) for r in REPS for a in core.ALL_ARMS}; expected_env={(r,a,s) for r in REPS for a in core.ALL_ARMS for s in HELDOUT_SEEDS}
    if len(steps)!=len(expected_steps) or steps.duplicated(['rep','step']).any() or set(steps[['rep','step']].itertuples(index=False,name=None))!=expected_steps: raise ValueError('step grid')
    if len(held)!=len(expected_held) or held.duplicated(['rep','arm']).any() or set(held[['rep','arm']].itertuples(index=False,name=None))!=expected_held: raise ValueError('heldout grid')
    if len(env)!=len(expected_env) or env.duplicated(['rep','arm','env_seed']).any() or set(env[['rep','arm','env_seed']].itertuples(index=False,name=None))!=expected_env: raise ValueError('environment grid')
    numeric=['rank_gap','hardness_diff','loss_variance_z_diff','parameter_z_diff','translation_diff','gradient_novelty_diff','span_diff','opposition_diff','step_max_general_identity_error']
    if not np.isfinite(steps[numeric].to_numpy(float)).all() or not np.isfinite(held[['mean_test','sd_test','p10_test','min_test','clean_test']].to_numpy(float)).all() or not np.isfinite(env[['accuracy']].to_numpy(float)).all(): raise ValueError('nonfinite')
    if float(steps.step_max_general_identity_error.abs().max())>MAX_ID: raise ValueError('identity')
    supported=steps.supported.astype(bool); live=steps[supported]
    if len(live) and ((live.loss_variance_z_diff.abs()>L_MARGIN+1e-12).any() or (live.parameter_z_diff.abs()>P_MARGIN+1e-12).any() or (live.translation_diff.abs()>T_MARGIN+1e-12).any() or (live.gradient_novelty_diff.abs()>G_MARGIN+1e-12).any() or (live.span_diff.abs()>S_MARGIN+1e-12).any() or (live.opposition_diff.abs()>O_MARGIN+1e-12).any()): raise ValueError('summary caliper')
    support_by=steps.assign(sb=supported).groupby('rep').sb.mean().reindex(REPS); pooled=float(supported.mean()); minimum=float(support_by.min())
    if pooled<MIN_POOLED_SUPPORT or minimum<MIN_REP_SUPPORT: raise ValueError('Stage-B support failure')
    hz=held.set_index(['rep','arm'])
    for key,d in state_digests.items():
        if hz.loc[key,'state_digest']!=d: raise ValueError('state digest mismatch')
    max_env_error=0.
    for (rep,arm),block in env.groupby(['rep','arm']):
        v=block.sort_values('env_seed').accuracy.to_numpy(np.float64); observed=np.array([v.mean(),v.std(ddof=1),np.quantile(v,.1),v.min()]); reported=hz.loc[(rep,arm),['mean_test','sd_test','p10_test','min_test']].to_numpy(float); max_env_error=max(max_env_error,float(np.max(np.abs(observed-reported))))
    if max_env_error>1e-12: raise ValueError('environment reaggregation')
    ref=steps.groupby('rep').mean(numeric_only=True).reindex(REPS); paired=[]; metrics=('mean_test','sd_test','p10_test','min_test','clean_test')
    for rep in REPS:
        row={'rep':rep,'support_fraction':float(support_by.loc[rep]),**{f:float(ref.loc[rep,f]) for f in ('rank_gap','hardness_diff','loss_variance_z_diff','parameter_z_diff','translation_diff','gradient_novelty_diff','span_diff','opposition_diff')}}
        for met in metrics:
            row[met+'_benefit']=float(hz.loc[(rep,'high'),met]-hz.loc[(rep,'low'),met]); row['high_'+met+'_vs_reference']=float(hz.loc[(rep,'high'),met]-hz.loc[(rep,'reference'),met]); row['low_'+met+'_vs_reference']=float(hz.loc[(rep,'low'),met]-hz.loc[(rep,'reference'),met])
        paired.append(row)
    paired=pd.DataFrame(paired); balances=[]; gates=True
    for col,margin in (('loss_variance_z_diff',L_MARGIN),('parameter_z_diff',P_MARGIN),('translation_diff',T_MARGIN),('gradient_novelty_diff',G_MARGIN),('span_diff',S_MARGIN),('opposition_diff',O_MARGIN)):
        q=core.equivalent(paired[col],margin); balances.append({'endpoint':col,**q}); gates &= bool(q['pass'])
    rank=core.estimate(paired.rank_gap); rank_pass=bool(rank['mean']>=RANK_MIN and rank['p_one_sided']<.05); gates &= rank_pass
    hardness=core.estimate(paired.hardness_diff); perf=core.estimate(paired.mean_test_benefit); perf_pass=bool(perf['mean']>0 and perf['p_one_sided']<.05)
    decision='RANK-ALLOCATION MATCH OR MANIPULATION FAILURE' if not gates else ('MATCHED RANK-RELAXATION SUPPORT' if perf_pass else 'NO RANK-RELAXATION PERFORMANCE SUPPORT')
    summary=[]
    for col in paired.columns:
        if col!='rep': summary.append({'endpoint':col,**core.estimate(paired[col])})
    d={'decision':decision,'n':30,'all_balance_manipulation_gates_pass':bool(gates),'rank_manipulation_pass':rank_pass,'performance_pass':perf_pass,'pooled_support':pooled,'minimum_replicate_support':minimum,'fallback_steps':int((~supported).sum()),'states_checked':state_count,'environment_rows':len(env),'max_environment_reaggregation_error':max_env_error,'maximum_general_identity_error':float(steps.step_max_general_identity_error.abs().max()),'protocol_hash':PH}
    out.mkdir(parents=True,exist_ok=True); paired.to_csv(out/'rank_allocation_primary30.csv',index=False);pd.DataFrame(summary).to_csv(out/'rank_allocation_summary30.csv',index=False);pd.DataFrame(balances).to_csv(out/'rank_allocation_balance30.csv',index=False);held.to_csv(out/'rank_allocation_heldout30.csv',index=False);ref.reset_index().to_csv(out/'rank_allocation_reference_summary30.csv',index=False);pd.DataFrame([d]).to_csv(out/'rank_allocation_decision30.csv',index=False)
    print(json.dumps(d),flush=True);print('PRIMARY',json.dumps(perf),flush=True);print('RANK',json.dumps(rank),flush=True);print('HARDNESS_DESCRIPTIVE',json.dumps(hardness),flush=True);print(pd.DataFrame(balances).to_string(index=False),flush=True)

def selftest():
    configure(); cal.selftest(); assert core.equivalent(np.zeros(30),.05)['pass']; a=np.linspace(1.6,1.8,30); e=core.estimate(a); assert e['mean']>=RANK_MIN and e['p_one_sided']<.05; print('RANK_CONFIRMATION_SELFTEST_PASS',PH)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('selftest','reference','reference-seal','train','arms-seal','evaluate','summarize'));p.add_argument('--start',type=int);p.add_argument('--end',type=int);p.add_argument('--input-dir');p.add_argument('--output-dir');p.add_argument('--global-seal');a=p.parse_args(); configure()
    if a.mode=='selftest': selftest(); return
    if a.output_dir is None: raise ValueError('output-dir required')
    out=Path(a.output_dir)
    if a.mode=='reference': reference(a.start,a.end,out)
    elif a.mode=='reference-seal': reference_seal(Path(a.input_dir),out)
    elif a.mode=='train': core.train(a.start,a.end,out,Path(a.global_seal))
    elif a.mode=='arms-seal': core.arms_seal(Path(a.input_dir),out)
    elif a.mode=='evaluate': core.evaluate(a.start,a.end,out,Path(a.global_seal))
    else: summarize(Path(a.input_dir),out,Path(a.global_seal))
if __name__=='__main__': main()

"""Issue91: overlap-restricted schedules with an identical-subset fallback.
Reuses the unchanged Issue86 train/evaluate/seal machinery in an isolated process.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats
import translation_matched_calibration as tc
import spatial_allocation_control as core
import parameter_matched_novelty_calibration as cal
import parameter_matched_novelty_confirmatory as pm
import cnn_regime_interaction as cr
import fixed_dose_response as fd
import transfer_specificity as base
from common import SmallCNN, head_gradient_directions, seed_everything

REPS=tuple(range(2200,2230)); ARMS=('u_high','u_low','m_high','m_low')
TRAIN_SEEDS=tuple(range(63000,63064)); HELDOUT_SEEDS=tuple(range(64000,64080))
OFFSET=2610000000; T_CALIPER=.20

def configure():
    # Isolated-process configuration; historical source files stay unchanged.
    core.REPS=REPS;core.ARMS=ARMS;core.ALL_ARMS=ARMS+('reference',)
    core.TRAIN_SEEDS=TRAIN_SEEDS;core.HELDOUT_SEEDS=HELDOUT_SEEDS;core.OFFSET=OFFSET
    core.PROTOCOL={**core.PROTOCOL,'issue':91,'reps':REPS,'arms':ARMS,
        'train_seeds':TRAIN_SEEDS,'heldout_seeds':HELDOUT_SEEDS,'base_seed_offset':OFFSET,
        'translation_caliper':T_CALIPER,'fallback':'identical_reference_top4',
        'min_pooled_support':.90,'min_replicate_support':.80}
    core.PH=hashlib.sha256(json.dumps(core.PROTOCOL,sort_keys=True).encode()).hexdigest()
    core.SOURCES=tuple(dict.fromkeys((*tc.SOURCES,'spatial_allocation_control.py','overlap_translation_novelty.py')))


def select_or_fallback(subsets,top4,family):
    pair=tc.choose_pair(subsets,None if family=='u' else T_CALIPER)
    if pair is not None:return {**pair,'fallback':False}
    if family=='u':raise RuntimeError('NO H/P REFERENCE PAIR')
    idx=next(i for i,r in enumerate(subsets) if tuple(r['local'])==tuple(top4))
    return {'high':idx,'low':idx,'eligible_pairs':0,'gradient_gap':0.,'hardness_diff':0.,
            'parameter_z_diff':0.,'translation_gap':0.,'fallback':True}


def support_gate(root):
    steps=core.load(root,'translation_reference_*.csv.gz')
    core.checked_grid(steps,['rep','step','family'],{(r,t,f) for r in REPS for t in range(80) for f in ('u','m')})
    if steps[steps.family=='u'].fallback.any():raise ValueError('unmatched fallback forbidden')
    m=steps[steps.family=='m'];supported=1-m.groupby('rep').fallback.mean()
    pooled=float(supported.mean());minimum=float(supported.min())
    if pooled<.90 or minimum<.80:raise RuntimeError(f'OVERLAP SUPPORT FAILURE pooled={pooled} minimum={minimum}')
    f=m[m.fallback]
    if not f.high_envs.eq(f.low_envs).all() or not f[['gradient_gap','hardness_diff','parameter_z_diff','translation_gap']].eq(0).all().all():raise ValueError('fallback asymmetry')
    return {'pooled_support':pooled,'minimum_replicate_support':minimum,'fallback_steps':len(f)}


def reference(start,end,out):
    core.check_range(start,end);base.configure_determinism(1)
    out.mkdir(parents=True,exist_ok=True);(out/'schedules').mkdir(exist_ok=True);(out/'states').mkdir(exist_ok=True)
    if list((out/'schedules').glob('*.json')) or list((out/'states').glob('*.pt')):raise ValueError('overwrite refused')
    x,y,*_=cr.data_split();ty=torch.tensor(y);envs=base.geometric_envs(x,TRAIN_SEEDS);par=tc.parameters(TRAIN_SEEDS)
    records=[];all_subsets=[];files={};schedule_hashes={}
    for rep in range(start,end):
        seed=OFFSET+4099*rep;sched=pm.build_schedule(len(ty),seed)
        if len(sched)!=80:raise ValueError('steps')
        seed_everything(seed);model=SmallCNN();model.train();opt=torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=.001)
        selections={a:[] for a in ARMS}
        for step,(b,cand) in enumerate(sched):
            logits,h=model(torch.cat([envs[e][b] for e in cand]));per=torch.nn.functional.cross_entropy(logits,ty[b].repeat(16),reduction='none').reshape(16,-1)
            losses=per.mean(1);gd=head_gradient_directions(logits,h,ty[b],16)
            subsets=tc.add_translation(cal.feasible_subsets(losses,gd@gd.T,cand,par),par)
            for idx,s in enumerate(subsets):
                all_subsets.append({'rep':rep,'step':step,'subset_id':idx,'local':';'.join(map(str,s['local'])),'envs':';'.join(map(str,s['envs'])),
                    **{f:s[f] for f in ('gradient_novelty','z_hard','z_param','translation_score','physical_diversity')}})
            top4=sorted(sorted(range(16),key=lambda j:(-float(losses[j].detach()),j))[:4])
            for family in ('u','m'):
                pair=select_or_fallback(subsets,top4,family)
                hi=subsets[pair['high']];lo=subsets[pair['low']];eh=list(hi['envs']);el=list(lo['envs'])
                selections[family+'_high'].append(eh);selections[family+'_low'].append(el)
                pa,pb=par[eh],par[el]
                rec={'rep':rep,'step':step,'family':family,**pair,'high_envs':';'.join(map(str,eh)),
                     'low_envs':';'.join(map(str,el)),'overlap':len(set(eh)&set(el))/4,
                     'raw_parameter_diversity_diff':hi['physical_diversity']-lo['physical_diversity']}
                for i,cname in enumerate(core.COORDS):
                    rec['mean_'+cname+'_diff']=float(pa[:,i].mean()-pb[:,i].mean())
                    rec['var_'+cname+'_diff']=float(pa[:,i].var()-pb[:,i].var())
                records.append(rec)
            chosen=sorted(sorted(range(16),key=lambda j:(-float(losses[j].detach()),j))[:4])
            opt.zero_grad(set_to_none=True);per[chosen].mean().backward();opt.step()
        pl={'rep':rep,'protocol_hash':core.PH,'base_schedule_digest':pm.schedule_digest(sched),**selections}
        sp=out/'schedules'/f'rep{rep}.json';core.write_json(sp,pl);schedule_hashes[sp.name]=fd.sha256(sp);files[str(sp.relative_to(out))]=fd.sha256(sp)
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};p=core.save_state(out,rep,'reference',state,fd.sha256(sp));files[str(p.relative_to(out))]=fd.sha256(p)
        print('TRANSLATION_REFERENCE_SEALED',rep,flush=True)
    for name,data in ((f'translation_reference_{start}_{end}.csv.gz',records),(f'translation_subsets_{start}_{end}.csv.gz',all_subsets)):
        p=out/name;pd.DataFrame(data).to_csv(p,index=False,compression='gzip');files[p.name]=fd.sha256(p)
    core.write_json(out/'reference_manifest.json',{'protocol':core.PROTOCOL,'protocol_hash':core.PH,'start':start,'end':end,
        'files':files,'schedule_hashes':schedule_hashes,'source_hashes':core.source_hashes(),'split_hashes':cr.split_hashes(),'runtime':fd.runtime()})

def equivalent(x,margin):
    e=core.estimate(x);mean,se,n=e['mean'],e['se'],e['n'];k=float(stats.t.ppf(.95,n-1))
    lo,hi=mean-k*se,mean+k*se
    pl=float(stats.t.sf((mean+margin)/se,n-1)) if se>0 else (0. if mean>-margin else 1.)
    pu=float(stats.t.cdf((mean-margin)/se,n-1)) if se>0 else (0. if mean<margin else 1.)
    return {'mean':mean,'se':se,'margin':margin,'ci90_low':lo,'ci90_high':hi,'p_lower':pl,'p_upper':pu,
            'pass':bool(lo>-margin and hi<margin and pl<.05 and pu<.05)}

def decide(gates,mp,up):
    if not gates:return 'MATCH OR MANIPULATION FAILURE'
    if mp and up:return 'OVERLAP-RESTRICTED TRANSLATION-BALANCED NOVELTY SUPPORT'
    if up:return 'UNMATCHED REPLICATION ONLY / MATCHED EFFECT NOT SUPPORTED'
    if mp:return 'MATCHED EFFECT ONLY / UNMATCHED CONTROL NOT REPLICATED'
    return 'NO PERFORMANCE REPLICATION'

def summarize(root,out):
    core.validate_manifests(root,'arms');support=support_gate(root)
    steps=core.load(root,'translation_reference_*.csv.gz');held=core.load(root,'spatial_heldout_*.csv');env=core.load(root,'spatial_environment_*.csv.gz')
    core.checked_grid(steps,['rep','step','family'],{(r,t,f) for r in REPS for t in range(80) for f in ('u','m')})
    core.checked_grid(held,['rep','arm'],{(r,a) for r in REPS for a in core.ALL_ARMS})
    core.checked_grid(env,['rep','arm','env_seed'],{(r,a,s) for r in REPS for a in core.ALL_ARMS for s in HELDOUT_SEEDS})
    metrics=['mean_test','sd_test','p10_test','min_test','clean_test']
    core.finite(steps,['gradient_gap','hardness_diff','parameter_z_diff','translation_gap']);core.finite(held,metrics);core.finite(env,['accuracy'])
    if (steps.hardness_diff.abs()>.05+1e-12).any() or (steps.parameter_z_diff.abs()>.05+1e-12).any():raise ValueError('HP caliper')
    if (steps[steps.family=='m'].translation_gap.abs()>T_CALIPER+1e-12).any():raise ValueError('T caliper')
    hz=held.set_index(['rep','arm']);err=0.;states=0
    for (rep,arm),block in env.groupby(['rep','arm']):
        v=block.sort_values('env_seed').accuracy.to_numpy();observed=[v.mean(),v.std(ddof=1),np.quantile(v,.1),v.min()]
        err=max(err,float(np.max(np.abs(np.array(observed)-hz.loc[(rep,arm),metrics[:4]].to_numpy(dtype=float)))))
    if err>1e-12:raise ValueError('environment reaggregation')
    for mp in root.rglob('arm_manifest.json'):
        for p in sorted((mp.parent/'states').glob('*.pt')):
            ck=torch.load(p,map_location='cpu',weights_only=True)
            if ck['protocol_hash']!=core.PH:raise ValueError('state protocol')
            key=(ck['rep'],ck['arm'])
            if pm.state_digest(ck['state_dict'])!=hz.loc[key,'state_digest']:raise ValueError('state digest')
            states+=1
    if states!=150:raise ValueError('state count')
    ref=steps.groupby(['rep','family']).mean(numeric_only=True);paired=[]
    for rep in REPS:
        row={'rep':rep}
        for f in ('u','m'):
            for met in metrics:row[f+'_'+met+'_benefit']=float(hz.loc[(rep,f+'_high'),met]-hz.loc[(rep,f+'_low'),met])
            for met in ('gradient_gap','hardness_diff','parameter_z_diff','translation_gap'):row[f+'_'+met]=float(ref.loc[(rep,f),met])
            for arm in (f+'_high',f+'_low'):row[arm+'_vs_reference']=float(hz.loc[(rep,arm),'mean_test']-hz.loc[(rep,'reference'),'mean_test'])
        row['benefit_difference_u_minus_m']=row['u_mean_test_benefit']-row['m_mean_test_benefit'];paired.append(row)
    paired=pd.DataFrame(paired);balances=[];summary=[];gates=True
    for f in ('u','m'):
        for met in ('hardness_diff','parameter_z_diff'):
            b=equivalent(paired[f+'_'+met],.05);balances.append({'endpoint':f+'_'+met,**b});gates &= b['pass']
        g=core.estimate(paired[f+'_gradient_gap']);gates &= g['mean']>=.05 and g['p_one_sided']<.05
    b=equivalent(paired['m_translation_gap'],T_CALIPER);balances.append({'endpoint':'m_translation_gap',**b});gates &= b['pass']
    for col in paired.columns:
        if col!='rep':summary.append({'endpoint':col,**core.estimate(paired[col])})
    m=core.estimate(paired.m_mean_test_benefit);u=core.estimate(paired.u_mean_test_benefit)
    mp=m['mean']>0 and m['p_one_sided']<.05;up=u['mean']>0 and u['p_one_sided']<.05
    d={'decision':decide(gates,mp,up),'n':30,'all_gates_pass':bool(gates),'matched_performance_pass':bool(mp),
       'unmatched_performance_pass':bool(up),**support,'translation_caliper':T_CALIPER,'protocol_hash':core.PH,'states_checked':states,'environment_rows':len(env),'max_environment_error':err}
    out.mkdir(parents=True,exist_ok=True)
    for n,df in (('paired30',paired),('summary30',pd.DataFrame(summary)),('balance30',pd.DataFrame(balances)),('heldout30',held),('reference_summary30',ref.reset_index()),('decision30',pd.DataFrame([d]))):df.to_csv(out/f'translation_matched_{n}.csv',index=False)
    print(json.dumps(d),flush=True)

def selftest():
    tc.selftest();assert equivalent(np.zeros(30),.05)['pass']
    assert not equivalent(np.full(30,.05),.05)['pass']
    assert decide(True,True,True)=='OVERLAP-RESTRICTED TRANSLATION-BALANCED NOVELTY SUPPORT'
    assert decide(True,False,True).startswith('UNMATCHED REPLICATION ONLY')
    assert decide(False,True,True)=='MATCH OR MANIPULATION FAILURE'
    rows=[{'local':(0,1,2,3),'envs':(0,1,2,3),'gradient_novelty':.1,'z_hard':0.,'z_param':0.,'translation_score':0.},
          {'local':(0,1,2,4),'envs':(0,1,2,4),'gradient_novelty':.5,'z_hard':0.,'z_param':0.,'translation_score':1.}]
    f=select_or_fallback(rows,[0,1,2,3],'m');assert f['fallback'] and f['high']==f['low']==0 and f['gradient_gap']==0
    assert not select_or_fallback(rows,[0,1,2,3],'u')['fallback']
    print('CONFIRMATORY_SYNTHETIC_PASS')

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('selftest','reference','train','evaluate','seal','summarize'))
    p.add_argument('--start',type=int);p.add_argument('--end',type=int);p.add_argument('--output-dir');p.add_argument('--input-dir');p.add_argument('--stage',choices=('reference','arms'))
    a=p.parse_args()
    if a.mode=='selftest':selftest();return
    configure();out=Path(a.output_dir)
    if a.mode in ('reference','train','evaluate'):{'reference':reference,'train':core.train,'evaluate':core.evaluate}[a.mode](a.start,a.end,out)
    elif a.mode=='seal':
        support_gate(Path(a.input_dir));core.seal(Path(a.input_dir),out,a.stage)
    else:summarize(Path(a.input_dir),out)
if __name__=='__main__':main()

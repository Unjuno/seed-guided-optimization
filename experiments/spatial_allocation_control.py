"""Issue86: fixed-Q spatial-allocation alternative; reference -> train -> evaluate.

Reference hardness/total-diversity are matched scalars, not all physical factors.
This is a schedule intervention, not an online gradnov-vs-loss-hard benchmark.
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
import parameter_matched_novelty_confirmatory as pm
import parameter_matched_novelty_calibration as cal
import cnn_regime_interaction as cr
import fixed_dose_response as fd
import transfer_specificity as base
from common import SmallCNN, environment_parameters, head_gradient_directions, seed_everything

REPS=tuple(range(1900,1930))
ARMS=('g_high','g_low','t_high','t_low')
ALL_ARMS=ARMS+('reference',)
TRAIN_SEEDS=tuple(range(57000,57064))
HELDOUT_SEEDS=tuple(range(58000,58080))
OFFSET=2310000000
C=.05
STEPS=80
PROTOCOL={'issue':86,'reps':REPS,'arms':ARMS,'baseline':'reference','K':16,'Q':4,
          'epochs':10,'batch':128,'lr':.005,'wd':.001,'train_seeds':TRAIN_SEEDS,
          'heldout_seeds':HELDOUT_SEEDS,'base_seed_offset':OFFSET,'base_seed_stride':4099,
          'caliper':C,'equivalence_margin':C,'spatial_score':'var_population(z_dx)+var_population(z_dy)',
          'feasible_ranks':tuple(range(2,11)),'n_train':988,'n_eval':809,'threads':1}
PH=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()
SOURCES=tuple(dict.fromkeys((*pm.LOCAL_SOURCES,'spatial_allocation_control.py')))
COORDS=('angle','dx','dy','blur','contrast','brightness','noise')

def source_hashes():
    return {n:fd.sha256(Path(__file__).parent/n) for n in SOURCES}

def check_range(start,end):
    if start is None or end is None or start>=end or not set(range(start,end))<=set(REPS):
        raise ValueError('unregistered range')

def write_json(path,obj):
    path.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False))

def checked_grid(df,keys,expected):
    if len(df)!=len(expected) or df.duplicated(keys).any() or set(df[keys].itertuples(index=False,name=None))!=expected:
        raise ValueError('invalid grid '+str(keys))

def finite(df,columns):
    if not np.isfinite(df[list(columns)].to_numpy(dtype=float)).all():
        raise ValueError('nonfinite scientific fields')

def parameters():
    p=np.stack([environment_parameters(s) for s in TRAIN_SEEDS]).astype(np.float64)
    return (p-p.mean(0))/(p.std(0,ddof=0)+1e-12)

def choose(rows,field):
    # Retain the exact Issue83 eligibility and deterministic tie-breaking.
    scored=[dict(r,gradient_novelty=float(r[field])) for r in rows]
    return cal.best_pair(scored,C)

def read_ref(out,start,end):
    ref=json.loads((out/'reference_manifest.json').read_text())
    if (ref['start'],ref['end'],ref['protocol_hash'])!=(start,end,PH): raise ValueError('reference protocol')
    if ref['source_hashes']!=source_hashes(): raise ValueError('scientific source changed')
    if ref['split_hashes']!=cr.split_hashes(): raise ValueError('image split changed')
    for n,h in ref['files'].items():
        if fd.sha256(out/n)!=h: raise ValueError('reference seal mismatch '+n)
    return ref

def save_state(out,rep,arm,state,schedule_sha):
    path=out/'states'/f'rep{rep}_{arm}.pt'
    if path.exists(): raise ValueError('state overwrite refused')
    torch.save({'state_dict':state,'rep':rep,'arm':arm,'protocol_hash':PH,'schedule_sha256':schedule_sha},path)
    return path

def reference(start,end,out):
    check_range(start,end); base.configure_determinism(1)
    out.mkdir(parents=True,exist_ok=True); (out/'schedules').mkdir(exist_ok=True); (out/'states').mkdir(exist_ok=True)
    if list((out/'schedules').glob('*.json')) or list((out/'states').glob('*.pt')): raise ValueError('reference overwrite')
    x,y,*_=cr.data_split(); ty=torch.tensor(y); envs=base.geometric_envs(x,TRAIN_SEEDS); par=parameters()
    records=[]; files={}; schedule_hashes={}
    for rep in range(start,end):
        seed=OFFSET+4099*rep; schedule=pm.build_schedule(len(ty),seed)
        assert len(schedule)==STEPS
        seed_everything(seed); model=SmallCNN(); model.train(); opt=torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=.001)
        selections={a:[] for a in ARMS}
        for step,(b,cand) in enumerate(schedule):
            logits,h=model(torch.cat([envs[e][b] for e in cand]))
            per=torch.nn.functional.cross_entropy(logits,ty[b].repeat(16),reduction='none').reshape(16,-1)
            losses=per.mean(1); dirs=head_gradient_directions(logits,h,ty[b],16); cos=dirs@dirs.T
            subsets=cal.feasible_subsets(losses,cos,cand,par)
            for row in subsets:
                v=par[list(row['envs'])].var(0,ddof=0); row['translation_score']=float(v[1]+v[2])
            byenv={tuple(row['envs']):row for row in subsets}
            for family,field in [('g','gradient_novelty'),('t','translation_score')]:
                pair=choose(subsets,field)
                if pair is None: raise RuntimeError(f'MATCH FEASIBILITY FAILURE {rep} {step}')
                hi=pm.parse_envs(pair['high_envs']); lo=pm.parse_envs(pair['low_envs'])
                a=byenv[tuple(hi)]; d=byenv[tuple(lo)]
                selections[family+'_high'].append(hi); selections[family+'_low'].append(lo)
                pa,pb=par[hi],par[lo]
                rec={'rep':rep,'step':step,'family':family,
                     'gradient_gap':a['gradient_novelty']-d['gradient_novelty'],
                     'translation_gap':a['translation_score']-d['translation_score'],
                     'hardness_diff':a['z_hard']-d['z_hard'], 'parameter_z_diff':a['z_param']-d['z_param'],
                     'raw_parameter_diversity_diff':a['physical_diversity']-d['physical_diversity'],
                     'overlap':len(set(hi)&set(lo))/4,'high_envs':pair['high_envs'],'low_envs':pair['low_envs']}
                for i,c in enumerate(COORDS):
                    rec['mean_'+c+'_diff']=float(pa[:,i].mean()-pb[:,i].mean())
                    rec['var_'+c+'_diff']=float(pa[:,i].var()-pb[:,i].var())
                records.append(rec)
            selected=sorted(sorted(range(16),key=lambda j:(-float(losses[j].detach()),j))[:4])
            opt.zero_grad(set_to_none=True); per[selected].mean().backward(); opt.step()
        payload={'rep':rep,'protocol_hash':PH,'base_schedule_digest':pm.schedule_digest(schedule),**selections}
        sp=out/'schedules'/f'rep{rep}.json'; write_json(sp,payload); schedule_hashes[sp.name]=fd.sha256(sp)
        files[str(sp.relative_to(out))]=fd.sha256(sp)
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        p=save_state(out,rep,'reference',state,fd.sha256(sp)); files[str(p.relative_to(out))]=fd.sha256(p)
        print('REFERENCE_SEALED',rep,flush=True)
    path=out/f'spatial_reference_{start}_{end}.csv.gz'; pd.DataFrame(records).to_csv(path,index=False,compression='gzip'); files[path.name]=fd.sha256(path)
    write_json(out/'reference_manifest.json',{'protocol':PROTOCOL,'protocol_hash':PH,'start':start,'end':end,
        'files':files,'schedule_hashes':schedule_hashes,'source_hashes':source_hashes(),'split_hashes':cr.split_hashes(),'runtime':fd.runtime()})

def train(start,end,out):
    check_range(start,end); base.configure_determinism(1); ref=read_ref(out,start,end)
    if (out/'arm_manifest.json').exists(): raise ValueError('arm overwrite')
    x,y,*_=cr.data_split(); ty=torch.tensor(y); envs=base.geometric_envs(x,TRAIN_SEEDS); files={}; rows=[]
    for rep in range(start,end):
        seed=OFFSET+4099*rep; schedule=pm.build_schedule(len(ty),seed); sp=out/'schedules'/f'rep{rep}.json'; pl=json.loads(sp.read_text())
        if pl['rep']!=rep or pl['protocol_hash']!=PH or pm.schedule_digest(schedule)!=pl['base_schedule_digest']: raise ValueError('schedule metadata')
        for arm in ARMS:
            if len(pl[arm])!=STEPS: raise ValueError('schedule truncation')
            state,diag=pm.train_arm(envs,ty,schedule,pl[arm],seed)
            p=save_state(out,rep,arm,state,fd.sha256(sp)); files[str(p.relative_to(out))]=fd.sha256(p)
            rows.append({'rep':rep,'arm':arm,**diag})
        p=out/'states'/f'rep{rep}_reference.pt'; files[str(p.relative_to(out))]=fd.sha256(p)
        print('ARMS_SEALED',rep,flush=True)
    p=out/f'spatial_training_{start}_{end}.csv';pd.DataFrame(rows).to_csv(p,index=False);files[p.name]=fd.sha256(p)
    write_json(out/'arm_manifest.json',{'protocol_hash':PH,'start':start,'end':end,'files':files,
       'reference_manifest_sha256':fd.sha256(out/'reference_manifest.json'),'source_hashes':source_hashes(),'runtime':fd.runtime()})

def read_arm(out,start,end):
    ref=read_ref(out,start,end); m=json.loads((out/'arm_manifest.json').read_text())
    if (m['start'],m['end'],m['protocol_hash'])!=(start,end,PH): raise ValueError('arm protocol')
    if m['reference_manifest_sha256']!=fd.sha256(out/'reference_manifest.json') or m['source_hashes']!=source_hashes(): raise ValueError('arm provenance')
    for n,h in m['files'].items():
        if fd.sha256(out/n)!=h: raise ValueError('arm seal mismatch '+n)
    return m

def evaluate(start,end,out):
    check_range(start,end); base.configure_determinism(1); read_arm(out,start,end)
    _,_,x,y,*_=cr.data_split(); labels=torch.tensor(y); clean=torch.tensor(x)
    held=[torch.tensor(base.geometric_environment(x,s)) for s in HELDOUT_SEEDS]; rows=[]; envrows=[]
    for rep in range(start,end):
        for arm in ALL_ARMS:
            path=out/'states'/f'rep{rep}_{arm}.pt'; ck=torch.load(path,map_location='cpu',weights_only=True)
            if (ck['rep'],ck['arm'],ck['protocol_hash'])!=(rep,arm,PH): raise ValueError('state metadata')
            model=SmallCNN(); model.load_state_dict(ck['state_dict']); model.eval(); vals=[]
            with torch.no_grad():
                ca=float((model(clean)[0].argmax(1)==labels).float().mean())
                for s,xx in zip(HELDOUT_SEEDS,held):
                    acc=float((model(xx)[0].argmax(1)==labels).float().mean()); vals.append(acc)
                    envrows.append({'rep':rep,'arm':arm,'env_seed':s,'accuracy':acc})
            v=np.array(vals,np.float64); rows.append({'rep':rep,'arm':arm,'mean_test':float(v.mean()),'sd_test':float(v.std(ddof=1)),
                'p10_test':float(np.quantile(v,.1)),'min_test':float(v.min()),'clean_test':ca,'state_digest':pm.state_digest(ck['state_dict'])})
        print('EVALUATED',rep,flush=True)
    read_arm(out,start,end)
    pd.DataFrame(rows).to_csv(out/f'spatial_heldout_{start}_{end}.csv',index=False)
    pd.DataFrame(envrows).to_csv(out/f'spatial_environment_{start}_{end}.csv.gz',index=False,compression='gzip')
    write_json(out/'evaluation_runtime.json',fd.runtime())

def estimate(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all(): raise ValueError('invalid statistical sample')
    mean=float(x.mean()); se=float(stats.sem(x)); k=float(stats.t.ppf(.975,len(x)-1))
    p=float(stats.t.sf(mean/se,len(x)-1)) if se>0 else (0. if mean>0 else 1. if mean<0 else .5)
    return {'n':len(x),'mean':mean,'se':se,'k':k,'ci95_low':mean-k*se,'ci95_high':mean+k*se,'p_one_sided':p,'positive_pairs':int((x>0).sum())}

def equivalent(x):
    e=estimate(x); mean,se=e['mean'],e['se']; k=float(stats.t.ppf(.95,e['n']-1))
    lo=mean-k*se;hi=mean+k*se
    pl=float(stats.t.sf((mean+C)/se,e['n']-1)) if se>0 else (0. if mean>-C else 1.)
    pu=float(stats.t.cdf((mean-C)/se,e['n']-1)) if se>0 else (0. if mean<C else 1.)
    return {'mean':mean,'se':se,'ci90_low':lo,'ci90_high':hi,'p_lower':pl,'p_upper':pu,'pass':bool(lo>-C and hi<C and pl<.05 and pu<.05)}

def decide(gates,gp,tp):
    if not gates:return 'MATCH OR MANIPULATION FAILURE'
    if gp and tp:return 'SPATIAL ALLOCATION ALTERNATIVE SUPPORTED'
    if gp:return 'GRADIENT REPLICATION ONLY / SPATIAL ALTERNATIVE NOT SUPPORTED'
    if tp:return 'SPATIAL EFFECT ONLY / GRADIENT REPLICATION FAILED'
    return 'NO PERFORMANCE REPLICATION'

def load(root,pattern):
    ps=sorted(root.rglob(pattern))
    if not ps:raise FileNotFoundError(pattern)
    return pd.concat([pd.read_csv(p,float_precision='round_trip') for p in ps],ignore_index=True)

def validate_manifests(root,stage):
    covered=[]; manifests=[]
    for p in sorted(root.rglob('reference_manifest.json')):
        m=json.loads(p.read_text()); start,end=m['start'],m['end'];check_range(start,end)
        read_ref(p.parent,start,end)
        if stage=='arms':read_arm(p.parent,start,end)
        covered.extend(range(start,end)); manifests.append({'start':start,'end':end,'reference_sha256':fd.sha256(p),
          'arm_sha256':fd.sha256(p.parent/'arm_manifest.json') if stage=='arms' else None})
    if sorted(covered)!=list(REPS):raise ValueError('global replicate coverage')
    return manifests

def seal(root,out,stage):
    manifests=validate_manifests(root,stage);out.mkdir(parents=True,exist_ok=True)
    write_json(out/f'{stage}_seal.json',{'protocol_hash':PH,'stage':stage,'shards':manifests,'reps':REPS})
    print('GLOBAL_SEAL',stage,PH,flush=True)

def summarize(root,out):
    validate_manifests(root,'arms')
    steps=load(root,'spatial_reference_*.csv.gz'); held=load(root,'spatial_heldout_*.csv');env=load(root,'spatial_environment_*.csv.gz')
    checked_grid(steps,['rep','step','family'],{(r,t,f) for r in REPS for t in range(STEPS) for f in ('g','t')})
    checked_grid(held,['rep','arm'],{(r,a) for r in REPS for a in ALL_ARMS})
    checked_grid(env,['rep','arm','env_seed'],{(r,a,s) for r in REPS for a in ALL_ARMS for s in HELDOUT_SEEDS})
    finite(steps,['gradient_gap','translation_gap','hardness_diff','parameter_z_diff']);finite(held,['mean_test','sd_test','p10_test','min_test','clean_test']);finite(env,['accuracy'])
    if (steps.hardness_diff.abs()>C+1e-10).any() or (steps.parameter_z_diff.abs()>C+1e-10).any():raise ValueError('per-step caliper failure')
    err=0.; state_count=0
    for rp in root.rglob('reference_manifest.json'):
        m=json.loads(rp.read_text())
        for r in range(m['start'],m['end']):
            for a in ALL_ARMS:
                ck=torch.load(rp.parent/'states'/f'rep{r}_{a}.pt',map_location='cpu',weights_only=True)
                row=held[(held.rep==r)&(held.arm==a)].iloc[0]
                if pm.state_digest(ck['state_dict'])!=row.state_digest:raise ValueError('state tensor digest')
                state_count+=1
                v=env[(env.rep==r)&(env.arm==a)].accuracy.to_numpy()
                for name,value in [('mean_test',v.mean()),('sd_test',v.std(ddof=1)),('p10_test',np.quantile(v,.1)),('min_test',v.min())]:
                    diff=abs(value-float(row[name]));err=max(err,diff)
                    if diff>1e-12:raise ValueError('environment reaggregation')
    numeric=[c for c in steps.columns if c not in ('rep','step','family','high_envs','low_envs')]
    ref=steps.groupby(['rep','family'])[numeric].mean().reset_index(); summaries=[];balances=[];paired=[]
    hp=held.set_index(['rep','arm']);rr=ref.set_index(['rep','family'])
    for r in REPS:
        d={'rep':r}
        for f in ('g','t'):
            for key in ['gradient_gap','translation_gap','hardness_diff','parameter_z_diff']:
                d[f+'_'+key]=float(rr.loc[(r,f),key])
            for metric in ['mean_test','clean_test','p10_test','min_test']:
                d[f+'_'+metric+'_benefit']=float(hp.loc[(r,f+'_high'),metric]-hp.loc[(r,f+'_low'),metric])
            for suffix in ['high','low']:
                d[f+'_'+suffix+'_vs_reference']=float(hp.loc[(r,f+'_'+suffix),'mean_test']-hp.loc[(r,'reference'),'mean_test'])
        d['gradient_minus_spatial_benefit']=d['g_mean_test_benefit']-d['t_mean_test_benefit'];paired.append(d)
    paired=pd.DataFrame(paired);gates=True
    for f in ('g','t'):
        metric=f+('_gradient_gap' if f=='g' else '_translation_gap');m=estimate(paired[metric]);manip=m['mean']>0 and m['p_one_sided']<.05;gates &= manip
        summaries.append({'endpoint':metric,**m})
        for key in ['hardness_diff','parameter_z_diff']:
            bal=equivalent(paired[f+'_'+key]);gates &= bal['pass'];balances.append({'endpoint':f+'_'+key,**bal})
    g=estimate(paired.g_mean_test_benefit);t=estimate(paired.t_mean_test_benefit)
    gp=g['mean']>0 and g['p_one_sided']<.05;tp=t['mean']>0 and t['p_one_sided']<.05
    for col in paired.columns:
        if col!='rep' and ('benefit' in col or 'vs_reference' in col):summaries.append({'endpoint':col,**estimate(paired[col])})
    out.mkdir(parents=True,exist_ok=True)
    paired.to_csv(out/'spatial_paired30.csv',index=False);ref.to_csv(out/'spatial_reference_summary30.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'spatial_summary30.csv',index=False);pd.DataFrame(balances).to_csv(out/'spatial_balance30.csv',index=False)
    held.to_csv(out/'spatial_heldout30.csv',index=False)
    decision={'decision':decide(gates,gp,tp),'n':30,'all_gates_pass':bool(gates),'gradient_performance_pass':bool(gp),'spatial_performance_pass':bool(tp),'protocol_hash':PH,
              'states_checked':state_count,'environment_rows':len(env),'max_environment_error':err}
    pd.DataFrame([decision]).to_csv(out/'spatial_decision30.csv',index=False);print(json.dumps(decision),flush=True)

def selftest():
    base.configure_determinism(1)
    losses=torch.linspace(2.,.5,16);gen=torch.Generator().manual_seed(3);g=torch.randn(16,12,generator=gen);g=g/g.norm(dim=1,keepdim=True)
    par=np.random.default_rng(2).normal(size=(64,7));rows=cal.feasible_subsets(losses,g@g.T,list(range(16)),par)
    for r in rows:r['translation_score']=float(par[list(r['envs']),1:3].var(0).sum())
    assert choose(rows,'gradient_novelty')==cal.best_pair(rows,C)
    a=choose(rows,'translation_score');assert a is not None
    r2=[dict(r,gradient_novelty=123-i) for i,r in enumerate(rows)]
    assert choose(rows,'translation_score')==choose(r2,'translation_score') # T independent of gradients
    for f in ['gradient_novelty','translation_score']:
        pair=choose(rows,f);assert pair is not None and abs(pair['hardness_diff'])<=C+1e-12 and abs(pair['parameter_z_diff'])<=C+1e-12
    assert equivalent(np.zeros(30))['pass'] and not equivalent(np.full(30,.06))['pass']
    assert decide(True,True,True)=='SPATIAL ALLOCATION ALTERNATIVE SUPPORTED'
    assert decide(True,True,False).startswith('GRADIENT REPLICATION ONLY')
    try:estimate([1,np.nan]);raise AssertionError('nonfinite accepted')
    except ValueError:pass
    try:checked_grid(pd.DataFrame({'a':[1,1]}),['a'],{(1,),(2,)});raise AssertionError('duplicate accepted')
    except ValueError:pass
    print('SELFTEST PASS',PH)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('selftest','reference','train','evaluate','seal','summarize'))
    p.add_argument('--start',type=int);p.add_argument('--end',type=int);p.add_argument('--output-dir');p.add_argument('--input-dir');p.add_argument('--stage',choices=('reference','arms'));a=p.parse_args()
    if a.mode=='selftest':selftest();return
    if not a.output_dir:raise ValueError('output directory required')
    out=Path(a.output_dir)
    if a.mode in ('reference','train','evaluate'):{'reference':reference,'train':train,'evaluate':evaluate}[a.mode](a.start,a.end,out)
    elif not a.input_dir:raise ValueError('input directory required')
    elif a.mode=='seal':seal(Path(a.input_dir),out,a.stage)
    else:summarize(Path(a.input_dir),out)
if __name__=='__main__':main()

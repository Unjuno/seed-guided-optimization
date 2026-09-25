"""Independent Issue115 archive audit. Never trains or evaluates a model.

Checks hashes, frozen grids, checkpoint tensor digests, all subset-pair choices,
reference contrasts, paired t intervals, TOST balance and the frozen decision.
Raw gradients/images are not reconstructed; the archived identity residual is
only threshold-checked. Python, NumPy, Pandas, SciPy and Torch are required.
"""
from __future__ import annotations
import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import torch

PH = 'e27c10bd5643f173632387011a41f703686e2d47f9cbb7b4cb58d47bfccd509a'
REPS = set(range(4100, 4160))
FIELDS = ['mean_rank','z_loss_variance','z_param','translation_score',
          'gradient_novelty','centered_erank','opposition_score','z_hard']
DIFFS = ['rank_gap','loss_variance_z_diff','parameter_z_diff','translation_diff',
         'gradient_novelty_diff','span_diff','opposition_diff','hardness_diff']
MARGINS = dict(zip(DIFFS[1:7], [.05,.05,.20,.05,.10,.15]))
II, JJ = np.triu_indices(455, 1)

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()

def require(condition, message: str):
    if not bool(condition): raise ValueError(message)

def ints(text): return tuple(map(int, str(text).split(';')))

def read(p): return pd.read_csv(p, float_precision='round_trip')

def est(x):
    x=np.asarray(x,float); require(x.ndim==1 and np.isfinite(x).all() and len(x)>1,'sample')
    mean=float(x.mean()); se=float(x.std(ddof=1)/np.sqrt(len(x))); k=float(stats.t.ppf(.975,len(x)-1))
    p=float(stats.t.sf(mean/se,len(x)-1)) if se else (0. if mean>0 else 1. if mean<0 else .5)
    return dict(n=len(x),mean=mean,se=se,k=k,ci95_low=mean-k*se,ci95_high=mean+k*se,
                p_one_sided=p,positive_pairs=int((x>0).sum()))

def audit(root: Path, out: Path):
    covered=[]; srows=[]; erows=[]; hrows=[]; num_subset=0; num_pairs=0; hashes=0; states=0; pairerr=0.
    cpus={s:set() for s in ('reference','arm','evaluation')}; sources=None; split=None
    seal=json.loads((root/'global/arms_seal.json').read_text())
    require(seal['protocol_hash']==PH,'global protocol')
    for mp in sorted((root/'downloaded').glob('*/reference_manifest.json')):
        d=mp.parent; ref=json.loads(mp.read_text()); start,end=ref['start'],ref['end']; covered.extend(range(start,end))
        require(ref['protocol_hash']==PH and not ref['heldout_constructed'] and ref['intervention_arms_trained']==0,'reference boundary')
        require(hashlib.sha256(json.dumps(ref['protocol'],sort_keys=True).encode()).hexdigest()==PH,'protocol digest')
        if sources is None: sources=ref['source_hashes'];split=ref['split_hashes']
        require(ref['source_hashes']==sources and ref['split_hashes']==split,'cross-shard provenance')
        for name,digest in sources.items(): require(sha(d/'source'/name)==digest,'source hash '+name);hashes+=1
        manifests={stage:json.loads((d/(stage+'_manifest.json')).read_text()) for stage in ('reference','arm','evaluation')}
        require(manifests['arm']['reference_manifest_sha256']==sha(mp),'reference chain')
        require(manifests['evaluation']['arm_manifest_sha256']==sha(d/'arm_manifest.json'),'arm chain')
        require(manifests['evaluation']['global_arms_seal_sha256']==sha(root/'global/arms_seal.json'),'global chain')
        require(not manifests['arm']['heldout_constructed'],'arm boundary')
        for stage,m in manifests.items():
            require(m['protocol_hash']==PH,'manifest protocol'); cpus[stage].add(m['runtime']['cpu_model'])
            for name,digest in m['files'].items(): require(sha(d/name)==digest,'file hash '+name); hashes+=1
        steps=read(d/f'rank_allocation_reference_{start}_{end}.csv.gz').set_index(['rep','step'])
        subs=read(d/f'rank_allocation_subsets_{start}_{end}.csv.gz')
        require(len(subs)==(end-start)*80*455,'subset count')
        require(not subs.duplicated(['rep','step','subset_id']).any(),'duplicate subsets')
        for key,b in subs.groupby(['rep','step'],sort=True):
            b=b.sort_values('subset_id'); require(b.subset_id.tolist()==list(range(455)),'subset grid')
            a=b[FIELDS].to_numpy(float); require(np.isfinite(a).all(),'subset finite')
            require(b.general_identity_error.abs().max()<=1e-10,'identity gate')
            locs=[ints(s) for s in b.local]; ranks=[ints(s) for s in b.ranks]; envs=[ints(s) for s in b.envs]
            require(all(len(set(t))==4 and 1 in r for t,r in zip(locs,ranks)),'subset shape/anchor')
            require(np.array_equal(a[:,0],np.mean(ranks,axis=1)),'rank reconstruction')
            delta=a[II]-a[JJ]
            good=np.flatnonzero(np.all(np.abs(delta[:,1:7])<=np.asarray([.05,.05,.20,.05,.10,.15])+1e-12,axis=1))
            row=steps.loc[key]; require(bool(row.supported)==(len(good)>0),'support reconstruction')
            if len(good):
                # Independently implement the published lexicographic ordering.
                z=min(good,key=lambda k:(-abs(delta[k,0]),abs(delta[k,6]),abs(delta[k,5]),abs(delta[k,4]),abs(delta[k,3]),abs(delta[k,2]),abs(delta[k,1]),locs[II[k]],locs[JJ[k]]))
                i,j=int(II[z]),int(JJ[z]); hi,lo=(i,j) if a[i,0]>=a[j,0] else (j,i)
                require(int(row.eligible_pairs)==len(good),'eligible count')
                require(ints(row.high_envs)==envs[hi] and ints(row.low_envs)==envs[lo],'optimal pair')
                err=float(np.max(np.abs(a[hi]-a[lo]-row[DIFFS].to_numpy(float))));pairerr=max(pairerr,err)
                require(err<=1e-12,'reference contrast');num_pairs+=1
            else:
                top=next(i for i,r in enumerate(ranks) if set(r)=={1,2,3,4})
                require(ints(row.high_envs)==envs[top] and ints(row.low_envs)==envs[top],'fallback subset')
                require((row[DIFFS]==0).all(),'fallback zero')
            num_subset+=455
        h=read(d/f'centered_span_heldout_{start}_{end}.csv'); e=read(d/f'centered_span_environment_{start}_{end}.csv.gz')
        hz=h.set_index(['rep','arm'])
        for rep in range(start,end):
            schedule=json.loads((d/'schedules'/f'rep{rep}.json').read_text())
            for arm in ('high','low'):
                require(len(schedule[arm])==80,'schedule length')
                for step,chosen in enumerate(schedule[arm]): require(tuple(chosen)==ints(steps.loc[(rep,step),arm+'_envs']),'schedule/pair')
            for arm in ('high','low','reference'):
                p=d/'states'/f'rep{rep}_{arm}.pt';ck=torch.load(p,weights_only=True,map_location='cpu'); hsh=hashlib.sha256()
                require((ck['rep'],ck['arm'],ck['protocol_hash'])==(rep,arm,PH),'state metadata')
                require(ck['schedule_sha256']==sha(d/'schedules'/f'rep{rep}.json'),'state schedule')
                for name,t in sorted(ck['state_dict'].items()):
                    a=t.detach().cpu().contiguous().numpy();require(np.isfinite(a).all(),'state finite')
                    hsh.update(name.encode());hsh.update(str(a.dtype).encode());hsh.update(str(a.shape).encode());hsh.update(a.tobytes())
                require(hsh.hexdigest()==hz.loc[(rep,arm),'state_digest'],'tensor digest'); states+=1
        srows.append(steps.reset_index()); hrows.append(h); erows.append(e)
        print('AUDITED',start,end,flush=True)
    require(sorted(covered)==sorted(REPS),'rep coverage')
    steps=pd.concat(srows);held=pd.concat(hrows);env=pd.concat(erows)
    require(len(steps)==4800 and not steps.duplicated(['rep','step']).any(),'step grid')
    require(len(held)==180 and not held.duplicated(['rep','arm']).any(),'heldout grid')
    require(len(env)==14400 and not env.duplicated(['rep','arm','env_seed']).any(),'environment grid')
    require(set(env.env_seed)==set(range(102000,102080)),'environment IDs')
    hz=held.set_index(['rep','arm']);env_error=0.
    for key,b in env.groupby(['rep','arm']):
        a=b.sort_values('env_seed').accuracy.to_numpy(float);require(len(a)==80 and ((a>=0)&(a<=1)).all(),'accuracy domain')
        obs=np.array([a.mean(),a.std(ddof=1),np.quantile(a,.1),a.min()]);stored=hz.loc[key,['mean_test','sd_test','p10_test','min_test']].to_numpy(float)
        env_error=max(env_error,float(np.max(np.abs(obs-stored))))
    require(env_error<=1e-12,'environment reaggregation')
    ref=steps.groupby('rep')[DIFFS].mean();pairs=[]
    for rep in sorted(REPS):
        row={'rep':rep,**ref.loc[rep].to_dict()}
        for met in ('mean_test','min_test','p10_test','clean_test','sd_test'):
            row[met+'_benefit']=float(hz.loc[(rep,'high'),met]-hz.loc[(rep,'low'),met])
        pairs.append(row)
    paired=pd.DataFrame(pairs);summary={k:est(paired[k]) for k in paired if k!='rep'}
    balances=[]
    for col,m in MARGINS.items():
        e=summary[col];k=stats.t.ppf(.95,59);lo=e['mean']-k*e['se'];hi=e['mean']+k*e['se']
        pl=stats.t.sf((e['mean']+m)/e['se'],59);pu=stats.t.cdf((e['mean']-m)/e['se'],59)
        balances.append({'endpoint':col,'margin':m,'ci90_low':lo,'ci90_high':hi,'p_lower':pl,'p_upper':pu,'pass':bool(lo>-m and hi<m and pl<.05 and pu<.05)})
    require(all(b['pass'] for b in balances),'balance gate')
    sup=steps.groupby('rep').supported.mean();require(sup.mean()>=.9 and sup.min()>=.8,'support gate')
    require(summary['rank_gap']['mean']>=1.5 and summary['rank_gap']['p_one_sided']<.05,'rank gate')
    mean_pass=summary['mean_test_benefit']['mean']>0 and summary['mean_test_benefit']['p_one_sided']<.05
    neg=est(-paired.min_test_benefit);tail_pass=neg['mean']>0 and neg['p_one_sided']<.05
    decision=('MEAN-TAIL TRADEOFF SUPPORT' if mean_pass and tail_pass else 'MEAN BENEFIT REPLICATED / TAIL DEGRADATION NOT CONFIRMED' if mean_pass else 'TAIL DEGRADATION ONLY / MEAN BENEFIT NOT REPLICATED' if tail_pass else 'NO MEAN-TAIL TRADEOFF REPLICATION')
    require(read(root/'aggregate/rank_tail_decision60.csv').iloc[0].decision==decision,'decision mismatch')
    official=read(root/'aggregate/rank_tail_primary60.csv').set_index('rep'); errors=[]
    for col in paired:
        if col!='rep':errors.append(float(np.max(np.abs(paired.set_index('rep')[col]-official[col]))))
    require(max(errors)<=1e-12,'paired mismatch')
    out.mkdir(parents=True,exist_ok=True)
    report={'audit':'PASS','experiment_decision':decision,'n':60,'subset_rows_checked':num_subset,'optimal_pairs_checked':num_pairs,'fallback_steps':int((~steps.supported).sum()),'states_checked':states,'hash_checks':hashes,'max_pair_error':pairerr,'max_environment_error':env_error,'max_paired_error':max(errors),'mean':summary['mean_test_benefit'],'minimum':summary['min_test_benefit'],'p_one_sided_minimum_negative':neg['p_one_sided'],'clean_exploratory':summary['clean_test_benefit'],'sd_exploratory':summary['sd_test_benefit'],'cpus':{s:sorted(v) for s,v in cpus.items()},'audit_runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,'torch':torch.__version__},'limits':'No raw-image inference, retraining, raw-gradient reconstruction, or cross-hardware replication.'}
    (out/'rank_tail_independent_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    pd.DataFrame([{'endpoint':k,**v} for k,v in summary.items()]).to_csv(out/'rank_tail_independent_statistics.csv',index=False)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();audit(a.input_dir,a.output_dir)

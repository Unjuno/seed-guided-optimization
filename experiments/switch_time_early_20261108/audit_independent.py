"""Independent raw-evidence audit of switch-time early study.

Does not use run_switch_early.analyze/holm or its pooled metrics. Tests:
source+manifest hashes, 192 tensor checkpoints, all shared-prefix traces,
full integer-grid raw counts, all 12-pool primary t11/Holm, clean guardrail,
and exact 4096-sign flip sensitivity (a post-hoc diagnostic).
"""
from __future__ import annotations
from collections import defaultdict
import csv, hashlib, itertools, json, math, os, statistics
from pathlib import Path
import numpy as np
from scipy.stats import t
import torch

ROOT=Path(__file__).resolve().parent
POLICIES=('M','T40','T60','T80')
SWITCH={'M':201,'T40':40,'T60':60,'T80':80}
CONTRASTS={'T60_vs_M':('M','T60'),'T60_vs_T40':('T40','T60'),'T60_vs_T80':('T80','T60')}


def req(ok,msg):
    if not ok:raise ValueError(msg)

def sha(path):
    with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def readrows(path):
    with open(path,newline='') as f: return list(csv.DictReader(f))

def means_stat(x, null=0, alternative='greater'):
    x=np.asarray(x,dtype=np.float64)
    mu=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(len(x)))
    req(se>0 and np.isfinite(se),'degenerate or nonfinite SE')
    z=(mu-null)/se
    p=float(t.sf(z,len(x)-1) if alternative=='greater' else t.cdf(z,len(x)-1))
    width=float(t.ppf(.975,len(x)-1)*se)
    return dict(mean=mu,se=se,ci_low=mu-width,ci_high=mu+width,p=p,positive=int((x>0).sum()))

def holm(probs):
    order=sorted(probs,key=lambda k:(probs[k],k))
    out={};prev=0
    for pos,name in enumerate(order):
        prev=max(prev,(len(probs)-pos)*probs[name]);out[name]=min(1.,prev)
    return out

def sf_exact(vec,null=0,alternative='greater'):
    x=np.asarray(vec,dtype=np.float64)-null
    obs=x.mean()
    count=0
    for signs in itertools.product((-1,1),repeat=len(x)):
        val=float(np.dot(np.asarray(signs,dtype=np.float64),x)/len(x))
        count+=int(val>=obs-1e-13 if alternative=='greater' else val<=obs+1e-13)
    return count / (2**len(x))


def compute():
    os.environ.setdefault('OMP_NUM_THREADS','1')
    torch.set_num_threads(1)
    frozen=json.loads((ROOT/'runs/freeze.json').read_text())
    seal=json.loads((ROOT/'runs/GLOBAL_SEAL.json').read_text())
    for p,d in frozen['sources'].items(): req(sha(ROOT/p)==d,'source drift '+p)
    req(frozen['config_hash']==seal['config_hash'],'config source mismatch')
    req(frozen['data_hashes']==seal['data_hashes'],'data split drift')
    req(seal['heldout_constructed'] is False,'seal behind evaluation')
    req(seal['states']==192 and len(seal['records'])==192,'global incomplete states')
    global_record={(int(q['pool']),int(q['seed']),q['policy']):q for q in seal['records']}
    req(len(global_record)==192,'duplicate records')
    check_trace=0;shared_prefix_pairs=0
    for p in range(12):
        td=ROOT/'runs/train'/f'pool{p}'
        m=json.loads((td/'manifest.json').read_text())
        req(m['heldout_constructed'] is False,'train exposed eval')
        req(m['sources']==frozen['sources'] and m['data_hashes']==frozen['data_hashes'],'training hashes drift')
        for fname,digest in m['files'].items():req(sha(td/fname)==digest,'training file drift')
        sr=readrows(td/'states.csv');trace=readrows(td/'trace.csv')
        req(len(sr)==16 and len(trace)==3200,'incomplete per-pool training')
        req({(int(x['seed']),x['policy']) for x in sr}=={(s,q) for s in range(4) for q in POLICIES},'train arm grid')
        req({(int(x['seed']),x['policy'],int(x['step'])) for x in trace}=={(s,q,j) for s in range(4) for q in POLICIES for j in range(1,201)},'train trace grid')
        d0={}
        for s in range(4):
            rows=[x for x in sr if int(x['seed'])==s]
            req(len({x['initial_digest'] for x in rows})==1,'initial weight mismatch')
            for r in rows:
                q=r['policy'];key=(p,s,q)
                req(int(r['selected_adam_steps'])==200-min(200,SWITCH[q]),'wrong policy A steps')
                req(int(r['selected_muon_steps'])==min(200,SWITCH[q]),'wrong policy M steps')
                f=td/r['file'];req(sha(f)==r['file_sha']==global_record[key]['file_sha'],'file SHA mismatch')
                cp=torch.load(f,map_location='cpu',weights_only=True)
                req(cp['heldout_constructed'] is False and cp['steps']==200,'invalid checkpoint')
                req(cp['initial_digest']==r['initial_digest'],'wrong checkpoint initial')
                # Check optimizer full state against original seal digest (value digest function hash).
                from sys import path as sys_path
                sys_path.insert(0,str(ROOT/'code'))
                from switch_engine import tree_digest
                digest=tree_digest(cp['engine'])
                req(digest==r['engine_digest']==global_record[key]['engine_digest'],'optimizer state tensor mismatch')
            tr={q:{int(x['step']):x for x in trace if int(x['seed'])==s and x['policy']==q} for q in POLICIES}
            for q in POLICIES:
                for step,x in tr[q].items():
                    expect='M' if (step-1)<SWITCH[q] else 'A'
                    req(x['active']==expect,'active optimizer mismatch')
                    for fld in ('grad_norm','update_norm_a','update_norm_m','update_cosine','train_ce'):
                        req(math.isfinite(float(x[fld])),'nonfinite training diagnostics')
            for a,b in itertools.combinations(POLICIES,2):
                k=min(SWITCH[a],SWITCH[b],200)
                for step in range(1,k+1):
                    for field in ('active','train_ce','grad_norm','update_norm_a','update_norm_m','update_cosine'):
                        req(tr[a][step][field]==tr[b][step][field],f'common prefix mismatch pool {p} seed {s} {a}-{b} step{step} {field}')
                    shared_prefix_pairs+=1
        check_trace+=len(trace)
    req(check_trace==38400,'wrong applied optimizer steps')
    evalrows={};cleanrows={};eval_count=0
    for p in range(12):
        d=ROOT/'runs/eval'/f'pool{p}'
        m=json.loads((d/'manifest.json').read_text())
        req(m['heldout_constructed'] is True and m['seal_sha']==seal['seal_sha'],'eval global seal')
        for name,dig in m['files'].items():req(sha(d/name)==dig,'eval file changed')
        shift=readrows(d/'heldout_counts.csv');clean=readrows(d/'clean_counts.csv')
        req(len(shift)==768 and len(clean)==16,'eval grid length')
        keys=set();nkeys=set()
        for z in shift:
            seed=int(z['seed']);q=z['policy'];j=int(z['environment']);k=(p,seed,q,j)
            req((q in POLICIES) and seed in range(4) and j in range(48) and k not in keys,'invalid/duplicate eval key')
            req(int(z['env_key'])==2105000+20000*p+j,'unexpected test environment key')
            n=int(z['n']);acc=int(z['correct']);loss=float(z['ce_sum'])
            req(n==809 and 0<=acc<=n and math.isfinite(loss) and loss>=0,'invalid count/CE')
            evalrows[k]=loss;keys.add(k)
        req(len(keys)==768,'missing eval cells')
        for z in clean:
            seed=int(z['seed']);q=z['policy'];k=(p,seed,q)
            req(q in POLICIES and seed in range(4) and k not in nkeys,'invalid/duplicate clean key')
            n=int(z['n']);acc=int(z['correct']);loss=float(z['ce_sum'])
            req(n==809 and 0<=acc<=n and math.isfinite(loss) and loss>=0,'invalid clean count/CE')
            cleanrows[k]=loss;nkeys.add(k)
        req(len(nkeys)==16,'missing clean cells')
        eval_count+=len(shift)
    req(eval_count==9216 and len(cleanrows)==192,'bad pooled eval records')
    pooled={}
    clean_pooled={}
    for p in range(12):
        for q in POLICIES:
            pooled[p,q]=math.fsum(evalrows[p,s,q,j] for s in range(4) for j in range(48))/(4*48*809)
            clean_pooled[p,q]=math.fsum(cleanrows[p,s,q] for s in range(4))/(4*809)
    comps={}
    for name,(base,target) in CONTRASTS.items():
        vec=[pooled[p,base]-pooled[p,target] for p in range(12)]
        out=means_stat(vec)
        out.update(vector=vec,p_signflip=sf_exact(vec))
        comps[name]=out
    corrected=holm({name:z['p'] for name,z in comps.items()})
    for name,z in comps.items():
        z['p_holm']=corrected[name]
        z['decision']='SUPPORT' if z['mean']>=.01 and z['p_holm']<.05 else 'NOT SUPPORTED'
    cv=[clean_pooled[p,'T60']-clean_pooled[p,'M'] for p in range(12)]
    guard=means_stat(cv,null=.05,alternative='less')
    guard['vector']=cv
    guard['p_signflip']=sf_exact(cv,null=.05,alternative='less')
    guard['decision']='PASS' if guard['mean']<.05 and guard['p']<.05 else 'NOT SUPPORTED'
    results=dict(n_pools=12,n_models=192,n_train_updates=check_trace,n_common_prefix_pair_steps=shared_prefix_pairs,
                 n_heldout_rows=eval_count,n_clean_rows=len(cleanrows),primary=comps,clean_guardrail=guard,
                 mean_shifted_ce={q:statistics.fmean([pooled[p,q] for p in range(12)]) for q in POLICIES},
                 mean_clean_ce={q:statistics.fmean([clean_pooled[p,q] for p in range(12)]) for q in POLICIES},
                 pool_shifted_ce={str(p):{q:pooled[p,q] for q in POLICIES} for p in range(12)},
                 pool_clean_ce={str(p):{q:clean_pooled[p,q] for q in POLICIES} for p in range(12)},
                 frozen_source_sha=frozen['sources'],sealed_file_sha=sha(ROOT/'runs/GLOBAL_SEAL.json'))
    # Compare with original trial's frozen output as an independent-arithmetic cross-check.
    a_path=ROOT/'results/summary.json'
    if a_path.exists():
        recorded=json.loads(a_path.read_text())
        for record in recorded['primary']:
            z=comps[record['contrast']]
            for src,dst in [('mean_ce_improvement','mean'),('se','se'),('ci95_low','ci_low'),('ci95_high','ci_high'),('p_raw','p'),('p_holm','p_holm')]:
                req(abs(float(record[src])-z[dst])<1e-10,'preregistered stat mismatch '+src)
            req(record['decision']==z['decision'],'preregistered outcome discrepancy')
        for src,dst in [('mean_clean_ce_penalty','mean'),('one_sided_p','p'),('ci95_low','ci_low'),('ci95_high','ci_high')]:
            req(abs(float(recorded['clean_guardrail'][src])-guard[dst])<1e-10,'clean guardrail discrepancy '+src)
        req(recorded['clean_guardrail']['decision']==guard['decision'],'clean label mismatch')
    return results

if __name__=='__main__':
    r=compute()
    out=ROOT/'results/INDEPENDENT_AUDIT.json'
    if out.exists():raise SystemExit('refuse to overwrite audit')
    out.write_text(json.dumps(r,indent=2,sort_keys=True,allow_nan=False))
    print('INDEPENDENT_AUDIT_PASS',r['n_models'],'models,',r['n_train_updates'],'train steps,',r['n_heldout_rows'],'heldout rows,',r['n_common_prefix_pair_steps'],'shared-prefix pair steps')
    for name,x in r['primary'].items():print(name,x['mean'],x['ci_low'],x['ci_high'],x['p_holm'],x['decision'])
    print('CLEAN_GUARDRAIL',r['clean_guardrail']['mean'],r['clean_guardrail']['p'],r['clean_guardrail']['decision'])

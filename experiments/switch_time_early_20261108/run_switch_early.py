"""Prospectively freeze and test fixed earlier fixed switch-time boundaries.

Four policies all share the initial Muon path and an identical 200-step schedule.
Training/test boundaries are enforced on disk. No online tuning.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from datetime import datetime,timezone

sys.path.insert(0,str(Path(__file__).resolve().parent/'code'))
import numpy as np
import torch
from scipy.stats import t
import legacy_study as legacy
import common
from switch_engine import ShadowEngine,tree_digest

ROOT=Path(__file__).resolve().parent
POLICIES=('M','T40','T60','T80')
SWITCH_AT={'M':201,'T40':40,'T60':60,'T80':80}
COMPARISONS={'T60_vs_M':('M','T60'),'T60_vs_T40':('T40','T60'),'T60_vs_T80':('T80','T60')}
CLEAN_MARGIN=0.05
CONFIG=dict(study='switch-time-early-20261108-v1',n_pools=12,n_initializations=4,
            steps=200,batch=100,n_fit=800,n_test=809,
            fit_environment_count=64,test_environment_count=48,
            eval_images='historical_809_reused',
            policies=POLICIES,switch_at=SWITCH_AT,
            init_base=13800000000,init_stride=10103,
            schedule_base=20900000000,schedule_stride=1423,
            env_base=2100000,env_stride=20000,fit_offset=0,test_offset=5000,
            alpha=.05,holm_family=3,minimum_point_improvement_CE=.01,
            epoch_count=25,train_size=800)
CONFIG_HASH=hashlib.sha256(json.dumps(CONFIG,sort_keys=True).encode()).hexdigest()
SOURCE_PATHS=('docs/SWITCH_TIME_EARLY_PROTOCOL_20261108.md','run_switch_early.py',
              'code/common.py','code/switch_engine.py','code/legacy_study.py','tests/test_early.py')
NPOOLS=12; NSEEDS=4

def active_optimizer(policy:str,zero_index_update:int)->str:
    require(policy in SWITCH_AT and 0<=zero_index_update<200,'invalid policy/step')
    return 'M' if zero_index_update<SWITCH_AT[policy] else 'A'

def now():return datetime.now(timezone.utc).isoformat()
def sha_bytes(x):return hashlib.sha256(x).hexdigest()
def sha_file(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def sha_array(x):return sha_bytes(np.ascontiguousarray(x).tobytes())
def sources():return {p:sha_file(ROOT/p) for p in SOURCE_PATHS}
def require(condition,msg):
    if not condition:raise ValueError(msg)
def write_json(path,content):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(content,f,indent=2,sort_keys=True,allow_nan=False)
def write_csv(path,rows):
    require(bool(rows),'empty rows')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))

def setup():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

def backend():
    return dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                device='cpu',threads=torch.get_num_threads(),
                mkldnn=torch.backends.mkldnn.enabled,
                clock='uncontrolled',cpu=platform.processor(),
                thread_env={x:os.environ.get(x) for x in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})

def init_key(pool,seed):
    require(pool in range(NPOOLS) and seed in range(NSEEDS),'unregistered cohort')
    return CONFIG['init_base'] + CONFIG['init_stride']*(NSEEDS*pool+seed)

def fit_keys(pool):
    return list(range(CONFIG['env_base']+CONFIG['env_stride']*pool,
                      CONFIG['env_base']+CONFIG['env_stride']*pool+64))

def test_keys(pool):
    return list(range(CONFIG['env_base']+CONFIG['env_stride']*pool+5000,
                      CONFIG['env_base']+CONFIG['env_stride']*pool+5000+48))

def schedule(pool):
    require(pool in range(NPOOLS),'bad pool')
    seed=CONFIG['schedule_base']+CONFIG['schedule_stride']*pool
    rng=np.random.default_rng(seed+1)
    gen=torch.Generator().manual_seed(seed)
    result=[]
    for epoch in range(25):
        for ix in torch.randperm(800,generator=gen).split(100):
            result.append((ix.numpy().astype(np.int64),rng.choice(64,4,replace=False).astype(np.int64)))
    require(len(result)==200,'wrong number of steps')
    return result

def schedule_sha(rows):
    h=hashlib.sha256()
    for b,e in rows:
        h.update(np.asarray(b,dtype='<i8').tobytes())
        h.update(np.asarray(e,dtype='<i8').tobytes())
    return h.hexdigest()

def image_data(part):
    require(part in ('fit','test'),'no control-based retuning')
    return legacy.image_data(part)

def data_hashes():
    return legacy.data_hashes()

def freeze():
    setup()
    require(not (ROOT/'runs/freeze.json').exists(),'freeze exists; do not overwrite')
    write_json(ROOT/'runs/freeze.json',dict(date=now(),config=CONFIG,config_hash=CONFIG_HASH,
                                        sources=sources(),data_hashes=data_hashes(),backend=backend(),
                                        no_heldout_generated=True))
    print('FREEZE_CREATED',CONFIG_HASH,flush=True)

def validate_freeze():
    path=ROOT/'runs/freeze.json';require(path.exists(),'no prospective freeze')
    z=json.loads(path.read_text())
    require(z['config_hash']==CONFIG_HASH and z['config']==json.loads(json.dumps(CONFIG,sort_keys=True)),'config changed after freeze')
    require(z['sources']==sources(),'source file changed after freeze')
    require(z['data_hashes']==data_hashes(),'dataset/split changed')
    b=backend()
    for x in ('python','torch','numpy','device','threads','mkldnn'):
        require(b[x]==z['backend'][x],f'backend drift {x}')
    return z

def fit_batches(pool):
    x,y=image_data('fit')
    env=torch.stack([torch.from_numpy(common.geometric_environment(x,key)) for key in fit_keys(pool)])
    z=[]
    for b,e in schedule(pool):
        tx=env[torch.from_numpy(e)][:,torch.from_numpy(b)].reshape(-1,8,8)
        labels=y[torch.from_numpy(b)].repeat(4)
        z.append((tx,labels))
    require(len(z)==200,'training fit batch count')
    return z

def train(pool):
    setup();validate_freeze()
    require(pool in range(NPOOLS),'bad training pool')
    require(not (ROOT/'runs/GLOBAL_SEAL.json').exists(),'holdout/seal already exists')
    d=ROOT/f'runs/train/pool{pool}'
    d.mkdir(parents=True,exist_ok=False)
    start=now()
    batches=fit_batches(pool);rows=[];traces=[]
    sch=schedule_sha(schedule(pool))
    for seed in range(NSEEDS):
        first=None
        for policy in POLICIES:
            engine=ShadowEngine(init_key(pool,seed))
            initial=tree_digest(engine.model.state_dict())
            if first is None:first=initial
            require(initial==first,'different initial tensors within block')
            for t,(xb,yb) in enumerate(batches):
                active=active_optimizer(policy,t)
                diag=engine.step(xb,yb,active)
                traces.append(dict(pool=pool,seed=seed,policy=policy,step=t+1,active=active,
                                   grad_norm=diag['hidden_grad_norm'],
                                   update_norm_a=diag['update_norm_a'],update_norm_m=diag['update_norm_m'],
                                   update_cosine=diag['update_cosine'],train_ce=diag['train_ce']))
            require(engine.steps==200,'incomplete path')
            path=d/f'seed{seed}_{policy}_step200.pt'
            state=engine.state_dict()
            torch.save(dict(engine=state,pool=pool,seed=seed,policy=policy,
                            steps=200,initial_digest=initial,config_hash=CONFIG_HASH,
                            schedule_sha=sch,heldout_constructed=False),path)
            rows.append(dict(pool=pool,seed=seed,policy=policy,steps=200,
                             initial_digest=initial,engine_digest=tree_digest(state),
                             file=path.name,file_sha=sha_file(path),schedule_sha=sch,
                             selected_adam_steps=max(0,200-SWITCH_AT[policy]),
                             selected_muon_steps=min(200,SWITCH_AT[policy])))
        print('TRAIN_SEED_DONE',pool,seed,flush=True)
    write_csv(d/'states.csv',rows)
    write_csv(d/'trace.csv',traces)
    write_json(d/'manifest.json',dict(pool=pool,started=start,finished=now(),
                config_hash=CONFIG_HASH,sources=sources(),data_hashes=data_hashes(),
                backend=backend(),files={p.name:sha_file(p) for p in sorted(d.iterdir()) if p.is_file()},
                schedule_sha=sch,train_keys=fit_keys(pool),
                heldout_constructed=False,models=16,steps=3200))
    print('TRAIN_POOL_DONE',pool,flush=True)

def seal():
    setup();validate_freeze()
    require(not (ROOT/'runs/GLOBAL_SEAL.json').exists(),'existing seal; cannot reseal')
    records=[]
    for pool in range(NPOOLS):
        d=ROOT/f'runs/train/pool{pool}'
        manifest=json.loads((d/'manifest.json').read_text())
        require(manifest['heldout_constructed'] is False,'test already constructed')
        require(manifest['config_hash']==CONFIG_HASH and manifest['sources']==sources(),'per-pool source drift')
        for name,digest in manifest['files'].items():require(sha_file(d/name)==digest,'file drift')
        rows=read_csv(d/'states.csv')
        require(len(rows)==16,'incomplete grid')
        require({(int(r['seed']),r['policy']) for r in rows}=={(i,p) for i in range(4) for p in POLICIES},'missing arms')
        for row in rows:
            cp=torch.load(d/row['file'],map_location='cpu',weights_only=True)
            require(sha_file(d/row['file'])==row['file_sha'],'file SHA mismatch')
            require(tree_digest(cp['engine'])==row['engine_digest'],'tensor SHA mismatch')
            require(cp['heldout_constructed'] is False and cp['steps']==200,'seal contract')
            records.append(dict(pool=pool,seed=int(row['seed']),policy=row['policy'],
                                file_sha=row['file_sha'],engine_digest=row['engine_digest']))
    require(len(records)==192,'wrong global count')
    sealhash=sha_bytes(json.dumps(records,sort_keys=True).encode())
    write_json(ROOT/'runs/GLOBAL_SEAL.json',dict(date=now(),seal_sha=sealhash,states=192,
                                                config_hash=CONFIG_HASH,sources=sources(),
                                                data_hashes=data_hashes(),
                                                heldout_constructed=False,records=records))
    print('GLOBAL_SEAL_PASS 192 FINAL STATES, HELDOUT NOT CONSTRUCTED',sealhash,flush=True)

def verify_global_seal():
    setup();validate_freeze()
    d=ROOT/'runs/GLOBAL_SEAL.json'
    require(d.exists(),'FORBIDDEN: heldout before global seal')
    g=json.loads(d.read_text())
    require(g['states']==192 and g['config_hash']==CONFIG_HASH
            and g['heldout_constructed'] is False,'wrong seal')
    for rec in g['records']:
        path=ROOT/f"runs/train/pool{rec['pool']}/seed{rec['seed']}_{rec['policy']}_step200.pt"
        require(sha_file(path)==rec['file_sha'],'post-seal file modification')
    return g

def evaluate(pool):
    setup();validate_freeze();verify_global_seal()
    require(pool in range(NPOOLS),'unregistered eval pool')
    d=ROOT/f'runs/eval/pool{pool}';d.mkdir(parents=True,exist_ok=False)
    x,y=image_data('test')
    require(len(y)==809,'wrong test size')
    # Construct evaluation transforms only AFTER all 192 final states were sealed.
    env=torch.stack([torch.from_numpy(common.geometric_environment(x,key)) for key in test_keys(pool)])
    rows=[];clean=[]
    for seed in range(NSEEDS):
        for policy in POLICIES:
            cp=torch.load(ROOT/f'runs/train/pool{pool}/seed{seed}_{policy}_step200.pt',weights_only=True,map_location='cpu')
            eng=ShadowEngine.from_state_dict(cp['engine']);eng.model.eval()
            with torch.no_grad():
                logits=eng.model(torch.from_numpy(x))[0]
                ce=torch.nn.functional.cross_entropy(logits,y,reduction='none')
                clean.append(dict(pool=pool,seed=seed,policy=policy,n=len(y),correct=int((logits.argmax(dim=1)==y).sum()),
                                  ce_sum=float(ce.double().sum())))
                for j,xx in enumerate(env):
                    logits=eng.model(xx)[0]
                    ce=torch.nn.functional.cross_entropy(logits,y,reduction='none')
                    rows.append(dict(pool=pool,seed=seed,policy=policy,environment=j,env_key=test_keys(pool)[j],
                                     n=len(y),correct=int((logits.argmax(dim=1)==y).sum()),
                                     ce_sum=float(ce.double().sum())))
    require(len(rows)==16*48 and len(clean)==16,'incomplete evaluation')
    write_csv(d/'heldout_counts.csv',rows);write_csv(d/'clean_counts.csv',clean)
    write_json(d/'manifest.json',dict(pool=pool,date=now(),seal_sha=verify_global_seal()['seal_sha'],
                heldout_constructed=True,test_keys=test_keys(pool),data_hashes=data_hashes(),
                files={p.name:sha_file(p) for p in sorted(d.iterdir()) if p.is_file()}))
    print('HELDOUT_EVAL_POOL_DONE',pool,'rows',len(rows),flush=True)

def holm(ps):
    order=sorted(ps,key=lambda k:(ps[k],k));out={};now_max=0
    for j,k in enumerate(order):
        now_max=max(now_max,(len(order)-j)*ps[k]);out[k]=min(1.,now_max)
    return out

def analyze():
    setup();validate_freeze();verify_global_seal()
    raw=[]
    for pool in range(NPOOLS):
        d=ROOT/f'runs/eval/pool{pool}'
        manifest=json.loads((d/'manifest.json').read_text())
        require(manifest['seal_sha']==verify_global_seal()['seal_sha'],'wrong evaluation seal')
        for name,digest in manifest['files'].items():require(sha_file(d/name)==digest,'eval file mutation')
        rows=read_csv(d/'heldout_counts.csv')
        require(len(rows)==768,'wrong eval rows')
        for row in rows:
            require(int(row['n'])==809 and 0<=int(row['correct'])<=809,'invalid counts')
            require(math.isfinite(float(row['ce_sum'])) and float(row['ce_sum'])>=0,'invalid CE')
        raw.extend(rows)
    require(len(raw)==9216,'wrong global eval count')
    pooled=[]
    for p in range(NPOOLS):
        for pol in POLICIES:
            rows=[r for r in raw if int(r['pool'])==p and r['policy']==pol]
            require(len(rows)==4*48,'wrong pool/arm size')
            value=math.fsum(float(r['ce_sum']) for r in rows)/(len(rows)*809)
            pooled.append(dict(pool=p,policy=pol,mean_ce=value))
    cmp=COMPARISONS
    contrasts=[];tests=[];ps={}
    for name,(base,target) in cmp.items():
        vec=[]
        for p in range(12):
            lookup={r['policy']:float(r['mean_ce']) for r in pooled if r['pool']==p}
            delta=lookup[base]-lookup[target]
            vec.append(delta);contrasts.append(dict(pool=p,contrast=name,ce_improvement=delta))
        x=np.asarray(vec,dtype=float);mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(len(x)))
        require(se>0,'degenerate standard error')
        pval=float(t.sf(mean/se,df=11));ciw=float(t.ppf(.975,11)*se)
        ps[name]=pval
        tests.append(dict(contrast=name,n_pools=12,mean_ce_improvement=mean,se=se,ci95_low=mean-ciw,
                          ci95_high=mean+ciw,p_raw=pval,positive_pools=int((x>0).sum())))
    adj=holm(ps)
    for row in tests:
        row.update(p_holm=adj[row['contrast']],decision=('SUPPORT' if row['mean_ce_improvement']>=.01 and adj[row['contrast']]<.05 else 'NOT SUPPORTED'))
    cleanraw=[]
    for pool in range(NPOOLS):
        d=ROOT/f'runs/eval/pool{pool}'
        rows=read_csv(d/'clean_counts.csv')
        require(len(rows)==16,'incomplete clean row count')
        require({(int(r['seed']),r['policy']) for r in rows} ==
                {(i,pol) for i in range(NSEEDS) for pol in POLICIES},'incomplete clean grid')
        for row in rows:
            require(int(row['n'])==809 and 0<=int(row['correct'])<=809,'invalid clean counts')
            require(math.isfinite(float(row['ce_sum'])) and float(row['ce_sum'])>=0,'invalid clean CE')
        cleanraw.extend(rows)
    cleanpool=[]
    for pool in range(NPOOLS):
        for pol in POLICIES:
            z=[r for r in cleanraw if int(r['pool'])==pool and r['policy']==pol]
            cleanpool.append(dict(pool=pool,policy=pol,
                       clean_ce=math.fsum(float(r['ce_sum']) for r in z)/(NSEEDS*809),
                       clean_accuracy=sum(int(r['correct']) for r in z)/(NSEEDS*809)))
    clean_deltas=[float(next(z['clean_ce'] for z in cleanpool if z['pool']==p and z['policy']=='T60'))-
                  float(next(z['clean_ce'] for z in cleanpool if z['pool']==p and z['policy']=='M'))
                  for p in range(NPOOLS)]
    x=np.asarray(clean_deltas,dtype=float)
    guard_mean=float(x.mean());guard_se=float(x.std(ddof=1)/np.sqrt(NPOOLS))
    require(guard_se>0,'degenerate guardrail standard error')
    guard_p=float(t.cdf((guard_mean-CLEAN_MARGIN)/guard_se,df=NPOOLS-1))
    guard_ci=float(t.ppf(.975,NPOOLS-1)*guard_se)
    guard=dict(contrast='T60_clean_penalty_vs_M',n_pools=NPOOLS,margin=CLEAN_MARGIN,
               mean_clean_ce_penalty=guard_mean,se=guard_se,
               ci95_low=guard_mean-guard_ci,ci95_high=guard_mean+guard_ci,
               one_sided_p=guard_p,
               decision='PASS' if guard_mean<CLEAN_MARGIN and guard_p<.05 else 'NOT SUPPORTED')
    write_csv(ROOT/'results/policy_pool_losses.csv',pooled)
    write_csv(ROOT/'results/policy_clean_pool_losses.csv',cleanpool)
    write_csv(ROOT/'results/primary_pool_differences.csv',contrasts)
    write_csv(ROOT/'results/primary_decisions.csv',tests)
    write_csv(ROOT/'results/clean_guardrail.csv',[guard])
    write_json(ROOT/'results/summary.json',dict(date=now(),config_hash=CONFIG_HASH,
                source_hashes=sources(),seal_sha=verify_global_seal()['seal_sha'],
                models=192,heldout_count_rows=9216,clean_count_rows=192,independent_pools=12,
                primary=tests,clean_guardrail=guard,
                mean_ce_by_policy={pol:float(np.mean([r['mean_ce'] for r in pooled if r['policy']==pol])) for pol in POLICIES},
                mean_clean_ce_by_policy={pol:float(np.mean([r['clean_ce'] for r in cleanpool if r['policy']==pol])) for pol in POLICIES},
                changes_not_allowed_after_outcomes=True))
    print('ANALYSIS_COMPLETE',json.dumps(tests,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','train','seal','evaluate','analyze']);p.add_argument('--pool',type=int)
    args=p.parse_args()
    if args.mode in ('train','evaluate') and args.pool is None:p.error('--pool required')
    if args.mode=='freeze':freeze()
    elif args.mode=='train':train(args.pool)
    elif args.mode=='seal':seal()
    elif args.mode=='evaluate':evaluate(args.pool)
    elif args.mode=='analyze':analyze()

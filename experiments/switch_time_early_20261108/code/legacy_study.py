"""Bounded train / global seal / final evaluation for frozen switching-seed pilot."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
from datetime import datetime,timezone
import hashlib
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import time
import numpy as np
import scipy
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
import torch.optim._muon
import common
from switch_engine import ShadowEngine,tree_digest,choose_branch,ADAM,MUON

ROOT=Path(__file__).resolve().parents[1]
POLICIES=('A','M','S','D')
NPOOLS=12
NSEEDS=4
GATES=tuple(range(40,181,20))
CHECKPOINTS=(40,100,200)
PART_COUNTS={'fit':800,'control':94,'selection':94,'test':809}
SOURCES=('PROTOCOL.md','code/common.py','code/switch_engine.py','code/run_study.py',
         'tests/test_engine.py','tests/test_study.py')
CONFIG=dict(id='switch-seed-20261031-v1',pools=12,seeds=4,policies=POLICIES,
            checkpoints=CHECKPOINTS,gates=GATES,lookahead=5,hysteresis=.005,
            steps=200,fit=800,control=94,selection=94,test=809,batch=100,Q=4,
            env_counts={'fit':64,'control':8,'selection':8,'test':48},
            init_base=6400000000,init_stride=7919,schedule_base=15500000000,schedule_stride=1201,
            env_base=1100000,env_stride=10000,env_offsets={'fit':0,'control':2000,'selection':3000,'test':5000},
            adam=ADAM,muon=MUON,stat_n=12,holm_family=5,alpha=.05,ce_point_gate=.01)
CONFIG_HASH=hashlib.sha256(json.dumps(CONFIG,sort_keys=True).encode()).hexdigest()


def now():return datetime.now(timezone.utc).isoformat()
def req(ok,msg):
    if not ok:raise ValueError(msg)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bytes_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def sources():return {p:sha(ROOT/p) for p in SOURCES}
def write_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(obj,f,sort_keys=True,indent=2,allow_nan=False)
def csv_write(path,rows):
    req(bool(rows),'refuse empty CSV')
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def csv_read(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def configure():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
def runtime():
    cpuinfo=Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else ''
    cpu=next((s.split(':',1)[1].strip() for s in cpuinfo.splitlines() if s.startswith('model name')),platform.processor())
    return dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
                cpu=cpu,cpu_count=os.cpu_count(),threads=torch.get_num_threads(),mkldnn=torch.backends.mkldnn.enabled,
                deterministic=torch.are_deterministic_algorithms_enabled(),clock='uncontrolled',device='cpu',
                muon_ns_dtype='bfloat16 inside torch2.10, float32 weights/gradients',
                torch_muon_source_sha=sha(inspect.getfile(torch.optim._muon)),
                thread_env={k:os.getenv(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})


def split_indices():
    _,y=load_digits(return_X_y=True);ids=np.arange(len(y))
    dev,test=train_test_split(ids,test_size=.45,random_state=314159,stratify=y)
    fit,res=train_test_split(dev,train_size=800,random_state=1031001,stratify=y[dev])
    control,selection=train_test_split(res,train_size=94,random_state=1031002,stratify=y[res])
    out=dict(fit=fit,control=control,selection=selection,test=test)
    req({k:len(v) for k,v in out.items()}==PART_COUNTS,'split sizes')
    req(len(set(np.concatenate(list(out.values()))))==1797,'overlapping images')
    return out


def data_hashes():
    x,y=load_digits(return_X_y=True)
    return dict(indices={k:bytes_sha(v) for k,v in split_indices().items()},images=bytes_sha(x),labels=bytes_sha(y))


def image_data(part):
    x,y=load_digits(return_X_y=True);idx=split_indices()[part]
    return (x[idx].astype(np.float32)/16).reshape(-1,8,8),torch.tensor(y[idx],dtype=torch.long)


def schedule(pool):
    req(pool in range(NPOOLS),'unregistered pool')
    seed=CONFIG['schedule_base']+CONFIG['schedule_stride']*pool
    rng=np.random.default_rng(seed+1);gen=torch.Generator().manual_seed(seed)
    rows=[]
    for _ in range(25):
        for batch in torch.randperm(800,generator=gen).split(100):
            rows.append((batch.numpy().astype(np.int64),rng.choice(64,4,replace=False).astype(np.int64)))
    req(len(rows)==200,'schedule length')
    return rows


def schedule_digest(rows):
    h=hashlib.sha256()
    for b,e in rows:
        h.update(np.asarray(b,dtype='<i8').tobytes());h.update(np.asarray(e,dtype='<i8').tobytes())
    return h.hexdigest()


def env_keys(pool,part):
    base=CONFIG['env_base']+CONFIG['env_stride']*pool+CONFIG['env_offsets'][part]
    return list(range(base,base+CONFIG['env_counts'][part]))


def init_key(pool,seed):return CONFIG['init_base']+CONFIG['init_stride']*(4*pool+seed)


def validate_freeze(out):
    f=json.loads((out/'freeze.json').read_text())
    req(f['config_hash']==CONFIG_HASH and f['sources']==sources(),'source/protocol mutation')
    req(f['data_hashes']==data_hashes(),'dataset mutation')
    r=runtime()
    for key in ('python','torch','numpy','scipy','sklearn','torch_muon_source_sha'):
        req(r[key]==f['runtime'][key],f'backend version drift {key}')
    return f


def require_global_seal(out):
    p=Path(out)/'GLOBAL_SEAL.json'
    if not p.exists():raise ValueError('heldout transforms forbidden before GLOBAL_SEAL')
    record=json.loads(p.read_text());req(record['test_constructed'] is False,'seal boundary')
    req(record['states']==576 and record['final_models']==192,'incomplete global seal')
    req(record['config_hash']==CONFIG_HASH,'wrong seal')
    return record


def environments(pool,part,out=None):
    if part=='test':
        req(out is not None,'missing barrier path');require_global_seal(out)
    x,y=image_data(part)
    xs=[torch.from_numpy(common.geometric_environment(x,key)) for key in env_keys(pool,part)]
    return torch.stack(xs),y


def fit_batches(envs,y,sched,start=0,end=200):
    batches=[]
    for b,e in sched[start:end]:
        xx=envs[torch.tensor(e)][:,torch.tensor(b)].reshape(-1,8,8)
        batches.append((xx,y[torch.tensor(b)].repeat(4)))
    return batches


@torch.no_grad()
def ce_on(engine,batches):
    engine.model.eval();total=0.;n=0
    for x,y in batches:
        ce=torch.nn.functional.cross_entropy(engine.model(x)[0],y,reduction='none')
        total+=float(ce.double().sum());n+=len(y)
    req(n>0 and np.isfinite(total),'bad control loss')
    return total/n


def probe_gate(engine,batches,control,incumbent,gate):
    """Return selected actual branch; exactly five applied + five rejected steps."""
    req(len(batches)==5,'probe length')
    before=tree_digest(engine.state_dict());branches={};scores={};traces=[];probes=[]
    for candidate in ('A','M'):
        fork=engine.fork();start=time.perf_counter()
        for j,(x,y) in enumerate(batches,1):
            d=fork.step(x,y,candidate)
            traces.append(dict(step=gate+j,gate=gate,candidate=candidate,applied=False,**d))
        train_secs=time.perf_counter()-start;start=time.perf_counter()
        score=ce_on(fork,control);score_secs=time.perf_counter()-start
        branches[candidate]=fork;scores[candidate]=score
        probes.append(dict(gate=gate,incumbent=incumbent,candidate=candidate,control_ce=score,chosen=False,
                           parent_digest=before,branch_digest=tree_digest(fork.state_dict()),
                           probe_train_seconds=train_secs,score_seconds=score_secs))
    req(before==tree_digest(engine.state_dict()),'probe leaked into incumbent')
    chosen=choose_branch(incumbent,scores)
    for row in probes:row['chosen']=row['candidate']==chosen
    for row in traces:row['applied']=row['candidate']==chosen
    return branches[chosen],chosen,probes,traces


def select_seeds(rows):
    req(len(rows)==16,'incomplete early selection grid')
    keys=[(r['policy'],int(r['seed'])) for r in rows]
    req(len(set(keys))==16 and set(keys)=={(p,i) for p in POLICIES for i in range(4)},'bad selection keys')
    req(all(np.isfinite(float(r['selection_ce'])) for r in rows),'nonfinite selection score')
    result={p:int(min((r for r in rows if r['policy']==p),key=lambda r:(float(r['selection_ce']),int(r['seed'])))['seed']) for p in POLICIES}
    b=min((r for r in rows if r['policy'] in ('A','M')),key=lambda r:(float(r['selection_ce']),r['policy'],int(r['seed'])))
    result['B']=dict(policy=b['policy'],seed=int(b['seed']))
    return result


def train_one(pool,seed,policy,batches,control,selection,d):
    key=init_key(pool,seed);engine=ShadowEngine(key)
    initial=tree_digest(engine.model.state_dict());active='A' if policy=='A' else 'M'
    states=[];traces=[];probes=[];scores=[];saved=set();start_wall=time.perf_counter()
    while engine.steps<200:
        t=engine.steps
        if policy=='D' and t in GATES:
            engine,active,ps,ts=probe_gate(engine,batches[t:t+5],control,active,t)
            probes.extend(ps);traces.extend(ts)
        else:
            active='A' if policy=='A' or (policy=='S' and t>=100) else ('M' if policy in ('M','S') else active)
            info=engine.step(*batches[t],active)
            traces.append(dict(step=engine.steps,gate=-1,candidate=active,applied=True,**info))
        t=engine.steps
        if t in CHECKPOINTS and t not in saved:
            state=engine.state_dict();name=f'seed{seed}_{policy}_step{t}.pt'
            cp=dict(engine=state,pool=pool,seed=seed,init_key=key,policy=policy,active=active,step=t,
                    initial_digest=initial,config_hash=CONFIG_HASH)
            torch.save(cp,d/name)
            states.append(dict(pool=pool,seed=seed,init_key=key,policy=policy,step=t,file=name,
                               file_sha=sha(d/name),model_digest=tree_digest(state['model']),
                               engine_digest=tree_digest(state),initial_digest=initial,active=active))
            saved.add(t)
            if t==40:scores.append(dict(pool=pool,seed=seed,policy=policy,step=40,selection_ce=ce_on(engine,selection),model_digest=tree_digest(state['model'])))
    for rows in (traces,probes):
        for row in rows:row.update(pool=pool,seed=seed,policy=policy)
    req(sum(r['applied'] for r in traces)==200,'applied update count')
    req(len(traces)==(240 if policy=='D' else 200),'executed update count')
    return states,traces,probes,scores,dict(pool=pool,seed=seed,policy=policy,seconds=time.perf_counter()-start_wall,
                                         applied_steps=200,executed_gradient_steps=len(traces),shadow_steps=2*len(traces))


def train_pool(pool,outstr):
    configure();out=Path(outstr);validate_freeze(out)
    d=out/'training'/f'pool{pool}';d.mkdir(parents=True,exist_ok=False)
    created=now();setup_start=time.perf_counter()
    envs,y=environments(pool,'fit');ce,cy=environments(pool,'control');se,sy=environments(pool,'selection')
    sched=schedule(pool);batches=fit_batches(envs,y,sched)
    control=[(ce[i],cy) for i in range(8)];selection=[(se[i],sy) for i in range(8)]
    setup_seconds=time.perf_counter()-setup_start
    sr=[];tr=[];pr=[];er=[];times=[]
    for seed in range(4):
        for policy in POLICIES:
            s,t,p,e,w=train_one(pool,seed,policy,batches,control,selection,d)
            sr+=s;tr+=t;pr+=p;er+=e;times.append(w)
        print(f'TRAIN_BLOCK_DONE pool={pool} seed={seed}; test not constructed',flush=True)
    choices=select_seeds(er)
    csv_write(d/'states.csv',sr);csv_write(d/'trace.csv',tr);csv_write(d/'probes.csv',pr)
    csv_write(d/'early_selection.csv',er);csv_write(d/'timing.csv',times);write_json(d/'choices.json',choices)
    files={p.name:sha(p) for p in sorted(d.iterdir()) if p.is_file()}
    manifest=dict(pool=pool,created=created,completed=now(),config_hash=CONFIG_HASH,sources=sources(),
                  data_hashes=data_hashes(),runtime=runtime(),schedule_hash=schedule_digest(sched),
                  environment_keys={part:env_keys(pool,part) for part in ('fit','control','selection')},
                  training_env_hash=bytes_sha(envs.numpy()),control_env_hash=bytes_sha(ce.numpy()),selection_env_hash=bytes_sha(se.numpy()),
                  files=files,test_constructed=False,setup_seconds=setup_seconds,states=len(sr),
                  applied_steps=sum(z['applied'] for z in tr),executed_steps=len(tr))
    write_json(d/'manifest.json',manifest)
    print(f'POOL_COMPLETE pool={pool} states={len(sr)} applied={manifest["applied_steps"]} executed={len(tr)}',flush=True)
    return pool


def verify_training_pool(pool,out,verify_tensors=True):
    d=out/'training'/f'pool{pool}';m=json.loads((d/'manifest.json').read_text())
    req(m['config_hash']==CONFIG_HASH and m['sources']==sources(),'source mismatch')
    req(m['data_hashes']==data_hashes() and m['test_constructed'] is False,'data/barrier mismatch')
    req(m['schedule_hash']==schedule_digest(schedule(pool)),'schedule mismatch')
    for f,dig in m['files'].items():req(sha(d/f)==dig,f'file mismatch {f}')
    rows=csv_read(d/'states.csv');expected={(i,p,t) for i in range(4) for p in POLICIES for t in CHECKPOINTS}
    req(len(rows)==48 and {(int(r['seed']),r['policy'],int(r['step'])) for r in rows}==expected,'state grid')
    for r in rows:
        req(r['file_sha']==sha(d/r['file']),'checkpoint file mismatch')
        if verify_tensors:
            cp=torch.load(d/r['file'],weights_only=True,map_location='cpu')
            req(tree_digest(cp['engine'])==r['engine_digest'] and tree_digest(cp['engine']['model'])==r['model_digest'],'tensor hash mismatch')
    es=csv_read(d/'early_selection.csv');choices=json.loads((d/'choices.json').read_text())
    req(choices==select_seeds(es),'late/invalid seed selection')
    req(choices['M']==choices['S']==choices['D'],'pre-gate identical states must select same seed')
    for i in range(4):
        initials={r['initial_digest'] for r in rows if int(r['seed'])==i};req(len(initials)==1,'unequal initial conditions')
        prefixes={r['engine_digest'] for r in rows if int(r['seed'])==i and r['step']=='40' and r['policy'] in ('M','S','D')};req(len(prefixes)==1,'initial Muon prefixes differ')
    req(m['states']==48 and m['applied_steps']==3200 and m['executed_steps']==3360,'step/state counts')
    return m


def freeze(out):
    configure();req(not out.exists(),'refuse overwrite freeze');out.mkdir(parents=True)
    write_json(out/'freeze.json',dict(timestamp=now(),config=CONFIG,config_hash=CONFIG_HASH,sources=sources(),
               data_hashes=data_hashes(),runtime=runtime(),test_constructed=False,real_training_started=False,
               registration='local; GitHub write failed before outcomes'))
    print('FROZEN_BEFORE_TRAINING',CONFIG_HASH,flush=True)


def seal(out):
    configure();validate_freeze(out);manifests={}
    for pool in range(NPOOLS):
        verify_training_pool(pool,out)
        manifests[str(pool)]=sha(out/'training'/f'pool{pool}'/'manifest.json')
    write_json(out/'GLOBAL_SEAL.json',dict(timestamp=now(),config_hash=CONFIG_HASH,sources=sources(),
               states=576,final_models=192,pools=12,test_constructed=False,manifests=manifests,
               freeze_hash=sha(out/'freeze.json')))
    print('GLOBAL_SEAL_COMPLETE: 576 states; test not constructed',flush=True)


@torch.no_grad()
def metrics(model,x,y):
    model.eval();logits=model(x)[0]
    ce=torch.nn.functional.cross_entropy(logits,y,reduction='none').double()
    req(torch.isfinite(ce).all().item(),'nonfinite evaluation')
    return dict(n=len(y),correct=int((logits.argmax(1)==y).sum()),ce_sum=float(ce.sum()))


def eval_pool(pool,outstr):
    configure();out=Path(outstr);validate_freeze(out);sealrec=require_global_seal(out)
    td=out/'training'/f'pool{pool}';req(sha(td/'manifest.json')==sealrec['manifests'][str(pool)],'seal changed')
    verify_training_pool(pool,out)
    d=out/'evaluation'/f'pool{pool}';d.mkdir(parents=True,exist_ok=False)
    envs,y=environments(pool,'test',out);clean,_=image_data('test');keys=env_keys(pool,'test')
    cr=[];er=[]
    states=[r for r in csv_read(td/'states.csv') if int(r['step'])==200]
    start=now()
    for r in states:
        cp=torch.load(td/r['file'],weights_only=True,map_location='cpu');model=common.MLP();model.load_state_dict(cp['engine']['model'])
        common_fields=dict(pool=pool,seed=int(r['seed']),policy=r['policy'],step=200)
        for j,key in enumerate(keys):er.append(dict(**common_fields,environment=j,env_key=key,**metrics(model,envs[j],y)))
        cr.append(dict(**common_fields,**metrics(model,torch.from_numpy(clean),y)))
    csv_write(d/'test_counts.csv',er);csv_write(d/'clean_counts.csv',cr)
    write_json(d/'manifest.json',dict(pool=pool,started=start,completed=now(),test_env_keys=keys,
               test_env_hash=bytes_sha(envs.numpy()),source_hashes=sources(),data_hashes=data_hashes(),
               seal_hash=sha(out/'GLOBAL_SEAL.json'),rows=len(er),clean_rows=len(cr),runtime=runtime(),
               files={f:sha(d/f) for f in ('test_counts.csv','clean_counts.csv')}))
    print(f'EVALUATION_COMPLETE pool={pool} rows={len(er)}',flush=True)
    return pool


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','train','seal','evaluate'])
    parser.add_argument('--out',default=str(ROOT/'experiment'));parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--pools',type=int,nargs='*');a=parser.parse_args();out=Path(a.out)
    if a.mode=='freeze':freeze(out);return
    if a.mode=='seal':seal(out);return
    configure();validate_freeze(out);pools=list(range(NPOOLS)) if a.pools is None else a.pools
    req(len(set(pools))==len(pools) and set(pools)<=set(range(NPOOLS)),'bad pool subset')
    req(1<=a.workers<=4,'worker bound')
    fn=train_pool if a.mode=='train' else eval_pool
    if a.mode=='evaluate':require_global_seal(out)
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as ex:
        futures={ex.submit(fn,p,str(out)):p for p in pools}
        for future in as_completed(futures):future.result()

if __name__=='__main__':main()

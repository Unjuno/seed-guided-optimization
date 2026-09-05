"""Issue #61: fixed-dose functional audit, not a difficulty-matched causal test.

Variable definitions and frozen tests: repository Issue #61. All scientific
quantities are dimensionless; accuracy outputs are fractions, not percentages.
Run selftest before train/evaluate. Existing training functions are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

REPS = tuple(range(1200, 1230))
METHODS = ('loss_hard', 'gradnov')
DOSES = (0.0, 0.01, 0.10, 0.25, 0.50, 1.0)
ALPHAS = (0.005, 0.05, 0.10)
CONDITIONS = tuple(('shared', v) for v in DOSES) + tuple(('nuisance', v) for v in ALPHAS)
TRAIN_SEEDS = tuple(range(41000, 41064))
SHARED_SEEDS = tuple(range(42000, 42080))
NUISANCE_SEEDS = tuple(range(43000, 43080))
PROTOCOL = {'issue': 61, 'reps': REPS, 'methods': METHODS, 'doses': DOSES,
            'alphas': ALPHAS, 'train_seeds': TRAIN_SEEDS,
            'shared_seeds': SHARED_SEEDS, 'nuisance_seeds': NUISANCE_SEEDS,
            'base_seed_offset': 1610000000, 'base_seed_stride': 4099}
PROTOCOL_HASH = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blend(x: np.ndarray, changed: np.ndarray, strength: float) -> np.ndarray:
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('strength outside [0,1]')
    if x.shape != changed.shape or not np.isfinite(x).all() or not np.isfinite(changed).all():
        raise ValueError('invalid images')
    if min(float(x.min()), float(changed.min())) < 0 or max(float(x.max()), float(changed.max())) > 1:
        raise ValueError('images outside [0,1]')
    return np.clip((1.0-strength)*x + strength*changed, 0, 1).astype(np.float32)


def runtime() -> dict:
    import scipy, sklearn
    cpu = Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else ''
    return {'python': sys.version, 'platform': platform.platform(),
            'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__,
            'sklearn': sklearn.__version__, 'torch': torch.__version__,
            'torch_config': torch.__config__.show(), 'threads': torch.get_num_threads(),
            'deterministic': torch.are_deterministic_algorithms_enabled(),
            'mkldnn_enabled': torch.backends.mkldnn.enabled,
            'cpu_model': next((v.split(':',1)[1].strip() for v in cpu.splitlines() if v.startswith('model name')), 'unreported'),
            'cpu_mhz_snapshot': next((v.split(':',1)[1].strip() for v in cpu.splitlines() if v.startswith('cpu MHz')), 'unreported'),
            'thread_env': {k: os.getenv(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
            'github_sha': os.getenv('GITHUB_SHA'), 'run_id': os.getenv('GITHUB_RUN_ID')}


def check_range(start: int, end: int) -> None:
    if start >= end or not set(range(start,end)).issubset(REPS):
        raise ValueError('unregistered replicate range')


def train(start: int, end: int, out: Path) -> None:
    import transfer_specificity as base
    check_range(start,end)
    base.configure_determinism(1)
    if (base.K,base.Q,base.BATCH,base.EPOCHS,base.LR,base.WD,base.NOVELTY_WEIGHT) != (16,4,128,10,.01,.001,.6):
        raise ValueError('base training protocol changed')
    out.mkdir(parents=True,exist_ok=True)
    states = out/'states'; states.mkdir(exist_ok=True)
    if list(states.glob('*.pt')): raise ValueError('refusing to overwrite sealed states')
    x,y,_,_ = base.load_digits_split(); y=torch.tensor(y)
    envs=base.geometric_envs(x,TRAIN_SEEDS)
    diagnostics=[]; hashes={}
    for rep in range(start,end):
        seed=1610000000+4099*rep
        sched=base.schedule(len(y),seed,TRAIN_SEEDS)
        for method in METHODS:
            model,diag=base.train_one(envs,y,sched,seed,method)
            p=states/f'rep{rep}_{method}.pt'
            torch.save({'state_dict':model.state_dict(),'rep':rep,'method':method,
                        'protocol_hash':PROTOCOL_HASH},p)
            hashes[p.name]=sha256(p)
            diagnostics.append({'rep':rep,'method':method,**diag})
        print(f'TRAINED rep={rep}; no evaluation environments constructed',flush=True)
    pd.DataFrame(diagnostics).to_csv(out/f'fixed_dose_training_{start}_{end}.csv',index=False)
    sources={n:sha256(Path(__file__).parent/n) for n in ('common.py','transfer_specificity.py','fixed_dose_response.py')}
    manifest={'protocol':PROTOCOL,'protocol_hash':PROTOCOL_HASH,'start':start,'end':end,
              'checkpoints':hashes,'source_hashes':sources,'runtime':runtime(),'n_train':len(y)}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('SEALED_TRAINING_COMPLETE '+PROTOCOL_HASH,flush=True)


def evaluate(start: int, end: int, out: Path) -> None:
    import transfer_specificity as base
    check_range(start,end); base.configure_determinism(1)
    manifest=json.loads((out/'manifest.json').read_text())
    if (manifest['start'],manifest['end'],manifest['protocol_hash']) != (start,end,PROTOCOL_HASH):
        raise ValueError('seal protocol/range mismatch')
    expected={f'rep{r}_{m}.pt' for r in range(start,end) for m in METHODS}
    if set(manifest['checkpoints']) != expected: raise ValueError('checkpoint set mismatch')
    for n,h in manifest['source_hashes'].items():
        if sha256(Path(__file__).parent/n)!=h: raise ValueError('source changed after seal')
    for n,h in manifest['checkpoints'].items():
        if sha256(out/'states'/n)!=h: raise ValueError('checkpoint changed after seal')
    # Only now construct fresh evaluation families. No calibration is performed.
    _,_,x,y=base.load_digits_split(); labels=torch.tensor(y); clean=torch.tensor(x)
    geo=[base.geometric_environment(x,s) for s in SHARED_SEEDS]
    envs={('shared',v):[(s,torch.tensor(blend(x,g,v))) for s,g in zip(SHARED_SEEDS,geo)] for v in DOSES if v>0}
    envs[('shared',0.0)]=[(-1,clean)]  # one clean evaluation, not 80 pseudo-replicates
    for v in ALPHAS:
        envs[('nuisance',v)]=[(s,torch.tensor(base.nuisance_environment(x,s,v))) for s in NUISANCE_SEEDS]
    input_rows=[]
    for (fam,v),items in envs.items():
        for s,xx in items:
            delta=xx.numpy().astype(np.float64)-x.astype(np.float64)
            input_rows.append(dict(family=fam,strength=v,env_seed=s,rms=float(np.sqrt(np.mean(delta**2))),linf=float(np.max(np.abs(delta)))))
    pd.DataFrame(input_rows).to_csv(out/f'fixed_dose_inputs_{start}_{end}.csv',index=False)
    rows=[]; perenv=[]
    for rep in range(start,end):
        for method in METHODS:
            p=out/'states'/f'rep{rep}_{method}.pt'
            ck=torch.load(p,map_location='cpu',weights_only=True)
            if (ck['rep'],ck['method'],ck['protocol_hash']) != (rep,method,PROTOCOL_HASH):
                raise ValueError('checkpoint metadata mismatch')
            model=base.MLP(); model.load_state_dict(ck['state_dict']); model.eval()
            with torch.no_grad():
                clean_logits=model(clean)[0]; clean_pred=clean_logits.argmax(1)
                for fam,v in CONDITIONS:
                    block=[]
                    for s,xx in envs[(fam,v)]:
                        logits=model(xx)[0]; pred=logits.argmax(1)
                        row=dict(rep=rep,method=method,family=fam,strength=v,env_seed=s,
                                 accuracy=float((pred==labels).float().mean()),
                                 nll=float(torch.nn.functional.cross_entropy(logits,labels)),
                                 prediction_disagreement=float((pred!=clean_pred).float().mean()))
                        block.append(row); perenv.append(row)
                    a=np.array([b['accuracy'] for b in block])
                    rows.append(dict(rep=rep,method=method,family=fam,strength=v,
                        mean_accuracy=float(a.mean()),mean_nll=float(np.mean([b['nll'] for b in block])),
                        disagreement=float(np.mean([b['prediction_disagreement'] for b in block])),
                        p10=float(np.quantile(a,.1)),minimum=float(a.min()),
                        n_env=len(block),n_examples=len(y),checkpoint_sha256=manifest['checkpoints'][p.name]))
    for n,h in manifest['checkpoints'].items():
        if sha256(out/'states'/n)!=h: raise ValueError('checkpoint modified during evaluation')
    pd.DataFrame(rows).to_csv(out/f'fixed_dose_metrics_{start}_{end}.csv',index=False)
    pd.DataFrame(perenv).to_csv(out/f'fixed_dose_environment_{start}_{end}.csv.gz',index=False,compression='gzip')
    print(f'EVALUATION_COMPLETE reps={start}:{end}, states unchanged, n_test={len(y)}',flush=True)


def estimate(values) -> dict:
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all(): raise ValueError('invalid paired sample')
    mean=float(x.mean()); se=float(stats.sem(x)); k=float(stats.t.ppf(.975,len(x)-1))
    if se==0: p=0.0 if mean>0 else (1.0 if mean<0 else .5)
    else: p=float(stats.ttest_1samp(x,0,alternative='greater').pvalue)
    return dict(n=len(x),mean=mean,se=se,ci95_low=mean-k*se,ci95_high=mean+k*se,p_one_sided=p)


def analyze(df: pd.DataFrame, reps=REPS):
    key=['rep','method','family','strength']
    expected={(r,m,f,v) for r in reps for m in METHODS for f,v in CONDITIONS}
    if len(df)!=len(expected) or df.duplicated(key).any() or set(df[key].itertuples(index=False,name=None))!=expected:
        raise ValueError('missing/duplicate/unregistered metric rows')
    if not np.isfinite(df[['mean_accuracy','mean_nll','disagreement','p10','minimum']].to_numpy()).all(): raise ValueError('nonfinite metrics')
    if not df.mean_accuracy.between(0,1).all() or not df.disagreement.between(0,1).all() or (df.mean_nll<0).any():
        raise ValueError('metric range error')
    if not df.checkpoint_sha256.str.fullmatch('[0-9a-f]{64}',na=False).all(): raise ValueError('invalid hash')
    if not df.n_examples.eq(445).all(): raise ValueError('image count mismatch')
    nenv=np.where((df.family=='shared') & (df.strength==0),1,80)
    if not np.array_equal(df.n_env.to_numpy(),nenv): raise ValueError('environment count mismatch')
    if df.groupby(['rep','method']).checkpoint_sha256.nunique().ne(1).any(): raise ValueError('states differ across doses')
    indexed=df.set_index(key); paired=[]
    for r in reps:
        row={'rep':r}
        for name,v in [('clean',0.0),('weak',.01),('full',1.0)]:
            row[name+'_benefit']=float(indexed.loc[(r,'gradnov','shared',v),'mean_accuracy']-indexed.loc[(r,'loss_hard','shared',v),'mean_accuracy'])
        row['interaction']=row['full_benefit']-row['clean_benefit']; paired.append(row)
    paired=pd.DataFrame(paired); full=estimate(paired.full_benefit); interaction=estimate(paired.interaction)
    fp=full['mean']>0 and full['p_one_sided']<.05; ip=interaction['mean']>0 and interaction['p_one_sided']<.05
    decision='DOSE-DEPENDENT BENEFIT PASS' if fp and ip else ('FULL EFFECT ONLY / NO DOSE INTERACTION' if fp else 'NO FULL-STRENGTH REPLICATION')
    summary=[]
    for f,v in CONDITIONS:
        part=df[(df.family==f)&(df.strength==v)].pivot(index='rep',columns='method',values='mean_accuracy')
        summary.append(dict(family=f,strength=v,loss_hard_mean=float(part.loss_hard.mean()),gradnov_mean=float(part.gradnov.mean()),**estimate(part.gradnov-part.loss_hard)))
    dec={'decision':decision,'full_pass':fp,'interaction_pass':ip,'protocol_hash':PROTOCOL_HASH,
         **{'full_'+k:v for k,v in full.items()},**{'interaction_'+k:v for k,v in interaction.items()}}
    return paired,pd.DataFrame(summary),pd.DataFrame([dec])


def summarize(root: Path, out: Path) -> None:
    paths=sorted(root.rglob('fixed_dose_metrics_*.csv'))
    if not paths: raise FileNotFoundError('no metrics')
    df=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    paired,summary,decision=analyze(df)
    manifests=sorted(root.rglob('manifest.json'))
    covered=[]; source_sets=[]
    for path in manifests:
        m=json.loads(path.read_text())
        if m['protocol_hash']!=PROTOCOL_HASH: raise ValueError('manifest protocol mismatch')
        covered.extend(range(m['start'],m['end'])); source_sets.append(m['source_hashes'])
        for name,h in m['checkpoints'].items():
            if sha256(path.parent/'states'/name)!=h: raise ValueError('aggregate checkpoint hash mismatch')
        for row in df[df.rep.between(m['start'],m['end']-1)].itertuples():
            if m['checkpoints'][f'rep{row.rep}_{row.method}.pt']!=row.checkpoint_sha256: raise ValueError('metric hash mismatch')
    if sorted(covered)!=list(REPS) or any(s!=source_sets[0] for s in source_sets): raise ValueError('manifest coverage/source mismatch')
    out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'fixed_dose_metrics30.csv',index=False)
    paired.to_csv(out/'fixed_dose_paired30.csv',index=False)
    summary.to_csv(out/'fixed_dose_summary30.csv',index=False)
    decision.to_csv(out/'fixed_dose_decision30.csv',index=False)
    print(decision.to_string(index=False),flush=True)


def selftest() -> None:
    rng=np.random.default_rng(7); x=rng.random((4,8,8),dtype=np.float32); y=rng.random((4,8,8),dtype=np.float32)
    assert np.array_equal(blend(x,y,0),x) and np.array_equal(blend(x,y,1),y)
    np.testing.assert_allclose(blend(x,y,.01)-x,.01*(y-x),atol=4*np.finfo(np.float32).eps,rtol=0)
    reps=tuple(range(4)); rows=[]
    for r in reps:
        for m in METHODS:
            for f,v in CONDITIONS:
                gain=(.01+.0001*r)+(.04+.0002*r)*v if f=='shared' else .01
                rows.append(dict(rep=r,method=m,family=f,strength=v,mean_accuracy=.5+(gain if m=='gradnov' else 0),mean_nll=1.,disagreement=0.,checkpoint_sha256=hashlib.sha256(f'{r}_{m}'.encode()).hexdigest(),p10=.5,minimum=.4,n_env=1 if (f=='shared' and v==0) else 80,n_examples=445))
    df=pd.DataFrame(rows); assert analyze(df,reps)[2].iloc[0].decision=='DOSE-DEPENDENT BENEFIT PASS'
    bads=[df.iloc[:-1],pd.concat([df,df.iloc[:1]],ignore_index=True)]
    bad=df.copy();bad.loc[0,'mean_accuracy']=np.nan;bads.append(bad)
    bad=df.copy();bad.loc[0,'checkpoint_sha256']='other';bads.append(bad)
    for bad in bads:
        try: analyze(bad,reps)
        except ValueError: pass
        else: raise AssertionError('invalid input accepted')
    no_inter=df.copy();no_inter.loc[no_inter.method=='gradnov','mean_accuracy']=.6
    assert analyze(no_inter,reps)[2].iloc[0].decision=='FULL EFFECT ONLY / NO DOSE INTERACTION'
    null=df.copy();null['mean_accuracy']=.5
    assert analyze(null,reps)[2].iloc[0].decision=='NO FULL-STRENGTH REPLICATION'
    print('SELFTEST PASS: blend endpoints; finite-precision identity; three decisions; four invalid-input rejections',flush=True)


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','evaluate','summarize','selftest'])
    p.add_argument('--start',type=int);p.add_argument('--end',type=int);p.add_argument('--output-dir',type=Path);p.add_argument('--input-dir',type=Path)
    a=p.parse_args()
    if a.mode=='selftest': selftest();return
    if a.output_dir is None: p.error('--output-dir required')
    if a.mode=='summarize':
        if a.input_dir is None: p.error('--input-dir required')
        summarize(a.input_dir,a.output_dir)
    else:
        if a.start is None or a.end is None: p.error('--start and --end required')
        (train if a.mode=='train' else evaluate)(a.start,a.end,a.output_dir)

if __name__=='__main__': main()

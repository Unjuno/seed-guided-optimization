"""Issue #76: CIFAR-10 / ResNet-20 finite-budget Q-scaling falsification."""
from __future__ import annotations

import argparse, hashlib, json, os, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from cifar_resnet_pilot import (
    CIFARResNet20, build_schedule, configure, head_gradient_directions,
    load_cifar, normalize, seed_everything, select_gradnov, select_loss,
)
from cifar_resnet_finetune_pilot import cache_schedule, cache_test, pretrain

REPS=tuple(range(50,80)); METHODS=("loss_hard","gradnov"); QS=(2,4,6,8)
K=8; BATCH=128; PRETRAIN_EPOCHS=10; FINETUNE_EPOCHS=2
PRETRAIN_LR=3e-3; FINETUNE_LR=1e-3; WD=1e-4; NOVELTY_WEIGHT=.6
TRAIN_PER_CLASS=600; TEST_PER_CLASS=300; TEST_ENVS=32
TRAIN_SEEDS=tuple(range(50000,50064)); HELDOUT_SEEDS=tuple(range(60000,60032))
PROTOCOL={
    "issue":76,"reps":REPS,"methods":METHODS,"q":QS,"K":K,"batch":BATCH,
    "pretrain_epochs":PRETRAIN_EPOCHS,"finetune_epochs":FINETUNE_EPOCHS,
    "pretrain_lr":PRETRAIN_LR,"finetune_lr":FINETUNE_LR,"wd":WD,"novelty_weight":NOVELTY_WEIGHT,
    "train_per_class":TRAIN_PER_CLASS,"test_per_class":TEST_PER_CLASS,"test_envs":TEST_ENVS,
    "train_seeds":TRAIN_SEEDS,"heldout_seeds":HELDOUT_SEEDS,
    "base_seed_offset":101000000,"base_seed_stride":7919,
}
PROTOCOL_HASH=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()


def sha256(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()

def state_digest(state)->str:
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        a=t.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

def runtime()->dict:
    cpu=Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else ''
    return {
        "python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"torch":torch.__version__,
        "threads":torch.get_num_threads(),"deterministic":torch.are_deterministic_algorithms_enabled(),
        "cpu_model":next((x.split(':',1)[1].strip() for x in cpu.splitlines() if x.startswith('model name')),'unreported'),
        "thread_env":{k:os.getenv(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
        "github_sha":os.getenv('GITHUB_SHA'),"run_id":os.getenv('GITHUB_RUN_ID')
    }

def check_range(start,end):
    if start>=end or not set(range(start,end)).issubset(REPS): raise ValueError('unregistered replicate range')

def selected_novelty(cos,chosen):
    idx=torch.tensor(chosen,dtype=torch.long)
    sub=cos.index_select(0,idx).index_select(1,idx)
    up=torch.triu_indices(len(chosen),len(chosen),offset=1)
    return float((1-sub[up[0],up[1]]).mean()) if up.shape[1] else 0.0

def finetune(prestate,cached,ytr,base_seed,method,q):
    seed_everything(base_seed)
    model=CIFARResNet20(); model.load_state_dict(prestate)
    opt=torch.optim.AdamW(model.parameters(),lr=FINETUNE_LR,weight_decay=WD)
    ns=[]; cs=[]; ss=[]; model.train()
    for step,idx,cand,u8 in cached:
        xb=normalize(u8.float()/255); logits,h=model(xb)
        per=F.cross_entropy(logits,ytr[idx].repeat(K),reduction='none').reshape(K,-1)
        losses=per.mean(1); gd=head_gradient_directions(logits,h,ytr[idx],K); cos=gd@gd.T
        chosen=select_loss(losses,q) if method=='loss_hard' else select_gradnov(losses,cos,q,w=NOVELTY_WEIGHT)
        chosen=sorted(chosen)
        ns.append(selected_novelty(cos,chosen)); cs.append(float(losses.mean().detach())); ss.append(float(losses[chosen].mean().detach()))
        opt.zero_grad(set_to_none=True); per[chosen].mean().backward(); opt.step()
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    return state,{"selected_pairwise_novelty":float(np.mean(ns)),"mean_candidate_loss":float(np.mean(cs)),"mean_selected_loss":float(np.mean(ss)),"state_digest":state_digest(state)}

def train(start,end,out,data_root):
    check_range(start,end); configure(); out.mkdir(parents=True,exist_ok=True); states=out/'states'; states.mkdir(exist_ok=True)
    if list(states.glob('*.pt')): raise ValueError('refusing checkpoint overwrite')
    xtr,ytr,_,_=load_cifar(data_root,TRAIN_PER_CLASS,TEST_PER_CLASS)
    if len(ytr)!=6000: raise ValueError('train subset mismatch')
    train_env=list(TRAIN_SEEDS); rows=[]; hashes={}
    for rep in range(start,end):
        base_seed=101000000+7919*rep
        prestate=pretrain(base_seed,xtr,ytr,PRETRAIN_EPOCHS,BATCH,PRETRAIN_LR)
        schedule=build_schedule(len(ytr),FINETUNE_EPOCHS,BATCH,64,K,base_seed+1000)
        cached=cache_schedule(xtr,schedule,train_env)
        for q in QS:
            for method in METHODS:
                state,d=finetune(prestate,cached,ytr,base_seed,method,q)
                p=states/f'rep{rep}_q{q}_{method}.pt'
                torch.save({"state_dict":state,"rep":rep,"q":q,"method":method,"protocol_hash":PROTOCOL_HASH},p)
                hashes[p.name]=sha256(p)
                rows.append({"rep":rep,"q":q,"method":method,"candidate_k":K,"coverage_ratio":q/K,**d})
        print(f'TRAINED rep={rep}; no heldout environments constructed',flush=True)
    pd.DataFrame(rows).to_csv(out/f'cifar_budget_training_{start}_{end}.csv',index=False)
    sources={n:sha256(Path(__file__).parent/n) for n in ('cifar_resnet_pilot.py','cifar_resnet_finetune_pilot.py','cifar_resnet_budget_scaling.py')}
    manifest={"protocol":PROTOCOL,"protocol_hash":PROTOCOL_HASH,"start":start,"end":end,"checkpoints":hashes,"source_hashes":sources,"runtime":runtime(),"n_train":6000,"n_test":3000}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('SEALED_TRAINING_COMPLETE '+PROTOCOL_HASH,flush=True)

def evaluate_model(model,heldout,yte,xte):
    model.eval(); acc=[]; env_rows=[]
    with torch.no_grad():
        for s,u8 in zip(HELDOUT_SEEDS,heldout):
            correct=0
            for idx in torch.arange(len(yte)).split(256):
                pred=model(normalize(u8[idx].float()/255))[0].argmax(1); correct+=int((pred==yte[idx]).sum())
            a=correct/len(yte); acc.append(a); env_rows.append((s,a))
        clean_correct=0
        for idx in torch.arange(len(yte)).split(256):
            pred=model(normalize(xte[idx]))[0].argmax(1); clean_correct+=int((pred==yte[idx]).sum())
    a=np.asarray(acc,np.float64)
    met={"mean_test":float(a.mean()),"sd_test":float(a.std(ddof=1)),"p10_test":float(np.quantile(a,.1)),"min_test":float(a.min()),"clean_test":clean_correct/len(yte)}
    return met,env_rows

def evaluate(start,end,out,data_root):
    check_range(start,end); configure(); m=json.loads((out/'manifest.json').read_text())
    if (m['start'],m['end'],m['protocol_hash'])!=(start,end,PROTOCOL_HASH): raise ValueError('manifest protocol/range mismatch')
    for n,h in m['source_hashes'].items():
        if sha256(Path(__file__).parent/n)!=h: raise ValueError('source changed after seal')
    for n,h in m['checkpoints'].items():
        if sha256(out/'states'/n)!=h: raise ValueError('checkpoint changed after seal')
    _,_,xte,yte=load_cifar(data_root,TRAIN_PER_CLASS,TEST_PER_CLASS)
    if len(yte)!=3000: raise ValueError('test subset mismatch')
    # Heldout construction occurs only in this stage, after all train-matrix jobs have completed.
    heldout=cache_test(xte,list(HELDOUT_SEEDS)); rows=[]; perenv=[]
    for rep in range(start,end):
        for q in QS:
            for method in METHODS:
                p=out/'states'/f'rep{rep}_q{q}_{method}.pt'; ck=torch.load(p,map_location='cpu',weights_only=True)
                if (ck['rep'],ck['q'],ck['method'],ck['protocol_hash'])!=(rep,q,method,PROTOCOL_HASH): raise ValueError('checkpoint metadata mismatch')
                state=ck['state_dict']; model=CIFARResNet20(); model.load_state_dict(state)
                met,env_rows=evaluate_model(model,heldout,yte,xte)
                rows.append({"rep":rep,"q":q,"method":method,"candidate_k":K,"coverage_ratio":q/K,"state_digest":state_digest(state),**met})
                perenv.extend({"rep":rep,"q":q,"method":method,"env_seed":s,"accuracy":a} for s,a in env_rows)
    for n,h in m['checkpoints'].items():
        if sha256(out/'states'/n)!=h: raise ValueError('checkpoint mutated during evaluation')
    pd.DataFrame(rows).to_csv(out/f'cifar_budget_heldout_{start}_{end}.csv',index=False)
    pd.DataFrame(perenv).to_csv(out/f'cifar_budget_environment_{start}_{end}.csv.gz',index=False,compression='gzip')
    print(f'EVALUATION_COMPLETE {start}:{end}; n_test=3000 heldout_envs=32',flush=True)

def selftest():
    configure(); losses=torch.arange(K,dtype=torch.float32); cos=torch.eye(K)
    for q in QS:
        a=sorted(select_loss(losses,q)); b=sorted(select_gradnov(losses,cos,q,w=NOVELTY_WEIGHT)); assert len(a)==len(b)==q and len(set(b))==q
    assert sorted(select_loss(losses,K))==sorted(select_gradnov(losses,cos,K,w=NOVELTY_WEIGHT))==list(range(K))
    s={"b":torch.tensor([1.,2.]),"a":torch.tensor([[3.]])}; assert state_digest(s)==state_digest({"a":s['a'].clone(),"b":s['b'].clone()})
    print('SELFTEST PASS',PROTOCOL_HASH)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=('selftest','train','evaluate')); ap.add_argument('--start',type=int); ap.add_argument('--end',type=int); ap.add_argument('--output-dir'); ap.add_argument('--data-root',default='.cache/cifar10'); a=ap.parse_args()
    if a.mode=='selftest': selftest(); return
    if a.start is None or a.end is None or a.output_dir is None: raise ValueError('range/output required')
    (train if a.mode=='train' else evaluate)(a.start,a.end,Path(a.output_dir),a.data_root)
if __name__=='__main__': main()

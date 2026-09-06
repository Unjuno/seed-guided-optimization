"""Issue #73: FashionMNIST Tiny Transformer finite-budget Q-scaling."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
import fashion_transformer_rep_rank_audit as base

REPS=tuple(range(30,60)); METHODS=("loss_hard","gradnov"); QS=(2,4,6,8)
K=8; PRETRAIN_EPOCHS=5; FINETUNE_EPOCHS=2; BATCH=128; PRETRAIN_LR=1e-3; FINETUNE_LR=8e-4; WD=1e-4; NOVELTY_WEIGHT=.6
TRAIN_PER_CLASS=300; TEST_PER_CLASS=100; TEST_ENVS=24
TRAIN_SEEDS=tuple(range(70000,70064)); HELDOUT_SEEDS=tuple(range(80000,80024))
PROTOCOL={"issue":73,"reps":REPS,"methods":METHODS,"q":QS,"K":K,"pretrain_epochs":PRETRAIN_EPOCHS,"finetune_epochs":FINETUNE_EPOCHS,
          "batch":BATCH,"pretrain_lr":PRETRAIN_LR,"finetune_lr":FINETUNE_LR,"wd":WD,"novelty_weight":NOVELTY_WEIGHT,
          "train_per_class":TRAIN_PER_CLASS,"test_per_class":TEST_PER_CLASS,"test_envs":TEST_ENVS,
          "train_seeds":TRAIN_SEEDS,"heldout_seeds":HELDOUT_SEEDS,"base_seed_offset":221000000,"base_seed_stride":8191}
PROTOCOL_HASH=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()

def sha256(path:Path): return hashlib.sha256(path.read_bytes()).hexdigest()

def state_digest(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        a=t.detach().cpu().contiguous().numpy(); h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

def runtime():
    cpu=Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else ''
    return {"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"torch":torch.__version__,
            "threads":torch.get_num_threads(),"deterministic":torch.are_deterministic_algorithms_enabled(),
            "cpu_model":next((x.split(':',1)[1].strip() for x in cpu.splitlines() if x.startswith('model name')),'unreported'),
            "thread_env":{k:os.getenv(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},"github_sha":os.getenv('GITHUB_SHA'),"run_id":os.getenv('GITHUB_RUN_ID')}

def check_range(start,end):
    if start>=end or not set(range(start,end)).issubset(REPS): raise ValueError('unregistered range')

def selected_novelty(cos,chosen):
    idx=torch.tensor(chosen,dtype=torch.long); sub=cos.index_select(0,idx).index_select(1,idx); up=torch.triu_indices(len(chosen),len(chosen),offset=1)
    return float((1-sub[up[0],up[1]]).mean()) if up.shape[1] else 0.0

def finetune(prestate,cached,ytr,mean,std,seed,method,q):
    base.seed_everything(seed); model=base.TinyPatchTransformer(); model.load_state_dict(prestate); opt=torch.optim.AdamW(model.parameters(),lr=FINETUNE_LR,weight_decay=WD)
    ns=[]; cs=[]; ss=[]; model.train()
    for step,idx,cand,u8 in cached:
        xb=base.normalize(u8.float()/255.0,mean,std); logits,h=model(xb)
        per=torch.nn.functional.cross_entropy(logits,ytr[idx].repeat(K),reduction='none').reshape(K,-1); losses=per.mean(1)
        directions=base.head_gradient_directions(logits,h,ytr[idx],K); cos=directions@directions.T
        chosen=base.select_loss(losses,q) if method=='loss_hard' else base.select_gradnov(losses,cos,q,weight=NOVELTY_WEIGHT)
        chosen=sorted(chosen)
        ns.append(selected_novelty(cos,chosen)); cs.append(float(losses.mean().detach())); ss.append(float(losses[chosen].mean().detach()))
        opt.zero_grad(set_to_none=True); per[chosen].mean().backward(); opt.step()
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    return state,{"selected_pairwise_novelty":float(np.mean(ns)),"mean_candidate_loss":float(np.mean(cs)),"mean_selected_loss":float(np.mean(ss)),"state_digest":state_digest(state)}

def train(start,end,out,data_root):
    check_range(start,end); base.configure(); out.mkdir(parents=True,exist_ok=True); states=out/'states'; states.mkdir(exist_ok=True)
    if list(states.glob('*.pt')): raise ValueError('refusing state overwrite')
    xtr,ytr,xte,yte,mean,std=base.load_fashion(data_root,TRAIN_PER_CLASS,TEST_PER_CLASS)
    if (len(ytr),len(yte))!=(3000,1000): raise ValueError('subset count')
    rows=[]; hashes={}
    for rep in range(start,end):
        seed=221000000+8191*rep
        pre=base.pretrain(seed,xtr,ytr,mean,std,PRETRAIN_EPOCHS,BATCH,PRETRAIN_LR)
        sched=base.build_schedule(len(ytr),FINETUNE_EPOCHS,BATCH,64,K,seed+1000)
        cached=base.cache_schedule(xtr,sched,list(TRAIN_SEEDS))
        for q in QS:
            for method in METHODS:
                state,d=finetune(pre,cached,ytr,mean,std,seed,method,q); p=states/f'rep{rep}_q{q}_{method}.pt'
                torch.save({"state_dict":state,"rep":rep,"q":q,"method":method,"protocol_hash":PROTOCOL_HASH},p); hashes[p.name]=sha256(p)
                rows.append({"rep":rep,"q":q,"method":method,"candidate_k":K,"coverage_ratio":q/K,**d})
        print(f'TRAINED rep={rep}; heldout not constructed',flush=True)
    pd.DataFrame(rows).to_csv(out/f'fashion_budget_training_{start}_{end}.csv',index=False)
    sources={n:sha256(Path(__file__).parent/n) for n in ('fashion_transformer_rep_rank_audit.py','fashion_transformer_budget_scaling.py')}
    manifest={"protocol":PROTOCOL,"protocol_hash":PROTOCOL_HASH,"start":start,"end":end,"checkpoints":hashes,"source_hashes":sources,
              "runtime":runtime(),"n_train":3000,"n_test":1000,"mean":mean,"std":std}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)); print('SEALED_TRAINING_COMPLETE '+PROTOCOL_HASH,flush=True)

def evaluate(start,end,out,data_root):
    check_range(start,end); base.configure(); m=json.loads((out/'manifest.json').read_text())
    if (m['start'],m['end'],m['protocol_hash'])!=(start,end,PROTOCOL_HASH): raise ValueError('manifest')
    for n,h in m['source_hashes'].items():
        if sha256(Path(__file__).parent/n)!=h: raise ValueError('source changed')
    for n,h in m['checkpoints'].items():
        if sha256(out/'states'/n)!=h: raise ValueError('checkpoint changed')
    xtr,ytr,xte,yte,mean,std=base.load_fashion(data_root,TRAIN_PER_CLASS,TEST_PER_CLASS)
    if not (np.isclose(mean,m['mean']) and np.isclose(std,m['std'])): raise ValueError('normalization drift')
    test_cache=base.cache_test(xte,list(HELDOUT_SEEDS)); rows=[]
    for rep in range(start,end):
        for q in QS:
            for method in METHODS:
                p=out/'states'/f'rep{rep}_q{q}_{method}.pt'; ck=torch.load(p,map_location='cpu',weights_only=True)
                if (ck['rep'],ck['q'],ck['method'],ck['protocol_hash'])!=(rep,q,method,PROTOCOL_HASH): raise ValueError('checkpoint metadata')
                state=ck['state_dict']; model=base.TinyPatchTransformer(); model.load_state_dict(state); met=base.evaluate_cached(model,test_cache,yte,xte,mean,std)
                rows.append({"rep":rep,"q":q,"method":method,"candidate_k":K,"coverage_ratio":q/K,"state_digest":state_digest(state),**met})
    for n,h in m['checkpoints'].items():
        if sha256(out/'states'/n)!=h: raise ValueError('checkpoint mutated')
    pd.DataFrame(rows).to_csv(out/f'fashion_budget_heldout_{start}_{end}.csv',index=False); print(f'EVALUATION_COMPLETE {start}:{end}; n_test=1000 heldout_envs=24',flush=True)

def selftest():
    base.configure(); losses=torch.arange(K,dtype=torch.float32); cos=torch.eye(K)
    for q in QS:
        a=sorted(base.select_loss(losses,q)); b=sorted(base.select_gradnov(losses,cos,q,weight=NOVELTY_WEIGHT)); assert len(a)==len(b)==q and len(set(b))==q
    assert sorted(base.select_loss(losses,K))==sorted(base.select_gradnov(losses,cos,K,weight=NOVELTY_WEIGHT))==list(range(K))
    s={"b":torch.tensor([1.,2.]),"a":torch.tensor([[3.]])}; assert state_digest(s)==state_digest({"a":s['a'].clone(),"b":s['b'].clone()})
    print('SELFTEST PASS',PROTOCOL_HASH)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=('selftest','train','evaluate')); ap.add_argument('--start',type=int); ap.add_argument('--end',type=int); ap.add_argument('--output-dir'); ap.add_argument('--data-root',default='.cache/fashionmnist'); a=ap.parse_args()
    if a.mode=='selftest': selftest(); return
    if a.start is None or a.end is None or a.output_dir is None: raise ValueError('range/output required')
    (train if a.mode=='train' else evaluate)(a.start,a.end,Path(a.output_dir),a.data_root)
if __name__=='__main__': main()

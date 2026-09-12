"""Issue #2 CUDA wall-clock benchmark harness.

This file intentionally refuses to run without CUDA. It records raw hardware-specific
measurements; it does not extrapolate CPU results to GPU.
"""
from __future__ import annotations
import argparse, json, platform, subprocess, threading, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from common import MLP, geometric_environment, head_gradient_directions, load_digits_split, seed_everything, select_hard_gradient_novel, select_loss_hard

TRAIN_SEEDS=tuple(range(110000,110064)); HELDOUT_SEEDS=tuple(range(111000,111080)); COUNTS=(2,4,8,16); Q_MAX=4
BATCH=128; LR=1e-2; WD=1e-3; NOV_W=.6

class NvidiaSampler:
    def __init__(self,interval=.5): self.interval=interval;self.rows=[];self.stop_evt=threading.Event();self.thread=None
    def _loop(self):
        while not self.stop_evt.is_set():
            try:
                p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw,clocks.sm','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=2,check=True)
                vals=p.stdout.strip().splitlines()[0].split(',');self.rows.append(tuple(float(x.strip()) for x in vals[:4]))
            except Exception: pass
            self.stop_evt.wait(self.interval)
    def __enter__(self): self.thread=threading.Thread(target=self._loop,daemon=True);self.thread.start();return self
    def __exit__(self,*_): self.stop_evt.set();self.thread.join(timeout=2)
    def summary(self):
        if not self.rows:return {'gpu_util_median':np.nan,'gpu_util_p10':np.nan,'gpu_util_p90':np.nan,'nvidia_memory_mib_median':np.nan,'power_w_median':np.nan,'sm_clock_mhz_median':np.nan,'nvidia_samples':0}
        a=np.asarray(self.rows,np.float64);return {'gpu_util_median':float(np.median(a[:,0])),'gpu_util_p10':float(np.quantile(a[:,0],.1)),'gpu_util_p90':float(np.quantile(a[:,0],.9)),'nvidia_memory_mib_median':float(np.median(a[:,1])),'power_w_median':float(np.median(a[:,2])),'sm_clock_mhz_median':float(np.median(a[:,3])),'nvidia_samples':len(a)}

def metadata():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA REQUIRED: Issue #2 cannot be measured on CPU')
    d=torch.cuda.current_device();p=torch.cuda.get_device_properties(d)
    return {'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'device':torch.cuda.get_device_name(d),'compute_capability':f'{p.major}.{p.minor}','total_memory_bytes':int(p.total_memory),'multiprocessors':int(p.multi_processor_count),'python':platform.python_version(),'platform':platform.platform()}

def schedule(n,seed,m):
    tg=torch.Generator().manual_seed(seed+1);er=np.random.default_rng(seed+2);out=[]
    for _ in range(10):
        for b in torch.randperm(n,generator=tg).split(BATCH):out.append((b,er.choice(len(TRAIN_SEEDS),m,replace=False).tolist()))
    return out

def pairnov(cos,sel):
    idx=torch.tensor(sel,device=cos.device);sub=cos.index_select(0,idx).index_select(1,idx);u=torch.triu_indices(len(sel),len(sel),1,device=cos.device);return float((1-sub[u[0],u[1]]).mean().detach().cpu())

def evaluate(model,xte,yte,held,device):
    model.eval();y=torch.tensor(yte,device=device);clean=torch.tensor(xte,device=device);vals=[]
    with torch.no_grad():
        logits,_=model(clean);clean_acc=float((logits.argmax(1)==y).float().mean().cpu())
        for xx in held:
            logits,_=model(xx);vals.append(float((logits.argmax(1)==y).float().mean().cpu()))
    a=np.asarray(vals,np.float64);return {'heldout_mean':float(a.mean()),'heldout_sd':float(a.std(ddof=1)),'heldout_p10':float(np.quantile(a,.1)),'heldout_min':float(a.min()),'clean_accuracy':clean_acc}

def run_condition(envs,ty,sched,seed,m,method,regime,fixed_updates,wall_seconds,device):
    q=min(Q_MAX,m);seed_everything(seed);model=MLP().to(device);opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD);model.train();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize(device)
    updates=0;selected_nov=[];candidate_loss=[];selected_loss=[];samples=0;start=time.perf_counter();i=0
    with NvidiaSampler() as sampler:
        while True:
            if regime=='fixed_updates' and updates>=fixed_updates:break
            b,cand=sched[i%len(sched)];i+=1;b=b.to(device);cand_t=torch.tensor(cand,device=device,dtype=torch.long)
            xb=envs.index_select(0,cand_t).index_select(1,b).reshape(m*len(b),*envs.shape[2:]);yb=ty.index_select(0,b)
            logits,h=model(xb);per=torch.nn.functional.cross_entropy(logits,yb.repeat(m),reduction='none').reshape(m,-1);losses=per.mean(1);dirs=head_gradient_directions(logits,h,yb,m);cos=dirs@dirs.T
            sel=select_loss_hard(losses,q) if method=='loss_hard' else select_hard_gradient_novel(losses,cos,q=q,novelty_weight=NOV_W);sel=sorted(sel)
            selected_nov.append(pairnov(cos,sel) if len(sel)>1 else 0.);candidate_loss.append(float(losses.mean().detach().cpu()));selected_loss.append(float(losses[sel].mean().detach().cpu()))
            opt.zero_grad(set_to_none=True);per[sel].mean().backward();opt.step();updates+=1;samples+=m*len(b)
            torch.cuda.synchronize(device)  # explicit practical wall-clock boundary
            elapsed=time.perf_counter()-start
            if regime=='fixed_wallclock' and elapsed>=wall_seconds:break
    torch.cuda.synchronize(device);elapsed=time.perf_counter()-start;sm=sampler.summary()
    diag={'updates':updates,'wall_seconds':elapsed,'updates_per_second':updates/elapsed,'candidate_images_per_second':samples/elapsed,'mean_selected_pairwise_novelty':float(np.mean(selected_nov)),'mean_candidate_loss':float(np.mean(candidate_loss)),'mean_selected_loss':float(np.mean(selected_loss)),'peak_allocated_bytes':int(torch.cuda.max_memory_allocated(device)),'peak_reserved_bytes':int(torch.cuda.max_memory_reserved(device)),**sm}
    return model,diag

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',required=True);ap.add_argument('--reps',type=int,default=3);ap.add_argument('--fixed-updates',type=int,default=80);ap.add_argument('--wall-seconds',type=float,default=30.);ap.add_argument('--counts',default='2,4,8,16');args=ap.parse_args()
    meta=metadata();device=torch.device('cuda');counts=tuple(int(x) for x in args.counts.split(','));
    if not counts or any(x not in COUNTS for x in counts):raise ValueError('counts must be subset of 2,4,8,16')
    xtr,ytr,xte,yte=load_digits_split();ty=torch.tensor(ytr,device=device);train=torch.stack([torch.tensor(geometric_environment(xtr,s)) for s in TRAIN_SEEDS]).to(device);held=[torch.tensor(geometric_environment(xte,s),device=device) for s in HELDOUT_SEEDS]
    rows=[];out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    for rep in range(args.reps):
        base=4910000000+4099*rep
        for m in counts:
            sched=schedule(len(ytr),base,m)
            for method in ('loss_hard','gradnov'):
                for regime in ('fixed_updates','fixed_wallclock'):
                    model,diag=run_condition(train,ty,sched,base,m,method,regime,args.fixed_updates,args.wall_seconds,device);perf=evaluate(model,xte,yte,held,device)
                    row={'rep':rep,'candidate_count':m,'selected_count':min(Q_MAX,m),'method':method,'regime':regime,**diag,**perf};rows.append(row);print(json.dumps(row),flush=True)
                    pd.DataFrame(rows).to_csv(out/'gpu_wallclock_raw.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'gpu_wallclock_raw.csv',index=False);(out/'gpu_metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps({'event':'GPU_BENCHMARK_COMPLETE','rows':len(rows),**meta}),flush=True)
if __name__=='__main__':main()

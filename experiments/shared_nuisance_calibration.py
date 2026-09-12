"""Issue110 Stage A: coherent high-dimensional nuisance matching calibration.

Training-only. No heldout environments are constructed in this stage.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, torch
import fixed_dose_response as fd
from common import MLP, configure_determinism, geometric_environment, head_gradient_directions, load_digits_split, seed_everything, select_hard_gradient_novel, select_loss_hard

REPS=tuple(range(3500,3510)); TRAIN_SEEDS=tuple(range(89000,89064)); OFFSET=3910000000
SEVERITIES=(.05,.10,.15,.20,.30,.40,.55,.70); K=16;Q=4;BATCH=128;EPOCHS=10;LR=.01;WD=.001;NOV_W=.6
N_TOL=.03; C_TOL=.10
PROTOCOL={"issue":110,"stage":"A","reps":REPS,"train_seeds":TRAIN_SEEDS,"offset":OFFSET,"stride":4099,
"severities":SEVERITIES,"pattern_scale":.30,"K":K,"Q":Q,"batch":BATCH,"epochs":EPOCHS,"lr":LR,"wd":WD,
"novelty_weight":NOV_W,"novelty_gain_tolerance":N_TOL,"candidate_loss_tolerance":C_TOL,
"nuisance":"environment-shared deterministic zero-mean unit-SD random pixel pattern","threads":1}
PH=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()
SOURCES=("common.py","fixed_dose_response.py","shared_nuisance_calibration.py")

def source_hashes():
    root=Path(__file__).parent; return {n:fd.sha256(root/n) for n in SOURCES}

def shared_nuisance_environment(images,seed,severity,offset=31):
    x=np.asarray(images,dtype=np.float32); rng=np.random.default_rng(int(seed)*7919+offset)
    p=rng.normal(0.,1.,size=x.shape[1:]).astype(np.float64); p=(p-p.mean())/(p.std(ddof=0)+1e-12); p=p.astype(np.float32)
    out=x+(0.30*float(severity))*p[None,...]
    return np.clip(out,0.,1.).astype(np.float32)

def pairnov(cos,selected):
    idx=torch.tensor(selected,dtype=torch.long); sub=cos.index_select(0,idx).index_select(1,idx); up=torch.triu_indices(len(selected),len(selected),offset=1)
    return float((1.-sub[up[0],up[1]]).mean())

def build_schedule(n,seed):
    tg=torch.Generator().manual_seed(seed+1); er=np.random.default_rng(seed+2); out=[]
    for _ in range(EPOCHS):
        for b in torch.randperm(n,generator=tg).split(BATCH): out.append((b,er.choice(len(TRAIN_SEEDS),K,replace=False).tolist()))
    return out

def train_one(envs,ty,sched,seed,method):
    seed_everything(seed); model=MLP(); model.train(); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    novelty=[];candidate_loss=[];selected_loss=[]
    for b,cand in sched:
        logits,h=model(torch.cat([envs[e][b] for e in cand])); per=torch.nn.functional.cross_entropy(logits,ty[b].repeat(K),reduction='none').reshape(K,-1); losses=per.mean(1); gd=head_gradient_directions(logits,h,ty[b],K); cos=gd@gd.T
        sel=select_loss_hard(losses,Q) if method=='loss_hard' else select_hard_gradient_novel(losses,cos,q=Q,novelty_weight=NOV_W); sel=sorted(sel)
        novelty.append(pairnov(cos,sel));candidate_loss.append(float(losses.mean().detach()));selected_loss.append(float(losses[sel].mean().detach()))
        opt.zero_grad(set_to_none=True);per[sel].mean().backward();opt.step()
    return {"selected_pairwise_novelty":float(np.mean(novelty)),"mean_candidate_loss":float(np.mean(candidate_loss)),"mean_selected_loss":float(np.mean(selected_loss))}

def check_range(start,end):
    if start is None or end is None or start>=end or not set(range(start,end))<=set(REPS): raise ValueError('unregistered calibration range')

def run(start,end,out):
    check_range(start,end);configure_determinism(1);out.mkdir(parents=True,exist_ok=True)
    x,y,_,_=load_digits_split();ty=torch.tensor(y)
    structured=[torch.tensor(geometric_environment(x,int(s))) for s in TRAIN_SEEDS]
    nuisance={a:[torch.tensor(shared_nuisance_environment(x,int(s),a,31)) for s in TRAIN_SEEDS] for a in SEVERITIES}
    rows=[]
    for rep in range(start,end):
        seed=OFFSET+4099*rep;sched=build_schedule(len(ty),seed)
        if len(sched)!=80:raise ValueError('step count')
        for family,a,envs in [('structured',None,structured)]+[('shared_nuisance',a,nuisance[a]) for a in SEVERITIES]:
            for method in ('loss_hard','gradnov'):
                d=train_one(envs,ty,sched,seed,method);rows.append({"rep":rep,"family":family,"severity":np.nan if a is None else a,"method":method,**d})
        print(f'SHARED_NUISANCE_CALIBRATION rep={rep}; heldout=0',flush=True)
    p=out/f'shared_nuisance_calibration_{start}_{end}.csv';pd.DataFrame(rows).to_csv(p,index=False)
    (out/'manifest.json').write_text(json.dumps({"protocol":PROTOCOL,"protocol_hash":PH,"start":start,"end":end,"source_hashes":source_hashes(),"split":"digits_default","file":p.name,"sha256":fd.sha256(p),"runtime":fd.runtime(),"heldout_constructed":False,"states_saved":0},indent=2,allow_nan=False))

def summarize(root,out):
    frames=[];covered=[]
    for mp in sorted(root.rglob('manifest.json')):
        m=json.loads(mp.read_text());
        if m['protocol_hash']!=PH or m['source_hashes']!=source_hashes() or m['heldout_constructed'] or m['states_saved']!=0: raise ValueError('provenance/boundary')
        p=mp.parent/m['file'];
        if fd.sha256(p)!=m['sha256']:raise ValueError('artifact hash')
        covered.extend(range(m['start'],m['end']));frames.append(pd.read_csv(p,float_precision='round_trip'))
    if sorted(covered)!=list(REPS):raise ValueError('rep coverage')
    df=pd.concat(frames,ignore_index=True); expected=10*(1+len(SEVERITIES))*2
    if len(df)!=expected or df.duplicated(['rep','family','severity','method']).any(): raise ValueError('row grid')
    if not np.isfinite(df[['selected_pairwise_novelty','mean_candidate_loss','mean_selected_loss']].to_numpy(float)).all():raise ValueError('nonfinite')
    def per_rep(block):
        z=block.set_index('method');return pd.Series({'novelty_gain':float(z.loc['gradnov','selected_pairwise_novelty']-z.loc['loss_hard','selected_pairwise_novelty']),'candidate_loss_level':float(z.mean_candidate_loss.mean()),'selected_loss_level':float(z.mean_selected_loss.mean())})
    s=df[df.family=='structured'].groupby('rep').apply(per_rep,include_groups=False).reindex(REPS)
    rows=[];selected=None;best_key=None
    for a in SEVERITIES:
        u=df[(df.family=='shared_nuisance')&np.isclose(df.severity,a)].groupby('rep').apply(per_rep,include_groups=False).reindex(REPS)
        dn=float(u.novelty_gain.mean()-s.novelty_gain.mean());dc=float(u.candidate_loss_level.mean()-s.candidate_loss_level.mean()); eligible=abs(dn)<=N_TOL and abs(dc)<=C_TOL;score=abs(dn)/N_TOL+abs(dc)/C_TOL
        row={"severity":a,"structured_novelty_gain":float(s.novelty_gain.mean()),"nuisance_novelty_gain":float(u.novelty_gain.mean()),"novelty_gain_diff":dn,"structured_candidate_loss":float(s.candidate_loss_level.mean()),"nuisance_candidate_loss":float(u.candidate_loss_level.mean()),"candidate_loss_diff":dc,"structured_selected_loss":float(s.selected_loss_level.mean()),"nuisance_selected_loss":float(u.selected_loss_level.mean()),"calibration_score":score,"eligible":bool(eligible)};rows.append(row)
        key=(score,a)
        if eligible and (best_key is None or key<best_key):best_key=key;selected=a
    decision={"decision":"SHARED-NUISANCE MATCH CALIBRATION PASS" if selected is not None else "SHARED-NUISANCE MATCH CALIBRATION FAIL","selected_severity":selected,"novelty_gain_tolerance":N_TOL,"candidate_loss_tolerance":C_TOL,"protocol_hash":PH,"heldout_constructed":False,"states_saved":0}
    out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/'shared_nuisance_calibration_summary.csv',index=False);(out/'shared_nuisance_calibration_decision.json').write_text(json.dumps(decision,indent=2,allow_nan=False));print(json.dumps(decision),flush=True);print(pd.DataFrame(rows).to_string(index=False),flush=True)
def selftest():
    x=np.zeros((3,8,8),np.float32);a=shared_nuisance_environment(x,123,.4);b=shared_nuisance_environment(x,123,.4);c=shared_nuisance_environment(x,124,.4);assert np.array_equal(a,b) and not np.array_equal(a,c);assert np.array_equal(a[0],a[1]);print('SHARED_NUISANCE_SELFTEST_PASS',PH)
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('selftest','run','summarize'));p.add_argument('--start',type=int);p.add_argument('--end',type=int);p.add_argument('--output-dir');p.add_argument('--input-dir');a=p.parse_args();
    if a.mode=='selftest':selftest()
    elif a.mode=='run':run(a.start,a.end,Path(a.output_dir))
    else:summarize(Path(a.input_dir),Path(a.output_dir))
if __name__=='__main__':main()

"""Issue89 Stage A: match translation allocation before arm/heldout experiments."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import parameter_matched_novelty_calibration as cal
import parameter_matched_novelty_confirmatory as pm
import cnn_regime_interaction as cr
import fixed_dose_response as fd
import transfer_specificity as base
from common import SmallCNN, environment_parameters, head_gradient_directions, seed_everything

REPS=tuple(range(2000,2010)); SEEDS=tuple(range(59000,59064))
GRID=(.05,.10,.20,.30,.50); HP=.05; SLACK=1e-12; MIN_GAP=.10
PROTOCOL={'issue':89,'stage':'A','reps':REPS,'train_seeds':SEEDS,'offset':2410000000,
          'stride':4099,'K':16,'Q':4,'steps':80,'epochs':10,'batch':128,'lr':.005,'wd':.001,
          'hp_caliper':HP,'t_grid':GRID,'min_gap':MIN_GAP,'threads':1}
PH=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()
SOURCES=(*pm.LOCAL_SOURCES,'translation_matched_calibration.py')

def source_hashes():
    return {n:fd.sha256(Path(__file__).parent/n) for n in SOURCES}

def parameters(seeds):
    p=np.stack([environment_parameters(s) for s in seeds]).astype(np.float64)
    return (p-p.mean(0))/(p.std(0,ddof=0)+1e-12)

def add_translation(rows,par):
    for r in rows:
        v=par[list(r['envs'])].var(0,ddof=0)
        r['translation_score']=float(v[1]+v[2])
    return rows

def choose_pair(rows,t_caliper=None):
    """Same H/P eligibility and tie ordering as Issue83; optionally constrain T."""
    fields=('gradient_novelty','z_hard','z_param','translation_score')
    a=np.array([[r[f] for f in fields] for r in rows],dtype=np.float64)
    if a.ndim!=2 or a.shape[1]!=4 or not np.isfinite(a).all(): raise ValueError('nonfinite subset table')
    if t_caliper is not None and (not np.isfinite(t_caliper) or t_caliper<=0): raise ValueError('bad T caliper')
    ii,jj=np.triu_indices(len(rows),1); diff=a[ii]-a[jj]
    keep=(np.abs(diff[:,1])<=HP+SLACK)&(np.abs(diff[:,2])<=HP+SLACK)
    if t_caliper is not None: keep &= np.abs(diff[:,3])<=t_caliper+SLACK
    loc=np.flatnonzero(keep)
    if not len(loc): return None
    k=min(loc,key=lambda z:(-abs(diff[z,0]),abs(diff[z,2]),abs(diff[z,1]),
                            tuple(rows[ii[z]]['local']),tuple(rows[jj[z]]['local'])))
    i,j=int(ii[k]),int(jj[k]); hi,lo=(i,j) if a[i,0]>=a[j,0] else (j,i)
    return {'high':hi,'low':lo,'eligible_pairs':int(len(loc)),
            'gradient_gap':float(a[hi,0]-a[lo,0]),'hardness_diff':float(a[hi,1]-a[lo,1]),
            'parameter_z_diff':float(a[hi,2]-a[lo,2]),'translation_gap':float(a[hi,3]-a[lo,3])}

def run(start,end,out):
    if start>=end or not set(range(start,end))<=set(REPS): raise ValueError('unregistered range')
    base.configure_determinism(1); out.mkdir(parents=True,exist_ok=True)
    if (out/'calibration_manifest.json').exists(): raise ValueError('overwrite refused')
    x,y,*_=cr.data_split(); ty=torch.tensor(y); envs=base.geometric_envs(x,SEEDS); par=parameters(SEEDS)
    records=[]; all_subsets=[]
    for rep in range(start,end):
        seed=PROTOCOL['offset']+4099*rep; sched=pm.build_schedule(len(ty),seed)
        if len(sched)!=80: raise ValueError('step count')
        seed_everything(seed); model=SmallCNN(); model.train(); opt=torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=.001)
        for step,(b,cand) in enumerate(sched):
            logits,h=model(torch.cat([envs[e][b] for e in cand]))
            per=torch.nn.functional.cross_entropy(logits,ty[b].repeat(16),reduction='none').reshape(16,-1)
            losses=per.mean(1); gd=head_gradient_directions(logits,h,ty[b],16)
            subsets=add_translation(cal.feasible_subsets(losses,gd@gd.T,cand,par),par)
            for idx,s in enumerate(subsets):
                all_subsets.append({'rep':rep,'step':step,'subset_id':idx,
                    'local':';'.join(map(str,s['local'])),'envs':';'.join(map(str,s['envs'])),
                    **{f:s[f] for f in ('gradient_novelty','z_hard','z_param','translation_score','physical_diversity')}})
            for c in GRID:
                pair=choose_pair(subsets,c)
                records.append({'rep':rep,'step':step,'t_caliper':c,'feasible':pair is not None,**(pair or {})})
            selected=sorted(sorted(range(16),key=lambda j:(-float(losses[j].detach()),j))[:4])
            opt.zero_grad(set_to_none=True); per[selected].mean().backward(); opt.step()
        print('CALIBRATION_REFERENCE_COMPLETE',rep,flush=True)
    r=out/f'translation_calibration_{start}_{end}.csv'; pd.DataFrame(records).to_csv(r,index=False)
    s=out/f'translation_subsets_{start}_{end}.csv.gz'; pd.DataFrame(all_subsets).to_csv(s,index=False,compression='gzip')
    m={'protocol':PROTOCOL,'protocol_hash':PH,'start':start,'end':end,'source_hashes':source_hashes(),
       'split_hashes':cr.split_hashes(),'files':{p.name:fd.sha256(p) for p in (r,s)},'runtime':fd.runtime(),
       'heldout_constructed':False,'intervention_arms_trained':0}
    (out/'calibration_manifest.json').write_text(json.dumps(m,indent=2,allow_nan=False))

def summarize(root,out):
    covered=[]; frames=[]
    for p in sorted(root.rglob('calibration_manifest.json')):
        m=json.loads(p.read_text())
        if m['protocol_hash']!=PH or m['source_hashes']!=source_hashes() or m['split_hashes']!=cr.split_hashes(): raise ValueError('provenance')
        covered.extend(range(m['start'],m['end']))
        for name,dig in m['files'].items():
            if fd.sha256(p.parent/name)!=dig: raise ValueError('file hash')
        frames.append(pd.read_csv(p.parent/f"translation_calibration_{m['start']}_{m['end']}.csv",float_precision='round_trip'))
    if sorted(covered)!=list(REPS): raise ValueError('replicate coverage')
    df=pd.concat(frames,ignore_index=True); keys=['rep','step','t_caliper']
    expected={(r,s,c) for r in REPS for s in range(80) for c in GRID}
    if len(df)!=4000 or df.duplicated(keys).any() or set(df[keys].itertuples(index=False,name=None))!=expected: raise ValueError('grid')
    rows=[]; chosen=None
    for c in GRID:
        v=df[df.t_caliper==c]; valid=v[v.feasible.eq(True)]
        if not np.isfinite(valid[['gradient_gap','hardness_diff','parameter_z_diff','translation_gap']].to_numpy()).all(): raise ValueError('nonfinite')
        if (valid.hardness_diff.abs()>HP+SLACK).any() or (valid.parameter_z_diff.abs()>HP+SLACK).any() or (valid.translation_gap.abs()>c+SLACK).any(): raise ValueError('caliper violation')
        gap=float(valid.gradient_gap.mean()) if len(valid) else None
        passed=len(valid)==800 and gap is not None and gap>=MIN_GAP
        rows.append({'t_caliper':c,'feasible_steps':len(valid),'total_steps':800,'mean_gradient_gap':gap,'pass':passed})
        if passed and chosen is None: chosen=c
    decision={'decision':'TRANSLATION MATCH CALIBRATION PASS' if chosen is not None else 'TRANSLATION MATCH CALIBRATION FAIL',
              'selected_t_caliper':chosen,'protocol_hash':PH,'heldout_constructed':False,'intervention_arms_trained':0}
    out.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out/'translation_calibration_summary.csv',index=False)
    (out/'translation_calibration_decision.json').write_text(json.dumps(decision,indent=2,allow_nan=False))
    print(json.dumps(decision),flush=True); print(pd.DataFrame(rows).to_string(index=False),flush=True)

def selftest():
    rows=[{'local':(0,i+1),'envs':(0,i+1),'gradient_novelty':g,'z_hard':0.,'z_param':0.,'translation_score':t}
          for i,(g,t) in enumerate(((.1,0.),(.5,1.),(.3,.02),(.8,2.)))]
    p=choose_pair(rows,.05); assert p['high']==2 and p['low']==0
    old=cal.best_pair(rows,.05); new=choose_pair(rows)
    assert new['gradient_gap']==old['gradient_gap']
    assert choose_pair(rows[:2],.05) is None
    for bad in (float('nan'),-1.):
        try: choose_pair(rows,bad)
        except ValueError: pass
        else: raise AssertionError('invalid caliper accepted')
    r=[dict(x) for x in rows]; r[0]['z_hard']=float('nan')
    try: choose_pair(r)
    except ValueError: pass
    else: raise AssertionError('nonfinite accepted')
    print('SYNTHETIC_SELFTEST_PASS',PH)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('selftest','run','summarize'))
    p.add_argument('--start',type=int);p.add_argument('--end',type=int);p.add_argument('--input-dir');p.add_argument('--output-dir')
    a=p.parse_args()
    if a.mode=='selftest':selftest()
    elif a.mode=='run':run(a.start,a.end,Path(a.output_dir))
    else:summarize(Path(a.input_dir),Path(a.output_dir))
if __name__=='__main__':main()

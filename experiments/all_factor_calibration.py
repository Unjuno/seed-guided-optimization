"""Issue105 Stage A: all-coordinate physical moment matching for gradient novelty."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, torch
import corrected_centered_span_calibration as cc
import parameter_matched_novelty_confirmatory as pm
import translation_matched_calibration as tc
import cnn_regime_interaction as cr
import fixed_dose_response as fd
import transfer_specificity as base
from common import SmallCNN, head_gradient_directions, seed_everything
REPS=tuple(range(3300,3310)); SEEDS=tuple(range(85000,85064)); OFFSET=3710000000
F_GRID=(.05,.10,.20,.30); H=.05;L=.05;P=.05;SLACK=1e-12;MIN_POOL=.90;MIN_REP=.80;MIN_G=.08;MIN_REP_G=.04;K=16;Q=4
COORDS=('angle','dx','dy','blur','contrast','brightness','noise')
PROTOCOL={"issue":105,"stage":"A","reps":REPS,"train_seeds":SEEDS,"offset":OFFSET,"stride":4099,"K":K,"Q":Q,"steps":80,"epochs":10,"batch":128,"lr":.005,"wd":.001,"subset_family":"loss-rank1 plus any3 ranks2-16","subset_count":455,"h_caliper":H,"loss_variance_z_caliper":L,"p_caliper":P,"physical_moment_grid":F_GRID,"coordinates":COORDS,"min_pooled_support":MIN_POOL,"min_rep_support":MIN_REP,"min_mean_gradient_gap":MIN_G,"min_rep_mean_gradient_gap":MIN_REP_G,"threads":1}
PH=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()
SOURCES=("common.py","centered_span_calibration.py","corrected_centered_span_calibration.py","parameter_matched_novelty_calibration.py","parameter_matched_novelty_confirmatory.py","translation_matched_calibration.py","cnn_regime_interaction.py","fixed_dose_response.py","transfer_specificity.py","all_factor_calibration.py")
def source_hashes():
    root=Path(__file__).parent;return {n:fd.sha256(root/n) for n in SOURCES}
def add_moments(rows,params):
    for r in rows:
        a=params[list(r['envs'])]
        for k,name in enumerate(COORDS):
            r['mean_'+name]=float(a[:,k].mean());r['var_'+name]=float(a[:,k].var(ddof=0))
    return rows
def choose_pair(rows,c):
    fields=('gradient_novelty','z_hard','z_loss_variance','z_param')+tuple('mean_'+x for x in COORDS)+tuple('var_'+x for x in COORDS)
    a=np.asarray([[r[f] for f in fields] for r in rows],np.float64);ii,jj=np.triu_indices(len(rows),1);d=a[ii]-a[jj]
    phys=np.max(np.abs(d[:,4:]),axis=1);keep=(np.abs(d[:,1])<=H+SLACK)&(np.abs(d[:,2])<=L+SLACK)&(np.abs(d[:,3])<=P+SLACK)&(phys<=c+SLACK);loc=np.flatnonzero(keep)
    if not len(loc):return None
    z=min(loc,key=lambda k:(-abs(d[k,0]),phys[k],abs(d[k,3]),abs(d[k,2]),abs(d[k,1]),tuple(rows[ii[k]]['local']),tuple(rows[jj[k]]['local'])))
    i,j=int(ii[z]),int(jj[z]);hi,lo=(i,j) if a[i,0]>=a[j,0] else (j,i)
    out={"high":hi,"low":lo,"eligible_pairs":int(len(loc)),"gradient_gap":float(a[hi,0]-a[lo,0]),"hardness_diff":float(a[hi,1]-a[lo,1]),"loss_variance_z_diff":float(a[hi,2]-a[lo,2]),"parameter_z_diff":float(a[hi,3]-a[lo,3]),"max_physical_moment_abs_diff":float(np.max(np.abs(a[hi,4:]-a[lo,4:]))),"high_local":";".join(map(str,rows[hi]['local'])),"low_local":";".join(map(str,rows[lo]['local'])),"high_ranks":";".join(map(str,rows[hi]['ranks'])),"low_ranks":";".join(map(str,rows[lo]['ranks']))}
    for x in COORDS:
        out['mean_'+x+'_diff']=float(rows[hi]['mean_'+x]-rows[lo]['mean_'+x]);out['var_'+x+'_diff']=float(rows[hi]['var_'+x]-rows[lo]['var_'+x])
    return out
def check_range(start,end):
    if start is None or end is None or start>=end or not set(range(start,end))<=set(REPS):raise ValueError('unregistered range')
def run(start,end,out):
    check_range(start,end);base.configure_determinism(1);out.mkdir(parents=True,exist_ok=True);x,y,*_=cr.data_split();ty=torch.tensor(y);envs=base.geometric_envs(x,SEEDS);params=tc.parameters(SEEDS);pairs=[];subs=[]
    for rep in range(start,end):
        seed=OFFSET+4099*rep;sched=pm.build_schedule(len(ty),seed);seed_everything(seed);model=SmallCNN();model.train();opt=torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=.001)
        for step,(b,cand) in enumerate(sched):
            logits,h=model(torch.cat([envs[e][b] for e in cand]));per=torch.nn.functional.cross_entropy(logits,ty[b].repeat(K),reduction='none').reshape(K,-1);losses=per.mean(1);gd=head_gradient_directions(logits,h,ty[b],K);rows=add_moments(cc.expanded_subsets(losses,gd,cand,params),params)
            for c in F_GRID:
                q=choose_pair(rows,c);pairs.append({"rep":rep,"step":step,"physical_caliper":c,"feasible":q is not None,**(q or {})})
            for n,r in enumerate(rows):
                z={"rep":rep,"step":step,"subset_id":n,"local":";".join(map(str,r['local'])),"ranks":";".join(map(str,r['ranks'])),"envs":";".join(map(str,r['envs'])),"gradient_novelty":r['gradient_novelty'],"z_hard":r['z_hard'],"z_loss_variance":r['z_loss_variance'],"z_param":r['z_param']}
                for x in COORDS:z['mean_'+x]=r['mean_'+x];z['var_'+x]=r['var_'+x]
                subs.append(z)
            sel=sorted(sorted(range(K),key=lambda j:(-float(losses[j].detach()),j))[:Q]);opt.zero_grad(set_to_none=True);per[sel].mean().backward();opt.step()
        print('ALL_FACTOR_REFERENCE_COMPLETE',rep,flush=True)
    p=out/f'all_factor_pairs_{start}_{end}.csv';s=out/f'all_factor_subsets_{start}_{end}.csv.gz';pd.DataFrame(pairs).to_csv(p,index=False);pd.DataFrame(subs).to_csv(s,index=False,compression='gzip');(out/'manifest.json').write_text(json.dumps({"protocol":PROTOCOL,"protocol_hash":PH,"start":start,"end":end,"source_hashes":source_hashes(),"split_hashes":cr.split_hashes(),"files":{q.name:fd.sha256(q) for q in (p,s)},"runtime":fd.runtime(),"heldout_constructed":False,"intervention_arms_trained":0},indent=2,allow_nan=False))
def summarize(root,out):
    frames=[];covered=[]
    for mp in sorted(root.rglob('manifest.json')):
        m=json.loads(mp.read_text());
        if m['protocol_hash']!=PH or m['source_hashes']!=source_hashes() or m['split_hashes']!=cr.split_hashes():raise ValueError('provenance')
        if m['heldout_constructed'] or m['intervention_arms_trained']!=0:raise ValueError('boundary')
        for n,d in m['files'].items():
            if fd.sha256(mp.parent/n)!=d:raise ValueError('hash')
        covered.extend(range(m['start'],m['end']));frames.append(pd.read_csv(mp.parent/f"all_factor_pairs_{m['start']}_{m['end']}.csv",float_precision='round_trip'))
    if sorted(covered)!=list(REPS):raise ValueError('coverage')
    df=pd.concat(frames,ignore_index=True);exp={(r,t,c) for r in REPS for t in range(80) for c in F_GRID}
    if len(df)!=len(exp) or df.duplicated(['rep','step','physical_caliper']).any() or set(df[['rep','step','physical_caliper']].itertuples(index=False,name=None))!=exp:raise ValueError('grid')
    results=[];selected=None
    for c in F_GRID:
        v=df[np.isclose(df.physical_caliper,c)];f=v.feasible.astype(bool);q=v[f]
        moment_cols=[f'{p}_{x}_diff' for x in COORDS for p in ('mean','var')]
        if len(q) and ((q.hardness_diff.abs()>H+SLACK).any() or(q.loss_variance_z_diff.abs()>L+SLACK).any()or(q.parameter_z_diff.abs()>P+SLACK).any()or(np.abs(q[moment_cols].to_numpy(float))>c+SLACK).any()):raise ValueError('caliper')
        sr=v.assign(ok=f).groupby('rep').ok.mean().reindex(REPS,fill_value=0.);gg=q.groupby('rep').gradient_gap.mean().reindex(REPS,fill_value=0.);pooled=float(f.mean());mins=float(sr.min());mg=float(q.gradient_gap.mean()) if len(q) else 0.;mrg=float(gg.min());passed=pooled>=MIN_POOL and mins>=MIN_REP and mg>=MIN_G and mrg>=MIN_REP_G
        results.append({"physical_caliper":c,"supported_steps":int(f.sum()),"total_steps":len(v),"pooled_support":pooled,"minimum_replicate_support":mins,"mean_gradient_gap_supported":mg,"minimum_replicate_mean_gradient_gap_supported":mrg,"mean_abs_hardness_diff_supported":float(q.hardness_diff.abs().mean()) if len(q) else None,"mean_abs_loss_variance_z_diff_supported":float(q.loss_variance_z_diff.abs().mean()) if len(q) else None,"mean_abs_parameter_z_diff_supported":float(q.parameter_z_diff.abs().mean()) if len(q) else None,"mean_max_physical_moment_abs_diff_supported":float(q.max_physical_moment_abs_diff.mean()) if len(q) else None,"pass":bool(passed)})
        if passed and selected is None:selected=c
    decision={"decision":"ALL-FACTOR MATCH CALIBRATION PASS" if selected is not None else "ALL-FACTOR MATCH CALIBRATION FAIL","selected_physical_caliper":selected,"protocol_hash":PH,"heldout_constructed":False,"intervention_arms_trained":0};out.mkdir(parents=True,exist_ok=True);pd.DataFrame(results).to_csv(out/'all_factor_calibration_summary.csv',index=False);(out/'all_factor_calibration_decision.json').write_text(json.dumps(decision,indent=2,allow_nan=False));print(json.dumps(decision),flush=True);print(pd.DataFrame(results).to_string(index=False),flush=True)
def selftest():
    base={"local":(0,1,2,3),"ranks":(1,2,3,4),"gradient_novelty":.6,"z_hard":0.,"z_loss_variance":0.,"z_param":0.};a=dict(base);b=dict(base);b['local']=(0,1,2,4);b['ranks']=(1,2,3,5);b['gradient_novelty']=.8
    for x in COORDS:a['mean_'+x]=a['var_'+x]=0.;b['mean_'+x]=b['var_'+x]=.01
    q=choose_pair([a,b],.05);assert q and abs(q['gradient_gap']-.2)<1e-12;print('ALL_FACTOR_SELFTEST_PASS',PH)
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('selftest','run','summarize'));p.add_argument('--start',type=int);p.add_argument('--end',type=int);p.add_argument('--output-dir');p.add_argument('--input-dir');a=p.parse_args();
    if a.mode=='selftest':selftest()
    elif a.mode=='run':run(a.start,a.end,Path(a.output_dir))
    else:summarize(Path(a.input_dir),Path(a.output_dir))
if __name__=='__main__':main()

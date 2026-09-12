"""Issue104 Stage A: mean loss-rank allocation calibration at matched geometry."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, torch
import corrected_centered_span_calibration as cc
import opposition_calibration as oc
import parameter_matched_novelty_confirmatory as pm
import translation_matched_calibration as tc
import cnn_regime_interaction as cr
import fixed_dose_response as fd
import transfer_specificity as base
from common import SmallCNN, head_gradient_directions, seed_everything
REPS=tuple(range(3100,3110)); SEEDS=tuple(range(81000,81064)); OFFSET=3510000000
L=.05; P=.05; T=.20; G=.05; S=.10; O=.15; SLACK=1e-12; MIN_POOL=.90; MIN_REP=.80; MIN_R=1.50; MIN_REP_R=.75; MAX_ID=1e-10; K=16;Q=4
PROTOCOL={"issue":104,"stage":"A","reps":REPS,"train_seeds":SEEDS,"offset":OFFSET,"stride":4099,"K":K,"Q":Q,"steps":80,"epochs":10,"batch":128,"lr":.005,"wd":.001,"subset_family":"loss-rank1 plus any3 ranks2-16","subset_count":455,"loss_variance_z_caliper":L,"p_caliper":P,"t_caliper":T,"g_caliper":G,"span_caliper":S,"opposition_caliper":O,"min_pooled_support":MIN_POOL,"min_rep_support":MIN_REP,"min_mean_rank_gap":MIN_R,"min_rep_mean_rank_gap":MIN_REP_R,"max_general_identity_error":MAX_ID,"threads":1}
PH=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()
SOURCES=("common.py","centered_span_calibration.py","corrected_centered_span_calibration.py","opposition_calibration.py","parameter_matched_novelty_calibration.py","parameter_matched_novelty_confirmatory.py","translation_matched_calibration.py","cnn_regime_interaction.py","fixed_dose_response.py","transfer_specificity.py","rank_allocation_calibration.py")
def source_hashes():
    root=Path(__file__).parent; return {n:fd.sha256(root/n) for n in SOURCES}
def enrich(rows,gd):
    oc.add_opposition(rows,gd)
    for r in rows: r['mean_rank']=float(np.mean(np.asarray(r['ranks'],dtype=np.float64)))
    return rows
def choose_pair(rows):
    fields=("mean_rank","z_loss_variance","z_param","translation_score","gradient_novelty","centered_erank","opposition_score","z_hard")
    a=np.asarray([[r[f] for f in fields] for r in rows],np.float64); ii,jj=np.triu_indices(len(rows),1); d=a[ii]-a[jj]
    keep=((np.abs(d[:,1])<=L+SLACK)&(np.abs(d[:,2])<=P+SLACK)&(np.abs(d[:,3])<=T+SLACK)&(np.abs(d[:,4])<=G+SLACK)&(np.abs(d[:,5])<=S+SLACK)&(np.abs(d[:,6])<=O+SLACK))
    loc=np.flatnonzero(keep)
    if not len(loc): return None
    z=min(loc,key=lambda k:(-abs(d[k,0]),abs(d[k,6]),abs(d[k,5]),abs(d[k,4]),abs(d[k,3]),abs(d[k,2]),abs(d[k,1]),tuple(rows[ii[k]]['local']),tuple(rows[jj[k]]['local'])))
    i,j=int(ii[z]),int(jj[z]); hi,lo=(i,j) if a[i,0]>=a[j,0] else (j,i)
    return {"high":hi,"low":lo,"eligible_pairs":int(len(loc)),"rank_gap":float(a[hi,0]-a[lo,0]),"hardness_diff":float(a[hi,7]-a[lo,7]),"loss_variance_z_diff":float(a[hi,1]-a[lo,1]),"parameter_z_diff":float(a[hi,2]-a[lo,2]),"translation_diff":float(a[hi,3]-a[lo,3]),"gradient_novelty_diff":float(a[hi,4]-a[lo,4]),"span_diff":float(a[hi,5]-a[lo,5]),"opposition_diff":float(a[hi,6]-a[lo,6]),"high_local":";".join(map(str,rows[hi]['local'])),"low_local":";".join(map(str,rows[lo]['local'])),"high_ranks":";".join(map(str,rows[hi]['ranks'])),"low_ranks":";".join(map(str,rows[lo]['ranks']))}
def check_range(start,end):
    if start is None or end is None or start>=end or not set(range(start,end))<=set(REPS): raise ValueError('unregistered range')
def run(start,end,out):
    check_range(start,end); base.configure_determinism(1); out.mkdir(parents=True,exist_ok=True)
    x,y,*_=cr.data_split(); ty=torch.tensor(y); envs=base.geometric_envs(x,SEEDS); params=tc.parameters(SEEDS); pairs=[]; subs=[]; max_id=0.
    for rep in range(start,end):
        seed=OFFSET+4099*rep; sched=pm.build_schedule(len(ty),seed); seed_everything(seed); model=SmallCNN();model.train();opt=torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=.001)
        if len(sched)!=80: raise ValueError('steps')
        for step,(b,cand) in enumerate(sched):
            logits,h=model(torch.cat([envs[e][b] for e in cand])); per=torch.nn.functional.cross_entropy(logits,ty[b].repeat(K),reduction='none').reshape(K,-1); losses=per.mean(1); gd=head_gradient_directions(logits,h,ty[b],K)
            rows=enrich(cc.expanded_subsets(losses,gd,cand,params),gd); step_id=max(abs(float(r['general_identity_error'])) for r in rows);max_id=max(max_id,step_id); q=choose_pair(rows)
            pairs.append({"rep":rep,"step":step,"feasible":q is not None,"step_max_general_identity_error":step_id,**(q or {})})
            for n,r in enumerate(rows): subs.append({"rep":rep,"step":step,"subset_id":n,"local":";".join(map(str,r['local'])),"ranks":";".join(map(str,r['ranks'])),"envs":";".join(map(str,r['envs'])),**{f:r[f] for f in ('mean_rank','z_hard','z_loss_variance','z_param','translation_score','gradient_novelty','centered_erank','opposition_score','general_identity_error')}})
            sel=sorted(sorted(range(K),key=lambda j:(-float(losses[j].detach()),j))[:Q]);opt.zero_grad(set_to_none=True);per[sel].mean().backward();opt.step()
        print('RANK_REFERENCE_COMPLETE',rep,flush=True)
    p=out/f'rank_pairs_{start}_{end}.csv';s=out/f'rank_subsets_{start}_{end}.csv.gz';pd.DataFrame(pairs).to_csv(p,index=False);pd.DataFrame(subs).to_csv(s,index=False,compression='gzip')
    (out/'manifest.json').write_text(json.dumps({"protocol":PROTOCOL,"protocol_hash":PH,"start":start,"end":end,"source_hashes":source_hashes(),"split_hashes":cr.split_hashes(),"files":{q.name:fd.sha256(q) for q in (p,s)},"runtime":fd.runtime(),"maximum_general_identity_error":max_id,"heldout_constructed":False,"intervention_arms_trained":0},indent=2,allow_nan=False))
def summarize(root,out):
    frames=[];covered=[];max_id=0.
    for mp in sorted(root.rglob('manifest.json')):
        m=json.loads(mp.read_text());
        if m['protocol_hash']!=PH or m['source_hashes']!=source_hashes() or m['split_hashes']!=cr.split_hashes(): raise ValueError('provenance')
        if m['heldout_constructed'] or m['intervention_arms_trained']!=0: raise ValueError('boundary')
        for n,d in m['files'].items():
            if fd.sha256(mp.parent/n)!=d: raise ValueError('hash')
        covered.extend(range(m['start'],m['end']));max_id=max(max_id,abs(float(m['maximum_general_identity_error'])));frames.append(pd.read_csv(mp.parent/f"rank_pairs_{m['start']}_{m['end']}.csv",float_precision='round_trip'))
    if sorted(covered)!=list(REPS): raise ValueError('coverage')
    df=pd.concat(frames,ignore_index=True); exp={(r,t) for r in REPS for t in range(80)}
    if len(df)!=len(exp) or df.duplicated(['rep','step']).any() or set(df[['rep','step']].itertuples(index=False,name=None))!=exp: raise ValueError('grid')
    f=df.feasible.astype(bool);q=df[f]
    if len(q) and ((q.loss_variance_z_diff.abs()>L+SLACK).any() or (q.parameter_z_diff.abs()>P+SLACK).any() or (q.translation_diff.abs()>T+SLACK).any() or (q.gradient_novelty_diff.abs()>G+SLACK).any() or (q.span_diff.abs()>S+SLACK).any() or (q.opposition_diff.abs()>O+SLACK).any()): raise ValueError('caliper')
    sr=df.assign(ok=f).groupby('rep').ok.mean().reindex(REPS,fill_value=0.); rg=q.groupby('rep').rank_gap.mean().reindex(REPS,fill_value=0.); pooled=float(f.mean()); mins=float(sr.min()); mr=float(q.rank_gap.mean()) if len(q) else 0.; mrr=float(rg.min()); ident=max_id<=MAX_ID
    passed=ident and pooled>=MIN_POOL and mins>=MIN_REP and mr>=MIN_R and mrr>=MIN_REP_R
    row={"supported_steps":int(f.sum()),"total_steps":len(df),"pooled_support":pooled,"minimum_replicate_support":mins,"mean_rank_gap_supported":mr,"minimum_replicate_mean_rank_gap_supported":mrr,"mean_hardness_diff_supported":float(q.hardness_diff.mean()) if len(q) else None,"mean_abs_loss_variance_z_diff_supported":float(q.loss_variance_z_diff.abs().mean()) if len(q) else None,"mean_abs_parameter_z_diff_supported":float(q.parameter_z_diff.abs().mean()) if len(q) else None,"mean_abs_translation_diff_supported":float(q.translation_diff.abs().mean()) if len(q) else None,"mean_abs_gradient_novelty_diff_supported":float(q.gradient_novelty_diff.abs().mean()) if len(q) else None,"mean_abs_span_diff_supported":float(q.span_diff.abs().mean()) if len(q) else None,"mean_abs_opposition_diff_supported":float(q.opposition_diff.abs().mean()) if len(q) else None,"maximum_general_identity_error":max_id,"identity_pass":ident,"pass":bool(passed)}
    decision={"decision":"RANK-ALLOCATION CALIBRATION PASS" if passed else "RANK-ALLOCATION CALIBRATION FAIL","protocol_hash":PH,"maximum_general_identity_error":max_id,"heldout_constructed":False,"intervention_arms_trained":0};out.mkdir(parents=True,exist_ok=True);pd.DataFrame([row]).to_csv(out/'rank_allocation_calibration_summary.csv',index=False);(out/'rank_allocation_calibration_decision.json').write_text(json.dumps(decision,indent=2,allow_nan=False));print(json.dumps(decision),flush=True);print(pd.DataFrame([row]).to_string(index=False),flush=True)
def selftest():
    rows=[{'local':(0,1,2,3),'ranks':(1,2,3,4),'mean_rank':2.5,'z_hard':1.,'z_loss_variance':0.,'z_param':0.,'translation_score':0.,'gradient_novelty':.5,'centered_erank':2.,'opposition_score':.8},{'local':(0,5,6,7),'ranks':(1,6,7,8),'mean_rank':5.5,'z_hard':0.,'z_loss_variance':.01,'z_param':.01,'translation_score':.01,'gradient_novelty':.51,'centered_erank':2.02,'opposition_score':.82}];q=choose_pair(rows);assert q and abs(q['rank_gap']-3.)<1e-12;print('RANK_SELFTEST_PASS',PH)
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('selftest','run','summarize'));p.add_argument('--start',type=int);p.add_argument('--end',type=int);p.add_argument('--output-dir');p.add_argument('--input-dir');a=p.parse_args();
    if a.mode=='selftest':selftest()
    elif a.mode=='run':run(a.start,a.end,Path(a.output_dir))
    else:summarize(Path(a.input_dir),Path(a.output_dir))
if __name__=='__main__':main()

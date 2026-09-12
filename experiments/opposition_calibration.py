"""Issue103 Stage A: pairwise-opposition calibration at matched G and centered span."""
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

REPS=tuple(range(2900,2910)); SEEDS=tuple(range(77000,77064)); OFFSET=3310000000
S_GRID=(0.02,0.05,0.10,0.20); H=.05; L=.05; P=.05; T=.20; G=.05; SLACK=1e-12
MIN_POOL=.90; MIN_REP=.80; MIN_O=.15; MIN_REP_O=.08; MAX_ID=1e-10; K=16; Q=4
PROTOCOL={"issue":103,"stage":"A","reps":REPS,"train_seeds":SEEDS,"offset":OFFSET,"stride":4099,
"K":K,"Q":Q,"steps":80,"epochs":10,"batch":128,"lr":.005,"wd":.001,
"subset_family":"loss-rank1 plus any3 ranks2-16","subset_count":455,
"h_caliper":H,"loss_variance_z_caliper":L,"p_caliper":P,"t_caliper":T,"g_caliper":G,
"span_grid":S_GRID,"min_pooled_support":MIN_POOL,"min_rep_support":MIN_REP,
"min_mean_opposition_gap":MIN_O,"min_rep_mean_opposition_gap":MIN_REP_O,
"opposition":"max pairwise (1-dot)","max_general_identity_error":MAX_ID,"threads":1}
PH=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()
SOURCES=("common.py","centered_span_calibration.py","corrected_centered_span_calibration.py",
"parameter_matched_novelty_calibration.py","parameter_matched_novelty_confirmatory.py",
"translation_matched_calibration.py","cnn_regime_interaction.py","fixed_dose_response.py",
"transfer_specificity.py","opposition_calibration.py")

def source_hashes():
    root=Path(__file__).parent; return {n:fd.sha256(root/n) for n in SOURCES}

def add_opposition(rows,gd):
    for r in rows:
        idx=torch.tensor(r["local"],dtype=torch.long)
        a=gd.index_select(0,idx).detach().double().cpu().numpy(); gram=a@a.T
        ii,jj=np.triu_indices(Q,1); d=1.0-gram[ii,jj]
        r["opposition_score"]=float(np.max(d)); r["negative_pair_count"]=int(np.sum(gram[ii,jj]<0.0))
    return rows

def choose_pair(rows,c_s):
    fields=("opposition_score","centered_erank","gradient_novelty","z_hard","z_loss_variance","z_param","translation_score")
    a=np.asarray([[r[f] for f in fields] for r in rows],np.float64); ii,jj=np.triu_indices(len(rows),1); d=a[ii]-a[jj]
    keep=((np.abs(d[:,3])<=H+SLACK)&(np.abs(d[:,4])<=L+SLACK)&(np.abs(d[:,5])<=P+SLACK)&
          (np.abs(d[:,6])<=T+SLACK)&(np.abs(d[:,2])<=G+SLACK)&(np.abs(d[:,1])<=c_s+SLACK))
    loc=np.flatnonzero(keep)
    if not len(loc): return None
    z=min(loc,key=lambda k:(-abs(d[k,0]),abs(d[k,1]),abs(d[k,2]),abs(d[k,6]),abs(d[k,5]),abs(d[k,4]),abs(d[k,3]),tuple(rows[ii[k]]["local"]),tuple(rows[jj[k]]["local"])))
    i,j=int(ii[z]),int(jj[z]); hi,lo=(i,j) if a[i,0]>=a[j,0] else (j,i)
    return {"high":hi,"low":lo,"eligible_pairs":int(len(loc)),"opposition_gap":float(a[hi,0]-a[lo,0]),
            "span_diff":float(a[hi,1]-a[lo,1]),"gradient_novelty_diff":float(a[hi,2]-a[lo,2]),
            "hardness_diff":float(a[hi,3]-a[lo,3]),"loss_variance_z_diff":float(a[hi,4]-a[lo,4]),
            "parameter_z_diff":float(a[hi,5]-a[lo,5]),"translation_diff":float(a[hi,6]-a[lo,6]),
            "high_local":";".join(map(str,rows[hi]["local"])),"low_local":";".join(map(str,rows[lo]["local"])),
            "high_ranks":";".join(map(str,rows[hi]["ranks"])),"low_ranks":";".join(map(str,rows[lo]["ranks"]))}

def check_range(start,end):
    if start is None or end is None or start>=end or not set(range(start,end))<=set(REPS): raise ValueError("unregistered range")

def run(start,end,out):
    check_range(start,end); base.configure_determinism(1); out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists(): raise ValueError('overwrite refused')
    x,y,*_=cr.data_split(); ty=torch.tensor(y); envs=base.geometric_envs(x,SEEDS); params=tc.parameters(SEEDS)
    pairs=[]; subsets_out=[]; max_id=0.
    for rep in range(start,end):
        seed=OFFSET+4099*rep; sched=pm.build_schedule(len(ty),seed); seed_everything(seed)
        model=SmallCNN(); model.train(); opt=torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=.001)
        if len(sched)!=80: raise ValueError('steps')
        for step,(b,cand) in enumerate(sched):
            logits,h=model(torch.cat([envs[e][b] for e in cand])); per=torch.nn.functional.cross_entropy(logits,ty[b].repeat(K),reduction='none').reshape(K,-1)
            losses=per.mean(1); gd=head_gradient_directions(logits,h,ty[b],K)
            rows=add_opposition(cc.expanded_subsets(losses,gd,cand,params),gd)
            step_id=max(abs(float(r['general_identity_error'])) for r in rows); max_id=max(max_id,step_id)
            for n,r in enumerate(rows):
                subsets_out.append({"rep":rep,"step":step,"subset_id":n,"local":";".join(map(str,r['local'])),"ranks":";".join(map(str,r['ranks'])),"envs":";".join(map(str,r['envs'])),
                    **{f:r[f] for f in ('opposition_score','negative_pair_count','gradient_novelty','centered_erank','general_identity_error','z_hard','z_loss_variance','z_param','translation_score')}})
            for c in S_GRID:
                q=choose_pair(rows,c); pairs.append({"rep":rep,"step":step,"span_caliper":c,"feasible":q is not None,"step_max_general_identity_error":step_id,**(q or {})})
            sel=sorted(sorted(range(K),key=lambda j:(-float(losses[j].detach()),j))[:Q]); opt.zero_grad(set_to_none=True); per[sel].mean().backward(); opt.step()
        print(f'OPPOSITION_REFERENCE_COMPLETE rep={rep}; no arms, no heldout',flush=True)
    p=out/f'opposition_pairs_{start}_{end}.csv'; s=out/f'opposition_subsets_{start}_{end}.csv.gz'
    pd.DataFrame(pairs).to_csv(p,index=False); pd.DataFrame(subsets_out).to_csv(s,index=False,compression='gzip')
    (out/'manifest.json').write_text(json.dumps({"protocol":PROTOCOL,"protocol_hash":PH,"start":start,"end":end,"source_hashes":source_hashes(),"split_hashes":cr.split_hashes(),"files":{q.name:fd.sha256(q) for q in (p,s)},"runtime":fd.runtime(),"maximum_general_identity_error":max_id,"heldout_constructed":False,"intervention_arms_trained":0},indent=2,allow_nan=False))

def summarize(root,out):
    frames=[]; covered=[]; max_id=0.
    for mp in sorted(root.rglob('manifest.json')):
        m=json.loads(mp.read_text())
        if m['protocol_hash']!=PH or m['source_hashes']!=source_hashes() or m['split_hashes']!=cr.split_hashes(): raise ValueError('provenance')
        if m['heldout_constructed'] or m['intervention_arms_trained']!=0: raise ValueError('boundary')
        for n,dig in m['files'].items():
            if fd.sha256(mp.parent/n)!=dig: raise ValueError('hash')
        covered.extend(range(m['start'],m['end'])); max_id=max(max_id,abs(float(m['maximum_general_identity_error'])))
        frames.append(pd.read_csv(mp.parent/f"opposition_pairs_{m['start']}_{m['end']}.csv",float_precision='round_trip'))
    if sorted(covered)!=list(REPS): raise ValueError('coverage')
    df=pd.concat(frames,ignore_index=True); exp={(r,t,c) for r in REPS for t in range(80) for c in S_GRID}
    if len(df)!=len(exp) or df.duplicated(['rep','step','span_caliper']).any() or set(df[['rep','step','span_caliper']].itertuples(index=False,name=None))!=exp: raise ValueError('grid')
    result=[]; selected=None; identity=max_id<=MAX_ID
    for c in S_GRID:
        v=df[np.isclose(df.span_caliper,c)].copy(); f=v.feasible.astype(bool); q=v[f]
        if len(q):
            if (q.hardness_diff.abs()>H+SLACK).any() or (q.loss_variance_z_diff.abs()>L+SLACK).any() or (q.parameter_z_diff.abs()>P+SLACK).any() or (q.translation_diff.abs()>T+SLACK).any() or (q.gradient_novelty_diff.abs()>G+SLACK).any() or (q.span_diff.abs()>c+SLACK).any(): raise ValueError('caliper')
        sr=v.assign(ok=f).groupby('rep').ok.mean().reindex(REPS,fill_value=0.); og=q.groupby('rep').opposition_gap.mean().reindex(REPS,fill_value=0.)
        pooled=float(f.mean()); mins=float(sr.min()); mo=float(q.opposition_gap.mean()) if len(q) else 0.; mro=float(og.min())
        passed=identity and pooled>=MIN_POOL and mins>=MIN_REP and mo>=MIN_O and mro>=MIN_REP_O
        result.append({"span_caliper":c,"supported_steps":int(f.sum()),"total_steps":len(v),"pooled_support":pooled,"minimum_replicate_support":mins,"mean_opposition_gap_supported":mo,"minimum_replicate_mean_opposition_gap_supported":mro,"mean_abs_span_diff_supported":float(q.span_diff.abs().mean()) if len(q) else None,"mean_abs_gradient_novelty_diff_supported":float(q.gradient_novelty_diff.abs().mean()) if len(q) else None,"mean_abs_hardness_diff_supported":float(q.hardness_diff.abs().mean()) if len(q) else None,"mean_abs_loss_variance_z_diff_supported":float(q.loss_variance_z_diff.abs().mean()) if len(q) else None,"mean_abs_parameter_z_diff_supported":float(q.parameter_z_diff.abs().mean()) if len(q) else None,"mean_abs_translation_diff_supported":float(q.translation_diff.abs().mean()) if len(q) else None,"maximum_general_identity_error":max_id,"identity_pass":identity,"pass":bool(passed)})
        if passed and selected is None: selected=c
    decision={"decision":"OPPOSITION CALIBRATION PASS" if selected is not None else "OPPOSITION CALIBRATION FAIL","selected_span_caliper":selected,"protocol_hash":PH,"maximum_general_identity_error":max_id,"heldout_constructed":False,"intervention_arms_trained":0}
    out.mkdir(parents=True,exist_ok=True); pd.DataFrame(result).to_csv(out/'opposition_calibration_summary.csv',index=False); (out/'opposition_calibration_decision.json').write_text(json.dumps(decision,indent=2,allow_nan=False))
    print(json.dumps(decision),flush=True); print(pd.DataFrame(result).to_string(index=False),flush=True)

def selftest():
    a=torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    rows=[{"local":(0,1,2,3),"ranks":(1,2,3,4),"opposition_score":2.,"centered_erank":2.,"gradient_novelty":4/3,"z_hard":0.,"z_loss_variance":0.,"z_param":0.,"translation_score":0.},
          {"local":(0,1,2,4),"ranks":(1,2,3,5),"opposition_score":1.8,"centered_erank":2.01,"gradient_novelty":1.32,"z_hard":0.,"z_loss_variance":0.,"z_param":0.,"translation_score":0.}]
    q=choose_pair(rows,.02); assert q and abs(q['opposition_gap']-.2)<1e-9
    g=cc.geometry(a); assert abs(g['general_identity_error'])<=1e-12
    print('OPPOSITION_SELFTEST_PASS',PH)

def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('selftest','run','summarize')); p.add_argument('--start',type=int); p.add_argument('--end',type=int); p.add_argument('--output-dir'); p.add_argument('--input-dir'); a=p.parse_args()
    if a.mode=='selftest': selftest()
    elif a.mode=='run': run(a.start,a.end,Path(a.output_dir))
    else: summarize(Path(a.input_dir),Path(a.output_dir))
if __name__=='__main__': main()

"""Independent Issue124 verifier; does not import the experiment implementation."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import stats

REPS=tuple(range(6000,6060)); METHODS=("gradnov","anchor_random"); K=16; EPOCHS=10; BATCH=128
OFFSET=7500000000; REG_COMMIT="bd81e737de8ec64bf88770495c8eb51b661da87f"
REG_BLOB="3c573b2e63dffe192d4e18cbb700f85ab4235cd8"; REG_PATH="docs/CLEAN_TAIL_CONFIRM_PROTOCOL.md"
TEST_STARTS=tuple(301000+2000*p for p in range(6))
SOURCES=("common.py","cnn_regime_interaction.py","fixed_dose_response.py","transfer_specificity.py",
         "parameter_matched_novelty_confirmatory.py","parameter_matched_novelty_calibration.py","clean_tail_confirm.py")
PROTOCOL={"issue":124,"name":"clean_tail_six_pool_confirmation","registration_commit":REG_COMMIT,
"registration_path":REG_PATH,"registration_blob_sha":REG_BLOB,"reps":REPS,"methods":METHODS,"K":16,"Q":4,
"batch":128,"epochs":10,"steps":80,"lr":.005,"weight_decay":.001,"novelty_weight":.6,
"base_seed_offset":OFFSET,"base_seed_stride":4099,"anchor_rng_offset":102021,
"pool_train_starts":tuple(300000+2000*p for p in range(6)),"pool_heldout_starts":TEST_STARTS,
"blocks_per_pool":10,"n_pools":6,"train_environments":64,"heldout_environments":80,
"primary_unit":"six_pool_level_contrasts","primary_df":5,"alpha":.05,"threads":1}
PH=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()

def req(x,m):
    if not bool(x): raise ValueError(m)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): h.update(c)
    return h.hexdigest()

def blob(p):
    b=Path(p).read_bytes(); return hashlib.sha1(f"blob {len(b)}\0".encode()+b).hexdigest()

def sdigest(state):
    h=hashlib.sha256()
    for n,t in sorted(state.items()):
        a=t.detach().cpu().contiguous().numpy()
        h.update(n.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()

def sched_digest(rep):
    seed=OFFSET+4099*rep; tg=torch.Generator().manual_seed(seed+1); rng=np.random.default_rng(seed+2); h=hashlib.sha256()
    for _ in range(EPOCHS):
        for b in torch.randperm(988,generator=tg).split(BATCH):
            cand=rng.choice(64,K,replace=False); h.update(b.numpy().astype(np.int64).tobytes()); h.update(np.asarray(cand,dtype=np.int64).tobytes())
    return h.hexdigest()

def pool(rep): return (rep-6000)//10

def test_seeds(rep):
    s=TEST_STARTS[pool(rep)]; return set(range(s,s+80))

def primary(x,negative=False):
    x=np.asarray(x,float); req(x.shape==(6,) and np.isfinite(x).all(),'primary sample')
    m=float(x.mean()); se=float(x.std(ddof=1)/np.sqrt(6)); k=float(stats.t.ppf(.975,5))
    if se==0: return dict(mean=m,se=se,k95=k,ci95_low=m,ci95_high=m,t=None,p_one_sided=None,pass_=False,degenerate=True)
    t=m/se; p=float(stats.t.cdf(t,5) if negative else stats.t.sf(t,5)); ok=(m<0 if negative else m>0) and p<.05
    return dict(mean=m,se=se,k95=k,ci95_low=m-k*se,ci95_high=m+k*se,t=float(t),p_one_sided=p,pass_=bool(ok),degenerate=False)

def eq(a,b,t=1e-12):
    if a is None or (isinstance(a,float) and np.isnan(a)): return b is None or (isinstance(b,float) and np.isnan(b))
    if b is None or (isinstance(b,float) and np.isnan(b)): return False
    return abs(float(a)-float(b))<=t

def run(root,seal_path,agg,out):
    here=Path(__file__).parent; req(blob(here.parent/REG_PATH)==REG_BLOB,'protocol blob')
    sources={n:sha(here/n) for n in SOURCES}; seal=json.loads(seal_path.read_text())
    req(seal['protocol_hash']==PH and seal['source_hashes']==sources and seal['states']==120 and seal['reps']==list(REPS),'seal')
    state_idx={(int(e['rep']),e['method']):e for e in seal['state_entries']}; req(len(state_idx)==120,'state index')
    held=[]; env=[]; train=[]; covered=[]; states=0; hashes=0; schedules=0; split=None
    for ep in sorted(root.rglob('evaluation_manifest.json')):
        d=ep.parent; em=json.loads(ep.read_text()); tm=json.loads((d/'training_manifest.json').read_text())
        req(tm['protocol_hash']==PH==em['protocol_hash'] and tm['source_hashes']==sources==em['source_hashes'],'manifest provenance')
        split=tm['split_hashes'] if split is None else split; req(tm['split_hashes']==split==em['split_hashes'],'split')
        req(not tm['heldout_constructed'] and em['heldout_constructed'],'boundary'); req(em['training_manifest_sha256']==sha(d/'training_manifest.json'),'manifest chain')
        req(em['global_seal_sha256']==sha(seal_path),'seal chain')
        for n,h in {**tm['files'],**em['files']}.items():
            req(sha(d/n)==h,'file hash'); hashes+=1
        covered.extend(range(tm['start'],tm['end'])); hh=pd.read_csv(d/'heldout.csv',float_precision='round_trip'); hi=hh.set_index(['rep','method'])
        for rep in range(tm['start'],tm['end']):
            q=sched_digest(rep); req(tm['schedule_digests'][str(rep)]==q,'schedule'); schedules+=1; initials=set()
            for method in METHODS:
                sp=d/'states'/f'rep{rep}_{method}.pt'; ck=torch.load(sp,map_location='cpu',weights_only=True); s=sdigest(ck['state_dict']); z=state_idx[(rep,method)]
                req(s==ck['state_digest']==hi.loc[(rep,method),'state_digest']==z['state_tensor_digest'],'tensor digest'); req(sha(sp)==z['state_file_sha256'],'state file')
                req(ck['schedule_digest']==q==z['schedule_digest'],'schedule chain'); req(ck['initial_digest']==tm['initial_digests'][str(rep)]==z['initial_digest'],'initial chain'); initials.add(ck['initial_digest']); states+=1
            req(len(initials)==1,'paired init')
        held.append(hh); env.append(pd.read_csv(d/'environment.csv.gz',float_precision='round_trip')); train.append(pd.read_csv(d/'training_steps.csv.gz',float_precision='round_trip'))
    req(sorted(covered)==list(REPS) and states==120 and schedules==60,'coverage')
    h=pd.concat(held,ignore_index=True); e=pd.concat(env,ignore_index=True); tr=pd.concat(train,ignore_index=True)
    req(len(h)==120 and len(e)==9600 and len(tr)==9600,'grid sizes')
    req(not h.duplicated(['rep','method']).any() and not e.duplicated(['rep','method','env_seed']).any() and not tr.duplicated(['rep','method','step']).any(),'duplicates')
    req(tr.groupby(['rep','method']).size().eq(80).all(),'training steps'); hi=h.set_index(['rep','method']); ce=ae=cl=0.
    for (rep,method),b in e.groupby(['rep','method']):
        rep=int(rep); req(set(map(int,b.env_seed))==test_seeds(rep),'env ids'); req((b.n_images==809).all(),'n images')
        ce=max(ce,float(np.max(np.abs(b.accuracy.to_numpy(float)-b.correct.to_numpy(float)/809.))))
        v=b.sort_values('env_seed').accuracy.to_numpy(float); obs=np.array([v.mean(),v.std(ddof=1),np.quantile(v,.1),v.min()])
        stored=hi.loc[(rep,method),['mean_test','sd_test','p10_test','min_test']].to_numpy(float); ae=max(ae,float(np.max(np.abs(obs-stored))))
        cl=max(cl,abs(float(hi.loc[(rep,method),'clean_test'])-float(hi.loc[(rep,method),'clean_correct'])/809.))
    req(ce<=1e-15 and ae<=1e-12 and cl<=1e-15,'count reconstruction')
    rows=[]
    for rep in REPS:
        g=hi.loc[(rep,'gradnov')]; a=hi.loc[(rep,'anchor_random')]
        rows.append(dict(rep=rep,pool=pool(rep),minimum_contrast=float(g.min_test-a.min_test),clean_contrast=float(g.clean_test-a.clean_test),
                         mean_contrast=float(g.mean_test-a.mean_test),p10_contrast=float(g.p10_test-a.p10_test),sd_contrast=float(g.sd_test-a.sd_test)))
    blocks=pd.DataFrame(rows); pools=blocks.groupby('pool',as_index=False)[['minimum_contrast','clean_contrast','mean_contrast','p10_contrast','sd_contrast']].mean()
    req(len(blocks)==60 and len(pools)==6,'contrasts')
    tb=pd.read_csv(agg/'clean_tail_block_contrasts60.csv',float_precision='round_trip'); tp=pd.read_csv(agg/'clean_tail_pool_contrasts6.csv',float_precision='round_trip')
    req(np.max(np.abs(tb.select_dtypes(include=[np.number]).to_numpy()-blocks.select_dtypes(include=[np.number]).to_numpy()))<=1e-12,'block reproduction')
    req(np.max(np.abs(tp.select_dtypes(include=[np.number]).to_numpy()-pools.select_dtypes(include=[np.number]).to_numpy()))<=1e-12,'pool reproduction')
    w=primary(pools.minimum_contrast); c=primary(pools.clean_contrast,negative=True)
    if w['degenerate'] or c['degenerate']: decision='INVALID / EXECUTION FAILURE'
    elif w['pass_'] and c['pass_']: decision='CLEAN-TAIL POLICY TRADEOFF SUPPORT'
    elif w['pass_']: decision='TAIL ADVANTAGE ONLY / CLEAN COST NOT CONFIRMED'
    elif c['pass_']: decision='CLEAN COST ONLY / TAIL ADVANTAGE NOT CONFIRMED'
    else: decision='NO CLEAN-TAIL POLICY TRADEOFF CONFIRMATION'
    pr=pd.read_csv(agg/'clean_tail_primary6.csv',float_precision='round_trip'); req(list(pr.endpoint)==['minimum_accuracy','clean_accuracy'],'primary rows')
    for row,calc in zip(pr.to_dict('records'),(w,c)):
        for k in ('mean','se','k95','ci95_low','ci95_high','t','p_one_sided'): req(eq(row[k],calc[k]),'stat '+k)
    dec=json.loads((agg/'clean_tail_decision60.json').read_text()); req(dec['decision']==decision and dec['states_checked']==120 and dec['environment_rows']==9600,'decision')
    out.mkdir(parents=True,exist_ok=True)
    report={'verification':'CLEAN_TAIL_INDEPENDENT_VERIFICATION_PASS','decision':decision,'protocol_hash':PH,'registration_commit':REG_COMMIT,
            'file_hashes_checked':hashes,'state_tensors_checked':states,'schedule_digests_reconstructed':schedules,
            'environment_count_rows_checked':len(e),'training_step_rows_checked':len(tr),'block_contrasts_reconstructed':60,
            'pool_contrasts_reconstructed':6,'maximum_accuracy_from_integer_count_error':ce,'maximum_environment_aggregate_error':ae,
            'maximum_clean_count_error':cl,'minimum_primary':w,'clean_primary':c,'raw_image_inference_or_retraining':False}
    (out/'clean_tail_independent_verification.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    blocks.to_csv(out/'clean_tail_verify_block_contrasts60.csv',index=False); pools.to_csv(out/'clean_tail_verify_pool_contrasts6.csv',index=False)
    print(json.dumps(report,allow_nan=False),flush=True)

def selftest():
    req(len(REPS)==60 and pool(6059)==5,'bounds')
    x=np.array([.01,.02,.03,0,.01,.02]); a=primary(x); b=stats.ttest_1samp(x,0,alternative='greater')
    req(abs(a['p_one_sided']-float(b.pvalue))<1e-12,'t test')
    print('CLEAN_TAIL_VERIFY_SYNTHETIC_PASS',PH,flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=('selftest','verify')); p.add_argument('--input-dir',type=Path)
    p.add_argument('--global-seal',type=Path); p.add_argument('--aggregate-dir',type=Path); p.add_argument('--output-dir',type=Path); a=p.parse_args()
    selftest() if a.mode=='selftest' else run(a.input_dir,a.global_seal,a.aggregate_dir,a.output_dir)
if __name__=='__main__': main()

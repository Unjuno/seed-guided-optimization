"""Issue116 Stage B: matched reusable-structure conversion test.

Global barriers:
1. train and seal all 120 states + diagnostics;
2. verify fresh training-only novelty/difficulty match;
3. only then construct heldout environments and evaluate.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import stats

import matched_novelty_reusability as old
import shared_nuisance_calibration as sn
import fixed_dose_response as fd
from common import MLP, configure_determinism, geometric_environment, load_digits_split

REPS=tuple(range(4300,4330)); TRAIN_SEEDS=tuple(range(105000,105064)); HELDOUT_SEEDS=tuple(range(106000,106080))
OFFSET=4710000000; SEVERITY=3.0; N_TOL=.03; C_TOL=.10
FAMILIES=('structured','shared_nuisance'); METHODS=('loss_hard','gradnov')
PROTOCOL={"issue":116,"stage":"B","reps":REPS,"train_seeds":TRAIN_SEEDS,"heldout_seeds":HELDOUT_SEEDS,
"offset":OFFSET,"stride":4099,"severity":SEVERITY,"novelty_gain_tolerance":N_TOL,"candidate_loss_tolerance":C_TOL,
"K":old.K,"Q":old.Q,"batch":old.BATCH_SIZE,"epochs":old.EPOCHS,"lr":old.LR,"wd":old.WEIGHT_DECAY,
"novelty_weight":old.NOVELTY_WEIGHT,"nuisance_pattern_scale":.30,"train_pattern_offset":31,"heldout_pattern_offset":71,"threads":1}
PH=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()
SOURCES=('common.py','fixed_dose_response.py','matched_novelty_reusability.py','shared_nuisance_calibration.py','strong_shared_nuisance_confirmation.py')

def configure():
    old.TRAIN_SEEDS=np.asarray(TRAIN_SEEDS,dtype=int); old.HELDOUT_SEEDS=np.asarray(HELDOUT_SEEDS,dtype=int)

def source_hashes():
    root=Path(__file__).parent; return {n:fd.sha256(root/n) for n in SOURCES}

def state_digest(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        a=t.detach().cpu().contiguous().numpy(); h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def write_json(path,obj): path.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False))
def check_range(start,end):
    if start is None or end is None or start>=end or not set(range(start,end))<=set(REPS): raise ValueError('unregistered range')

def train_envs(x):
    return {
      'structured':[torch.tensor(geometric_environment(x,int(s))) for s in TRAIN_SEEDS],
      'shared_nuisance':[torch.tensor(sn.shared_nuisance_environment(x,int(s),SEVERITY,31)) for s in TRAIN_SEEDS],
    }

def train(start,end,out):
    configure();check_range(start,end);configure_determinism(1);out.mkdir(parents=True,exist_ok=True);(out/'states').mkdir(exist_ok=True)
    if (out/'train_manifest.json').exists(): raise ValueError('overwrite refused')
    x,y,_,_=load_digits_split();ty=torch.tensor(y);env=train_envs(x);rows=[];files={};states={}
    for rep in range(start,end):
        seed=OFFSET+4099*rep;sched=old.build_schedule(len(ty),seed)
        if len(sched)!=80: raise ValueError('step count')
        for family in FAMILIES:
            for method in METHODS:
                model,diag=old.train_one(env[family],ty,sched,seed,method); state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; dig=state_digest(state)
                p=out/'states'/f'rep{rep}_{family}_{method}.pt';torch.save({'state_dict':state,'rep':rep,'family':family,'method':method,'severity':SEVERITY,'protocol_hash':PH,'state_digest':dig},p)
                files[str(p.relative_to(out))]=fd.sha256(p);states[f'{rep}:{family}:{method}']=dig
                rows.append({'rep':rep,'family':family,'method':method,'severity':np.nan if family=='structured' else SEVERITY,**diag,'state_digest':dig})
        print(f'STRONG_REUSABILITY_TRAIN_SEALED rep={rep}; heldout=0',flush=True)
    p=out/f'strong_reusability_training_{start}_{end}.csv';pd.DataFrame(rows).to_csv(p,index=False);files[p.name]=fd.sha256(p)
    write_json(out/'train_manifest.json',{'protocol':PROTOCOL,'protocol_hash':PH,'start':start,'end':end,'source_hashes':source_hashes(),'files':files,'state_digests':states,'runtime':fd.runtime(),'heldout_constructed':False,'states_saved':4*(end-start)})

def validate_train_shard(root,start,end):
    m=json.loads((root/'train_manifest.json').read_text())
    if (m['start'],m['end'],m['protocol_hash'])!=(start,end,PH) or m['source_hashes']!=source_hashes() or m['heldout_constructed']:raise ValueError('train provenance')
    if m['states_saved']!=4*(end-start):raise ValueError('state count')
    for n,d in m['files'].items():
        if fd.sha256(root/n)!=d:raise ValueError('train hash')
    return m

def diagnostics(root):
    ps=sorted(root.rglob('strong_reusability_training_*.csv'))
    if not ps:raise FileNotFoundError('training diagnostics')
    return pd.concat([pd.read_csv(p,float_precision='round_trip') for p in ps],ignore_index=True)

def family_match(df):
    expected={(r,f,m) for r in REPS for f in FAMILIES for m in METHODS}
    if len(df)!=len(expected) or df.duplicated(['rep','family','method']).any() or set(df[['rep','family','method']].itertuples(index=False,name=None))!=expected:raise ValueError('diagnostic grid')
    if not np.isfinite(df[['selected_pairwise_novelty','mean_candidate_loss','mean_selected_loss']].to_numpy(float)).all():raise ValueError('nonfinite diagnostics')
    rec=[]
    for rep in REPS:
        z=df[df.rep==rep].set_index(['family','method'])
        ns=float(z.loc[('structured','gradnov'),'selected_pairwise_novelty']-z.loc[('structured','loss_hard'),'selected_pairwise_novelty'])
        nn=float(z.loc[('shared_nuisance','gradnov'),'selected_pairwise_novelty']-z.loc[('shared_nuisance','loss_hard'),'selected_pairwise_novelty'])
        cs=float(z.loc['structured'].mean_candidate_loss.mean());cn=float(z.loc['shared_nuisance'].mean_candidate_loss.mean())
        rec.append({'rep':rep,'structured_novelty_gain':ns,'nuisance_novelty_gain':nn,'novelty_gain_diff':nn-ns,'structured_candidate_loss':cs,'nuisance_candidate_loss':cn,'candidate_loss_diff':cn-cs})
    r=pd.DataFrame(rec);dn=float(r.novelty_gain_diff.mean());dc=float(r.candidate_loss_diff.mean());return r,dn,dc

def train_seal(root,out):
    configure();covered=[];shards=[];allstates={}
    for mp in sorted(root.rglob('train_manifest.json')):
        m=json.loads(mp.read_text());start,end=int(m['start']),int(m['end']);check_range(start,end);validate_train_shard(mp.parent,start,end);covered.extend(range(start,end));shards.append({'start':start,'end':end,'manifest_sha256':fd.sha256(mp)});allstates.update(m['state_digests'])
    if sorted(covered)!=list(REPS) or len(allstates)!=120:raise ValueError('global train coverage')
    df=diagnostics(root);perrep,dn,dc=family_match(df);passed=abs(dn)<=N_TOL and abs(dc)<=C_TOL
    out.mkdir(parents=True,exist_ok=True);perrep.to_csv(out/'fresh_training_match30.csv',index=False)
    seal={'protocol_hash':PH,'stage':'train','reps':list(REPS),'severity':SEVERITY,'novelty_gain_diff':dn,'candidate_loss_diff':dc,'novelty_gain_tolerance':N_TOL,'candidate_loss_tolerance':C_TOL,'match_pass':bool(passed),'state_digests':allstates,'shards':sorted(shards,key=lambda z:z['start'])}
    write_json(out/'train_seal.json',seal);print(json.dumps(seal),flush=True)
    if not passed:raise RuntimeError(f'STRONG SHARED-NUISANCE CONFIRMATORY MATCH FAILURE dN={dn} dC={dc}')

def validate_seal(path):
    s=json.loads(path.read_text())
    if s['protocol_hash']!=PH or s['stage']!='train' or s['reps']!=list(REPS) or not s['match_pass'] or float(s['severity'])!=SEVERITY:raise ValueError('global train seal')
    return s

def evaluate(start,end,out,seal_path):
    configure();check_range(start,end);configure_determinism(1);seal=validate_seal(seal_path);validate_train_shard(out,start,end)
    _,_,x,y=load_digits_split();labels=torch.tensor(y);clean=torch.tensor(x).flatten(1)
    held={
      'structured':[torch.tensor(geometric_environment(x,int(s))) for s in HELDOUT_SEEDS],
      'shared_nuisance':[torch.tensor(sn.shared_nuisance_environment(x,int(s),SEVERITY,71)) for s in HELDOUT_SEEDS],
    }
    rows=[];envrows=[]
    for rep in range(start,end):
        for family in FAMILIES:
            for method in METHODS:
                p=out/'states'/f'rep{rep}_{family}_{method}.pt';ck=torch.load(p,map_location='cpu',weights_only=True)
                key=f'{rep}:{family}:{method}'
                if (ck['rep'],ck['family'],ck['method'],ck['protocol_hash'])!=(rep,family,method,PH) or ck['state_digest']!=seal['state_digests'][key] or state_digest(ck['state_dict'])!=ck['state_digest']:raise ValueError('state seal')
                model=MLP();model.load_state_dict(ck['state_dict']);model.eval();vals=[]
                with torch.no_grad():
                    ca=float((model(clean)[0].argmax(1)==labels).float().mean())
                    for s,xx in zip(HELDOUT_SEEDS,held[family]):
                        logits,_=model(xx.flatten(1));a=float((logits.argmax(1)==labels).float().mean());vals.append(a);envrows.append({'rep':rep,'family':family,'method':method,'env_seed':s,'accuracy':a})
                v=np.asarray(vals,np.float64);rows.append({'rep':rep,'family':family,'method':method,'mean_test':float(v.mean()),'sd_test':float(v.std(ddof=1)),'p10_test':float(np.quantile(v,.1)),'min_test':float(v.min()),'clean_test':ca,'state_digest':ck['state_digest']})
        print(f'STRONG_REUSABILITY_EVALUATED rep={rep}',flush=True)
    hp=out/f'strong_reusability_heldout_{start}_{end}.csv';ep=out/f'strong_reusability_environment_{start}_{end}.csv.gz';pd.DataFrame(rows).to_csv(hp,index=False);pd.DataFrame(envrows).to_csv(ep,index=False,compression='gzip')
    write_json(out/'evaluation_manifest.json',{'protocol_hash':PH,'start':start,'end':end,'source_hashes':source_hashes(),'train_manifest_sha256':fd.sha256(out/'train_manifest.json'),'global_seal_sha256':fd.sha256(seal_path),'files':{hp.name:fd.sha256(hp),ep.name:fd.sha256(ep)},'runtime':fd.runtime()})

def validate_eval(root,start,end,seal_path):
    validate_train_shard(root,start,end);m=json.loads((root/'evaluation_manifest.json').read_text())
    if (m['start'],m['end'],m['protocol_hash'])!=(start,end,PH) or m['source_hashes']!=source_hashes() or m['train_manifest_sha256']!=fd.sha256(root/'train_manifest.json') or m['global_seal_sha256']!=fd.sha256(seal_path):raise ValueError('eval provenance')
    for n,d in m['files'].items():
        if fd.sha256(root/n)!=d:raise ValueError('eval hash')

def load_frames(root,pattern):
    ps=sorted(root.rglob(pattern));
    if not ps:raise FileNotFoundError(pattern)
    return pd.concat([pd.read_csv(p,float_precision='round_trip') for p in ps],ignore_index=True)
def estimate(x):
    x=np.asarray(x,np.float64)
    if len(x)<2 or not np.isfinite(x).all():raise ValueError('sample')
    mean=float(x.mean());se=float(stats.sem(x));k=float(stats.t.ppf(.975,len(x)-1));p=float(stats.t.sf(mean/se,len(x)-1)) if se>0 else (0. if mean>0 else 1. if mean<0 else .5)
    return {'n':len(x),'mean':mean,'se':se,'ci95_low':mean-k*se,'ci95_high':mean+k*se,'p_one_sided':p,'positive_pairs':int((x>0).sum())}

def summarize(root,out,seal_path):
    configure();seal=validate_seal(seal_path);covered=[]
    for mp in sorted(root.rglob('evaluation_manifest.json')):
        m=json.loads(mp.read_text());start,end=int(m['start']),int(m['end']);check_range(start,end);validate_eval(mp.parent,start,end,seal_path);covered.extend(range(start,end))
    if sorted(covered)!=list(REPS):raise ValueError('eval coverage')
    train=diagnostics(root);perrep,dn,dc=family_match(train)
    if abs(dn)>N_TOL or abs(dc)>C_TOL or abs(dn-float(seal['novelty_gain_diff']))>1e-12 or abs(dc-float(seal['candidate_loss_diff']))>1e-12:raise ValueError('training match changed')
    held=load_frames(root,'strong_reusability_heldout_*.csv');env=load_frames(root,'strong_reusability_environment_*.csv.gz')
    expected={(r,f,m) for r in REPS for f in FAMILIES for m in METHODS};expected_env={(r,f,m,s) for r in REPS for f in FAMILIES for m in METHODS for s in HELDOUT_SEEDS}
    if len(held)!=len(expected) or held.duplicated(['rep','family','method']).any() or set(held[['rep','family','method']].itertuples(index=False,name=None))!=expected:raise ValueError('held grid')
    if len(env)!=len(expected_env) or env.duplicated(['rep','family','method','env_seed']).any() or set(env[['rep','family','method','env_seed']].itertuples(index=False,name=None))!=expected_env:raise ValueError('env grid')
    if not np.isfinite(held[['mean_test','sd_test','p10_test','min_test','clean_test']].to_numpy(float)).all() or not np.isfinite(env.accuracy.to_numpy(float)).all():raise ValueError('nonfinite heldout')
    hz=held.set_index(['rep','family','method']);maxerr=0.;paired=[]
    for (rep,f,m),block in env.groupby(['rep','family','method']):
        v=block.sort_values('env_seed').accuracy.to_numpy(np.float64);obs=np.array([v.mean(),v.std(ddof=1),np.quantile(v,.1),v.min()]);repv=hz.loc[(rep,f,m),['mean_test','sd_test','p10_test','min_test']].to_numpy(float);maxerr=max(maxerr,float(np.max(np.abs(obs-repv))))
    if maxerr>1e-12:raise ValueError('reaggregation')
    for rep in REPS:
        bs=float(hz.loc[(rep,'structured','gradnov'),'mean_test']-hz.loc[(rep,'structured','loss_hard'),'mean_test']);bn=float(hz.loc[(rep,'shared_nuisance','gradnov'),'mean_test']-hz.loc[(rep,'shared_nuisance','loss_hard'),'mean_test'])
        row={'rep':rep,'structured_benefit':bs,'nuisance_benefit':bn,'conversion_contrast':bs-bn}
        for f in FAMILIES:
            for met in ('sd_test','p10_test','min_test','clean_test'):
                row[f+'_'+met+'_benefit']=float(hz.loc[(rep,f,'gradnov'),met]-hz.loc[(rep,f,'loss_hard'),met])
        paired.append(row)
    paired=pd.DataFrame(paired);primary=estimate(paired.conversion_contrast);sp=estimate(paired.structured_benefit);npf=estimate(paired.nuisance_benefit);performance=primary['mean']>0 and primary['p_one_sided']<.05
    decision='STRONG MATCHED REUSABLE-STRUCTURE CONVERSION SUPPORT' if performance else 'NO STRONG MATCHED REUSABLE-STRUCTURE CONVERSION SUPPORT'
    summary=[]
    for c in paired.columns:
        if c!='rep':summary.append({'endpoint':c,**estimate(paired[c])})
    d={'decision':decision,'n':30,'severity':SEVERITY,'training_match_pass':True,'novelty_gain_diff':dn,'candidate_loss_diff':dc,'performance_pass':bool(performance),'environment_rows':len(env),'max_environment_reaggregation_error':maxerr,'protocol_hash':PH}
    out.mkdir(parents=True,exist_ok=True);paired.to_csv(out/'strong_reusability_primary30.csv',index=False);pd.DataFrame(summary).to_csv(out/'strong_reusability_summary30.csv',index=False);held.to_csv(out/'strong_reusability_heldout30.csv',index=False);perrep.to_csv(out/'strong_reusability_training_match30.csv',index=False);pd.DataFrame([d]).to_csv(out/'strong_reusability_decision30.csv',index=False)
    print(json.dumps(d),flush=True);print('PRIMARY',json.dumps(primary),flush=True);print('STRUCTURED',json.dumps(sp),flush=True);print('NUISANCE',json.dumps(npf),flush=True)

def selftest():
    configure();x=np.zeros((2,8,8),np.float32);a=sn.shared_nuisance_environment(x,11,SEVERITY,31);assert np.array_equal(a[0],a[1]);assert len(REPS)==30 and SEVERITY==3.;print('STRONG_REUSABILITY_SELFTEST_PASS',PH)
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('selftest','train','train-seal','evaluate','summarize'));p.add_argument('--start',type=int);p.add_argument('--end',type=int);p.add_argument('--input-dir');p.add_argument('--output-dir');p.add_argument('--global-seal');a=p.parse_args();configure()
    if a.mode=='selftest':selftest();return
    if a.output_dir is None:raise ValueError('output-dir required')
    out=Path(a.output_dir)
    if a.mode=='train':train(a.start,a.end,out)
    elif a.mode=='train-seal':train_seal(Path(a.input_dir),out)
    elif a.mode=='evaluate':evaluate(a.start,a.end,out,Path(a.global_seal))
    else:summarize(Path(a.input_dir),out,Path(a.global_seal))
if __name__=='__main__':main()

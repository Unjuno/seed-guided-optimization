"""Prospective online gradient-information controls. See docs/ONLINE_CONTROL_PROTOCOL.md.

No historical experimental source is modified. Modes are synthetic selftest,
train, global seal, evaluate, and aggregate. All 240 states must be sealed before
any heldout environment is constructed. Statistics use training blocks, not rows.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats
import common
import cnn_regime_interaction as cr
import fixed_dose_response as fd
import transfer_specificity as envbase
import parameter_matched_novelty_confirmatory as pm

REPS=tuple(range(5000,5060))
METHODS=('loss_hard','gradnov','anchor_random','sham_gradnov')
COMPARATORS=('loss_hard','anchor_random','sham_gradnov')
K,Q,BATCH,EPOCHS=16,4,128,10
OFFSET=6500000000
PROTOCOL={'name':'online_information_controls','reps':REPS,'methods':METHODS,
          'comparators':COMPARATORS,'K':K,'Q':Q,'batch':BATCH,'epochs':EPOCHS,
          'steps':80,'lr':.005,'weight_decay':.001,'novelty_weight':.6,
          'base_seed_offset':OFFSET,'base_seed_stride':4099,
          'pool_train_starts':(200000,202000,204000),
          'pool_heldout_starts':(201000,203000,205000),
          'blocks_per_pool':20,'train_environments':64,'heldout_environments':80,
          'alpha':.05,'multiplicity':'Holm3','threads':1,
          'registration':'docs/ONLINE_CONTROL_PROTOCOL.md'}
PH=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()
SOURCES=('common.py','cnn_regime_interaction.py','fixed_dose_response.py',
         'transfer_specificity.py','parameter_matched_novelty_confirmatory.py',
         'parameter_matched_novelty_calibration.py','online_information_controls.py')

def require(ok,message):
    if not bool(ok): raise ValueError(message)

def write(path,obj):
    path.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n')

def hashes(): return {n:fd.sha256(Path(__file__).parent/n) for n in SOURCES}

def pool(rep):
    require(rep in REPS,'unregistered repetition')
    return (rep-5000)//20

def seeds(rep,heldout=False):
    key='pool_heldout_starts' if heldout else 'pool_train_starts'
    n=80 if heldout else 64
    s=PROTOCOL[key][pool(rep)]
    return tuple(range(s,s+n))

def check_range(start,end):
    require(start is not None and end is not None and start<end,'range')
    require(set(range(start,end))<=set(REPS),'unregistered range')

def schedule(seed):
    tg=torch.Generator().manual_seed(seed+1);rng=np.random.default_rng(seed+2)
    return [(b,rng.choice(64,K,replace=False).tolist()) for _ in range(EPOCHS)
            for b in torch.randperm(988,generator=tg).split(BATCH)]

def choose(method,losses,gram,rng):
    require(method in METHODS,'method')
    require(len(losses)==K and gram.shape==(K,K),'selector shape')
    if method=='loss_hard': selected=common.select_loss_hard(losses,Q)
    elif method=='gradnov': selected=common.select_hard_gradient_novel(losses,gram,Q,.6)
    elif method=='anchor_random':
        anchor=int(torch.argmax(losses))
        rest=[i for i in range(K) if i!=anchor]
        selected=[anchor,*map(int,rng.choice(rest,Q-1,replace=False))]
    else:
        perm=torch.as_tensor(rng.permutation(K),dtype=torch.long)
        sham=gram.index_select(0,perm).index_select(1,perm)
        selected=common.select_hard_gradient_novel(losses,sham,Q,.6)
    selected=sorted(selected)
    require(len(selected)==Q and len(set(selected))==Q and min(selected)>=0 and max(selected)<K,'selected indices')
    return selected

def train(start,end,out):
    check_range(start,end);envbase.configure_determinism(1)
    require(not out.exists() or not any(out.iterdir()),'refuse overwrite')
    out.mkdir(parents=True,exist_ok=True);(out/'states').mkdir()
    x,y,*_=cr.data_split();require(len(x)==988,'training split');y=torch.as_tensor(y)
    files={};records=[];initials={};schedules={};cache={}
    for rep in range(start,end):
        p=pool(rep)
        if p not in cache: cache[p]=envbase.geometric_envs(x,seeds(rep))
        envs=cache[p];seed=OFFSET+4099*rep;sched=schedule(seed)
        require(len(sched)==80,'schedule size');schedules[str(rep)]=pm.schedule_digest(sched)
        for mi,method in enumerate(METHODS):
            common.seed_everything(seed);model=common.SmallCNN();model.train()
            initial=pm.state_digest(model.state_dict())
            initials.setdefault(str(rep),initial);require(initials[str(rep)]==initial,'unpaired initialization')
            opt=torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=.001)
            rng=np.random.default_rng(seed+100003+mi*1009)
            for step,(b,cand) in enumerate(sched):
                logits,h=model(torch.cat([envs[e][b] for e in cand]))
                per=torch.nn.functional.cross_entropy(logits,y[b].repeat(K),reduction='none').reshape(K,-1)
                losses=per.mean(1);dirs=common.head_gradient_directions(logits,h,y[b],K)
                gram=dirs@dirs.T
                require(torch.isfinite(losses).all() and torch.isfinite(gram).all(),'training finite')
                selected=choose(method,losses,gram,rng)
                order=sorted(range(K),key=lambda i:(-float(losses[i].detach()),i));ranks={i:r+1 for r,i in enumerate(order)}
                sub=gram[selected][:,selected].detach().double().numpy();ii,jj=np.triu_indices(Q,1)
                rec={'rep':rep,'pool':p,'method':method,'step':step,'batch_size':len(b),
                     'selected_local':';'.join(map(str,selected)),
                     'selected_env_seed':';'.join(str(seeds(rep)[cand[i]]) for i in selected),
                     'mean_selected_rank':float(np.mean([ranks[i] for i in selected])),
                     'mean_candidate_loss':float(losses.mean().detach()),
                     'mean_selected_loss':float(losses[selected].mean().detach()),
                     'selected_pairwise_novelty':float(np.mean(1-sub[ii,jj]))}
                records.append(rec)
                opt.zero_grad(set_to_none=True);per[selected].mean().backward();opt.step()
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            require(all(torch.isfinite(t).all() for t in state.values()),'state finite')
            path=out/'states'/f'rep{rep}_{method}.pt'
            torch.save({'state_dict':state,'rep':rep,'method':method,'pool':p,
                        'protocol_hash':PH,'state_digest':pm.state_digest(state),
                        'initial_digest':initial,'schedule_digest':schedules[str(rep)]},path)
            files[str(path.relative_to(out))]=fd.sha256(path)
        print('ONLINE_TRAINED',rep,'heldout_not_constructed',flush=True)
    path=out/'training_steps.csv.gz';pd.DataFrame(records).to_csv(path,index=False,compression='gzip');files[path.name]=fd.sha256(path)
    write(out/'training_manifest.json',{'protocol':PROTOCOL,'protocol_hash':PH,
          'start':start,'end':end,'files':files,'source_hashes':hashes(),'split_hashes':cr.split_hashes(),
          'runtime':fd.runtime(),'initial_digests':initials,'schedule_digests':schedules,'heldout_constructed':False})

def validate_train(out):
    path=out/'training_manifest.json';m=json.loads(path.read_text());check_range(m['start'],m['end'])
    require(m['protocol_hash']==PH and m['source_hashes']==hashes(),'training provenance')
    require(m['split_hashes']==cr.split_hashes() and not m['heldout_constructed'],'split/boundary')
    require(hashlib.sha256(json.dumps(m['protocol'],sort_keys=True).encode()).hexdigest()==PH,'protocol content')
    for n,h in m['files'].items(): require(fd.sha256(out/n)==h,'training file hash '+n)
    return m

def seal(root,out):
    reps=[];entries=[];states=0
    for path in sorted(root.rglob('training_manifest.json')):
        m=validate_train(path.parent);reps.extend(range(m['start'],m['end']))
        for rep in range(m['start'],m['end']):
            for method in METHODS:
                ck=torch.load(path.parent/'states'/f'rep{rep}_{method}.pt',map_location='cpu',weights_only=True)
                require((ck['rep'],ck['method'],ck['protocol_hash'])==(rep,method,PH),'state metadata')
                require(pm.state_digest(ck['state_dict'])==ck['state_digest'],'tensor digest');states+=1
        entries.append({'start':m['start'],'end':m['end'],'training_manifest_sha256':fd.sha256(path)})
    require(sorted(reps)==list(REPS) and states==240,'global coverage')
    out.mkdir(parents=True,exist_ok=True)
    write(out/'global_seal.json',{'protocol_hash':PH,'reps':list(REPS),'states':states,'shards':entries,'heldout_constructed':False})
    print('GLOBAL_ONLINE_STATE_SEAL',states,PH,flush=True)

def validate_seal(path,m):
    s=json.loads(path.read_text())
    require(s['protocol_hash']==PH and s['reps']==list(REPS) and s['states']==240 and not s['heldout_constructed'],'global seal')
    entries=[e for e in s['shards'] if (e['start'],e['end'])==(m['start'],m['end'])]
    require(len(entries)==1,'seal shard')
    return entries[0]

def evaluate(start,end,out,global_seal):
    check_range(start,end);envbase.configure_determinism(1);m=validate_train(out)
    require((m['start'],m['end'])==(start,end),'evaluation range')
    entry=validate_seal(global_seal,m)
    require(entry['training_manifest_sha256']==fd.sha256(out/'training_manifest.json'),'global seal hash')
    require(not (out/'evaluation_manifest.json').exists(),'evaluation overwrite')
    # No heldout construction appears before validation of the global barrier.
    _,_,x,y,*_=cr.data_split();require(len(x)==809,'evaluation split');labels=torch.as_tensor(y);clean=torch.as_tensor(x)
    cache={};rows=[];envrows=[]
    for rep in range(start,end):
        p=pool(rep)
        if p not in cache:cache[p]=[torch.as_tensor(envbase.geometric_environment(x,s)) for s in seeds(rep,True)]
        for method in METHODS:
            ck=torch.load(out/'states'/f'rep{rep}_{method}.pt',map_location='cpu',weights_only=True)
            require((ck['rep'],ck['method'],ck['protocol_hash'])==(rep,method,PH),'eval state metadata')
            require(pm.state_digest(ck['state_dict'])==ck['state_digest'],'eval tensor digest')
            model=common.SmallCNN();model.load_state_dict(ck['state_dict']);model.eval();vals=[]
            with torch.no_grad():
                clean_correct=int((model(clean)[0].argmax(1)==labels).sum())
                for env_seed,xx in zip(seeds(rep,True),cache[p]):
                    correct=int((model(xx)[0].argmax(1)==labels).sum());a=correct/len(labels);vals.append(a)
                    envrows.append({'rep':rep,'pool':p,'method':method,'env_seed':env_seed,'correct':correct,'n_images':len(labels),'accuracy':a})
            a=np.asarray(vals,np.float64)
            rows.append({'rep':rep,'pool':p,'method':method,'mean_test':float(a.mean()),'sd_test':float(a.std(ddof=1)),
                         'p10_test':float(np.quantile(a,.1)),'min_test':float(a.min()),'clean_test':clean_correct/len(labels),
                         'clean_correct':clean_correct,'state_digest':ck['state_digest']})
        print('ONLINE_EVALUATED',rep,flush=True)
    validate_train(out);files={}
    for name,data in (('heldout.csv',rows),('environment.csv.gz',envrows)):
        p=out/name;pd.DataFrame(data).to_csv(p,index=False);files[name]=fd.sha256(p)
    write(out/'evaluation_manifest.json',{'protocol_hash':PH,'start':start,'end':end,'files':files,'source_hashes':hashes(),
          'training_manifest_sha256':fd.sha256(out/'training_manifest.json'),'global_seal_sha256':fd.sha256(global_seal),
          'runtime':fd.runtime(),'heldout_constructed':True})

def estimate(x):
    x=np.asarray(x,float);require(x.ndim==1 and len(x)>1 and np.isfinite(x).all(),'sample')
    n=len(x);mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(n));k=float(stats.t.ppf(.975,n-1))
    p=float(stats.t.sf(mean/se,n-1)) if se else (0. if mean>0 else 1. if mean<0 else .5)
    return dict(n=n,mean=mean,se=se,k=k,ci95_low=mean-k*se,ci95_high=mean+k*se,p_one_sided=p,positive_pairs=int((x>0).sum()))

def holm(p):
    a=np.asarray(p,float);require(a.ndim==1 and np.isfinite(a).all() and ((a>=0)&(a<=1)).all(),'p domain')
    order=np.argsort(a,kind='stable');out=np.empty_like(a);out[order]=np.minimum(1.,np.maximum.accumulate(a[order]*np.arange(len(a),0,-1)))
    return out

def aggregate(root,out,global_seal):
    covered=[];held=[];environments=[];train=[];cpus=[];states=0
    for mp in sorted(root.rglob('evaluation_manifest.json')):
        d=mp.parent;m=json.loads(mp.read_text());tm=validate_train(d);entry=validate_seal(global_seal,tm)
        require((m['start'],m['end'])==(tm['start'],tm['end']),'eval range')
        require(m['protocol_hash']==PH and m['source_hashes']==hashes(),'eval provenance')
        require(m['training_manifest_sha256']==entry['training_manifest_sha256']==fd.sha256(d/'training_manifest.json'),'manifest chain')
        require(m['global_seal_sha256']==fd.sha256(global_seal),'global chain')
        for n,h in m['files'].items():require(fd.sha256(d/n)==h,'evaluation hash')
        covered.extend(range(m['start'],m['end']))
        h=pd.read_csv(d/'heldout.csv',float_precision='round_trip');hidx=h.set_index(['rep','method'])
        for rep in range(m['start'],m['end']):
            for method in METHODS:
                ck=torch.load(d/'states'/f'rep{rep}_{method}.pt',weights_only=True,map_location='cpu')
                require(pm.state_digest(ck['state_dict'])==hidx.loc[(rep,method),'state_digest'],'aggregate tensor digest');states+=1
        held.append(h);environments.append(pd.read_csv(d/'environment.csv.gz',float_precision='round_trip'))
        train.append(pd.read_csv(d/'training_steps.csv.gz',float_precision='round_trip'))
        cpus.append({'start':m['start'],'training':tm['runtime'],'evaluation':m['runtime']})
    require(sorted(covered)==list(REPS) and states==240,'aggregate coverage')
    held=pd.concat(held,ignore_index=True);env=pd.concat(environments,ignore_index=True);training=pd.concat(train,ignore_index=True)
    require(len(held)==240 and not held.duplicated(['rep','method']).any(),'heldout grid')
    require(set(held[['rep','method']].itertuples(index=False,name=None))=={(r,m) for r in REPS for m in METHODS},'heldout IDs')
    require(len(env)==19200 and not env.duplicated(['rep','method','env_seed']).any(),'env grid')
    require(len(training)==19200 and not training.duplicated(['rep','method','step']).any(),'training grid')
    require(np.isfinite(held[['mean_test','sd_test','p10_test','min_test','clean_test']].to_numpy(float)).all(),'held finite')
    hz=held.set_index(['rep','method']);err=0.
    for (rep,method),b in env.groupby(['rep','method']):
        require(set(b.env_seed)==set(seeds(rep,True)),'env IDs')
        require((b.n_images==809).all() and ((b.correct>=0)&(b.correct<=809)).all(),'correct count')
        a=b.sort_values('env_seed').accuracy.to_numpy(float)
        require(np.max(np.abs(b.accuracy-b.correct/809))<=1e-15,'accuracy reconstruction')
        obs=np.array([a.mean(),a.std(ddof=1),np.quantile(a,.1),a.min()]);stored=hz.loc[(rep,method),['mean_test','sd_test','p10_test','min_test']].to_numpy(float)
        err=max(err,float(np.max(np.abs(obs-stored))))
    require(err<=1e-12,'environment aggregate')
    pairs=[];summary=[];exploratory=[]
    for rep in REPS:
        row={'rep':rep,'pool':pool(rep)}
        for method in METHODS:row[method+'_mean_test']=float(hz.loc[(rep,method),'mean_test'])
        for comp in COMPARATORS:
            row['gradnov_minus_'+comp]=float(hz.loc[(rep,'gradnov'),'mean_test']-hz.loc[(rep,comp),'mean_test'])
        pairs.append(row)
    pairs=pd.DataFrame(pairs)
    for comp in COMPARATORS:summary.append({'comparator':comp,**estimate(pairs['gradnov_minus_'+comp])})
    adj=holm([r['p_one_sided'] for r in summary])
    for r,p in zip(summary,adj):r.update(p_holm=float(p),supported=bool(r['mean']>0 and p<.05))
    if all(r['supported'] for r in summary):decision='ONLINE GRADIENT-INFORMATION CONTROL SUPPORT'
    elif summary[0]['supported']:decision='ONLINE EFFECT WITHOUT FULL CONTROL SEPARATION'
    else:decision='ONLINE BASELINE SUPERIORITY NOT CONFIRMED'
    for a,b in [('gradnov',c) for c in COMPARATORS]+[('anchor_random','loss_hard')]:
        for met in ('mean_test','sd_test','p10_test','min_test','clean_test'):
            v=np.array([hz.loc[(r,a),met]-hz.loc[(r,b),met] for r in REPS])
            exploratory.append({'arm':a,'comparator':b,'endpoint':met,'scope':'all_fixed_pools','exploratory':True,**estimate(v)})
        for p in range(3):
            ids=[r for r in REPS if pool(r)==p];v=np.array([hz.loc[(r,a),'mean_test']-hz.loc[(r,b),'mean_test'] for r in ids])
            exploratory.append({'arm':a,'comparator':b,'endpoint':'mean_test','scope':f'pool{p}','exploratory':True,**estimate(v)})
    out.mkdir(parents=True,exist_ok=True)
    for name,data in [('online_controls_paired60.csv',pairs),('online_controls_summary60.csv',pd.DataFrame(summary)),
                      ('online_controls_exploratory60.csv',pd.DataFrame(exploratory)),('online_controls_heldout60.csv',held),
                      ('online_controls_training_summary60.csv',training.groupby(['rep','pool','method']).mean(numeric_only=True).reset_index())]:data.to_csv(out/name,index=False)
    d={'decision':decision,'protocol_hash':PH,'n':60,'states_checked':states,'environment_rows':len(env),
       'max_environment_reaggregation_error':err,'primary_tests':3,'multiplicity':'Holm3','pools':3,'raw_image_retraining_in_audit':False}
    write(out/'online_controls_decision60.json',d);write(out/'online_controls_runtime.json',cpus)
    print(json.dumps(d),flush=True);print(pd.DataFrame(summary).to_string(index=False),flush=True)

def selftest():
    losses=torch.linspace(2,.1,K);a=torch.arange(K*7,dtype=torch.float32).reshape(K,7)+1;a=a/a.norm(dim=1,keepdim=True);gram=a@a.T
    for method in METHODS:
        first=choose(method,losses,gram,np.random.default_rng(17));second=choose(method,losses,gram,np.random.default_rng(17))
        require(first==second and 0 in first and len(first)==4,'selector selftest')
    require(np.allclose(holm([.01,.04,.03]),[.03,.06,.06]),'Holm selftest')
    require(np.allclose(holm([.9,.001,.002]),[.9,.003,.004]),'Holm order selftest')
    require(len(schedule(1234))==80 and len(schedule(1234)[-1][0])==92,'schedule synthetic shape')
    require(set(seeds(5000)).isdisjoint(seeds(5000,True)) and pool(5059)==2,'pool boundaries')
    v=np.linspace(-.1,.2,60);e=estimate(v);p=float(stats.ttest_rel(v,np.zeros(60),alternative='greater').pvalue)
    require(abs(e['p_one_sided']-p)<1e-12,'independent paired test')
    print('ONLINE_CONTROLS_SYNTHETIC_PASS',PH,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('selftest','train','seal','evaluate','aggregate'));p.add_argument('--start',type=int);p.add_argument('--end',type=int);p.add_argument('--input-dir',type=Path);p.add_argument('--output-dir',type=Path);p.add_argument('--global-seal',type=Path);a=p.parse_args()
    if a.mode=='selftest':selftest()
    elif a.mode=='train':train(a.start,a.end,a.output_dir)
    elif a.mode=='seal':seal(a.input_dir,a.output_dir)
    elif a.mode=='evaluate':evaluate(a.start,a.end,a.output_dir,a.global_seal)
    else:aggregate(a.input_dir,a.output_dir,a.global_seal)
if __name__=='__main__':main()

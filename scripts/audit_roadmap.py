"""Independent artifact/statistical audit; no training or raw-image inference.
Does not import scientific execution or selection functions.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats

def need(ok,msg):
    if not bool(ok):raise ValueError(msg)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dh(state):
    h=hashlib.sha256()
    for n,t in sorted(state.items()):
        a=t.detach().cpu().contiguous().numpy()
        for b in (n.encode(),str(a.dtype).encode(),str(a.shape).encode(),a.tobytes()):h.update(b)
    return h.hexdigest()
def holm(p):
    ix=sorted(range(len(p)),key=lambda k:(p[k],k));o=[0.]*len(p);q=0.
    for rank,k in enumerate(ix):q=max(q,min(1.,p[k]*(len(p)-rank)));o[k]=q
    return o

def expected_weight(name,loss,gram,rng):
    policy=name.split('__')[-1];w=np.zeros(16,dtype=np.float32)
    if policy=='mean16':w[:]=1/16;return w
    if policy=='mix16':w[:]=.8/16;w[int(np.argmax(loss))]+=.2;return w
    if policy=='uniform4':ix=rng.choice(16,4,replace=False)
    elif policy in ('anchor4','anchor_random'):
        i=int(np.argmax(loss));ix=[i,*rng.choice([v for v in range(16) if v!=i],3,replace=False)]
    elif policy=='loss_hard':ix=torch.topk(torch.tensor(loss),4).indices.numpy()
    elif policy=='gradnov':
        l=torch.tensor(loss);g=torch.tensor(gram);chosen=[int(l.argmax())]
        while len(chosen)<4:
            nov=torch.min(1-g[:,chosen],dim=1).values
            nz=(nov-nov.mean())/(nov.std(correction=0)+1e-8)
            lz=(l-l.mean())/(l.std(correction=0)+1e-8)
            score=lz+.6*nz
            rem=[j for j in range(16) if j not in chosen]
            chosen.append(max(rem,key=lambda j:float(score[j])))
        ix=chosen
    else:raise ValueError('method')
    w[list(ix)]=.25;return w

def audit(root,out):
    torch.set_num_threads(1)
    dec=json.loads((root/'aggregate/decision.json').read_text());proto=dec['protocol'];study=proto['study'];psha=hashlib.sha256(json.dumps(proto,sort_keys=True).encode()).hexdigest();need(psha==dec['protocol_hash'],'protocol hash')
    known={'tail':(6000,7500000000,300000,301000,['gradnov','anchor_random']),
           'fashion':(7000,8500000000,400000,401000,['loss_hard','gradnov','anchor_random','uniform4']),
           'optimizer':(8000,9500000000,500000,501000,[o+'__'+m for o in ['AdamW','SGD'] for m in ['uniform4','mean16','anchor4','mix16']])}
    need(study in known,'registered study')
    need(tuple(proto[k] for k in ['first_rep','offset','train_base','eval_base','methods'])==known[study],'registered allocation')
    need(all(proto[k]==v for k,v in {'pools':6,'blocks_per_pool':10,'K':16,'sampled_Q':4,'batch':128,'epochs':10,'steps':80,'n_train':988,'n_eval':809,'head_novelty_weight':.6,'anchor_rng_offset':102021,'uniform_rng_offset':103030,'train_pool_size':64,'eval_pool_size':80,'pool_key_stride':2000}.items()),'fixed protocol fields')
    glob=json.loads((root/'global/global_seal.json').read_text());need(glob['protocol_hash']==psha and set(glob['pools'])==set(map(str,range(6))) and not glob['heldout_constructed'],'seal')
    for n,h in dec['files'].items():need(sha(root/'aggregate'/n)==h,'aggregate bytes')
    dfs=[];seen=set();state_count=0;trace_steps=0;max_weight_error=0.;imagesets={False:set(),True:set()};cpus=[]
    for em in sorted((root/'downloaded').rglob('evaluation_manifest.json')):
        p=em.parent;e=json.loads(em.read_text());t=json.loads((p/'training_manifest.json').read_text());pool=t['pool'];need(pool not in seen,'duplicate pool');seen.add(pool)
        need(glob['pools'][str(pool)]==sha(p/'training_manifest.json')==e['training_manifest_sha256'],'training seal linkage')
        need(e['global_seal_sha256']==sha(root/'global/global_seal.json'),'global seal linkage')
        need(t['protocol_hash']==psha==e['protocol_hash'] and t['protocol']==proto and not t['heldout_constructed'],'protocol/boundary')
        need(t['source_hashes']==e['source_hashes']==dec['source_hashes']==glob['source_hashes'],'source provenance')
        for n,h in t['source_hashes'].items():need(sha(p/'source'/n)==h,'source bytes')
        for n,h in t['files'].items():need(sha(p/n)==h,'training bytes '+n)
        need(sha(p/'counts.csv.gz')==e['counts_sha256'] and sha(p/'evaluation_inputs.npz')==e['inputs_sha256'],'eval bytes')
        need(t['runtime']['time']<=glob['sealed_at']<=e['runtime']['time'],'timestamp barrier')
        need(t['runtime']['python'].startswith('3.12.') and e['runtime']['python'].startswith('3.12.'),'Python runtime')
        for rt in (t['runtime'],e['runtime']):
            need(rt['threads']==1 and rt['deterministic'],'determinism')
            for k,v in proto['versions'].items():need(rt[k]==v,'dependency runtime')
        cpus.append({'pool':pool,'training_cpu':t['runtime']['cpu'],'evaluation_cpu':e['runtime']['cpu'],'training_clock':t['runtime']['clock_MHz_uncontrolled']})
        for is_eval,name,m in [(False,'train_inputs.npz',t),(True,'evaluation_inputs.npz',e)]:
            z=np.load(p/name);a=z['images'];y=z['labels'];ix=z['indices'];side=28 if study=='fashion' else 8;n=809 if is_eval else 988
            need(a.shape==(n,side,side) and a.dtype==np.float32 and y.shape==ix.shape==(n,),'image shape')
            need(np.isfinite(a).all() and a.min()>=0 and a.max()<=1 and set(y)==set(range(10)),'data domain')
            for k,v in [('images',a),('labels',y),('indices',ix)]:need(hashlib.sha256(v.tobytes()).hexdigest()==m['input_hashes'][k],'input arrays')
            imagesets[is_eval].add(json.dumps(m['input_hashes'],sort_keys=True))
        need(t['reps']==list(range(proto['first_rep']+pool*10,proto['first_rep']+(pool+1)*10)),'rep grid')
        d=pd.read_csv(p/'counts.csv.gz');ids=['pool','rep','method','env_seed'];seeds=[-1,*range(proto['eval_base']+2000*pool,proto['eval_base']+2000*pool+80)];expected=set(itertools.product([pool],t['reps'],proto['methods'],seeds))
        need(len(d)==len(expected) and not d.duplicated(ids).any() and set(d[ids].itertuples(index=False,name=None))==expected,'count grid')
        need(d.study.eq(study).all(),'study binding');a=d[['correct','n_images']].to_numpy(float);need(np.isfinite(a).all() and (a==np.floor(a)).all() and (a[:,1]==809).all() and ((a[:,0]>=0)&(a[:,0]<=809)).all(),'integer counts')
        for rep in t['reps']:
            seed=proto['offset']+4099*rep;tg=torch.Generator().manual_seed(seed+1);er=np.random.default_rng(seed+2);sched=[];hs=hashlib.sha256()
            for _ in range(10):
                for b in torch.randperm(988,generator=tg).split(128):
                    cand=er.choice(64,16,replace=False);sched.append(cand);hs.update(b.numpy().astype(np.int64).tobytes());hs.update(cand.astype(np.int64).tobytes())
            need(hs.hexdigest()==t['schedule_digests'][str(rep)],'schedule independent reconstruction')
            for method in proto['methods']:
                key=f'rep{rep}_{method}';ck=torch.load(p/'states'/(key+'.pt'),weights_only=True,map_location='cpu')
                need((ck['study'],ck['pool'],ck['rep'],ck['method'],ck['protocol_hash'])==(study,pool,rep,method,psha),'state metadata')
                need(dh(ck['state_dict'])==ck['state_digest']==t['state_digests'][key],'state tensor integrity')
                need(ck['initial_digest']==t['initial_digests'][str(rep)] and ck['schedule_digest']==hs.hexdigest(),'paired initialization/schedule')
                need(d[(d.rep==rep)&(d.method==method)].state_digest.eq(ck['state_digest']).all(),'counts tied to state')
                state_count+=1;trace=np.load(p/'trace'/(key+'.npz'));ls=trace['losses'];ws=trace['weights'];gs=trace['gram'];need(ls.shape==ws.shape==(80,16) and gs.shape==(80,16,16),'trace size')
                need(all(np.isfinite(x).all() for x in [ls,ws,gs]),'trace finite');need(np.array_equal(trace['candidates'],sched),'trace schedule')
                rng=np.random.default_rng(seed+(103030 if method.split('__')[-1]=='uniform4' else 102021))
                for l,w,g in zip(ls,ws,gs):
                    ew=expected_weight(method,l,g,rng);err=float(np.max(np.abs(w-ew)));max_weight_error=max(max_weight_error,err);need(err==0,'policy/weight replay');trace_steps+=1
        dfs.append(d)
    need(seen==set(range(6)) and all(len(v)==1 for v in imagesets.values()),'pools/data identity');need(state_count==dec['states'],'states')
    df=pd.concat(dfs,ignore_index=True);stored=pd.read_csv(root/'aggregate/counts.csv.gz');sort=['pool','rep','method','env_seed'];pd.testing.assert_frame_equal(df.sort_values(sort).reset_index(drop=True),stored.sort_values(sort).reset_index(drop=True))
    need(len(df)==dec['count_rows'],'row count');records=[]
    for (pool,rep,meth),d in df.groupby(['pool','rep','method']):
        v=d.loc[d.env_seed>=0,'correct'].to_numpy(dtype=float)/809
        records.append(dict(pool=pool,rep=rep,method=meth,mean=v.mean(),min=v.min(),clean=float(d.loc[d.env_seed==-1,'correct'].iloc[0]/809),p10=np.quantile(v,.1),lower10_mean=np.sort(v)[:8].mean(),sd=v.std(ddof=1)))
    md=pd.DataFrame(records);mr=pd.read_csv(root/'aggregate/metrics.csv',float_precision='round_trip');diff=max(float(np.max(np.abs(md.sort_values(['pool','rep','method'])[m].to_numpy()-mr.sort_values(['pool','rep','method'])[m].to_numpy()))) for m in ['mean','min','clean','p10','lower10_mean','sd']);need(diff<1e-14,'endpoint reaggregation')
    checks=[];pvals=[];pr=pd.read_csv(root/'aggregate/primary.csv',float_precision='round_trip');max_stat_error=0.
    expected_tests=([('gradnov_minus_anchor_random','min','greater',0.),('gradnov_minus_anchor_random','clean','less',0.)] if study=='tail' else [('anchor_random_minus_loss_hard','mean','greater',.005),('gradnov_minus_anchor_random','mean','greater',.005),('gradnov_minus_anchor_random','min','greater',.005)] if study=='fashion' else [(c,'mean','two-sided',0.) for c in [o+'_'+m for o in ['AdamW','SGD'] for m in ['objective','estimator','interaction']]+['three_way']])
    need(list(pr[['contrast','endpoint','alternative','null']].itertuples(index=False,name=None))==expected_tests and pr.n.eq(6).all(),'preregistered primary tests')
    independent_blocks=[]
    for endpoint in ['mean','min','clean','p10','lower10_mean','sd']:
        piv=md.pivot(index=['pool','rep'],columns='method',values=endpoint);cs={}
        if study=='optimizer':
            for op in ('AdamW','SGD'):
                u,e,a,m=[piv[op+'__'+z] for z in ['uniform4','mean16','anchor4','mix16']]
                cs[op+'_objective']=.5*(a+m-u-e);cs[op+'_estimator']=.5*(e+m-u-a);cs[op+'_interaction']=m-e-a+u
            cs['three_way']=cs['SGD_interaction']-cs['AdamW_interaction']
        else:
            cs['gradnov_minus_anchor_random']=piv.gradnov-piv.anchor_random
            if study=='fashion':
                cs['anchor_random_minus_loss_hard']=piv.anchor_random-piv.loss_hard
                cs['anchor_random_minus_uniform4']=piv.anchor_random-piv.uniform4
        for c,values in cs.items():
            for (pool,rep),value in values.items():independent_blocks.append({'pool':pool,'rep':rep,'endpoint':endpoint,'contrast':c,'difference':value})
    ib=pd.DataFrame(independent_blocks)
    for filename,ind,sortcols in [('block_contrasts.csv',ib,['pool','rep','endpoint','contrast']),('pool_contrasts.csv',ib.groupby(['pool','endpoint','contrast'],as_index=False).difference.mean(),['pool','endpoint','contrast'])]:
        supplied=pd.read_csv(root/'aggregate'/filename,float_precision='round_trip');ind=ind.sort_values(sortcols).reset_index(drop=True);supplied=supplied.sort_values(sortcols).reset_index(drop=True)
        need(ind[sortcols].equals(supplied[sortcols]),'contrast metadata '+filename)
        need(np.max(np.abs(ind.difference-supplied.difference))<1e-14,'contrast values '+filename)
    for row in pr.itertuples():
        piv=md.pivot(index=['pool','rep'],columns='method',values=row.endpoint)
        if study=='optimizer':
            v={}
            for op in ('AdamW','SGD'):
                u,e,a,m=[piv[op+'__'+s] for s in ['uniform4','mean16','anchor4','mix16']]
                v[op+'_objective']=.5*(a+m-u-e);v[op+'_estimator']=.5*(e+m-u-a);v[op+'_interaction']=m-e-a+u
            v['three_way']=v['SGD_interaction']-v['AdamW_interaction'];b=v[row.contrast]
        else:
            first,second=row.contrast.split('_minus_');b=piv[first]-piv[second]
        pools=b.groupby('pool').mean().to_numpy();test=stats.ttest_1samp(pools,popmean=row.null,alternative=row.alternative);ci=stats.ttest_1samp(pools,0).confidence_interval(.95)
        se=float(stats.sem(pools));need(se>0,'SE');pc=float(test.pvalue);pvals.append(pc)
        for actual,target in [(pools.mean(),row.mean),(np.median(pools),row.median),(se,row.se),(ci.low,row.ci95_low),(ci.high,row.ci95_high),(pc,row.p)]:max_stat_error=max(max_stat_error,abs(actual-target))
        checks.append({'contrast':row.contrast,'endpoint':row.endpoint,'pool_values':pools.tolist(),'mean':float(pools.mean()),'se':se,'ci95':[float(ci.low),float(ci.high)],'p_independent':pc})
    adj=pvals if study=='tail' else holm(pvals);need(np.max(np.abs(np.array(adj)-pr.p_adjusted))<1e-12 and max_stat_error<1e-12,'statistics/adjustment')
    need(np.array_equal(np.array(adj)<.05,pr['pass']),'decisions')
    if study=='tail':
        t,c=(np.array(adj)<.05).tolist()
        label='CLEAN-TAIL POLICY TRADEOFF SUPPORT' if t and c else 'TAIL ADVANTAGE ONLY / CLEAN COST NOT CONFIRMED' if t else 'CLEAN COST ONLY / TAIL ADVANTAGE NOT CONFIRMED' if c else 'NO CLEAN-TAIL POLICY TRADEOFF CONFIRMATION'
        need(dec['decision']==label,'compound decision label')
    out.parent.mkdir(parents=True,exist_ok=True)
    result={'status':'INDEPENDENT ARTIFACT AND STATISTICAL AUDIT PASS','scope':'source/file/state/data/schedule/selector/count/statistic audit; not all-model retraining or raw-image reinference','study':study,'protocol_hash':psha,'states_checked':state_count,'trace_steps_checked':trace_steps,'count_rows_checked':len(df),'block_contrast_rows_checked':len(ib),'pool_contrast_rows_checked':len(ib.groupby(['pool','endpoint','contrast'])),'max_weight_error':max_weight_error,'max_endpoint_error':diff,'max_statistic_error':max_stat_error,'independent_primary':checks,'adjusted_p':list(map(float,adj)),'cpus':cpus}
    out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:result[k] for k in ['status','study','states_checked','trace_steps_checked','count_rows_checked','max_statistic_error']},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();audit(a.root,a.out)

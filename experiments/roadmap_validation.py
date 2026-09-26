"""Issues124/125: fixed six-pool tests, immutable training/evaluation barriers.
Only selftest may run outside the registered Python3.12 scientific environment.
"""
from __future__ import annotations
import argparse, datetime, gzip, hashlib, itertools, json, platform, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.ndimage import rotate, shift, gaussian_filter
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
import common

K,Q,BATCH,EPOCHS=16,4,128,10
VERSIONS={'torch':'2.10.0+cpu','numpy':'2.3.5','pandas':'2.2.3','scipy':'1.17.0','sklearn':'1.8.0'}
SPECS={
 'tail':{'issue':124,'first_rep':6000,'offset':7500000000,'train_base':300000,'eval_base':301000,'methods':['gradnov','anchor_random'],'data':'digits','primary':'conjunction: min greater0 AND clean less0; pool t5'},
 'fashion':{'issue':125,'first_rep':7000,'offset':8500000000,'train_base':400000,'eval_base':401000,'methods':['loss_hard','gradnov','anchor_random','uniform4'],'data':'fashion28','primary':'Holm3 directional differences greater.005; pool t5'},
 'optimizer':{'issue':125,'first_rep':8000,'offset':9500000000,'train_base':500000,'eval_base':501000,'methods':[o+'__'+m for o in ['AdamW','SGD'] for m in ['uniform4','mean16','anchor4','mix16']],'data':'digits','primary':'Holm7 two-sided factorial contrasts; pool t5'},
}
FASHION_MD5={'train-images-idx3-ubyte.gz':'8d4fb7e6c68d591d4c3dfef9ec88bf0d','train-labels-idx1-ubyte.gz':'25c81989df183df01b3e8a0aad5dffbe','t10k-images-idx3-ubyte.gz':'bef4ecab320f06d8554ea6380940ec79','t10k-labels-idx1-ubyte.gz':'bb300cfdad3c16e7a12a480ee83cd310'}

def require(ok,msg):
    if not bool(ok): raise ValueError(msg)

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,sort_keys=True,indent=2,allow_nan=False)

def protocol(study):
    return {'study':study,**SPECS[study],'pools':6,'blocks_per_pool':10,'K':16,'sampled_Q':4,'batch':128,'epochs':10,'steps':80,'n_train':988,'n_eval':809,'head_novelty_weight':.6,'anchor_rng_offset':102021,'uniform_rng_offset':103030,'train_pool_size':64,'eval_pool_size':80,'pool_key_stride':2000,'versions':VERSIONS,'python':'3.12','dtype':'float32','statistics':'float64','threads':1,'AdamW':{'lr':.005,'weight_decay':.001,'betas':[.9,.999],'eps':1e-8,'foreach':False,'fused':False},'SGD':{'lr':.05,'momentum':.9,'weight_decay':.001,'dampening':0,'nesterov':False,'foreach':False},'fashion_train_sample':926001,'fashion_test_sample':926002,'fashion_spatial_scale':3.5,'fashion_md5':FASHION_MD5,'no_optional_stopping':True}

def ph(study):return hashlib.sha256(json.dumps(protocol(study),sort_keys=True).encode()).hexdigest()
def sources():return {p.name:sha(p) for p in [Path(__file__),Path(common.__file__)]}

def configure(strict=True):
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    torch.use_deterministic_algorithms(True)
    if strict:
        require(platform.python_version().startswith('3.12.'),'registered Python3.12 required')
        require({k:str(v) for k,v in zip(VERSIONS,[torch.__version__,np.__version__,pd.__version__,scipy.__version__,sklearn.__version__])}==VERSIONS,'scientific versions')

def runtime():
    cpu=Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else ''
    return {'time':now(),'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'platform':platform.platform(),'cpu':[l.split(':',1)[1].strip() for l in cpu.splitlines() if l.startswith('model name')][:1],'clock_MHz_uncontrolled':[float(l.split(':')[1]) for l in cpu.splitlines() if l.startswith('cpu MHz')],'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'mkldnn':torch.backends.mkldnn.enabled,'deterministic':torch.are_deterministic_algorithms_enabled(),'torch_config':torch.__config__.show()}

def tensor_hash(state):
    h=hashlib.sha256()
    for n,t in sorted(state.items()):
        a=t.detach().cpu().contiguous().numpy()
        for v in [n.encode(),str(a.dtype).encode(),str(a.shape).encode(),a.tobytes()]:h.update(v)
    return h.hexdigest()

def reps(study,pool):
    require(pool in range(6),'pool')
    first=SPECS[study]['first_rep']+10*pool
    return range(first,first+10)

def keys(study,pool,evaluation=False):
    start=SPECS[study]['eval_base' if evaluation else 'train_base']+2000*pool
    return range(start,start+(80 if evaluation else 64))

def fetch_idx(name,cache):
    cache.mkdir(parents=True,exist_ok=True);p=cache/name
    if not p.exists():
        url='https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'+name
        with urllib.request.urlopen(url,timeout=120) as r:p.write_bytes(r.read())
    require(hashlib.md5(p.read_bytes()).hexdigest()==FASHION_MD5[name],'official Fashion checksum')
    b=gzip.decompress(p.read_bytes());dims=b[3];require(b[:3]==b'\x00\x00\x08' and dims in [1,3],'IDX header')
    shape=np.frombuffer(b,dtype='>u4',count=dims,offset=4).astype(int)
    a=np.frombuffer(b,dtype=np.uint8,offset=4*(dims+1)).copy();require(a.size==np.prod(shape),'IDX length')
    return a.reshape(tuple(shape))

def data(study,evaluation,cache):
    if SPECS[study]['data']=='digits':
        x,y=load_digits(return_X_y=True);x=(x.astype(np.float32)/16).reshape(-1,8,8);y=y.astype(np.int64)
        tr,te=train_test_split(np.arange(len(y)),test_size=.45,random_state=314159,stratify=y)
        ix=te if evaluation else tr
    else:
        prefix='t10k' if evaluation else 'train'
        x=fetch_idx(prefix+'-images-idx3-ubyte.gz',cache);y=fetch_idx(prefix+'-labels-idx1-ubyte.gz',cache).astype(np.int64)
        ix,_=train_test_split(np.arange(len(y)),train_size=809 if evaluation else 988,random_state=926002 if evaluation else 926001,stratify=y)
        x=x.astype(np.float32)/255
    xx=x[ix];yy=y[ix];require(len(ix)==(809 if evaluation else 988),'image count')
    return xx,yy,ix

class FashionCNN(common.SmallCNN):
    def forward(self,x):
        if x.ndim==3:x=x[:,None]
        f=self.features(x);f=torch.nn.functional.adaptive_avg_pool2d(f,(4,4));h=self.project(f)
        return self.head(h),h

def model_for(study):return FashionCNN() if study=='fashion' else common.SmallCNN()

def transform(x,s,study):
    if study!='fashion':return common.geometric_environment(x,s)
    a,dx,dy,blur,contrast,bright,noise=common.environment_parameters(s);out=np.empty_like(x);rng=np.random.default_rng(s*4001+17)
    for i,img in enumerate(x):
        z=rotate(img,float(a),reshape=False,order=1,mode='constant',cval=0,prefilter=False)
        z=shift(z,(float(dy)*3.5,float(dx)*3.5),order=1,mode='constant',cval=0,prefilter=False)
        if blur>1e-6:z=gaussian_filter(z,float(blur)*3.5,mode='nearest')
        out[i]=np.clip((z-.5)*contrast+.5+bright+rng.normal(0,noise,z.shape),0,1)
    return out.astype(np.float32)

def schedule(seed):
    tg=torch.Generator().manual_seed(seed+1);rng=np.random.default_rng(seed+2)
    out=[(b,rng.choice(64,16,replace=False).tolist()) for _ in range(10) for b in torch.randperm(988,generator=tg).split(128)]
    require(len(out)==80,'schedule');return out

def schedule_hash(sched):
    h=hashlib.sha256()
    for b,c in sched:
        h.update(b.numpy().astype(np.int64).tobytes());h.update(np.array(c,dtype=np.int64).tobytes())
    return h.hexdigest()

def weights(method,losses,gram,rng):
    policy=method.split('__')[-1];w=torch.zeros_like(losses)
    if policy=='mean16':w.fill_(1/16);return w
    if policy=='mix16':w.fill_(.8/16);w[int(losses.argmax())]+=.2;return w
    if policy=='loss_hard':sel=common.select_loss_hard(losses,4)
    elif policy=='gradnov':sel=common.select_hard_gradient_novel(losses,gram,4,.6)
    elif policy in ['anchor_random','anchor4']:
        anchor=int(losses.argmax());sel=[anchor,*map(int,rng.choice([i for i in range(16) if i!=anchor],3,replace=False))]
    elif policy=='uniform4':sel=list(map(int,rng.choice(16,4,replace=False)))
    else:raise ValueError('unknown method')
    require(len(set(sel))==4,'selection count');w[sorted(sel)]=.25;return w

def optimizer(method,model):
    if method.startswith('SGD__'):return torch.optim.SGD(model.parameters(),lr=.05,momentum=.9,weight_decay=.001,dampening=0,nesterov=False,foreach=False)
    return torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=.001,betas=(.9,.999),eps=1e-8,foreach=False,fused=False)

def train(study,pool,out):
    configure();require(not out.exists(),'training overwrite');out.mkdir(parents=True);(out/'states').mkdir();(out/'trace').mkdir()
    x,y,ix=data(study,False,out/'cache');yy=torch.tensor(y);envs=[torch.tensor(transform(x,s,study)) for s in keys(study,pool)];files={};initials={};schedules={};digests={}
    np.savez_compressed(out/'train_inputs.npz',images=x,labels=y,indices=ix);files['train_inputs.npz']=sha(out/'train_inputs.npz')
    for rep in reps(study,pool):
        seed=SPECS[study]['offset']+4099*rep;sched=schedule(seed);schedules[str(rep)]=schedule_hash(sched)
        for method in SPECS[study]['methods']:
            common.seed_everything(seed);model=model_for(study);model.train();init=tensor_hash(model.state_dict());initials.setdefault(str(rep),init);require(initials[str(rep)]==init,'pairing')
            opt=optimizer(method,model);policy=method.split('__')[-1];rng=np.random.default_rng(seed+(103030 if policy=='uniform4' else 102021))
            ls=[];ws=[];gs=[]
            for b,cand in sched:
                logits,h=model(torch.cat([envs[e][b] for e in cand]));per=torch.nn.functional.cross_entropy(logits,yy[b].repeat(16),reduction='none').reshape(16,-1);losses=per.mean(1)
                dirs=common.head_gradient_directions(logits,h,yy[b],16);gram=dirs@dirs.T
                require(torch.isfinite(losses).all() and torch.isfinite(gram).all(),'finite training')
                w=weights(method,losses,gram,rng);ls.append(losses.detach().numpy().copy());ws.append(w.detach().numpy().copy());gs.append(gram.detach().numpy().copy())
                opt.zero_grad(set_to_none=True)
                if study=='optimizer':objective=(w*losses).sum()
                else:objective=per[torch.nonzero(w).flatten()].mean()
                objective.backward();opt.step()
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};require(all(torch.isfinite(t).all() for t in state.values()),'finite state');d=tensor_hash(state)
            name=f'rep{rep}_{method}';p=out/'states'/(name+'.pt');torch.save({'state_dict':state,'state_digest':d,'initial_digest':init,'schedule_digest':schedules[str(rep)],'protocol_hash':ph(study),'study':study,'pool':pool,'rep':rep,'method':method},p);files[str(p.relative_to(out))]=sha(p);digests[name]=d
            p=out/'trace'/(name+'.npz');np.savez_compressed(p,losses=np.array(ls),weights=np.array(ws),gram=np.array(gs),candidates=np.array([c for b,c in sched]));files[str(p.relative_to(out))]=sha(p)
        print('TRAINED',study,pool,rep,'heldout_not_constructed',flush=True)
    write(out/'training_manifest.json',{'protocol':protocol(study),'protocol_hash':ph(study),'pool':pool,'reps':list(reps(study,pool)),'files':files,'state_digests':digests,'initial_digests':initials,'schedule_digests':schedules,'source_hashes':sources(),'input_hashes':{'images':ah(x),'labels':ah(y),'indices':ah(ix)},'environment_hash':ah(np.stack([e.numpy() for e in envs])),'runtime':runtime(),'heldout_constructed':False})

def validate_train(study,out):
    m=json.loads((out/'training_manifest.json').read_text());require(m['protocol']==protocol(study) and m['protocol_hash']==ph(study) and m['source_hashes']==sources() and not m['heldout_constructed'],'training provenance')
    require(m['reps']==list(reps(study,m['pool'])),'repetitions')
    for n,d in m['files'].items():require(sha(out/n)==d,'file hash '+n)
    for rep in m['reps']:
        require(m['schedule_digests'][str(rep)]==schedule_hash(schedule(SPECS[study]['offset']+4099*rep)),'schedule hash')
        for method in SPECS[study]['methods']:
            ck=torch.load(out/'states'/f'rep{rep}_{method}.pt',map_location='cpu',weights_only=True)
            require((ck['study'],ck['pool'],ck['rep'],ck['method'],ck['protocol_hash'])==(study,m['pool'],rep,method,ph(study)),'state metadata')
            require(tensor_hash(ck['state_dict'])==ck['state_digest']==m['state_digests'][f'rep{rep}_{method}'],'tensor digest')
            require(ck['initial_digest']==m['initial_digests'][str(rep)] and ck['schedule_digest']==m['schedule_digests'][str(rep)],'state pairing')
    return m

def seal(study,root,out):
    configure();entries={};inputs=set()
    for p in sorted(root.rglob('training_manifest.json')):
        m=validate_train(study,p.parent);pool=m['pool'];require(str(pool) not in entries,'duplicate pool');entries[str(pool)]=sha(p);inputs.add(json.dumps(m['input_hashes'],sort_keys=True))
    require(set(entries)==set(map(str,range(6))) and len(inputs)==1,'all six pools/same images');out.mkdir(parents=True,exist_ok=True)
    write(out/'global_seal.json',{'protocol_hash':ph(study),'source_hashes':sources(),'pools':entries,'states':60*len(SPECS[study]['methods']),'heldout_constructed':False,'sealed_at':now()});print('GLOBAL_SEAL',study,flush=True)

def check_seal(study,out,global_seal):
    m=validate_train(study,out);s=json.loads(global_seal.read_text())
    require(s['protocol_hash']==ph(study) and s['source_hashes']==sources() and set(s['pools'])==set(map(str,range(6))) and s['states']==60*len(SPECS[study]['methods']) and not s['heldout_constructed'],'seal content')
    require(s['pools'][str(m['pool'])]==sha(out/'training_manifest.json'),'seal binding');return m

def evaluate(study,pool,out,global_seal):
    configure();m=check_seal(study,out,global_seal);require(m['pool']==pool and not (out/'evaluation_manifest.json').exists(),'evaluation boundary/overwrite')
    x,y,ix=data(study,True,out/'cache');labels=torch.tensor(y);envs=[(-1,torch.tensor(x))]+[(s,torch.tensor(transform(x,s,study))) for s in keys(study,pool,True)];records=[]
    np.savez_compressed(out/'evaluation_inputs.npz',images=x,labels=y,indices=ix)
    for rep in reps(study,pool):
        for method in SPECS[study]['methods']:
            ck=torch.load(out/'states'/f'rep{rep}_{method}.pt',map_location='cpu',weights_only=True);model=model_for(study);model.load_state_dict(ck['state_dict']);model.eval()
            with torch.no_grad():
                for s,xx in envs:
                    # Evaluation batching has no BN/dropout and bounds native28 memory.
                    correct=sum(int((model(b)[0].argmax(1)==yb).sum()) for b,yb in zip(xx.split(128),labels.split(128)))
                    records.append({'study':study,'pool':pool,'rep':rep,'method':method,'env_seed':s,'correct':correct,'n_images':809,'state_digest':ck['state_digest']})
        print('EVALUATED',study,pool,rep,flush=True)
    validate_train(study,out);p=out/'counts.csv.gz';pd.DataFrame(records).to_csv(p,index=False,compression='gzip')
    write(out/'evaluation_manifest.json',{'protocol_hash':ph(study),'source_hashes':sources(),'pool':pool,'global_seal_sha256':sha(global_seal),'training_manifest_sha256':sha(out/'training_manifest.json'),'counts_sha256':sha(p),'inputs_sha256':sha(out/'evaluation_inputs.npz'),'environment_hash':ah(np.stack([e.numpy() for s,e in envs])),'input_hashes':{'images':ah(x),'labels':ah(y),'indices':ah(ix)},'runtime':runtime()})

def estimate(x,alternative='two-sided',null=0):
    x=np.asarray(x,np.float64);require(x.ndim==1 and len(x)>1 and np.isfinite(x).all(),'sample');n=len(x);mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(n));require(se>0,'zero-SE analysis exception');k=float(stats.t.ppf(.975,n-1));t=(mean-null)/se
    p=float(stats.t.sf(t,n-1) if alternative=='greater' else stats.t.cdf(t,n-1) if alternative=='less' else 2*stats.t.sf(abs(t),n-1))
    return {'n':n,'mean':mean,'median':float(np.median(x)),'se':se,'k':k,'ci95_low':mean-k*se,'ci95_high':mean+k*se,'null':null,'alternative':alternative,'p':p,'positive':int((x>0).sum())}

def holm(ps):
    x=np.array(ps,float);i=np.argsort(x,kind='stable');a=np.empty_like(x);a[i]=np.minimum(1,np.maximum.accumulate(x[i]*np.arange(len(x),0,-1)));return a

def aggregate(study,root,out,global_seal):
    configure();frames=[];covered=[];eval_inputs=set();backends=[]
    for p in sorted(root.rglob('evaluation_manifest.json')):
        m=json.loads(p.read_text());tm=check_seal(study,p.parent,global_seal)
        require(m['protocol_hash']==ph(study) and m['source_hashes']==sources() and m['global_seal_sha256']==sha(global_seal) and m['training_manifest_sha256']==sha(p.parent/'training_manifest.json'),'evaluation provenance')
        require(m['counts_sha256']==sha(p.parent/'counts.csv.gz') and m['inputs_sha256']==sha(p.parent/'evaluation_inputs.npz'),'evaluation files')
        d=pd.read_csv(p.parent/'counts.csv.gz');pool=m['pool'];require(pool==tm['pool'],'pool binding');covered.append(pool)
        expected=set(itertools.product(reps(study,pool),SPECS[study]['methods'],[-1,*keys(study,pool,True)]));cols=['rep','method','env_seed']
        require(len(d)==len(expected) and not d.duplicated(cols).any() and set(d[cols].itertuples(index=False,name=None))==expected and d.pool.eq(pool).all() and d.study.eq(study).all(),'counts grid')
        a=d[['correct','n_images']].to_numpy(float);require(np.isfinite(a).all() and (a==np.floor(a)).all() and (a[:,1]==809).all() and ((a[:,0]>=0)&(a[:,0]<=809)).all(),'integer counts')
        for r in d.itertuples():require(r.state_digest==tm['state_digests'][f'rep{r.rep}_{r.method}'],'count state binding')
        frames.append(d);eval_inputs.add(json.dumps(m['input_hashes'],sort_keys=True));backends.append({'pool':pool,'training':tm['runtime'],'evaluation':m['runtime']})
    require(sorted(covered)==list(range(6)) and len(eval_inputs)==1,'coverage/evaluation images')
    d=pd.concat(frames,ignore_index=True);metrics=[]
    for (pool,rep,method),b in d.groupby(['pool','rep','method']):
        v=b[b.env_seed!=-1].correct.to_numpy(float)/809
        metrics.append({'pool':pool,'rep':rep,'method':method,'mean':float(v.mean()),'min':float(v.min()),'p10':float(np.quantile(v,.1)),'lower10_mean':float(np.sort(v)[:8].mean()),'sd':float(v.std(ddof=1)),'clean':float(b.loc[b.env_seed==-1,'correct'].iloc[0]/809)})
    md=pd.DataFrame(metrics);z=md.set_index(['pool','rep','method']);contrasts=[]
    for pool in range(6):
        for rep in reps(study,pool):
            for endpoint in ['mean','min','clean','p10','lower10_mean','sd']:
                v={m:float(z.loc[(pool,rep,m),endpoint]) for m in SPECS[study]['methods']}
                if study=='optimizer':
                    c={}
                    for op in ['AdamW','SGD']:
                        u,e,a,m=[v[op+'__'+n] for n in ['uniform4','mean16','anchor4','mix16']]
                        c.update({op+'_objective':((a-u)+(m-e))/2,op+'_estimator':((e-u)+(m-a))/2,op+'_interaction':(m-e)-(a-u)})
                    c['three_way']=c['SGD_interaction']-c['AdamW_interaction']
                else:
                    c={a+'_minus_'+b:v[a]-v[b] for a,b in [('gradnov','anchor_random')] if a in v and b in v}
                    if study=='fashion':c.update({'anchor_random_minus_loss_hard':v['anchor_random']-v['loss_hard'],'anchor_random_minus_uniform4':v['anchor_random']-v['uniform4']})
                contrasts.extend({'pool':pool,'rep':rep,'endpoint':endpoint,'contrast':k,'difference':x} for k,x in c.items())
    bd=pd.DataFrame(contrasts);poold=bd.groupby(['pool','endpoint','contrast'],as_index=False).difference.mean();primary=[]
    tests=([('gradnov_minus_anchor_random','min','greater',0),('gradnov_minus_anchor_random','clean','less',0)] if study=='tail' else [('anchor_random_minus_loss_hard','mean','greater',.005),('gradnov_minus_anchor_random','mean','greater',.005),('gradnov_minus_anchor_random','min','greater',.005)] if study=='fashion' else [(c,'mean','two-sided',0) for c in [o+'_'+m for o in ['AdamW','SGD'] for m in ['objective','estimator','interaction']]+['three_way']])
    for c,e,a,h0 in tests:
        vals=poold[(poold.contrast==c)&(poold.endpoint==e)].sort_values('pool').difference.to_numpy();require(len(vals)==6,'six primary units');primary.append({'contrast':c,'endpoint':e,**estimate(vals,a,h0)})
    prim=pd.DataFrame(primary);prim['p_adjusted']=prim.p if study=='tail' else holm(prim.p);prim['pass']=prim.p_adjusted<.05
    if study=='tail':
        tail,clean=prim['pass'].tolist();label='CLEAN-TAIL POLICY TRADEOFF SUPPORT' if tail and clean else 'TAIL ADVANTAGE ONLY / CLEAN COST NOT CONFIRMED' if tail else 'CLEAN COST ONLY / TAIL ADVANTAGE NOT CONFIRMED' if clean else 'NO CLEAN-TAIL POLICY TRADEOFF CONFIRMATION'
    else:label='FIXED STUDY COMPLETE; SEE EACH HOLM-ADJUSTED PRIMARY CONTRAST'
    out.mkdir(parents=True,exist_ok=False)
    for name,frame in [('metrics',md),('block_contrasts',bd),('pool_contrasts',poold),('primary',prim)]:frame.to_csv(out/(name+'.csv'),index=False)
    d.to_csv(out/'counts.csv.gz',index=False,compression='gzip');write(out/'backends.json',{'backends':backends})
    write(out/'decision.json',{'decision':label,'protocol':protocol(study),'protocol_hash':ph(study),'source_hashes':sources(),'n_pools':6,'training_blocks':60,'states':60*len(SPECS[study]['methods']),'count_rows':len(d),'primary':prim.to_dict('records'),'files':{p.name:sha(p) for p in out.iterdir()},'completed_at':now()})
    print('DECISION',label,flush=True);print(prim.to_string(index=False),flush=True)

def selftest():
    configure(False);loss=torch.arange(16,dtype=torch.float32);gram=torch.eye(16);rng=np.random.default_rng(1)
    for s in SPECS:
        for m in SPECS[s]['methods']:
            w=weights(m,loss,gram,rng);require(torch.isfinite(w).all() and abs(float(w.sum())-1)<1e-6,'weight simplex')
    require(np.allclose(holm([.04,.01,.03]),[.06,.03,.06]),'Holm')
    x=np.arange(6)/100;require(abs(estimate(x)['p']-stats.ttest_1samp(x,0).pvalue)<1e-12,'t test')
    require(estimate(-x,'less')['p']==estimate(x,'greater')['p'],'test direction')
    avg=[]
    for c in itertools.combinations(range(15),3):
        w=np.zeros(16);w[[15,*c]]=.25;avg.append(w)
    require(np.allclose(np.mean(avg,0),weights('mix16',loss,gram,rng),atol=1e-8,rtol=0),'anchored expectation')
    model=FashionCNN();require(model(torch.zeros(2,28,28))[0].shape==(2,10),'native28 shape')
    require(schedule_hash(schedule(7500000000))==schedule_hash(schedule(7500000000)),'schedule determinism')
    print('ROADMAP_SYNTHETIC_PASS',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['selftest','train','seal','evaluate','aggregate']);ap.add_argument('--study',choices=list(SPECS));ap.add_argument('--pool',type=int);ap.add_argument('--root',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--global-seal',type=Path);a=ap.parse_args()
    if a.mode=='selftest':selftest();return
    require(a.study is not None and a.out is not None,'arguments')
    if a.mode=='train':train(a.study,a.pool,a.out)
    elif a.mode=='seal':seal(a.study,a.root,a.out)
    elif a.mode=='evaluate':evaluate(a.study,a.pool,a.out,a.global_seal)
    else:aggregate(a.study,a.root,a.out,a.global_seal)
if __name__=='__main__':main()

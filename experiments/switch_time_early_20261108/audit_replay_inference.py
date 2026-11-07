"""Replay 3 frozen 200-step runs and independently infer two heldout count rows.

Audit only: no modification of frozen experiment code or model checkpoints.
"""
from pathlib import Path
import csv,json,hashlib,sys
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
import common,legacy_study
from switch_engine import ShadowEngine,tree_digest
sys.path.insert(0,str(ROOT))
import run_switch_early as r

torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

replay_cases=[(0,0,'M'),(6,2,'T60'),(11,3,'T40')]
replays=[]
for pool,seed,policy in replay_cases:
    e=ShadowEngine(r.init_key(pool,seed))
    for index,(xb,yb) in enumerate(r.fit_batches(pool)):
        e.step(xb,yb,r.active_optimizer(policy,index))
    saved=torch.load(ROOT/f'runs/train/pool{pool}/seed{seed}_{policy}_step200.pt',weights_only=True,map_location='cpu')
    expect=tree_digest(saved['engine']);actual=tree_digest(e.state_dict())
    if expect!=actual: raise ValueError(f'REPLAY MISMATCH {(pool,seed,policy)}')
    replays.append(dict(pool=pool,seed=seed,policy=policy,steps=e.steps,tensor_digest=expect))
    print('BITWISE_REPLAY_PASS',pool,seed,policy,flush=True)

inference_cases=[(0,0,'M',0),(11,3,'T60',47)]
x,y=legacy_study.image_data('test')
checks=[]
for pool,seed,policy,env_idx in inference_cases:
    cp=torch.load(ROOT/f'runs/train/pool{pool}/seed{seed}_{policy}_step200.pt',weights_only=True,map_location='cpu')
    e=ShadowEngine.from_state_dict(cp['engine']);e.model.eval()
    env_key=2105000+20000*pool+env_idx
    img=torch.from_numpy(common.geometric_environment(x,env_key))
    with torch.no_grad():
        logits=e.model(img)[0]
        counts=int((logits.argmax(1)==y).sum())
        ce=float(torch.nn.functional.cross_entropy(logits,y,reduction='none').double().sum())
    path=ROOT/f'runs/eval/pool{pool}/heldout_counts.csv'
    with path.open(newline='') as f:
        matches=[z for z in csv.DictReader(f) if int(z['seed'])==seed and z['policy']==policy and int(z['environment'])==env_idx]
    if len(matches)!=1: raise ValueError('missing eval record')
    record=matches[0]
    err=abs(ce-float(record['ce_sum']))
    if counts!=int(record['correct']) or err>1e-8: raise ValueError('RAW_IMAGE_REINFERENCE_MISMATCH')
    checks.append(dict(pool=pool,seed=seed,policy=policy,environment=env_idx,correct=counts,ce_sum=ce,ce_error=err))
    print('RAW_IMAGE_INFERENCE_PASS',pool,seed,policy,env_idx,'ce_error',err,flush=True)
receipt=dict(replays=replays,inference=checks,backend=dict(torch=torch.__version__,threads=torch.get_num_threads(),mkldnn=torch.backends.mkldnn.enabled))
(ROOT/'results'/'REPLAY_INFERENCE_RECEIPT.json').write_text(json.dumps(receipt,indent=2,sort_keys=True))
print('REPLAY_AND_RAW_IMAGE_AUDIT_PASS',len(replays),'replays',len(checks),'inferences')

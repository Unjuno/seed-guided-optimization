"""Exact finite-law first-step check. Synthetic, not an SGO performance test."""
from __future__ import annotations
import argparse, json, platform
from fractions import Fraction as F
from pathlib import Path
import torch

LAWS={'A':[(F(-1),F(1,2)),(F(3),F(1,2))],
      'B':[(F(1,3),F(9,10)),(F(7),F(1,10))]}

def moments(law):
    mean=sum(g*p for g,p in law);second=sum(g*g*p for g,p in law)
    return {'mean':mean,'second_moment':second,'variance':second-mean*mean}

def exact_update(law,eta,eps):
    return -eta*sum(p*g/(abs(g)+eps) for g,p in law)

def pytorch_update(g,dtype,kind):
    theta=torch.nn.Parameter(torch.tensor(0.,dtype=dtype))
    if kind=='AdamW':
        opt=torch.optim.AdamW([theta],lr=.005,betas=(.9,.999),eps=1e-8,
                             weight_decay=.001,amsgrad=False,foreach=False,fused=False)
    else:
        opt=torch.optim.SGD([theta],lr=.005,momentum=.9,weight_decay=.001,
                           dampening=0,nesterov=False,foreach=False)
    theta.grad=torch.tensor(float(g),dtype=dtype);opt.step()
    return float(theta.detach())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for law in LAWS.values():
        assert sum(p for g,p in law)==1
        assert moments(law)=={'mean':F(1),'second_moment':F(5),'variance':F(4)}
    eta,eps=F(1,200),F(1,100000000)
    ua=exact_update(LAWS['A'],eta,eps);ub=exact_update(LAWS['B'],eta,eps)
    assert ua!=ub and abs(ub-ua)>F(499,100000)
    assert -ua/eta==eps/((1+eps)*(3+eps))
    assert -ub/eta==F(9,10)/(1+3*eps)+F(7,10)/(7+eps)
    records=[]
    for dtype in [torch.float64,torch.float32]:
        for kind in ['AdamW','SGD']:
            for name,law in LAWS.items():
                got=sum(float(p)*pytorch_update(g,dtype,kind) for g,p in law)
                exp=float(exact_update(law,eta,eps)) if kind=='AdamW' else -float(eta)
                error=abs(got-exp);limit=1e-15 if dtype==torch.float64 else 1e-9
                assert error<limit,(dtype,kind,name,error)
                records.append({'dtype':str(dtype),'optimizer':kind,'law':name,
                                'expected_update':got,'exact_formula':exp,'abs_error':error,'tolerance':limit})
    report={'decision':'SYNTHETIC EXACT-MOMENT COUNTEREXAMPLE VERIFIED',
            'scope':'First scalar step at zero parameter and zero optimizer state; not SGO mechanism identification',
            'laws':{name:[{'gradient':str(g),'probability':str(p)} for g,p in law] for name,law in LAWS.items()},
            'exact_moments':{name:{k:str(v) for k,v in moments(law).items()} for name,law in LAWS.items()},
            'learning_rate':float(eta),'epsilon':float(eps),'exact_updates':{'A':str(ua),'B':str(ub)},
            'records':records,'maximum_float64_error':max(r['abs_error'] for r in records if r['dtype']=='torch.float64'),
            'maximum_float32_error':max(r['abs_error'] for r in records if r['dtype']=='torch.float32'),
            'runtime':{'python':platform.python_version(),'torch':str(torch.__version__),'threads':1,
                       'device':'CPU','foreach':False,'fused':False,'quantization':False}}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()

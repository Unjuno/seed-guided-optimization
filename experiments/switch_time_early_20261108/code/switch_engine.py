"""Warm common-gradient optimizer histories, with isolated finite-horizon forks.

Only one proposed hidden weight is applied. Both hidden optimizer histories see
identical realized gradients on the actual trajectory; neither runs on its own
independent training trajectory. Output and bias parameters always use AdamW.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from typing import Any
import torch
import common

ADAM = dict(lr=.005, weight_decay=.001, betas=(.9,.999), eps=1e-8,
            foreach=False, fused=False)
MUON = dict(lr=.005, weight_decay=.001, momentum=.95, nesterov=True,
            ns_steps=5, ns_coefficients=(3.4445,-4.775,2.0315), eps=1e-7,
            adjust_lr_fn='match_rms_adamw')


def tree_digest(value: Any) -> str:
    """Stable value hash including all tensor bytes; not serialization metadata."""
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            a=v.detach().cpu().contiguous().numpy()
            h.update(b'tensor');h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
        elif isinstance(v,dict):
            h.update(b'dict')
            for k in sorted(v,key=lambda x:(type(x).__name__,repr(x))):visit(k);visit(v[k])
        elif isinstance(v,(list,tuple)):
            h.update(type(v).__name__.encode());h.update(str(len(v)).encode())
            for x in v:visit(x)
        else:
            h.update(type(v).__name__.encode());h.update(json.dumps(v,allow_nan=False).encode())
    visit(value)
    return h.hexdigest()


def choose_branch(incumbent: str, scores: dict[str,float], relative_margin:float=.005)->str:
    if incumbent not in ('A','M') or set(scores)!= {'A','M'}:
        raise ValueError('invalid policy names')
    if not all(math.isfinite(float(x)) and x>=0 for x in scores.values()):
        raise ValueError('invalid probe CE')
    if not 0<=relative_margin<1:raise ValueError('invalid hysteresis')
    other='A' if incumbent=='M' else 'M'
    return other if scores[other] < scores[incumbent]*(1-relative_margin) else incumbent


class ShadowEngine:
    def __init__(self, seed:int):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model=common.MLP()
        self.model.train()
        self.hidden=self.model.body[0].weight
        self.shadow_a=torch.nn.Parameter(self.hidden.detach().clone())
        self.shadow_m=torch.nn.Parameter(self.hidden.detach().clone())
        self.hidden_a=torch.optim.AdamW([self.shadow_a],**ADAM)
        self.hidden_m=torch.optim.Muon([self.shadow_m],**MUON)
        rest=[p for p in self.model.parameters() if p is not self.hidden]
        if len(rest)!=3 or self.hidden.shape!=(128,64):raise ValueError('model partition changed')
        self.head=torch.optim.AdamW(rest,**ADAM)
        self.steps=0

    def state_dict(self)->dict:
        return copy.deepcopy(dict(model=self.model.state_dict(),hidden_a=self.hidden_a.state_dict(),
                                  hidden_m=self.hidden_m.state_dict(),head=self.head.state_dict(),steps=self.steps))

    @classmethod
    def from_state_dict(cls, state:dict)->'ShadowEngine':
        obj=cls(0)
        obj.model.load_state_dict(state['model'])
        obj.hidden_a.load_state_dict(copy.deepcopy(state['hidden_a']))
        obj.hidden_m.load_state_dict(copy.deepcopy(state['hidden_m']))
        obj.head.load_state_dict(copy.deepcopy(state['head']))
        obj.steps=int(state['steps'])
        return obj

    def fork(self)->'ShadowEngine':
        return self.from_state_dict(self.state_dict())

    def step(self,x:torch.Tensor,y:torch.Tensor,active:str)->dict[str,float]:
        if active not in ('A','M'):raise ValueError('unknown active optimizer')
        self.model.train();self.model.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(self.model(x)[0],y)
        if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
        loss.backward()
        g=self.hidden.grad.detach().clone()
        before=self.hidden.detach().clone()
        for p,opt in ((self.shadow_a,self.hidden_a),(self.shadow_m,self.hidden_m)):
            with torch.no_grad():p.copy_(before)
            p.grad=g.clone()
            opt.step()
        with torch.no_grad():
            da=self.shadow_a-before;dm=self.shadow_m-before
            self.hidden.copy_(self.shadow_a if active=='A' else self.shadow_m)
        self.head.step();self.steps+=1
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):raise ValueError('nonfinite weight')
        na=float(da.norm());nm=float(dm.norm());ng=float(g.norm())
        cos=float(torch.sum(da*dm)/(da.norm()*dm.norm()).clamp_min(1e-30))
        return dict(train_ce=float(loss.detach()),hidden_grad_norm=ng,
                    update_norm_a=na,update_norm_m=nm,update_cosine=cos)

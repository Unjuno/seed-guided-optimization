"""POST-OUTCOME descriptive objective-mixture exploration, never a registered gate."""
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
POL=('M','T40','T60','T80')

def read(name):
    with (ROOT/'results'/name).open(newline='') as f:return list(csv.DictReader(f))
shift=read('policy_pool_losses.csv');clean=read('policy_clean_pool_losses.csv')
S=np.array([[float(next(z['mean_ce'] for z in shift if int(z['pool'])==p and z['policy']==q)) for q in POL] for p in range(12)])
C=np.array([[float(next(z['clean_ce'] for z in clean if int(z['pool'])==p and z['policy']==q)) for q in POL] for p in range(12)])
s=S.mean(0);c=C.mean(0)
assert set(POL)==set(z['policy'] for z in shift)

crossings={}
for i,a in enumerate(POL):
    for j,b in enumerate(POL):
        if j<=i:continue
        dC=c[i]-c[j];dS=s[i]-s[j];denom=dC-dS
        if abs(denom)>1e-14:
            cross=float(dC/denom)
            crossings[a+'__'+b]=cross

# Reconstruct the exact lower envelope at interval midpoints between all in-range crossings.
breaks=[0.]+sorted([x for x in crossings.values() if 0<x<1])+[1.]
segments=[]
for lo,hi in zip(breaks[:-1],breaks[1:]):
    mid=(lo+hi)/2
    vals=mid*s+(1-mid)*c
    winner=POL[int(vals.argmin())]
    if segments and segments[-1]['policy']==winner:
        segments[-1]['w_max']=hi
    else:segments.append(dict(w_min=lo,w_max=hi,policy=winner))

# Nonparametric pool-level bootstrap: descriptive only, fixed reused Digits images.
rng=np.random.default_rng(20261108)
nsim=10000
I=rng.integers(0,12,size=(nsim,12))
SB=S[I].mean(1);CB=C[I].mean(1)
weights=[0.0,0.5,0.7,0.8,0.9,1.0]
frequency=[]
for w in weights:
    utility=w*SB+(1-w)*CB
    indices=utility.argmin(1)
    freq=np.bincount(indices,minlength=4)/nsim
    frequency.append(dict(w=w,**{p:float(v) for p,v in zip(POL,freq)}))

# Fixed w=.8 descriptive pool averages
w=.8
candidates=w*s+(1-w)*c
winner=POL[int(candidates.argmin())]
res=dict(date='POSTHOC_20261108',policies=POL,shift_ce={k:float(v) for k,v in zip(POL,s)},clean_ce={k:float(v) for k,v in zip(POL,c)},
         pairwise_crossings=crossings,lower_envelope=segments,bootstrap=frequency,
         weighted_ce_at_w08={k:float(v) for k,v in zip(POL,candidates)},winner_w08=winner,
         sample_mean_pareto_dominated=['T40'] if s[1]>=s[2] and c[1]>=c[2] else [],
         per_pool_T60_lower_shift_than_T80=int((S[:,2]<S[:,3]).sum()),
         per_pool_T60_lower_shift_than_T40=int((S[:,2]<S[:,1]).sum()),
         per_pool_M_lower_clean_than_T60=int((C[:,0]<C[:,2]).sum()))
path=ROOT/'results'/'POSTHOC_TRADEOFF.json'
if path.exists():raise SystemExit('no overwrites')
path.write_text(json.dumps(res,indent=2,sort_keys=True))
print(json.dumps(res,indent=2))
# Plot default matplotlib palette, no imposed categorical claims from plot.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
wvals=np.linspace(0,1,501)
fig,ax=plt.subplots(figsize=(8.1,5.0))
for i,q in enumerate(POL):ax.plot(wvals,wvals*s[i]+(1-wvals)*c[i],label=q)
for seg in segments:
    if 0<seg['w_min']<1:ax.axvline(seg['w_min'],linestyle=':',alpha=.5)
ax.set_xlabel('Weight on transformed-image CE (w)')
ax.set_ylabel('Sample-mean CE of objective mixture')
ax.set_title('Exploratory policy preference by evaluation objective — not validated threshold')
ax.grid(alpha=.15);ax.legend()
fig.tight_layout();fig.savefig(ROOT/'results'/'POSTHOC_OBJECTIVE_FRONTIER.png',dpi=160);plt.close(fig)

"""Read-only checker for prospectively registered Issue #133.

Validates pool-level statistics only, not source hashes or models.
"""
import csv,math,statistics
from pathlib import Path
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]
NAMES=('T60_vs_M','T60_vs_T40','T60_vs_T80')
def rows(name):
    with (ROOT/'results'/name).open(newline='') as f:
        return list(csv.DictReader(f))
def near(x,y,label):
    if not math.isfinite(x) or abs(x-y)>1e-10:
        raise ValueError(f'{label}: {x} != {y}')
def holm(p):
    order=sorted(p,key=lambda a:(p[a],NAMES.index(a)))
    result={}; running=0.0
    for j,k in enumerate(order):
        running=max(running,(3-j)*p[k]);result[k]=min(1.,running)
    return result
def main():
    p=rows('switch_early_20261108_primary.csv')
    d=rows('switch_early_20261108_pools.csv')
    g=rows('switch_early_20261108_clean_guardrail_pools.csv')
    if len(p)!=3 or len(d)!=36 or len(g)!=12:
        raise ValueError('incomplete data')
    if {r['contrast'] for r in p}!=set(NAMES):
        raise ValueError('hypothesis family')
    if {(r['contrast'],int(r['pool'])) for r in d}!={(n,i) for n in NAMES for i in range(12)}:
        raise ValueError('duplicate/missing cells')
    if {int(r['pool']) for r in g}!=set(range(12)):
        raise ValueError('clean pool grid')
    ps={}
    for name in NAMES:
        values=[float(next(r['ce_improvement'] for r in d if r['contrast']==name and int(r['pool'])==j)) for j in range(12)]
        mu=statistics.fmean(values);se=statistics.stdev(values)/math.sqrt(12)
        if not se>0 or not all(map(math.isfinite,values)):
            raise ValueError('invalid precision')
        pp=float(t.sf(mu/se,11)); k=float(t.ppf(.975,11))*se
        z=next(r for r in p if r['contrast']==name)
        for key,v in [('mean_ce_improvement',mu),('se',se),('ci95_low',mu-k),('ci95_high',mu+k),('p_raw',pp)]:
            near(float(z[key]),v,key)
        if int(z['positive_pools'])!=sum(v>0 for v in values) or int(z['n_pools'])!=12:
            raise ValueError('count mismatch')
        ps[name]=pp
    hp=holm(ps)
    for row in p:
        name=row['contrast'];near(float(row['p_holm']),hp[name],'Holm3')
        decision='SUPPORT' if float(row['mean_ce_improvement'])>=.01 and hp[name]<.05 else 'NOT SUPPORTED'
        if row['decision']!=decision:
            raise ValueError('frozen decision mismatch')
    v=[float(next(r['clean_ce_penalty_T60_minus_M'] for r in g if int(r['pool'])==j)) for j in range(12)]
    mean=statistics.fmean(v);se=statistics.stdev(v)/math.sqrt(12)
    if se<=0:raise ValueError('zero guardrail SE')
    cp=float(t.cdf((mean-.05)/se,11))
    near(mean,0.10870868189889434,'clean mean');near(cp,0.9884222621688508,'clean p')
    if mean<.05 and cp<.05:
        raise ValueError('unexpected noninferiority result')
    print('EARLY_SWITCH_PUBLIC_STATS_PASS: 3 Holm3 comparisons + clean noninferiority, 12 pools')
    print('This does not authenticate 192 checkpoint hashes or reconstruct raw image inference.')
if __name__=='__main__':
    main()

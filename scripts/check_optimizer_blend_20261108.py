"""Public statistic-only checker of prospectively frozen Issue135 optimizer blends.

Reads 12 pool contrasts for each of four components, recomputes t11,
one-sided noninferiority p, pointwise 95% CI, Holm4 and conjunctive claims.
Does NOT verify training/source/checkpoint hashes, raw images or new datasets.
"""
import csv,math,statistics
from pathlib import Path
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]
CONTRASTS=('B25_shift_vs_M','B25_clean_guardrail_vs_M',
           'B50_shift_vs_M','B50_clean_guardrail_vs_M')
def read(name):
    with (ROOT/'results'/name).open(newline='') as f:
        return list(csv.DictReader(f))
def close(a,b,label):
    if not math.isfinite(a) or abs(a-b)>1e-10:raise ValueError(f'{label} mismatch {a} vs {b}')
def holm(ps):
    names=sorted(ps,key=lambda x:(ps[x],CONTRASTS.index(x)))
    out={};v=0.
    for j,name in enumerate(names):
        v=max(v,(len(names)-j)*ps[name]);out[name]=min(1.,v)
    return out
def main():
    raw=read('optimizer_blend_20261108_pool_effects.csv')
    pub=read('optimizer_blend_20261108_primary.csv')
    means=read('optimizer_blend_20261108_policy_means.csv')
    if len(raw)!=48 or len(pub)!=4 or len(means)!=4:raise ValueError('incomplete output')
    if {(z['contrast'],int(z['pool'])) for z in raw}!={(c,j) for c in CONTRASTS for j in range(12)}:
        raise ValueError('wrong pool/contrast grid')
    if {z['contrast'] for z in pub}!=set(CONTRASTS):raise ValueError('hypothesis names mismatch')
    if {z['policy'] for z in means}!={'M','B25','B50','T80'}:raise ValueError('policy grid mismatch')
    ps={}
    for name in CONTRASTS:
        row=next(z for z in pub if z['contrast']==name)
        v=[float(next(z['value'] for z in raw if z['contrast']==name and int(z['pool'])==j))
           for j in range(12)]
        if not all(math.isfinite(x) for x in v):raise ValueError('nonfinite value')
        mean=statistics.fmean(v)
        se=statistics.stdev(v)/math.sqrt(12)
        if se<=0:raise ValueError('zero standard error')
        expected_null=0.05 if 'clean_guardrail' in name else 0.0
        expected_alt='less' if 'clean_guardrail' in name else 'greater'
        if float(row['null'])!=expected_null or row['alternative']!=expected_alt or int(row['n_pools'])!=12:
            raise ValueError('registered null/alternative changed')
        z=(mean-expected_null)/se
        p=float(t.cdf(z,11) if expected_alt=='less' else t.sf(z,11))
        ci=float(t.ppf(.975,11)*se)
        for key,val in [('mean',mean),('se',se),('ci95_low',mean-ci),('ci95_high',mean+ci),('p_raw',p)]:
            close(float(row[key]),val,key)
        ps[name]=p
    adjusted=holm(ps)
    outcomes={}
    for name in CONTRASTS:
        row=next(z for z in pub if z['contrast']==name)
        close(float(row['p_holm4']),adjusted[name],'Holm4')
        ok=(float(row['mean'])<.05 if 'clean_guardrail' in name else float(row['mean'])>=.01) and adjusted[name]<.05
        label='SUPPORT' if ok else 'NOT SUPPORTED'
        if row['decision']!=label:raise ValueError('decision drift')
        outcomes[name]=ok
    for q in ('B25','B50'):
        print(q,'joint clean-safe shifted benefit =',
              'SUPPORT' if outcomes[q+'_shift_vs_M'] and outcomes[q+'_clean_guardrail_vs_M']
              else 'NOT CONFIRMED')
    m={r['policy']:r for r in means}
    for q in ('B25','B50'):
        close(float(m['M']['shift_ce'])-float(m[q]['shift_ce']),
              float(next(z['mean'] for z in pub if z['contrast']==q+'_shift_vs_M')),
              'policy shift mean')
        close(float(m[q]['clean_ce'])-float(m['M']['clean_ce']),
              float(next(z['mean'] for z in pub if z['contrast']==q+'_clean_guardrail_vs_M')),
              'policy clean mean')
    print('OPTIMIZER_BLEND_PUBLIC_HOLM4_CHECK_PASS: 12 pools, four preregistered tests')
    print('Statistics-only: no raw checkpoint, model or inference validation.')
if __name__=='__main__':main()

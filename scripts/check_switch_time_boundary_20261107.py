"""Public read-only check of preregistered Issue131 switch-time statistics.

Checks ONLY registered 36 pool contrasts, 3 t11/Holm3 tests and printed
policy means. Does not hash/reinfer 192 tensor states or prove preregistration.
Full original experiment evidence is a separate owner-supplied ZIP.
"""
import csv
from pathlib import Path
import math
import statistics
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
COMPARISONS=('T100_vs_M','T100_vs_T80','T100_vs_T120')


def read(name):
    with (ROOT/'results'/name).open(newline='') as f:
        return list(csv.DictReader(f))


def close(actual, expected, field):
    if not math.isfinite(actual) or abs(actual-expected)>1e-10:
        raise ValueError(f'{field} mismatch {actual} != {expected}')


def holm(probs):
    order=sorted(probs,key=lambda x:(probs[x],COMPARISONS.index(x)))
    result={}
    max_val=0.
    for k,name in enumerate(order):
        max_val=max(max_val,(len(probs)-k)*probs[name])
        result[name]=min(1.,max_val)
    return result


def main():
    primary=read('switch_time_20261107_primary.csv')
    rows=read('switch_time_20261107_pool_differences.csv')
    policies=read('switch_time_20261107_policy_means.csv')
    if len(primary)!=3 or len(rows)!=36 or len(policies)!=4:
        raise ValueError('missing or duplicate registered row')
    if {z['contrast'] for z in primary}!=set(COMPARISONS):
        raise ValueError('changed primary comparison family')
    if {(z['contrast'],int(z['pool'])) for z in rows} != {
        (name,pool) for name in COMPARISONS for pool in range(12)}:
        raise ValueError('incomplete 12-pool contrast grid')
    if {z['policy'] for z in policies}!={'M','T80','T100','T120'}:
        raise ValueError('four-arm policy grid changed')
    ps={}
    for name in COMPARISONS:
        v=[float(next(r['ce_improvement'] for r in rows
                      if r['contrast']==name and int(r['pool'])==i)) for i in range(12)]
        if not all(math.isfinite(x) for x in v):
            raise ValueError('nonfinite pool CE')
        mu=statistics.fmean(v)
        se=statistics.stdev(v)/math.sqrt(12)
        if se<=0:raise ValueError('zero pool SE')
        p=float(t.sf(mu/se,11))
        ci=float(t.ppf(.975,11)*se)
        row=next(r for r in primary if r['contrast']==name)
        for field,value in [('mean_ce_improvement',mu),('se',se),
                            ('ci95_low',mu-ci),('ci95_high',mu+ci),('p_raw',p)]:
            close(float(row[field]),value,field)
        if int(row['n_pools'])!=12 or int(row['positive_pools'])!=sum(x>0 for x in v):
            raise ValueError('pool count/sign disagreement')
        ps[name]=p
    adjusted=holm(ps)
    for row in primary:
        name=row['contrast']
        close(float(row['p_holm']),adjusted[name],'Holm adjusted p')
        status=('SUPPORT' if float(row['mean_ce_improvement'])>=.01
                and adjusted[name]<.05 else 'NOT SUPPORTED')
        if row['decision']!=status:
            raise ValueError('registered outcome changed')
    print('SWITCH_TIME_REGISTERED_PUBLIC_CHECK_PASS: 3 tests, 36 pool contrasts, 12 pools.')
    print('This is statistics-only; code/source/seal/checkpoint audit is separate.')


if __name__=='__main__':
    main()

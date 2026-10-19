"""Recalculate all PR126 preregistered primary statistics from six-pool CSVs.

Inputs are public summary CSVs. This does not retrain or check state hashes.
Confidence intervals are pointwise, and n=6 environment pools for each test.
"""
import csv
from pathlib import Path
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    with (ROOT / 'results' / name).open(newline='') as f:
        return list(csv.DictReader(f))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p, kind='stable')
    out = np.empty_like(p)
    out[order] = np.minimum(1., np.maximum.accumulate(
        p[order] * (len(p) - np.arange(len(p)))))
    return out


def near(a, b):
    if not np.isfinite(a) or abs(a - b) > 1e-10:
        raise ValueError(f'published statistic mismatch {a} != {b}')


def main():
    primary = load('roadmap_validation_20260926_primary.csv')
    pools = load('roadmap_validation_20260926_pool_contrasts.csv')
    expected = {
        'tail': {
            'gradnov_minus_anchor_random:min:greater:0',
            'gradnov_minus_anchor_random:clean:less:0'},
        'fashion': {
            'anchor_random_minus_loss_hard:mean:greater:0.005',
            'gradnov_minus_anchor_random:mean:greater:0.005',
            'gradnov_minus_anchor_random:min:greater:0.005'},
        'optimizer': {
            'AdamW_objective:mean:two-sided:0',
            'AdamW_estimator:mean:two-sided:0',
            'AdamW_interaction:mean:two-sided:0',
            'SGD_objective:mean:two-sided:0',
            'SGD_estimator:mean:two-sided:0',
            'SGD_interaction:mean:two-sided:0',
            'three_way:mean:two-sided:0'}}
    if len(primary) != 12 or len(pools) != 72:
        raise ValueError('missing/extra registered rows')
    poolkeys = set()
    for item in pools:
        key = (item['study'], item['contrast'], item['endpoint'],
               int(item['pool']))
        if key in poolkeys or key[3] not in range(6):
            raise ValueError('duplicate/invalid pool')
        poolkeys.add(key)
    keys = set()
    for study, required in expected.items():
        group = [p for p in primary if p['study'] == study]
        actual = {
            ':'.join([r['contrast'], r['endpoint'], r['alternative'],
                      str(float(r['null'])) if float(r['null']) else '0'])
            for r in group}
        if actual != required:
            raise ValueError('unregistered contrasts ' + study)
        ps = []
        for r in group:
            key = (study, r['contrast'], r['endpoint'])
            if key in keys:
                raise ValueError('duplicate primary')
            keys.add(key)
            block = [x for x in pools if
                     (x['study'], x['contrast'], x['endpoint']) == key]
            if len(block) != 6 or {int(x['pool']) for x in block} != set(range(6)):
                raise ValueError('pool grid')
            v = np.asarray([float(next(x['difference'] for x in block
                                       if int(x['pool']) == j))
                            for j in range(6)])
            if not np.isfinite(v).all():
                raise ValueError('nonfinite')
            mean = float(v.mean())
            se = float(v.std(ddof=1) / np.sqrt(6))
            if se <= 0:
                raise ValueError('zero standard error')
            delta = (mean - float(r['null'])) / se
            direction = r['alternative']
            p = float(t.sf(delta, 5) if direction == 'greater'
                      else t.cdf(delta, 5) if direction == 'less'
                      else 2 * t.sf(abs(delta), 5))
            k = float(t.ppf(.975, 5))
            near(float(r['mean']), mean)
            near(float(r['ci95_low']), mean - k * se)
            near(float(r['ci95_high']), mean + k * se)
            ps.append(p)
        adjusted = ps if study == 'tail' else holm(ps)
        for r, pa in zip(group, adjusted):
            near(float(r['p_adjusted']), float(pa))
            mean, null = float(r['mean']), float(r['null'])
            direction = r['alternative']
            directional = (
                mean > null if direction == 'greater'
                else mean < null if direction == 'less'
                else True)
            passed = directional and pa < .05
            if str(bool(passed)).lower() != r['pass'].lower():
                raise ValueError('published PASS mismatch')
    if len(keys) != 12 or len(poolkeys) != 72:
        raise ValueError('bad registered grid')
    print('ROADMAP VALIDATION PUBLIC STATISTICS VERIFIED: 12 tests, 72 pool observations')


if __name__ == '__main__':
    main()

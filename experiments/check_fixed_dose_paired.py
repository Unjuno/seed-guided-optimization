"""Independently recompute Issue #61 primary statistics from public paired CSVs.

This does not train a model, select a run, or establish causal mediation.
Run from the repository root with: python experiments/check_fixed_dose_paired.py
All benefit/SE/CI fields are dimensionless accuracy fractions; multiply by 100
for percentage points. See docs/FIXED_DOSE_RESPONSE.md for variable definitions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    paired = pd.read_csv(args.input_dir / 'fixed_dose_paired30.csv')
    expected = ['rep', 'clean_benefit', 'weak_benefit', 'full_benefit', 'interaction']
    if list(paired.columns) != expected or sorted(paired.rep.tolist()) != list(range(1200, 1230)):
        raise ValueError('incorrect paired grid or schema')
    if not np.isfinite(paired[expected[1:]].to_numpy()).all():
        raise ValueError('nonfinite measurements')
    np.testing.assert_allclose(paired.full_benefit-paired.clean_benefit,
                               paired.interaction, rtol=0, atol=1e-14)
    decision = pd.read_csv(args.input_dir / 'fixed_dose_decision30.csv')
    if len(decision) != 1:
        raise ValueError('exactly one frozen decision required')
    decision = decision.iloc[0]
    rows = []
    for column in expected[1:]:
        values = paired[column].to_numpy(dtype=np.float64)
        mean = float(values.mean())
        se = float(values.std(ddof=1)/np.sqrt(len(values)))
        k = float(stats.t.ppf(.975, len(values)-1))
        if se == 0:
            p = 0.0 if mean > 0 else (1.0 if mean < 0 else .5)
        else:
            p = float(stats.t.sf(mean/se, len(values)-1))
        rows.append(dict(metric=column, n=len(values), mean=mean, se=se, k=k,
            ci95_low=mean-k*se, ci95_high=mean+k*se, p_one_sided=p,
            p_two_sided=2*min(p,1-p), positive=int((values>0).sum()),
            negative=int((values<0).sum())))
    lookup = {row['metric']:row for row in rows}
    for column, prefix in [('full_benefit','full'), ('interaction','interaction')]:
        for field in ('mean','se','ci95_low','ci95_high','p_one_sided'):
            if not np.isclose(lookup[column][field], decision[prefix+'_'+field], rtol=0, atol=1e-13):
                raise ValueError(f'frozen statistic mismatch: {prefix}_{field}')
    full = lookup['full_benefit']; interaction = lookup['interaction']
    fp = full['mean'] > 0 and full['p_one_sided'] < .05
    ip = interaction['mean'] > 0 and interaction['p_one_sided'] < .05
    label = 'DOSE-DEPENDENT BENEFIT PASS' if fp and ip else (
        'FULL EFFECT ONLY / NO DOSE INTERACTION' if fp else 'NO FULL-STRENGTH REPLICATION')
    if label != decision.decision:
        raise ValueError('frozen scientific decision mismatch')
    result = pd.DataFrame(rows)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print('INDEPENDENT CHECK PASS:', label)


if __name__ == '__main__':
    main()

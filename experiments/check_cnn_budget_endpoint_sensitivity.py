"""Post-hoc endpoint sensitivity of Issue 70; no retraining or decision replacement.

Definitions and limitations: docs/CNN_BUDGET_ENDPOINT_SENSITIVITY.md.
Inputs are published paired attenuation rows and the frozen Q16 identity report.
All reported differences and standard errors are accuracy fractions (SI unit 1).
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPS = tuple(range(1500, 1530))
COLUMNS = ['benefit_low', 'benefit_high', 'benefit_attenuation']


def analyze(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) != 30 or frame.rep.duplicated().any() or set(frame.rep) != set(REPS):
        raise ValueError('expected exactly registered paired reps 1500-1529')
    x = frame.sort_values('rep')[COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all() or np.any(np.abs(x[:, :2]) > 1):
        raise ValueError('invalid accuracy differences')
    low, high, registered = x.T
    if not np.allclose(registered, low - high, rtol=0, atol=1e-12):
        raise ValueError('registered contrast arithmetic mismatch')
    rows = []
    for name, values in [('registered_low_minus_high', low - high),
                         ('posthoc_low_minus_q12', low - 2 * high)]:
        mean = float(values.mean())
        se = float(values.std(ddof=1) / np.sqrt(len(values)))
        k = float(stats.t.ppf(.975, len(values) - 1))
        p = float(stats.t.sf(mean / se, len(values) - 1)) if se > 0 else (0.0 if mean > 0 else 1.0 if mean < 0 else .5)
        rows.append(dict(analysis=name, n=len(values), mean=mean, se=se, k=k,
                         ci95_low=mean-k*se, ci95_high=mean+k*se,
                         p_one_sided_descriptive=p, positive_pairs=int((values > 0).sum())))
    return pd.DataFrame(rows)


def selftest() -> None:
    f = pd.DataFrame({'rep': REPS, 'benefit_low': np.linspace(.01, .04, 30),
                      'benefit_high': .01})
    f['benefit_attenuation'] = f.benefit_low - f.benefit_high
    a = analyze(f)
    np.testing.assert_allclose(a['mean'], [.015, .005], atol=1e-14, rtol=0)
    bad = [f.iloc[:-1], pd.concat([f.iloc[:-1], f.iloc[:1]])]
    b = f.copy(); b.loc[0, 'benefit_low'] = np.nan; bad.append(b)
    b = f.copy(); b.loc[0, 'benefit_attenuation'] += .01; bad.append(b)
    for b in bad:
        try:
            analyze(b)
        except ValueError:
            continue
        raise AssertionError('invalid synthetic input accepted')
    print('SELFTEST PASS: arithmetic and missing/duplicate/nonfinite/inconsistent data checks')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=Path('results'))
    p.add_argument('--output', type=Path)
    p.add_argument('--selftest', action='store_true')
    args = p.parse_args()
    if args.selftest:
        selftest(); return
    decision = pd.read_csv(args.input_dir / 'cnn_budget_decision30.csv')
    mismatch = pd.read_csv(args.input_dir / 'cnn_budget_q16_mismatches.csv')
    if (len(decision) != 1 or int(decision.iloc[0]['n']) != 30
            or str(decision.iloc[0]['q16_identity_pass']) != 'True'
            or int(decision.iloc[0]['q16_mismatch_count']) != 0 or not mismatch.empty):
        raise ValueError('published exact Q16 identity premise not established')
    source = args.input_dir / 'cnn_budget_attenuation30.csv'
    result = analyze(pd.read_csv(source, float_precision='round_trip'))
    result['input_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print('POST-HOC SENSITIVITY ONLY: the original frozen decision is not replaced.')


if __name__ == '__main__':
    main()

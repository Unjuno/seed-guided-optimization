"""Verify tracked Issue121 statistics only; no model training or image access."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy import stats

COMPARATORS = ('loss_hard', 'anchor_random', 'sham_gradnov')


def holm(values: list[float]) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid p values')
    result = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(p)), key=lambda i: (p[i], i))):
        running = max(running, (len(p) - rank) * p[index])
        result[index] = min(1.0, running)
    return result


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def verify(root: Path) -> None:
    paired = read(root / 'results/online_controls_contrasts60.csv')
    summary = read(root / 'results/online_controls_summary60.csv')
    if [int(r['rep']) for r in paired] != list(range(5000, 5060)):
        raise ValueError('Missing, duplicate or unordered repetitions')
    if any(int(r['pool']) != (int(r['rep']) - 5000) // 20 for r in paired):
        raise ValueError('Pool mapping changed')
    if [r['comparator'] for r in summary] != list(COMPARATORS):
        raise ValueError('Comparator family changed')
    p_values = []
    means = []
    for comparator, stored in zip(COMPARATORS, summary):
        x = np.array([float(r['gradnov_minus_' + comparator]) for r in paired])
        if not np.isfinite(x).all() or (np.abs(x) > 1).any():
            raise ValueError('Nonfinite or out-of-domain accuracy contrast')
        n = len(x)
        mean = float(x.mean())
        se = float(x.std(ddof=1) / np.sqrt(n))
        k = float(stats.t.ppf(.975, n - 1))
        if se <= 0:
            raise ValueError('Degenerate sample; inspect explicitly')
        p = float(stats.ttest_rel(x, np.zeros(n), alternative='greater').pvalue)
        values = dict(n=n, mean=mean, se=se, k=k, ci95_low=mean-k*se,
                      ci95_high=mean+k*se, p_one_sided=p,
                      positive_pairs=int((x > 0).sum()))
        for field, value in values.items():
            if not np.isclose(float(stored[field]), value, atol=1e-13, rtol=1e-10):
                raise ValueError(f'{comparator}: {field} mismatch')
        p_values.append(p)
        means.append(mean)
    adjusted = holm(p_values)
    for row, p, mean in zip(summary, adjusted, means):
        if not np.isclose(float(row['p_holm']), p, atol=1e-13, rtol=1e-10):
            raise ValueError('Holm mismatch')
        if row['supported'] not in ('True', 'False'):
            raise ValueError('Malformed boolean')
        if (row['supported'] == 'True') != (mean > 0 and p < .05):
            raise ValueError('Support label mismatch')
    print('ONLINE CONTROL PUBLIC STATISTICS VERIFIED; not a raw-image/state audit')


def selftest() -> None:
    if not np.allclose(holm([.01, .04, .03]), [.03, .06, .06]):
        raise AssertionError('Holm reference case')
    for malformed in ([float('nan')], [1.1], [-.1]):
        try:
            holm(malformed)
        except ValueError:
            continue
        raise AssertionError('Malformed p value accepted')
    print('ONLINE CONTROL STATISTICAL SELFTEST PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    selftest()
    verify(args.root)

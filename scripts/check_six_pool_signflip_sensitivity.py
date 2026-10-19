"""Post-hoc six-pool sign-flip sensitivity, NOT the preregistered t5 test."""
from pathlib import Path
import csv
import itertools

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    with (ROOT / 'results' / name).open(newline='') as f:
        return list(csv.DictReader(f))


def holm(probs):
    order = sorted(range(len(probs)), key=lambda i: (probs[i], i))
    adjusted = [0.] * len(probs)
    previous = 0.
    for position, i in enumerate(order):
        previous = max(previous, (len(probs) - position) * probs[i])
        adjusted[i] = min(1., previous)
    return adjusted


def calc(values, alternative, null):
    centered = [v - null for v in values]
    observed = sum(centered) / 6
    exceed = 0
    for signs in itertools.product((-1, 1), repeat=6):
        shuffled = sum(a * b for a, b in zip(signs, centered)) / 6
        tol = 1e-13
        if alternative == 'greater':
            hits = shuffled >= observed - tol
        elif alternative == 'less':
            hits = shuffled <= observed + tol
        elif alternative == 'two-sided':
            hits = abs(shuffled) >= abs(observed) - tol
        else:
            raise ValueError('unsupported alternative')
        exceed += int(hits)
    return exceed / 64


def main():
    primary = read('roadmap_validation_20260926_primary.csv')
    pools = read('roadmap_validation_20260926_pool_contrasts.csv')
    if len(primary) != 12 or len(pools) != 72:
        raise ValueError('missing/extra rows')
    for study in ('tail', 'fashion', 'optimizer'):
        rows = []
        for r in [x for x in primary if x['study'] == study]:
            selected = [x for x in pools if
                        (x['study'], x['contrast'], x['endpoint']) ==
                        (study, r['contrast'], r['endpoint'])]
            if len(selected) != 6 or {int(x['pool']) for x in selected} != set(range(6)):
                raise ValueError('pool support/grid')
            values = [float(next(x['difference'] for x in selected
                                 if int(x['pool']) == j)) for j in range(6)]
            rows.append((r, calc(values, r['alternative'], float(r['null']))))
        adjusted = ([p for _, p in rows] if study == 'tail'
                    else holm([p for _, p in rows]))
        print(study)
        for (r, p), q in zip(rows, adjusted):
            print(r['contrast'], r['endpoint'],
                  'p_signflip=', format(p, '.8f'),
                  'p_holm_sensitivity=', format(q, '.8f'),
                  'frozen_t5_pass=', r['pass'])


if __name__ == '__main__':
    main()

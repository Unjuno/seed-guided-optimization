"""Read-only recovery integrity checks for Issue 76; never trains or selects outcomes.

The original numerical experiment and statistical decision are unchanged.
No CIFAR inputs or labels are loaded by this verifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROTOCOL_HASH = 'e388d13d5890e8b60e939f085403763d5065360bd3cdaecc7aa05de510f144d1'
SOURCES = {
    'cifar_resnet_pilot.py': 'cc0b454c9c7a6fd798cae7675d7db07530d54407c124bdf91b8c5936f0c09e4d',
    'cifar_resnet_finetune_pilot.py': '446c41a33fa86106d7400b03c00e0aff7e3e22b2278e2b0dddf4f4aefbf5deee',
    'cifar_resnet_budget_scaling.py': '59c6afb6fa8ebd324b7353a475390de03d1d51f6023ee862cb73a1253dec1dc0',
}
SUMMARY_SHA = '12ee458bc32f23d43411075c21f16fe031ae410c5057f662bf549542b2196dd3'
OLD_RUN = '34004715086'
OLD_STARTS = (50, 55, 65, 70, 75)
OLD_REPS = tuple(r for s in OLD_STARTS for r in range(s, s + 5))
RANGES = {(s, s + 5) for s in OLD_STARTS} | {(s, s + 1) for s in range(60, 65)}
QS = (2, 4, 6, 8)
METHODS = ('loss_hard', 'gradnov')
KEY = ['rep', 'q', 'method']
TRAIN = ['selected_pairwise_novelty', 'mean_candidate_loss', 'mean_selected_loss']
HELD = ['mean_test', 'sd_test', 'p10_test', 'min_test', 'clean_test']


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(state: dict) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        a = tensor.detach().cpu().contiguous().numpy()
        require(bool(np.isfinite(a).all()), 'nonfinite model tensor')
        for part in (name.encode(), str(a.dtype).encode(), str(a.shape).encode(), a.tobytes()):
            h.update(part)
    return h.hexdigest()


def sources() -> None:
    require(sha(Path(__file__).parent / 'summarize_cifar_resnet_budget_scaling.py') == SUMMARY_SHA, 'frozen summarizer changed')
    for name, expected in SOURCES.items():
        require(sha(Path(__file__).parent / name) == expected, 'scientific source changed: ' + name)


def grid(df: pd.DataFrame, reps: tuple, metrics: list) -> None:
    exp = {(r, q, m) for r in reps for q in QS for m in METHODS}
    require(len(df) == len(exp) and not df.duplicated(KEY).any()
            and set(df[KEY].itertuples(index=False, name=None)) == exp, 'invalid scientific grid')
    require(bool(np.isfinite(df[metrics].to_numpy(float)).all()), 'nonfinite scientific metric')
    require(bool(df.candidate_k.eq(8).all()) and bool(df.coverage_ratio.eq(df.q / 8).all()), 'budget metadata')
    require(bool(df.state_digest.str.fullmatch('[0-9a-f]{64}', na=False).all()), 'state digest format')
    for r in reps:
        pair = df[(df.rep == r) & (df.q == 8)].set_index('method')
        require(pair.loc['loss_hard', 'state_digest'] == pair.loc['gradnov', 'state_digest'], 'Q8 state mismatch')
        x = pair.loc['loss_hard', metrics].to_numpy(dtype=np.float64)
        y = pair.loc['gradnov', metrics].to_numpy(dtype=np.float64)
        require(x.tobytes() == y.tobytes(), 'Q8 scientific metric mismatch')


def inspect_shard(path: Path, evaluated: bool = False) -> tuple[dict, tuple, float]:
    m = json.loads((path / 'manifest.json').read_text())
    start, end = m['start'], m['end']
    require((start, end) in RANGES, 'unapproved shard, including excluded partial 60:65')
    require(m['protocol_hash'] == PROTOCOL_HASH, 'protocol hash')
    encoded = json.dumps(m['protocol'], sort_keys=True).encode()
    require(hashlib.sha256(encoded).hexdigest() == PROTOCOL_HASH, 'protocol content')
    require(m['source_hashes'] == SOURCES, 'manifest source hashes')
    require((m['n_train'], m['n_test']) == (6000, 3000), 'image counts')
    rt = m['runtime']
    require(rt['threads'] == 4 and rt['deterministic'] is True, 'execution mode changed')
    require(all(rt['thread_env'][k] == '4' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')), 'thread environment')
    for name, version in [('numpy', '2.3.5'), ('pandas', '2.2.3'), ('torch', '2.10.0+cpu')]:
        require(rt[name] == version, 'training runtime changed: ' + name)
    require(str(rt['python']).startswith('3.12.14 '), 'training Python version')
    if start in OLD_STARTS:
        require(str(rt['run_id']) == OLD_RUN, 'completed old training was replaced')
    else:
        require(str(rt['run_id']) != OLD_RUN, 'unsealed partial data reused')
    for name, expected in SOURCES.items():
        require(sha(path / 'source' / name) == expected, 'archived source changed')
    require(sha(path / 'source' / 'summarize_cifar_resnet_budget_scaling.py') == SUMMARY_SHA, 'archived summarizer changed')
    reps = tuple(range(start, end))
    trpath = path / f'cifar_budget_training_{start}_{end}.csv'
    tr = pd.read_csv(trpath, float_precision='round_trip')
    grid(tr, reps, TRAIN)
    names = {f'rep{r}_q{q}_{method}.pt' for r in reps for q in QS for method in METHODS}
    require(set(m['checkpoints']) == names, 'manifest checkpoint coverage')
    require({p.name for p in (path / 'states').glob('*.pt')} == names, 'checkpoint files coverage')
    indexed = tr.set_index(KEY)
    for r, q, method in indexed.index:
        name = f'rep{r}_q{q}_{method}.pt'
        p = path / 'states' / name
        require(sha(p) == m['checkpoints'][name], 'checkpoint file mutated')
        ck = torch.load(p, map_location='cpu', weights_only=True)
        require((ck['rep'], ck['q'], ck['method'], ck['protocol_hash']) == (r, q, method, PROTOCOL_HASH), 'checkpoint metadata')
        require(digest(ck['state_dict']) == indexed.loc[(r, q, method), 'state_digest'], 'canonical tensor digest')
    error = 0.0
    if evaluated:
        he = pd.read_csv(path / f'cifar_budget_heldout_{start}_{end}.csv', float_precision='round_trip')
        grid(he, reps, HELD)
        for metric in ('mean_test', 'p10_test', 'min_test', 'clean_test'):
            require(bool(he[metric].between(0, 1).all()), 'accuracy range')
        require(bool(he.sd_test.ge(0).all()), 'negative SD')
        hi = he.set_index(KEY).sort_index()
        require(hi.state_digest.equals(indexed.sort_index().state_digest), 'training/evaluation state mismatch')
        ev = pd.read_csv(path / f'cifar_budget_environment_{start}_{end}.csv.gz', float_precision='round_trip')
        ek = KEY + ['env_seed']
        exp = {(r, q, method, s) for r in reps for q in QS for method in METHODS for s in range(60000, 60032)}
        require(len(ev) == len(exp) and not ev.duplicated(ek).any()
                and set(ev[ek].itertuples(index=False, name=None)) == exp, 'environment grid')
        require(bool(ev.accuracy.between(0, 1).all()), 'invalid environment accuracy')
        for key, group in ev.groupby(KEY):
            a = group.sort_values('env_seed').accuracy.to_numpy(float)
            want = np.array([a.mean(), a.std(ddof=1), np.quantile(a, .1), a.min()])
            got = hi.loc[key, ['mean_test', 'sd_test', 'p10_test', 'min_test']].to_numpy(float)
            error = max(error, float(np.max(np.abs(want - got))))
        require(error <= 1e-12, 'environment aggregation mismatch')
    cert = {'start': start, 'end': end, 'manifest_sha256': sha(path / 'manifest.json'),
            'training_csv_sha256': sha(trpath), 'checkpoints': m['checkpoints'], 'runtime': rt}
    return cert, reps, error


def verify(root: Path, scope: str, evaluated: bool, seal: Path | None) -> dict:
    sources()
    paths = sorted(root.rglob('manifest.json'))
    require(bool(paths), 'no manifests')
    certs, covered, error = {}, [], 0.0
    for f in paths:
        c, reps, err = inspect_shard(f.parent, evaluated)
        require(str(c['start']) not in certs, 'duplicate shard')
        certs[str(c['start'])] = c
        covered.extend(reps)
        error = max(error, err)
    target = OLD_REPS if scope == 'original' else tuple(range(50, 80)) if scope == 'all' else tuple(covered)
    require(sorted(covered) == sorted(target) and len(covered) == len(set(covered)), 'replicate coverage')
    if scope == 'shard':
        require(len(certs) == 1, 'expected one shard')
    if seal is not None:
        s = json.loads(seal.read_text())
        require(s['protocol_hash'] == PROTOCOL_HASH and s['replicates'] == list(range(50, 80)), 'global seal incomplete')
        for name, c in certs.items():
            require(s['shards'].get(name) == c, 'global sealed training data changed')
    return {'protocol_hash': PROTOCOL_HASH, 'replicates': sorted(covered), 'checkpoint_count': 8 * len(covered),
            'q8_training_identity_pairs': len(covered), 'environment_rows_verified': 256 * len(covered) if evaluated else 0,
            'max_environment_reaggregation_error': error, 'shards': certs}


def selftest() -> None:
    s = {'b': torch.tensor([1., 2.]), 'a': torch.tensor([[3.]])}
    require(digest(s) == digest(dict(reversed(list(s.items())))), 'digest ordering')
    require(digest(s) != digest({**s, 'b': torch.tensor([1., 3.])}), 'digest mutation')
    rows = [{'rep': r, 'q': q, 'method': m, 'candidate_k': 8, 'coverage_ratio': q / 8,
             'state_digest': 'a' * 64, **{k: .1 for k in TRAIN}} for r in (1, 2) for q in QS for m in METHODS]
    df = pd.DataFrame(rows)
    grid(df, (1, 2), TRAIN)
    bad = [df.iloc[:-1], pd.concat([df.iloc[:-1], df.iloc[:1]])]
    x = df.copy(); x.loc[0, TRAIN[0]] = np.nan; bad.append(x)
    x = df.copy(); x.loc[0, 'candidate_k'] = 7; bad.append(x)
    x = df.copy(); x.loc[7, TRAIN[0]] = .2; bad.append(x)
    x = df.copy(); x.loc[0, 'q'] = 3; bad.append(x)
    for x in bad:
        try:
            grid(x, (1, 2), TRAIN)
        except ValueError:
            continue
        raise AssertionError('invalid synthetic grid accepted')
    print('SELFTEST PASS: digest order/mutation, missing/duplicate/NaN/budget/Q8/unregistered-Q rejection')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('sources', 'selftest', 'verify'))
    p.add_argument('--root', type=Path)
    p.add_argument('--scope', choices=('original', 'all', 'shard'), default='all')
    p.add_argument('--evaluated', action='store_true')
    p.add_argument('--seal', type=Path)
    p.add_argument('--report', type=Path)
    a = p.parse_args()
    if a.mode == 'selftest':
        selftest(); return
    if a.mode == 'sources':
        sources(); print('ORIGINAL SCIENTIFIC SOURCE HASHES PASS'); return
    if a.root is None:
        p.error('--root required')
    report = verify(a.root, a.scope, a.evaluated, a.seal)
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'shards'}, sort_keys=True))


if __name__ == '__main__':
    main()

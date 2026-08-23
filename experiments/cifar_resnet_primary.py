from __future__ import annotations

import argparse, json, time
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F

from experiments.cifar_resnet_pilot import (
    CIFARResNet20, build_schedule, configure, head_gradient_directions,
    load_cifar, normalize, seed_everything, select_gradnov, select_loss,
)
from experiments.cifar_resnet_finetune_pilot import (
    cache_schedule, cache_test, evaluate_cached, pretrain,
)


def run(a):
    configure()
    xtr, ytr, xte, yte = load_cifar(a.data_root, a.train_per_class, a.test_per_class)
    train_env = [20000 + i for i in range(64)]
    # Fresh final environment pool: disjoint from tuning (25k) and pilot (30k).
    test_env = [40000 + i for i in range(a.test_envs)]

    rep = a.rep
    base = 91_000_000 + rep * 7919
    test_cache = cache_test(xte, test_env)
    state = pretrain(base, xtr, ytr, a.pretrain_epochs, a.batch_size, a.pretrain_lr)

    base_model = CIFARResNet20(); base_model.load_state_dict(state)
    base_metrics = evaluate_cached(base_model, test_cache, yte, xte)
    schedule = build_schedule(len(ytr), a.finetune_epochs, a.batch_size, 64, a.k, base + 1000)
    cached = cache_schedule(xtr, schedule, train_env)

    rows = []
    for method in ('loss4', 'gradnov4'):
        seed_everything(base)
        model = CIFARResNet20(); model.load_state_dict(state)
        opt = torch.optim.AdamW(model.parameters(), lr=a.finetune_lr, weight_decay=1e-4)
        st = time.time(); model.train()
        for step, idx, cand, u8 in cached:
            xb = normalize(u8.float() / 255)
            logits, h = model(xb)
            per = F.cross_entropy(logits, ytr[idx].repeat(a.k), reduction='none').reshape(a.k, -1)
            loss = per.mean(1)
            if method == 'loss4':
                chosen = select_loss(loss, a.q)
            else:
                gd = head_gradient_directions(logits, h, ytr[idx], a.k)
                chosen = select_gradnov(loss, gd @ gd.T, a.q)
            opt.zero_grad(set_to_none=True)
            per[chosen].mean().backward()
            opt.step()
        row = {
            'rep': rep, 'method': method, 'train_seconds': time.time() - st,
            'pretrain_epochs': a.pretrain_epochs, 'finetune_epochs': a.finetune_epochs,
            'n_train': len(ytr), 'n_test': len(yte), 'candidate_k': a.k, 'backward_q': a.q,
            'base_clean': base_metrics['clean_test'], 'base_mean': base_metrics['mean_test'],
            **evaluate_cached(model, test_cache, yte, xte),
        }
        rows.append(row); print(json.dumps(row), flush=True)

    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='.cache/cifar10')
    p.add_argument('--output', required=True)
    p.add_argument('--rep', type=int, required=True)
    p.add_argument('--pretrain-epochs', type=int, default=10)
    p.add_argument('--finetune-epochs', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--train-per-class', type=int, default=600)
    p.add_argument('--test-per-class', type=int, default=300)
    p.add_argument('--test-envs', type=int, default=32)
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--q', type=int, default=4)
    p.add_argument('--pretrain-lr', type=float, default=3e-3)
    p.add_argument('--finetune-lr', type=float, default=1e-3)
    run(p.parse_args())

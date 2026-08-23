from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from common import (
    MLP, configure_determinism, environment_metrics, geometric_environment,
    head_gradient_directions, load_digits_split, seed_everything,
    select_hard_gradient_novel, select_loss_hard,
)


def make_optimizer(name: str, model: torch.nn.Module):
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
    if name == "sgd_momentum":
        # LR=0.2 selected by an independent loss-hard LR sweep; see results/sgd_lr_sweep_*.csv.
        return torch.optim.SGD(model.parameters(), lr=2e-1, momentum=0.9, weight_decay=1e-4)
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=20)
    p.add_argument("--output", default="optimizer_ablation.csv")
    args = p.parse_args()

    configure_determinism(1)
    batch_size, epochs, k, q = 128, 10, 16, 4
    xtr, ytr, xte, yte = load_digits_split()
    train = [torch.tensor(geometric_environment(xtr, s)) for s in range(13000, 13064)]
    unseen = [torch.tensor(geometric_environment(xte, s)) for s in range(14000, 14080)]
    clean = torch.tensor(xte).flatten(1)
    ty, ey = torch.tensor(ytr), torch.tensor(yte)

    rows = []
    t0 = time.time()
    for rep in range(args.start, args.end):
        base = 23000000 + rep * 2281
        tg = torch.Generator().manual_seed(base + 1)
        er = np.random.default_rng(base + 2)
        schedule = []
        for _ in range(epochs):
            for b in torch.randperm(len(ty), generator=tg).split(batch_size):
                schedule.append((b, er.choice(64, k, replace=False).tolist()))

        for optimizer_name in ("adamw", "sgd_momentum"):
            for selector in ("loss4", "gradnov4"):
                seed_everything(base)
                model = MLP()
                opt = make_optimizer(optimizer_name, model)
                st = time.time()
                for b, cand in schedule:
                    xb = torch.cat([train[e][b] for e in cand])
                    logits, h = model(xb)
                    per = torch.nn.functional.cross_entropy(logits, ty[b].repeat(k), reduction="none").reshape(k, -1)
                    loss = per.mean(1)
                    gd = head_gradient_directions(logits, h, ty[b], k)
                    cosine = gd @ gd.T
                    idx = select_loss_hard(loss, q) if selector == "loss4" else select_hard_gradient_novel(loss, cosine, q)
                    opt.zero_grad(set_to_none=True)
                    per[idx].mean().backward()
                    opt.step()
                metrics = environment_metrics(model, unseen, ey, clean)
                rows.append({"rep": rep, "optimizer": optimizer_name, "selector": selector, "train_seconds": time.time() - st, **metrics})
        print(f"rep {rep} done {time.time()-t0:.1f}s", flush=True)

    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

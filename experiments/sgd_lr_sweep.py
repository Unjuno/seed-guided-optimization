from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import torch

from common import (
    MLP,
    configure_determinism,
    environment_metrics,
    geometric_environment,
    load_digits_split,
    seed_everything,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=10)
    p.add_argument("--output", default="sgd_lr_sweep.csv")
    args = p.parse_args()

    configure_determinism(1)
    batch_size, epochs, k = 128, 10, 16
    lrs = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00]

    xtr, ytr, xte, yte = load_digits_split()
    train = [torch.tensor(geometric_environment(xtr, s)) for s in range(13000, 13064)]
    unseen = [torch.tensor(geometric_environment(xte, s)) for s in range(14000, 14080)]
    clean = torch.tensor(xte).flatten(1)
    ty, ey = torch.tensor(ytr), torch.tensor(yte)

    rows = []
    for rep in range(args.start, args.end):
        base = 24000000 + rep * 2297
        tg = torch.Generator().manual_seed(base + 1)
        er = np.random.default_rng(base + 2)
        schedule = []
        for _ in range(epochs):
            for b in torch.randperm(len(ty), generator=tg).split(batch_size):
                schedule.append((b, er.choice(64, k, replace=False).tolist()))

        for lr in lrs:
            seed_everything(base)
            model = MLP()
            opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

            # Tune only the baseline loss-hard selector.
            for b, cand in schedule:
                xb = torch.cat([train[e][b] for e in cand])
                logits, _ = model(xb)
                per = torch.nn.functional.cross_entropy(
                    logits, ty[b].repeat(k), reduction="none"
                ).reshape(k, -1)
                loss = per.mean(1)
                idx = torch.topk(loss, 4).indices
                opt.zero_grad(set_to_none=True)
                per[idx].mean().backward()
                opt.step()

            rows.append({"rep": rep, "lr": lr, **environment_metrics(model, unseen, ey, clean)})

    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

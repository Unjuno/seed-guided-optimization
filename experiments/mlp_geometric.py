from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from common import (
    MLP, configure_determinism, environment_metrics, environment_parameters,
    farthest_subset, geometric_environment, head_gradient_directions,
    load_digits_split, rng_fingerprint, seed_everything,
    select_hard_gradient_novel, select_hard_parameter_novel, select_loss_hard,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=20)
    p.add_argument("--output", default="mlp_geometric.csv")
    args = p.parse_args()

    configure_determinism(1)
    batch_size, epochs, k, q, lr = 128, 10, 16, 4, 1e-2
    xtr, ytr, xte, yte = load_digits_split()
    train_seeds = np.arange(13000, 13064, dtype=int)
    train = [torch.tensor(geometric_environment(xtr, int(s))) for s in train_seeds]
    unseen = [torch.tensor(geometric_environment(xte, s)) for s in range(14000, 14080)]
    clean = torch.tensor(xte).flatten(1)
    ty, ey = torch.tensor(ytr), torch.tensor(yte)

    params = np.stack([environment_parameters(int(s)) for s in train_seeds])
    params = (params - params.mean(0)) / (params.std(0) + 1e-8)
    rng7 = np.stack([rng_fingerprint(int(s), 7) for s in train_seeds])
    rng7 = (rng7 - rng7.mean(0)) / (rng7.std(0) + 1e-8)

    methods = ["loss4", "gradnov4", "paramnov4", "rng12_gradnov4"]
    rows = []
    t0 = time.time()
    for rep in range(args.start, args.end):
        base = 21000000 + rep * 2203
        tg = torch.Generator().manual_seed(base + 1)
        er = np.random.default_rng(base + 2)
        schedule = []
        for _ in range(epochs):
            for b in torch.randperm(len(ty), generator=tg).split(batch_size):
                schedule.append((b, er.choice(64, k, replace=False).tolist()))

        for method in methods:
            seed_everything(base)
            model = MLP()
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
            st = time.time()
            env_forwards = 0
            for b, cand16 in schedule:
                if method == "rng12_gradnov4":
                    local = farthest_subset(rng7[np.array(cand16)], 12)
                    cand = [cand16[i] for i in local]
                else:
                    cand = cand16
                m = len(cand)
                env_forwards += m
                xb = torch.cat([train[e][b] for e in cand])
                logits, h = model(xb)
                per = torch.nn.functional.cross_entropy(logits, ty[b].repeat(m), reduction="none").reshape(m, -1)
                loss = per.mean(1)
                gd = head_gradient_directions(logits, h, ty[b], m)
                cosine = gd @ gd.T
                if method == "loss4":
                    idx = select_loss_hard(loss, q)
                elif method in ("gradnov4", "rng12_gradnov4"):
                    idx = select_hard_gradient_novel(loss, cosine, q)
                else:
                    idx = select_hard_parameter_novel(loss, params[np.array(cand)], q)
                opt.zero_grad(set_to_none=True)
                per[idx].mean().backward()
                opt.step()
            metrics = environment_metrics(model, unseen, ey, clean)
            rows.append({"rep": rep, "method": method, "train_seconds": time.time() - st, "env_forwards": env_forwards, **metrics})
        print(f"rep {rep} done {time.time()-t0:.1f}s", flush=True)

    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

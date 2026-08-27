from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

K = 16
Q = 4
BATCH = 64
EPOCHS = 20
LR = 3e-3
GRID = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
METHODS = ["beta0", "beta1.5", "beta3", "relative0.15"]

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def configure_data(regime: str):
    X, y = load_breast_cancer(return_X_y=True)
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.30, random_state=314159, stratify=y
    )
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    Xtr = ((Xtr - mu) / sd).astype(np.float32)
    Xte = ((Xte - mu) / sd).astype(np.float32)

    if regime == "low":
        noise_lo, noise_hi = 0.02, 0.15
        mask_hi = 0.08
        gain_lo, gain_hi = 0.90, 1.10
        shift_lo, shift_hi = -0.08, 0.08
        train_start, test_start = 21000, 22000
    elif regime == "high":
        noise_lo, noise_hi = 0.05, 0.55
        mask_hi = 0.25
        gain_lo, gain_hi = 0.72, 1.28
        shift_lo, shift_hi = -0.25, 0.25
        train_start, test_start = 23000, 24000
    else:
        raise ValueError(regime)

    def env(Xb: np.ndarray, seed: int, offset: int) -> np.ndarray:
        r = np.random.default_rng(seed)
        sigma = r.uniform(noise_lo, noise_hi)
        mask_p = r.uniform(0.0, mask_hi)
        gain = r.uniform(gain_lo, gain_hi)
        shift = r.uniform(shift_lo, shift_hi)
        rr = np.random.default_rng(seed * 2017 + offset)
        noise = rr.normal(0.0, sigma, Xb.shape).astype(np.float32)
        mask = (rr.random(Xb.shape) >= mask_p).astype(np.float32)
        return ((Xb * gain + shift + noise) * mask).astype(np.float32)

    train = [torch.tensor(env(Xtr, s, 11)) for s in range(train_start, train_start + 64)]
    unseen = [torch.tensor(env(Xte, s, 37)) for s in range(test_start, test_start + 80)]
    clean = torch.tensor(Xte)
    return train, unseen, clean, torch.tensor(ytr), torch.tensor(yte)


class Net(torch.nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(in_dim, 64), torch.nn.ReLU())
        self.head = torch.nn.Linear(64, 2)

    def forward(self, x):
        h = self.body(x)
        return self.head(h), h


def zscore(x):
    x = torch.as_tensor(x, dtype=torch.float32)
    return (x - x.mean()) / (x.std(unbiased=False) + 1e-8)


def head_gradient_directions(logits, h, yb):
    b = len(yb)
    zz = logits.detach().reshape(K, b, -1)
    hh = h.detach().reshape(K, b, -1)
    p = torch.softmax(zz, -1)
    oh = torch.nn.functional.one_hot(yb, 2).float()[None]
    residual = (p - oh) / b
    gw = torch.einsum("mbc,mbh->mch", residual, hh).reshape(K, -1)
    gb = residual.sum(1)
    g = torch.cat([gw, gb], 1)
    return g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)


def select(loss, cos, beta):
    selected = [int(torch.argmax(loss))]
    while len(selected) < Q:
        remaining = [i for i in range(K) if i not in selected]
        novelty = zscore((1 - cos[:, selected]).min(1).values)
        score = zscore(loss.detach()) + beta * novelty
        selected.append(max(remaining, key=lambda i: float(score[i])))
    return selected


def pair_cosine(cos, idx):
    a = cos[idx][:, idx]
    tri = torch.triu_indices(len(idx), len(idx), 1)
    return float(a[tri[0], tri[1]].mean())


def relative_select(loss, cos, rho=0.15):
    s0 = select(loss, cos, 0.0)
    s5 = select(loss, cos, 5.0)
    c0 = pair_cosine(cos, s0)
    c5 = pair_cosine(cos, s5)
    target = c5 + rho * (c0 - c5)
    options = []
    for beta in GRID:
        s = select(loss, cos, beta)
        c = pair_cosine(cos, s)
        options.append((abs(c - target), beta, s, c))
    _, beta, s, c = min(options, key=lambda x: (x[0], x[1]))
    return s, beta, c, target, c0, c5


def holm_adjust(ps):
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[idx])
        adj[idx] = min(1.0, running)
    return adj


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["mean_test", "sd_test", "p10_test", "min_test", "clean_test"]
    rows = []
    base = df[df.method == "beta0"].set_index("rep")
    for method in ["beta1.5", "beta3", "relative0.15"]:
        other = df[df.method == method].set_index("rep")
        common = base.index.intersection(other.index)
        local = []
        ps = []
        for metric in metrics:
            d = other.loc[common, metric] - base.loc[common, metric]
            p = float(ttest_rel(other.loc[common, metric], base.loc[common, metric]).pvalue)
            local.append({
                "comparison": f"{method}-beta0",
                "metric": metric,
                "n": len(common),
                "delta": float(d.mean()),
                "p": p,
            })
            ps.append(p)
        for row, adj in zip(local, holm_adjust(ps)):
            row["p_holm_5"] = float(adj)
        rows.extend(local)

    # Direct test of the adaptive controller against the fixed robust operating point.
    a = df[df.method == "relative0.15"].set_index("rep")
    b = df[df.method == "beta1.5"].set_index("rep")
    common = a.index.intersection(b.index)
    local = []
    ps = []
    for metric in metrics:
        d = a.loc[common, metric] - b.loc[common, metric]
        p = float(ttest_rel(a.loc[common, metric], b.loc[common, metric]).pvalue)
        local.append({
            "comparison": "relative0.15-beta1.5",
            "metric": metric,
            "n": len(common),
            "delta": float(d.mean()),
            "p": p,
        })
        ps.append(p)
    for row, adj in zip(local, holm_adjust(ps)):
        row["p_holm_5"] = float(adj)
    rows.extend(local)
    return pd.DataFrame(rows)


def run(regime: str, start: int, end: int, output: str):
    train, unseen, clean, ytr, yte = configure_data(regime)
    in_dim = train[0].shape[1]
    rows = []
    started = time.time()

    for rep in range(start, end):
        base = (31_000_000 if regime == "low" else 41_000_000) + rep * 2029
        gen = torch.Generator().manual_seed(base + 1)
        env_rng = np.random.default_rng(base + 2)
        schedule = []
        for _ in range(EPOCHS):
            for batch_idx in torch.randperm(len(ytr), generator=gen).split(BATCH):
                schedule.append((batch_idx, env_rng.choice(64, K, replace=False).tolist()))

        for method in METHODS:
            torch.manual_seed(base)
            np.random.seed(base % (2**32 - 1))
            random.seed(base)
            model = Net(in_dim)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
            betas, pair_cos, selected_losses, targets, c0s, c5s = [], [], [], [], [], []
            st = time.time()
            model.train()

            for batch_idx, candidates in schedule:
                xb = torch.cat([train[e][batch_idx] for e in candidates])
                logits, h = model(xb)
                per = torch.nn.functional.cross_entropy(
                    logits, ytr[batch_idx].repeat(K), reduction="none"
                ).reshape(K, -1)
                env_loss = per.mean(1)
                gd = head_gradient_directions(logits, h, ytr[batch_idx])
                cos = gd @ gd.T

                if method == "relative0.15":
                    idx, beta, c, target, c0, c5 = relative_select(env_loss, cos, 0.15)
                    targets.append(target)
                    c0s.append(c0)
                    c5s.append(c5)
                else:
                    beta = float(method.replace("beta", ""))
                    idx = select(env_loss, cos, beta)
                    c = pair_cosine(cos, idx)

                betas.append(beta)
                pair_cos.append(c)
                selected_losses.append(float(env_loss[idx].detach().mean()))
                opt.zero_grad(set_to_none=True)
                per[idx].mean().backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                acc = np.asarray([
                    float((model(x)[0].argmax(1) == yte).float().mean()) for x in unseen
                ])
                clean_acc = float((model(clean)[0].argmax(1) == yte).float().mean())

            row = {
                "regime": regime,
                "rep": rep,
                "method": method,
                "mean_test": float(acc.mean()),
                "sd_test": float(acc.std(ddof=1)),
                "p10_test": float(np.quantile(acc, 0.1)),
                "min_test": float(acc.min()),
                "clean_test": clean_acc,
                "train_seconds": time.time() - st,
                "beta_mean": float(np.mean(betas)),
                "selected_pair_cos": float(np.mean(pair_cos)),
                "selected_loss": float(np.mean(selected_losses)),
                "target_cos": float(np.mean(targets)) if targets else np.nan,
                "c0": float(np.mean(c0s)) if c0s else np.nan,
                "c5": float(np.mean(c5s)) if c5s else np.nan,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

        print(regime, "rep", rep, "done", round(time.time() - started, 1), flush=True)

    df = pd.DataFrame(rows)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summarize(df).to_csv(out.with_name(out.stem + "_paired.csv"), index=False)
    means = df.groupby("method")[[
        "mean_test", "sd_test", "p10_test", "min_test", "clean_test",
        "beta_mean", "selected_pair_cos", "selected_loss", "train_seconds"
    ]].mean()
    means.to_csv(out.with_name(out.stem + "_means.csv"))
    print("\nMETHOD MEANS")
    print(means.to_string())
    print("\nPAIRED")
    print(summarize(df).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("regime", choices=["low", "high"])
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=30)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run(a.regime, a.start, a.end, a.output)

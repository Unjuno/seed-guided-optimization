from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import ttest_rel
from torchvision.datasets import FashionMNIST

AUDIT_ENV_INDICES = [1, 9, 17, 25, 33, 41, 49, 57]
RANK_TOLERANCE = 0.01


def configure():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)


def seed_everything(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)


def stratified_subset(targets, per_class: int, seed: int) -> np.ndarray:
    y = np.asarray(targets)
    rng = np.random.default_rng(seed)
    out = []
    for c in range(10):
        idx = np.flatnonzero(y == c)
        out.extend(rng.choice(idx, size=per_class, replace=False).tolist())
    out = np.asarray(out, dtype=np.int64)
    rng.shuffle(out)
    return out


def load_fashion(root: str, train_per_class: int, test_per_class: int):
    tr = FashionMNIST(root, train=True, download=True)
    te = FashionMNIST(root, train=False, download=True)
    tri = stratified_subset(tr.targets, train_per_class, 314159)
    tei = stratified_subset(te.targets, test_per_class, 271828)
    xtr = tr.data[tri].unsqueeze(1).float() / 255.0
    ytr = tr.targets[tri].long()
    xte = te.data[tei].unsqueeze(1).float() / 255.0
    yte = te.targets[tei].long()
    mean = float(xtr.mean())
    std = float(xtr.std(unbiased=False)) + 1e-6
    return xtr, ytr, xte, yte, mean, std


def env_parameters(seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    return np.asarray(
        [
            r.uniform(-20.0, 20.0),
            r.uniform(-3.0, 3.0),
            r.uniform(-3.0, 3.0),
            r.uniform(0.75, 1.25),
            r.uniform(-0.12, 0.12),
            r.uniform(0.0, 0.08),
        ],
        dtype=np.float32,
    )


def apply_environment(x: torch.Tensor, seed: int, step: int) -> torch.Tensor:
    angle, dx, dy, contrast, bright, sigma = map(float, env_parameters(seed))
    rad = math.radians(angle)
    c, s = math.cos(rad), math.sin(rad)
    theta = torch.tensor(
        [[c, -s, 2.0 * dx / 28.0], [s, c, 2.0 * dy / 28.0]],
        dtype=x.dtype,
    ).unsqueeze(0).expand(len(x), -1, -1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    z = F.grid_sample(
        x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    z = (z - 0.5) * contrast + 0.5 + bright
    if sigma > 0:
        g = torch.Generator(device="cpu").manual_seed(
            (seed * 1_000_003 + step * 97 + 31) % (2**63 - 1)
        )
        z = z + sigma * torch.randn(z.shape, generator=g, dtype=z.dtype)
    return z.clamp(0, 1)


def normalize(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (x - mean) / std


class TinyPatchTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        dim = 48
        self.patch = torch.nn.Conv2d(1, dim, kernel_size=7, stride=7)
        self.cls = torch.nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = torch.nn.Parameter(torch.zeros(1, 17, dim))
        layer = torch.nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            dim_feedforward=96,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=2)
        self.norm = torch.nn.LayerNorm(dim)
        self.head = torch.nn.Linear(dim, 10)
        torch.nn.init.normal_(self.cls, std=0.02)
        torch.nn.init.normal_(self.pos, std=0.02)

    def forward(self, x):
        p = self.patch(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(len(x), -1, -1)
        t = torch.cat([cls, p], dim=1) + self.pos
        t = self.encoder(t)
        h = self.norm(t[:, 0])
        return self.head(h), h


def zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / (x.std(unbiased=False) + 1e-8)


def head_gradient_directions(logits, features, y, k):
    b = len(y)
    zz = logits.detach().reshape(k, b, -1)
    hh = features.detach().reshape(k, b, -1)
    probs = torch.softmax(zz, -1)
    residual = (probs - F.one_hot(y, 10).float()[None]) / b
    gw = torch.einsum("kbc,kbh->kch", residual, hh).reshape(k, -1)
    gb = residual.sum(1)
    g = torch.cat([gw, gb], 1)
    return g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)


def select_loss(loss, q):
    return torch.topk(loss, q).indices.tolist()


def select_gradnov(loss, cosine, q, weight=0.6):
    selected = [int(torch.argmax(loss))]
    while len(selected) < q:
        rem = [i for i in range(len(loss)) if i not in selected]
        novelty = zscore((1 - cosine[:, selected]).min(1).values)
        score = zscore(loss.detach()) + weight * novelty
        selected.append(max(rem, key=lambda i: float(score[i])))
    return selected


def build_schedule(n, epochs, batch, env_pool, k, base):
    tg = torch.Generator().manual_seed(base + 1)
    er = np.random.default_rng(base + 2)
    out = []
    step = 0
    for _ in range(epochs):
        for idx in torch.randperm(n, generator=tg).split(batch):
            out.append((step, idx, er.choice(env_pool, k, replace=False).tolist()))
            step += 1
    return out


def cache_schedule(xtr, schedule, train_env):
    out = []
    for step, idx, cand in schedule:
        seeds = [train_env[e] for e in cand]
        xb = torch.cat([apply_environment(xtr[idx], seed, step) for seed in seeds], 0)
        out.append((step, idx, cand, (xb * 255).round().to(torch.uint8)))
    return out


def pretrain(base, xtr, ytr, mean, std, epochs, batch, lr):
    seed_everything(base)
    model = TinyPatchTransformer()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    tg = torch.Generator().manual_seed(base + 333)
    model.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(ytr), generator=tg).split(batch):
            logits, _ = model(normalize(xtr[idx], mean, std))
            loss = F.cross_entropy(logits, ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def effective_rank_features(h: np.ndarray) -> float:
    h = np.asarray(h, dtype=np.float64)
    h = h - h.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(h, compute_uv=False)
    values = singular * singular
    total = float(values.sum())
    if total <= 1e-15:
        return 1.0
    p = values[values > 1e-15] / total
    return float(np.exp(-(p * np.log(p)).sum()))


def representation_audit(model, xtr, train_env, probe_idx, mean, std) -> float:
    model.eval()
    features = []
    with torch.no_grad():
        for j, env_idx in enumerate(AUDIT_ENV_INDICES):
            z = apply_environment(xtr[probe_idx], train_env[env_idx], 880_000 + j)
            _, h = model(normalize(z, mean, std))
            features.append(h.cpu().numpy())
    return effective_rank_features(np.concatenate(features, axis=0))


def cache_test(xte, test_env):
    return [
        (apply_environment(xte, seed, 990_000 + i) * 255).round().to(torch.uint8)
        for i, seed in enumerate(test_env)
    ]


def evaluate_cached(model, test_cache, yte, xte, mean, std, batch=256):
    model.eval()
    acc = []
    with torch.no_grad():
        for u8 in test_cache:
            correct = 0
            for idx in torch.arange(len(yte)).split(batch):
                x = u8[idx].float() / 255.0
                pred = model(normalize(x, mean, std))[0].argmax(1)
                correct += int((pred == yte[idx]).sum())
            acc.append(correct / len(yte))
        clean = (
            sum(
                int((model(normalize(xte[idx], mean, std))[0].argmax(1) == yte[idx]).sum())
                for idx in torch.arange(len(yte)).split(batch)
            )
            / len(yte)
        )
    a = np.asarray(acc)
    return {
        "mean_test": float(a.mean()),
        "sd_test": float(a.std(ddof=1)),
        "p10_test": float(np.quantile(a, 0.1)),
        "min_test": float(a.min()),
        "clean_test": float(clean),
    }


def clone_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def frozen_direction(delta_rank: float) -> str:
    if abs(delta_rank) < RANK_TOLERANCE:
        return "UNCERTAIN"
    return "POSITIVE" if delta_rank > 0 else "NONPOSITIVE"


def paired_summary(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df.method == "loss4"].set_index("rep")
    novel = df[df.method == "gradnov4"].set_index("rep")
    common = base.index.intersection(novel.index)
    rows = []
    for metric in ["rep_eff_rank", "mean_test", "p10_test", "min_test", "clean_test"]:
        delta = novel.loc[common, metric] - base.loc[common, metric]
        p = float(ttest_rel(novel.loc[common, metric], base.loc[common, metric]).pvalue) if len(common) >= 3 else float("nan")
        rows.append({"metric": metric, "n": len(common), "delta": float(delta.mean()), "p_unadjusted": p})
    return pd.DataFrame(rows)


def run(a):
    configure()
    xtr, ytr, xte, yte, mean, std = load_fashion(a.data_root, a.train_per_class, a.test_per_class)
    train_env = [50_000 + i for i in range(64)]
    test_env = [60_000 + i for i in range(a.test_envs)]
    rows = []
    started = time.time()

    for rep in range(a.start, a.end):
        base = 121_000_000 + rep * 8191
        state = pretrain(base, xtr, ytr, mean, std, a.pretrain_epochs, a.batch_size, a.pretrain_lr)
        schedule = build_schedule(len(ytr), a.finetune_epochs, a.batch_size, 64, a.k, base + 1000)
        cached = cache_schedule(xtr, schedule, train_env)
        probe_rng = np.random.default_rng(base + 7001)
        probe_idx = torch.tensor(probe_rng.choice(len(ytr), size=min(a.probe_size, len(ytr)), replace=False), dtype=torch.long)

        pre_model = TinyPatchTransformer()
        pre_model.load_state_dict(state)
        pre_rank = representation_audit(pre_model, xtr, train_env, probe_idx, mean, std)

        sealed = {}
        for method in ("loss4", "gradnov4"):
            seed_everything(base)
            model = TinyPatchTransformer()
            model.load_state_dict(state)
            opt = torch.optim.AdamW(model.parameters(), lr=a.finetune_lr, weight_decay=1e-4)
            st = time.time()
            model.train()
            for step, idx, cand, u8 in cached:
                xb = normalize(u8.float() / 255.0, mean, std)
                logits, h = model(xb)
                per = F.cross_entropy(logits, ytr[idx].repeat(a.k), reduction="none").reshape(a.k, -1)
                loss = per.mean(1)
                if method == "loss4":
                    chosen = select_loss(loss, a.q)
                else:
                    directions = head_gradient_directions(logits, h, ytr[idx], a.k)
                    chosen = select_gradnov(loss, directions @ directions.T, a.q, weight=0.6)
                opt.zero_grad(set_to_none=True)
                per[chosen].mean().backward()
                opt.step()

            rep_rank = representation_audit(model, xtr, train_env, probe_idx, mean, std)
            training_only = {
                "rep": rep,
                "method": method,
                "pre_rep_eff_rank": pre_rank,
                "rep_eff_rank": rep_rank,
                "rep_eff_rank_from_pretrain": rep_rank - pre_rank,
                "train_seconds": time.time() - st,
            }
            sealed[method] = {"training_only": training_only, "state": clone_state(model)}
            print("TRAINING_ONLY " + json.dumps(training_only), flush=True)

        delta_rank = sealed["gradnov4"]["training_only"]["rep_eff_rank"] - sealed["loss4"]["training_only"]["rep_eff_rank"]
        prediction = {
            "rep": rep,
            "delta_rep_eff_rank": delta_rank,
            "rank_tolerance": RANK_TOLERANCE,
            "predicted_mean_direction": frozen_direction(delta_rank),
        }
        print("SEALED_PREDICTION " + json.dumps(prediction), flush=True)

        test_cache = cache_test(xte, test_env)
        for method in ("loss4", "gradnov4"):
            model = TinyPatchTransformer()
            model.load_state_dict(sealed[method]["state"])
            heldout = evaluate_cached(model, test_cache, yte, xte, mean, std)
            row = {
                **sealed[method]["training_only"],
                "predicted_mean_direction": prediction["predicted_mean_direction"],
                "rep_delta_rep_eff_rank": delta_rank,
                "pretrain_epochs": a.pretrain_epochs,
                "finetune_epochs": a.finetune_epochs,
                "n_train": len(ytr),
                "n_test": len(yte),
                "candidate_k": a.k,
                "backward_q": a.q,
                "probe_size": len(probe_idx),
                **heldout,
            }
            rows.append(row)
            print("FINAL " + json.dumps(row), flush=True)

        out = Path(a.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(out, index=False)
        paired_summary(frame).to_csv(out.with_name(out.stem + "_paired.csv"), index=False)
        print(f"rep {rep} completed; elapsed={time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=".cache/fashionmnist")
    p.add_argument("--output", required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--pretrain-epochs", type=int, default=5)
    p.add_argument("--finetune-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-per-class", type=int, default=300)
    p.add_argument("--test-per-class", type=int, default=100)
    p.add_argument("--test-envs", type=int, default=24)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--q", type=int, default=4)
    p.add_argument("--probe-size", type=int, default=256)
    p.add_argument("--pretrain-lr", type=float, default=1e-3)
    p.add_argument("--finetune-lr", type=float, default=8e-4)
    run(p.parse_args())

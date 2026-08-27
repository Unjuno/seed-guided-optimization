from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from fashion_transformer_rep_rank_audit import (
    AUDIT_ENV_INDICES,
    RANK_TOLERANCE,
    apply_environment,
    build_schedule,
    cache_schedule,
    cache_test,
    clone_state,
    configure,
    effective_rank_features,
    evaluate_cached,
    frozen_direction,
    head_gradient_directions,
    load_fashion,
    normalize,
    paired_summary,
    seed_everything,
    select_gradnov,
    select_loss,
)


class MediumPatchTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        dim = 96
        self.patch = torch.nn.Conv2d(1, dim, kernel_size=7, stride=7)
        self.cls = torch.nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = torch.nn.Parameter(torch.zeros(1, 17, dim))
        layer = torch.nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=8,
            dim_feedforward=192,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=4)
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


def pretrain(base, xtr, ytr, mean, std, epochs, batch, lr):
    seed_everything(base)
    model = MediumPatchTransformer()
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


def representation_audit(model, xtr, train_env, probe_idx, mean, std):
    model.eval()
    features = []
    with torch.no_grad():
        for j, env_idx in enumerate(AUDIT_ENV_INDICES):
            z = apply_environment(xtr[probe_idx], train_env[env_idx], 980_000 + j)
            _, h = model(normalize(z, mean, std))
            features.append(h.cpu().numpy())
    return effective_rank_features(np.concatenate(features, axis=0))


def run(a):
    configure()
    xtr, ytr, xte, yte, mean, std = load_fashion(
        a.data_root, a.train_per_class, a.test_per_class
    )
    train_env = [50_000 + i for i in range(64)]
    test_env = [60_000 + i for i in range(a.test_envs)]
    rows = []
    started = time.time()

    for rep in range(a.start, a.end):
        base = 141_000_000 + rep * 8191
        state = pretrain(
            base, xtr, ytr, mean, std,
            a.pretrain_epochs, a.batch_size, a.pretrain_lr
        )
        schedule = build_schedule(
            len(ytr), a.finetune_epochs, a.batch_size, 64, a.k, base + 1000
        )
        cached = cache_schedule(xtr, schedule, train_env)
        probe_rng = np.random.default_rng(base + 7001)
        probe_idx = torch.tensor(
            probe_rng.choice(
                len(ytr), size=min(a.probe_size, len(ytr)), replace=False
            ),
            dtype=torch.long,
        )

        pre_model = MediumPatchTransformer()
        pre_model.load_state_dict(state)
        pre_rank = representation_audit(
            pre_model, xtr, train_env, probe_idx, mean, std
        )

        sealed = {}
        for method in ("loss4", "gradnov4"):
            seed_everything(base)
            model = MediumPatchTransformer()
            model.load_state_dict(state)
            opt = torch.optim.AdamW(
                model.parameters(), lr=a.finetune_lr, weight_decay=1e-4
            )
            st = time.time()
            model.train()
            for step, idx, cand, u8 in cached:
                xb = normalize(u8.float() / 255.0, mean, std)
                logits, h = model(xb)
                per = F.cross_entropy(
                    logits, ytr[idx].repeat(a.k), reduction="none"
                ).reshape(a.k, -1)
                loss = per.mean(1)
                if method == "loss4":
                    chosen = select_loss(loss, a.q)
                else:
                    directions = head_gradient_directions(
                        logits, h, ytr[idx], a.k
                    )
                    chosen = select_gradnov(
                        loss, directions @ directions.T, a.q, weight=0.6
                    )
                opt.zero_grad(set_to_none=True)
                per[chosen].mean().backward()
                opt.step()

            rep_rank = representation_audit(
                model, xtr, train_env, probe_idx, mean, std
            )
            training_only = {
                "rep": rep,
                "method": method,
                "pre_rep_eff_rank": pre_rank,
                "rep_eff_rank": rep_rank,
                "rep_eff_rank_from_pretrain": rep_rank - pre_rank,
                "train_seconds": time.time() - st,
            }
            sealed[method] = {
                "training_only": training_only,
                "state": clone_state(model),
            }
            print("TRAINING_ONLY " + json.dumps(training_only), flush=True)

        delta_rank = (
            sealed["gradnov4"]["training_only"]["rep_eff_rank"]
            - sealed["loss4"]["training_only"]["rep_eff_rank"]
        )
        prediction = {
            "rep": rep,
            "delta_rep_eff_rank": delta_rank,
            "rank_tolerance": RANK_TOLERANCE,
            "predicted_mean_direction": frozen_direction(delta_rank),
        }
        print("SEALED_PREDICTION " + json.dumps(prediction), flush=True)

        test_cache = cache_test(xte, test_env)
        for method in ("loss4", "gradnov4"):
            model = MediumPatchTransformer()
            model.load_state_dict(sealed[method]["state"])
            heldout = evaluate_cached(
                model, test_cache, yte, xte, mean, std
            )
            row = {
                **sealed[method]["training_only"],
                "predicted_mean_direction":
                    prediction["predicted_mean_direction"],
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
        paired_summary(frame).to_csv(
            out.with_name(out.stem + "_paired.csv"), index=False
        )
        print(
            f"rep {rep} completed; elapsed={time.time() - started:.1f}s",
            flush=True,
        )


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

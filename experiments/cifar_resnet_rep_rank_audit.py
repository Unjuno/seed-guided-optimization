from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import ttest_rel

from cifar_resnet_pilot import (
    CIFARResNet20,
    apply_environment,
    build_schedule,
    configure,
    head_gradient_directions,
    load_cifar,
    normalize,
    seed_everything,
    select_gradnov,
    select_loss,
)
from cifar_resnet_finetune_pilot import (
    cache_schedule,
    cache_test,
    evaluate_cached,
    pretrain,
)

AUDIT_ENV_INDICES = [1, 9, 17, 25, 33, 41, 49, 57]
RANK_TOLERANCE = 0.01


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


def representation_audit(model, xtr, train_env, probe_idx) -> float:
    """Training-pool-only hidden representation effective rank."""
    model.eval()
    features = []
    with torch.no_grad():
        for j, env_idx in enumerate(AUDIT_ENV_INDICES):
            transformed = apply_environment(
                xtr[probe_idx], train_env[env_idx], 770_000 + j
            )
            _, h = model(normalize(transformed))
            features.append(h.cpu().numpy())
    return effective_rank_features(np.concatenate(features, axis=0))


def paired_summary(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df.method == "loss4"].set_index("rep")
    novel = df[df.method == "gradnov4"].set_index("rep")
    common = base.index.intersection(novel.index)
    rows = []
    for metric in ["rep_eff_rank", "mean_test", "p10_test", "min_test", "clean_test"]:
        delta = novel.loc[common, metric] - base.loc[common, metric]
        p = (
            float(ttest_rel(novel.loc[common, metric], base.loc[common, metric]).pvalue)
            if len(common) >= 3
            else float("nan")
        )
        rows.append(
            {
                "metric": metric,
                "n": len(common),
                "delta": float(delta.mean()),
                "p_unadjusted": p,
            }
        )
    return pd.DataFrame(rows)


def frozen_direction(delta_rank: float) -> str:
    if abs(delta_rank) < RANK_TOLERANCE:
        return "UNCERTAIN"
    return "POSITIVE" if delta_rank > 0 else "NONPOSITIVE"


def clone_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def run(a):
    configure()
    xtr, ytr, xte, yte = load_cifar(
        a.data_root, a.train_per_class, a.test_per_class
    )
    train_env = [20_000 + i for i in range(64)]
    test_env = [40_000 + i for i in range(a.test_envs)]

    rows = []
    started = time.time()
    for rep in range(a.start, a.end):
        base = 91_000_000 + rep * 7919
        state = pretrain(
            base,
            xtr,
            ytr,
            a.pretrain_epochs,
            a.batch_size,
            a.pretrain_lr,
        )
        schedule = build_schedule(
            len(ytr),
            a.finetune_epochs,
            a.batch_size,
            64,
            a.k,
            base + 1000,
        )
        cached = cache_schedule(xtr, schedule, train_env)
        probe_rng = np.random.default_rng(base + 7001)
        probe_idx = torch.tensor(
            probe_rng.choice(
                len(ytr), size=min(a.probe_size, len(ytr)), replace=False
            ),
            dtype=torch.long,
        )

        pre_model = CIFARResNet20()
        pre_model.load_state_dict(state)
        pre_rank = representation_audit(pre_model, xtr, train_env, probe_idx)

        sealed = {}
        for method in ("loss4", "gradnov4"):
            seed_everything(base)
            model = CIFARResNet20()
            model.load_state_dict(state)
            opt = torch.optim.AdamW(
                model.parameters(), lr=a.finetune_lr, weight_decay=1e-4
            )
            st = time.time()
            model.train()
            for step, idx, cand, u8 in cached:
                xb = normalize(u8.float() / 255)
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
                    chosen = select_gradnov(loss, directions @ directions.T, a.q)
                opt.zero_grad(set_to_none=True)
                per[chosen].mean().backward()
                opt.step()

            rep_rank = representation_audit(model, xtr, train_env, probe_idx)
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

        # Seal and emit the preregistered predictor before any held-out metric is
        # computed for either method in this replicate.
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

        # Held-out construction and evaluation occur only after both training-only
        # model states and the predictor have been sealed.
        test_cache = cache_test(xte, test_env)
        for method in ("loss4", "gradnov4"):
            model = CIFARResNet20()
            model.load_state_dict(sealed[method]["state"])
            heldout = evaluate_cached(model, test_cache, yte, xte)
            training_only = sealed[method]["training_only"]
            row = {
                **training_only,
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
        paired_summary(frame).to_csv(
            out.with_name(out.stem + "_paired.csv"), index=False
        )
        print(
            f"rep {rep} completed; elapsed={time.time() - started:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=".cache/cifar10")
    p.add_argument("--output", required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--pretrain-epochs", type=int, default=10)
    p.add_argument("--finetune-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-per-class", type=int, default=600)
    p.add_argument("--test-per-class", type=int, default=300)
    p.add_argument("--test-envs", type=int, default=32)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--q", type=int, default=4)
    p.add_argument("--probe-size", type=int, default=256)
    p.add_argument("--pretrain-lr", type=float, default=3e-3)
    p.add_argument("--finetune-lr", type=float, default=1e-3)
    run(p.parse_args())

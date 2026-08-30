from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (
    MLP,
    configure_determinism,
    geometric_environment,
    head_gradient_directions,
    load_digits_split,
    seed_everything,
    select_hard_gradient_novel,
    select_loss_hard,
)

K = 16
Q = 4
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-2
WEIGHT_DECAY = 1e-3
NOVELTY_WEIGHT = 0.6
TRAIN_SEEDS = np.arange(13000, 13064, dtype=int)
HELDOUT_SEEDS = np.arange(16000, 16080, dtype=int)
AUDIT_ENV_INDICES = (1, 9, 17, 25, 33, 41, 49, 57)
PROBE_SIZE = 192
INTERVENTIONS = ("native", "spread", "concentrate")


def effective_rank_features(features: np.ndarray, standardize: bool = False) -> float:
    h = np.asarray(features, dtype=np.float64)
    h = h - h.mean(axis=0, keepdims=True)
    if standardize:
        sd = h.std(axis=0, ddof=0)
        nz = sd > 1e-12
        if not np.any(nz):
            return 1.0
        h = h[:, nz] / sd[nz][None, :]
    s = np.linalg.svd(h, compute_uv=False)
    values = s * s
    total = float(values.sum())
    if total <= 1e-15:
        return 1.0
    p = values[values > 1e-15] / total
    return float(np.exp(-(p * np.log(p)).sum()))


def probe_features(
    model: torch.nn.Module,
    train_environments: list[torch.Tensor],
    probe_idx: torch.Tensor,
) -> np.ndarray:
    model.eval()
    features = []
    with torch.no_grad():
        for env_idx in AUDIT_ENV_INDICES:
            _, h = model(train_environments[env_idx][probe_idx])
            features.append(h.cpu().numpy())
    return np.concatenate(features, axis=0)


def spread_scales(features: np.ndarray) -> np.ndarray:
    sigma = np.asarray(features, dtype=np.float64).std(axis=0, ddof=0)
    positive = sigma > 1e-12
    if not np.any(positive):
        return np.ones(features.shape[1], dtype=np.float32)
    g = float(np.exp(np.log(sigma[positive]).mean()))
    ratio = g / np.maximum(sigma, 1e-12)
    exponents = np.rint(np.log2(ratio))
    exponents = np.clip(exponents, -3, 3).astype(np.int64)
    return np.power(2.0, exponents).astype(np.float32)


def concentrate_scales(features: np.ndarray) -> np.ndarray:
    sigma = np.asarray(features, dtype=np.float64).std(axis=0, ddof=0)
    channels = np.arange(len(sigma))
    order = np.lexsort((channels, -sigma))
    scales = np.full(len(sigma), 2.0 ** -3, dtype=np.float32)
    scales[order[:16]] = 2.0 ** 3
    return scales


def reparameterize(model: MLP, scales: np.ndarray) -> MLP:
    out = MLP()
    out.load_state_dict({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    d = torch.as_tensor(scales, dtype=out.body[0].weight.dtype)
    with torch.no_grad():
        out.body[0].weight.mul_(d[:, None])
        out.body[0].bias.mul_(d)
        out.head.weight.div_(d[None, :])
    return out


def build_schedule(n_train: int, base: int) -> list[tuple[torch.Tensor, list[int]]]:
    tg = torch.Generator().manual_seed(base + 1)
    env_rng = np.random.default_rng(base + 2)
    schedule: list[tuple[torch.Tensor, list[int]]] = []
    for _ in range(EPOCHS):
        for batch_idx in torch.randperm(n_train, generator=tg).split(BATCH_SIZE):
            candidates = env_rng.choice(len(TRAIN_SEEDS), K, replace=False).tolist()
            schedule.append((batch_idx, candidates))
    return schedule


def train_model(
    method: str,
    base: int,
    schedule: list[tuple[torch.Tensor, list[int]]],
    train_environments: list[torch.Tensor],
    labels: torch.Tensor,
) -> MLP:
    seed_everything(base)
    model = MLP()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model.train()
    for batch_idx, candidates in schedule:
        xb = torch.cat([train_environments[e][batch_idx] for e in candidates])
        logits, h = model(xb)
        per = torch.nn.functional.cross_entropy(
            logits,
            labels[batch_idx].repeat(K),
            reduction="none",
        ).reshape(K, -1)
        losses = per.mean(1)
        if method == "loss_hard":
            selected = select_loss_hard(losses, Q)
        else:
            directions = head_gradient_directions(logits, h, labels[batch_idx], K)
            selected = select_hard_gradient_novel(
                losses,
                directions @ directions.T,
                q=Q,
                novelty_weight=NOVELTY_WEIGHT,
            )
        selected = sorted(selected)
        opt.zero_grad(set_to_none=True)
        per[selected].mean().backward()
        opt.step()
    return model


def train_shard(start: int, end: int, output_dir: Path) -> None:
    configure_determinism(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    states_dir = output_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)

    xtr, ytr, _, _ = load_digits_split()
    labels = torch.tensor(ytr)
    train_environments = [
        torch.tensor(geometric_environment(xtr, int(seed))) for seed in TRAIN_SEEDS
    ]

    rows: list[dict] = []
    started = time.time()

    for rep in range(start, end):
        base = 410_000_000 + rep * 6151
        schedule = build_schedule(len(labels), base)
        probe_rng = np.random.default_rng(base + 3)
        probe_idx = torch.tensor(
            probe_rng.choice(len(labels), min(PROBE_SIZE, len(labels)), replace=False),
            dtype=torch.long,
        )

        for method in ("loss_hard", "gradnov"):
            st = time.time()
            native = train_model(method, base, schedule, train_environments, labels)
            native_features = probe_features(native, train_environments, probe_idx)
            spread_d = spread_scales(native_features)
            concentrate_d = concentrate_scales(native_features)
            models = {
                "native": native,
                "spread": reparameterize(native, spread_d),
                "concentrate": reparameterize(native, concentrate_d),
            }
            scales = {
                "native": np.ones(native_features.shape[1], dtype=np.float32),
                "spread": spread_d,
                "concentrate": concentrate_d,
            }

            for intervention in INTERVENTIONS:
                model = models[intervention]
                features = probe_features(model, train_environments, probe_idx)
                d = scales[intervention]
                row = {
                    "rep": rep,
                    "method": method,
                    "intervention": intervention,
                    "rep_eff_rank": effective_rank_features(features),
                    "std_rep_eff_rank": effective_rank_features(features, standardize=True),
                    "scale_min": float(d.min()),
                    "scale_max": float(d.max()),
                    "scale_log2_mean_abs": float(np.mean(np.abs(np.log2(d)))),
                    "probe_size": len(probe_idx),
                    "train_seconds_native": time.time() - st,
                }
                rows.append(row)
                torch.save(
                    {
                        "state_dict": {
                            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                        },
                        "rep": rep,
                        "method": method,
                        "intervention": intervention,
                    },
                    states_dir / f"rep{rep:03d}_{method}_{intervention}.pt",
                )
                print("TRAINING_ONLY " + json.dumps(row), flush=True)

        pd.DataFrame(rows).to_csv(
            output_dir / f"rank_intervention_diagnostics_{start}_{end}.csv", index=False
        )
        print(
            f"training rep {rep} completed; elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    pd.DataFrame(rows).to_csv(
        output_dir / f"rank_intervention_diagnostics_{start}_{end}.csv", index=False
    )
    print(
        json.dumps(
            {
                "event": "TRAINING_ONLY_COMPLETE",
                "start": start,
                "end": end,
                "heldout_constructed": False,
            }
        ),
        flush=True,
    )


def outputs_and_metrics(
    model: MLP,
    heldout: list[torch.Tensor],
    labels: torch.Tensor,
    clean: torch.Tensor,
) -> tuple[list[torch.Tensor], dict[str, float]]:
    model.eval()
    logits_all: list[torch.Tensor] = []
    acc: list[float] = []
    with torch.no_grad():
        for x in heldout:
            logits, _ = model(x)
            logits = logits.cpu()
            logits_all.append(logits)
            acc.append(float((logits.argmax(1) == labels).float().mean()))
        clean_logits, _ = model(clean)
        clean_logits = clean_logits.cpu()
        logits_all.append(clean_logits)
        clean_acc = float((clean_logits.argmax(1) == labels).float().mean())
    a = np.asarray(acc, dtype=np.float64)
    metrics = {
        "mean_test": float(a.mean()),
        "sd_test": float(a.std(ddof=1)),
        "p10_test": float(np.quantile(a, 0.1)),
        "min_test": float(a.min()),
        "clean_test": clean_acc,
    }
    return logits_all, metrics


def evaluate_shard(start: int, end: int, output_dir: Path) -> None:
    configure_determinism(1)
    _, _, xte, yte = load_digits_split()
    labels = torch.tensor(yte)

    # Fresh held-out environments are intentionally constructed only after all
    # training-only interventions and states have been sealed.
    heldout = [
        torch.tensor(geometric_environment(xte, int(seed))) for seed in HELDOUT_SEEDS
    ]
    clean = torch.tensor(xte).flatten(1)

    rows: list[dict] = []
    states_dir = output_dir / "states"
    for rep in range(start, end):
        for method in ("loss_hard", "gradnov"):
            models: dict[str, MLP] = {}
            for intervention in INTERVENTIONS:
                checkpoint = torch.load(
                    states_dir / f"rep{rep:03d}_{method}_{intervention}.pt",
                    map_location="cpu",
                )
                model = MLP()
                model.load_state_dict(checkpoint["state_dict"])
                models[intervention] = model

            native_logits, native_metrics = outputs_and_metrics(
                models["native"], heldout, labels, clean
            )
            for intervention in INTERVENTIONS:
                if intervention == "native":
                    logits = native_logits
                    metrics = native_metrics
                    pred_identity = True
                    max_logit_diff = 0.0
                else:
                    logits, metrics = outputs_and_metrics(
                        models[intervention], heldout, labels, clean
                    )
                    pred_identity = True
                    max_logit_diff = 0.0
                    for z0, z1 in zip(native_logits, logits):
                        pred_identity = pred_identity and bool(
                            torch.equal(z0.argmax(1), z1.argmax(1))
                        )
                        max_logit_diff = max(
                            max_logit_diff,
                            float((z0 - z1).abs().max()),
                        )

                row = {
                    "rep": rep,
                    "method": method,
                    "intervention": intervention,
                    "predictions_identical_to_native": pred_identity,
                    "max_abs_logit_diff": max_logit_diff,
                    **metrics,
                }
                rows.append(row)
                print("HELDOUT " + json.dumps(row), flush=True)

    pd.DataFrame(rows).to_csv(
        output_dir / f"rank_intervention_heldout_{start}_{end}.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if args.mode == "train":
        train_shard(args.start, args.end, output_dir)
    else:
        evaluate_shard(args.start, args.end, output_dir)


if __name__ == "__main__":
    main()

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
    environment_metrics,
    geometric_environment,
    head_gradient_directions,
    load_digits_split,
    seed_everything,
    select_hard_gradient_novel,
    select_loss_hard,
)


K = 16
Q_VALUES = (2, 4, 8, 12, 16)
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-2
WEIGHT_DECAY = 1e-3
NOVELTY_WEIGHT = 0.6
TRAIN_SEEDS = np.arange(13000, 13064, dtype=int)
HELDOUT_SEEDS = np.arange(15000, 15080, dtype=int)
AUDIT_ENV_INDICES = (1, 9, 17, 25, 33, 41, 49, 57)
PROBE_SIZE = 192


def effective_rank_features(features: np.ndarray) -> float:
    h = np.asarray(features, dtype=np.float64)
    h = h - h.mean(axis=0, keepdims=True)
    s = np.linalg.svd(h, compute_uv=False)
    values = s * s
    total = float(values.sum())
    if total <= 1e-15:
        return 1.0
    p = values[values > 1e-15] / total
    return float(np.exp(-(p * np.log(p)).sum()))


def representation_rank(
    model: torch.nn.Module,
    train_environments: list[torch.Tensor],
    probe_idx: torch.Tensor,
) -> float:
    model.eval()
    features = []
    with torch.no_grad():
        for env_idx in AUDIT_ENV_INDICES:
            _, h = model(train_environments[env_idx][probe_idx])
            features.append(h.cpu().numpy())
    return effective_rank_features(np.concatenate(features, axis=0))


def selected_pairwise_novelty(cosine: torch.Tensor, selected: list[int]) -> float:
    idx = torch.tensor(selected, dtype=torch.long)
    sub = cosine.index_select(0, idx).index_select(1, idx)
    upper = torch.triu_indices(len(selected), len(selected), offset=1)
    if upper.shape[1] == 0:
        return 0.0
    vals = 1.0 - sub[upper[0], upper[1]]
    return float(vals.mean())


def build_schedule(n_train: int, base: int) -> list[tuple[torch.Tensor, list[int]]]:
    tg = torch.Generator().manual_seed(base + 1)
    env_rng = np.random.default_rng(base + 2)
    schedule: list[tuple[torch.Tensor, list[int]]] = []
    for _ in range(EPOCHS):
        for batch_idx in torch.randperm(n_train, generator=tg).split(BATCH_SIZE):
            candidates = env_rng.choice(len(TRAIN_SEEDS), K, replace=False).tolist()
            schedule.append((batch_idx, candidates))
    return schedule


def train_shard(start: int, end: int, output_dir: Path) -> None:
    configure_determinism(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    states_dir = output_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)

    xtr, ytr, _, _ = load_digits_split()
    ty = torch.tensor(ytr)
    train_environments = [
        torch.tensor(geometric_environment(xtr, int(seed))) for seed in TRAIN_SEEDS
    ]

    rows: list[dict] = []
    started = time.time()

    for rep in range(start, end):
        base = 310_000_000 + rep * 4099
        schedule = build_schedule(len(ty), base)
        probe_rng = np.random.default_rng(base + 3)
        probe_idx = torch.tensor(
            probe_rng.choice(len(ty), min(PROBE_SIZE, len(ty)), replace=False),
            dtype=torch.long,
        )

        for q in Q_VALUES:
            for method in ("loss_hard", "gradnov"):
                seed_everything(base)
                model = MLP()
                opt = torch.optim.AdamW(
                    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
                )
                step_novelty: list[float] = []
                step_selected_loss: list[float] = []
                step_candidate_loss: list[float] = []
                st = time.time()

                model.train()
                for batch_idx, candidates in schedule:
                    xb = torch.cat([train_environments[e][batch_idx] for e in candidates])
                    logits, h = model(xb)
                    per = torch.nn.functional.cross_entropy(
                        logits,
                        ty[batch_idx].repeat(K),
                        reduction="none",
                    ).reshape(K, -1)
                    losses = per.mean(1)
                    directions = head_gradient_directions(logits, h, ty[batch_idx], K)
                    cosine = directions @ directions.T

                    if method == "loss_hard":
                        selected = select_loss_hard(losses, q)
                    else:
                        selected = select_hard_gradient_novel(
                            losses,
                            cosine,
                            q=q,
                            novelty_weight=NOVELTY_WEIGHT,
                        )

                    # Frozen Q=K identity control: both methods must backpropagate
                    # through the same complete candidate set in the same order.
                    selected = sorted(selected)

                    step_novelty.append(selected_pairwise_novelty(cosine, selected))
                    step_selected_loss.append(float(losses[selected].mean().detach()))
                    step_candidate_loss.append(float(losses.mean().detach()))

                    opt.zero_grad(set_to_none=True)
                    per[selected].mean().backward()
                    opt.step()

                rep_rank = representation_rank(model, train_environments, probe_idx)
                row = {
                    "rep": rep,
                    "method": method,
                    "q": q,
                    "candidate_k": K,
                    "coverage_ratio": q / K,
                    "rep_eff_rank": rep_rank,
                    "selected_pairwise_novelty": float(np.mean(step_novelty)),
                    "mean_selected_loss": float(np.mean(step_selected_loss)),
                    "mean_candidate_loss": float(np.mean(step_candidate_loss)),
                    "probe_size": len(probe_idx),
                    "train_seconds": time.time() - st,
                }
                rows.append(row)
                torch.save(
                    {
                        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "rep": rep,
                        "method": method,
                        "q": q,
                    },
                    states_dir / f"rep{rep:03d}_q{q:02d}_{method}.pt",
                )
                print("TRAINING_ONLY " + json.dumps(row), flush=True)

        pd.DataFrame(rows).to_csv(
            output_dir / f"budget_coverage_diagnostics_{start}_{end}.csv", index=False
        )
        print(
            f"training rep {rep} completed; elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    pd.DataFrame(rows).to_csv(
        output_dir / f"budget_coverage_diagnostics_{start}_{end}.csv", index=False
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


def evaluate_shard(start: int, end: int, output_dir: Path) -> None:
    configure_determinism(1)
    _, _, xte, yte = load_digits_split()
    ey = torch.tensor(yte)

    # Fresh held-out environments are intentionally constructed only in this
    # evaluation phase, after every training-only state/diagnostic is sealed.
    heldout = [
        torch.tensor(geometric_environment(xte, int(seed))) for seed in HELDOUT_SEEDS
    ]
    clean = torch.tensor(xte).flatten(1)

    rows: list[dict] = []
    states_dir = output_dir / "states"
    for rep in range(start, end):
        for q in Q_VALUES:
            for method in ("loss_hard", "gradnov"):
                path = states_dir / f"rep{rep:03d}_q{q:02d}_{method}.pt"
                checkpoint = torch.load(path, map_location="cpu")
                model = MLP()
                model.load_state_dict(checkpoint["state_dict"])
                metrics = environment_metrics(model, heldout, ey, clean)
                row = {
                    "rep": rep,
                    "method": method,
                    "q": q,
                    "candidate_k": K,
                    "coverage_ratio": q / K,
                    **metrics,
                }
                rows.append(row)
                print("HELDOUT " + json.dumps(row), flush=True)

    pd.DataFrame(rows).to_csv(
        output_dir / f"budget_coverage_heldout_{start}_{end}.csv", index=False
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

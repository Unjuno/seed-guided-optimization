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
Q = 4
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-2
WEIGHT_DECAY = 1e-3
NOVELTY_WEIGHT = 0.6
TRAIN_SEEDS = np.arange(18000, 18064, dtype=int)
HELDOUT_SEEDS = np.arange(19000, 19080, dtype=int)
LAMBDA_GRID = (0.20, 0.35, 0.50, 0.65, 0.80, 0.95, 1.10)


def unstructured_environment(images: np.ndarray, seed: int, severity: float, offset: int) -> np.ndarray:
    """Independent per-example pixel nuisance with no shared geometric factor."""
    x = np.asarray(images, dtype=np.float32)
    rng = np.random.default_rng(seed * 7919 + offset)
    sigma = 0.28 * severity
    drop_p = min(0.28 * severity, 0.45)
    impulse_p = min(0.06 * severity, 0.12)
    noise = rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
    keep = rng.random(x.shape) >= drop_p
    salt = rng.random(x.shape) < impulse_p
    pepper = rng.random(x.shape) < impulse_p
    out = (x + noise) * keep
    out = np.where(salt, 1.0, out)
    out = np.where(pepper, 0.0, out)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def selected_pairwise_novelty(cosine: torch.Tensor, selected: list[int]) -> float:
    idx = torch.tensor(selected, dtype=torch.long)
    sub = cosine.index_select(0, idx).index_select(1, idx)
    upper = torch.triu_indices(len(selected), len(selected), offset=1)
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


def make_train_environments(xtr: np.ndarray, family: str, severity: float | None) -> list[torch.Tensor]:
    if family == "structured":
        return [torch.tensor(geometric_environment(xtr, int(seed))) for seed in TRAIN_SEEDS]
    if family == "unstructured":
        if severity is None:
            raise ValueError("unstructured family requires severity")
        return [
            torch.tensor(unstructured_environment(xtr, int(seed), severity, offset=31))
            for seed in TRAIN_SEEDS
        ]
    raise ValueError(f"unknown family {family}")


def train_one(
    train_environments: list[torch.Tensor],
    ty: torch.Tensor,
    schedule: list[tuple[torch.Tensor, list[int]]],
    base: int,
    method: str,
) -> tuple[MLP, dict]:
    seed_everything(base)
    model = MLP()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    step_novelty: list[float] = []
    step_selected_loss: list[float] = []
    step_candidate_loss: list[float] = []
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
            selected = select_loss_hard(losses, Q)
        elif method == "gradnov":
            selected = select_hard_gradient_novel(
                losses, cosine, q=Q, novelty_weight=NOVELTY_WEIGHT
            )
        else:
            raise ValueError(method)
        selected = sorted(selected)
        step_novelty.append(selected_pairwise_novelty(cosine, selected))
        step_selected_loss.append(float(losses[selected].mean().detach()))
        step_candidate_loss.append(float(losses.mean().detach()))
        opt.zero_grad(set_to_none=True)
        per[selected].mean().backward()
        opt.step()
    return model, {
        "selected_pairwise_novelty": float(np.mean(step_novelty)),
        "mean_selected_loss": float(np.mean(step_selected_loss)),
        "mean_candidate_loss": float(np.mean(step_candidate_loss)),
    }


def calibrate_shard(start: int, end: int, output_dir: Path) -> None:
    configure_determinism(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    xtr, ytr, _, _ = load_digits_split()
    ty = torch.tensor(ytr)
    structured_envs = make_train_environments(xtr, "structured", None)
    nuisance_envs = {
        severity: make_train_environments(xtr, "unstructured", severity)
        for severity in LAMBDA_GRID
    }
    rows: list[dict] = []
    started = time.time()
    for rep in range(start, end):
        base = 510_000_000 + rep * 4099
        schedule = build_schedule(len(ty), base)
        conditions: list[tuple[str, float | None, list[torch.Tensor]]] = [
            ("structured", None, structured_envs)
        ] + [
            ("unstructured", severity, nuisance_envs[severity])
            for severity in LAMBDA_GRID
        ]
        for family, severity, envs in conditions:
            for method in ("loss_hard", "gradnov"):
                _, diag = train_one(envs, ty, schedule, base, method)
                row = {
                    "rep": rep,
                    "family": family,
                    "severity": np.nan if severity is None else severity,
                    "method": method,
                    **diag,
                }
                rows.append(row)
                print("CALIBRATION " + json.dumps(row), flush=True)
        pd.DataFrame(rows).to_csv(
            output_dir / f"matched_novelty_calibration_{start}_{end}.csv", index=False
        )
        print(f"calibration rep {rep} done; elapsed={time.time()-started:.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(
        output_dir / f"matched_novelty_calibration_{start}_{end}.csv", index=False
    )
    print(json.dumps({"event": "CALIBRATION_TRAINING_ONLY_COMPLETE", "heldout_constructed": False}), flush=True)


def confirm_train_shard(start: int, end: int, output_dir: Path, severity: float) -> None:
    configure_determinism(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    states_dir = output_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    xtr, ytr, _, _ = load_digits_split()
    ty = torch.tensor(ytr)
    envs = {
        "structured": make_train_environments(xtr, "structured", None),
        "unstructured": make_train_environments(xtr, "unstructured", severity),
    }
    rows: list[dict] = []
    started = time.time()
    for rep in range(start, end):
        base = 610_000_000 + rep * 4099
        schedule = build_schedule(len(ty), base)
        for family in ("structured", "unstructured"):
            for method in ("loss_hard", "gradnov"):
                model, diag = train_one(envs[family], ty, schedule, base, method)
                row = {
                    "rep": rep,
                    "family": family,
                    "severity": np.nan if family == "structured" else severity,
                    "method": method,
                    **diag,
                }
                rows.append(row)
                torch.save(
                    {
                        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "rep": rep,
                        "family": family,
                        "method": method,
                        "severity": severity,
                    },
                    states_dir / f"rep{rep:03d}_{family}_{method}.pt",
                )
                print("CONFIRM_TRAINING_ONLY " + json.dumps(row), flush=True)
        pd.DataFrame(rows).to_csv(
            output_dir / f"matched_novelty_diagnostics_{start}_{end}.csv", index=False
        )
        print(f"confirm training rep {rep} done; elapsed={time.time()-started:.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(
        output_dir / f"matched_novelty_diagnostics_{start}_{end}.csv", index=False
    )
    print(json.dumps({
        "event": "CONFIRM_TRAINING_ONLY_COMPLETE",
        "start": start,
        "end": end,
        "sealed_severity": severity,
        "heldout_constructed": False,
    }), flush=True)


def confirm_evaluate_shard(start: int, end: int, output_dir: Path, severity: float) -> None:
    configure_determinism(1)
    _, _, xte, yte = load_digits_split()
    ey = torch.tensor(yte)
    heldout = {
        "structured": [
            torch.tensor(geometric_environment(xte, int(seed))) for seed in HELDOUT_SEEDS
        ],
        "unstructured": [
            torch.tensor(unstructured_environment(xte, int(seed), severity, offset=71))
            for seed in HELDOUT_SEEDS
        ],
    }
    clean = torch.tensor(xte).flatten(1)
    rows: list[dict] = []
    states_dir = output_dir / "states"
    for rep in range(start, end):
        for family in ("structured", "unstructured"):
            for method in ("loss_hard", "gradnov"):
                path = states_dir / f"rep{rep:03d}_{family}_{method}.pt"
                ck = torch.load(path, map_location="cpu")
                if float(ck["severity"]) != float(severity):
                    raise ValueError("sealed severity mismatch")
                model = MLP()
                model.load_state_dict(ck["state_dict"])
                metrics = environment_metrics(model, heldout[family], ey, clean)
                row = {
                    "rep": rep,
                    "family": family,
                    "severity": np.nan if family == "structured" else severity,
                    "method": method,
                    **metrics,
                }
                rows.append(row)
                print("CONFIRM_HELDOUT " + json.dumps(row), flush=True)
    pd.DataFrame(rows).to_csv(
        output_dir / f"matched_novelty_heldout_{start}_{end}.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("calibrate", "train", "evaluate"))
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--severity", type=float)
    args = parser.parse_args()
    out = Path(args.output_dir)
    if args.mode == "calibrate":
        calibrate_shard(args.start, args.end, out)
    elif args.mode == "train":
        if args.severity is None:
            raise ValueError("--severity required for confirmatory train")
        confirm_train_shard(args.start, args.end, out, args.severity)
    else:
        if args.severity is None:
            raise ValueError("--severity required for confirmatory evaluate")
        confirm_evaluate_shard(args.start, args.end, out, args.severity)


if __name__ == "__main__":
    main()

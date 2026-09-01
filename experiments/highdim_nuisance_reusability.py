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
TRAIN_SEEDS = np.arange(20000, 20064, dtype=int)
HELDOUT_SEEDS = np.arange(21000, 21080, dtype=int)
LAMBDA_GRID = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def highdim_nuisance_environment(images: np.ndarray, seed: int, severity: float) -> np.ndarray:
    """Environment-specific 64-D pixel permutation mixture shared across examples."""
    x = np.asarray(images, dtype=np.float32).reshape(len(images), 64)
    rng = np.random.default_rng(seed * 8191 + 97)
    perm = rng.permutation(64)
    out = (1.0 - severity) * x + severity * x[:, perm]
    return out.reshape(len(images), 8, 8).astype(np.float32)


def selected_pairwise_novelty(cosine: torch.Tensor, selected: list[int]) -> float:
    idx = torch.tensor(selected, dtype=torch.long)
    sub = cosine.index_select(0, idx).index_select(1, idx)
    upper = torch.triu_indices(len(selected), len(selected), offset=1)
    return float((1.0 - sub[upper[0], upper[1]]).mean())


def build_schedule(n_train: int, base: int) -> list[tuple[torch.Tensor, list[int]]]:
    tg = torch.Generator().manual_seed(base + 1)
    env_rng = np.random.default_rng(base + 2)
    schedule = []
    for _ in range(EPOCHS):
        for batch_idx in torch.randperm(n_train, generator=tg).split(BATCH_SIZE):
            candidates = env_rng.choice(len(TRAIN_SEEDS), K, replace=False).tolist()
            schedule.append((batch_idx, candidates))
    return schedule


def make_environments(images: np.ndarray, family: str, severity: float | None) -> list[torch.Tensor]:
    if family == "structured":
        return [torch.tensor(geometric_environment(images, int(seed))) for seed in TRAIN_SEEDS]
    if family == "highdim":
        if severity is None:
            raise ValueError("highdim family requires severity")
        return [
            torch.tensor(highdim_nuisance_environment(images, int(seed), severity))
            for seed in TRAIN_SEEDS
        ]
    raise ValueError(family)


def train_one(envs, labels, schedule, base: int, method: str):
    seed_everything(base)
    model = MLP()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    novelty = []
    selected_loss = []
    candidate_loss = []
    model.train()
    for batch_idx, candidates in schedule:
        xb = torch.cat([envs[e][batch_idx] for e in candidates])
        logits, h = model(xb)
        per = torch.nn.functional.cross_entropy(
            logits, labels[batch_idx].repeat(K), reduction="none"
        ).reshape(K, -1)
        losses = per.mean(1)
        directions = head_gradient_directions(logits, h, labels[batch_idx], K)
        cosine = directions @ directions.T
        if method == "loss_hard":
            selected = select_loss_hard(losses, Q)
        else:
            selected = select_hard_gradient_novel(
                losses, cosine, q=Q, novelty_weight=NOVELTY_WEIGHT
            )
        selected = sorted(selected)
        novelty.append(selected_pairwise_novelty(cosine, selected))
        selected_loss.append(float(losses[selected].mean().detach()))
        candidate_loss.append(float(losses.mean().detach()))
        opt.zero_grad(set_to_none=True)
        per[selected].mean().backward()
        opt.step()
    return model, {
        "selected_pairwise_novelty": float(np.mean(novelty)),
        "mean_selected_loss": float(np.mean(selected_loss)),
        "mean_candidate_loss": float(np.mean(candidate_loss)),
    }


def calibrate(start: int, end: int, outdir: Path) -> None:
    configure_determinism(1)
    outdir.mkdir(parents=True, exist_ok=True)
    xtr, ytr, _, _ = load_digits_split()
    labels = torch.tensor(ytr)
    structured = make_environments(xtr, "structured", None)
    highdim = {s: make_environments(xtr, "highdim", s) for s in LAMBDA_GRID}
    rows = []
    started = time.time()
    for rep in range(start, end):
        base = 710_000_000 + rep * 4099
        schedule = build_schedule(len(labels), base)
        conditions = [("structured", None, structured)] + [
            ("highdim", severity, highdim[severity]) for severity in LAMBDA_GRID
        ]
        for family, severity, envs in conditions:
            for method in ("loss_hard", "gradnov"):
                _, diag = train_one(envs, labels, schedule, base, method)
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
            outdir / f"highdim_calibration_{start}_{end}.csv", index=False
        )
        print(f"calibration rep {rep} done elapsed={time.time()-started:.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(
        outdir / f"highdim_calibration_{start}_{end}.csv", index=False
    )
    print(json.dumps({"event":"CALIBRATION_COMPLETE","heldout_constructed":False}), flush=True)


def confirm_train(start: int, end: int, outdir: Path, severity: float) -> None:
    configure_determinism(1)
    outdir.mkdir(parents=True, exist_ok=True)
    states = outdir / "states"
    states.mkdir(parents=True, exist_ok=True)
    xtr, ytr, _, _ = load_digits_split()
    labels = torch.tensor(ytr)
    envs = {
        "structured": make_environments(xtr, "structured", None),
        "highdim": make_environments(xtr, "highdim", severity),
    }
    rows = []
    for rep in range(start, end):
        base = 810_000_000 + rep * 4099
        schedule = build_schedule(len(labels), base)
        for family in ("structured", "highdim"):
            for method in ("loss_hard", "gradnov"):
                model, diag = train_one(envs[family], labels, schedule, base, method)
                row = {
                    "rep": rep,
                    "family": family,
                    "severity": np.nan if family == "structured" else severity,
                    "method": method,
                    **diag,
                }
                rows.append(row)
                torch.save({
                    "state_dict": {k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    "rep": rep,
                    "family": family,
                    "method": method,
                    "severity": severity,
                }, states / f"rep{rep:03d}_{family}_{method}.pt")
                print("CONFIRM_TRAINING_ONLY " + json.dumps(row), flush=True)
        pd.DataFrame(rows).to_csv(
            outdir / f"highdim_diagnostics_{start}_{end}.csv", index=False
        )
    pd.DataFrame(rows).to_csv(
        outdir / f"highdim_diagnostics_{start}_{end}.csv", index=False
    )
    print(json.dumps({
        "event":"CONFIRM_TRAINING_ONLY_COMPLETE",
        "sealed_severity":severity,
        "heldout_constructed":False,
    }), flush=True)


def confirm_evaluate(start: int, end: int, outdir: Path, severity: float) -> None:
    configure_determinism(1)
    _, _, xte, yte = load_digits_split()
    labels = torch.tensor(yte)
    heldout = {
        "structured": [torch.tensor(geometric_environment(xte, int(seed))) for seed in HELDOUT_SEEDS],
        "highdim": [
            torch.tensor(highdim_nuisance_environment(xte, int(seed), severity))
            for seed in HELDOUT_SEEDS
        ],
    }
    clean = torch.tensor(xte).flatten(1)
    states = outdir / "states"
    rows = []
    for rep in range(start, end):
        for family in ("structured", "highdim"):
            for method in ("loss_hard", "gradnov"):
                ck = torch.load(states / f"rep{rep:03d}_{family}_{method}.pt", map_location="cpu")
                if float(ck["severity"]) != float(severity):
                    raise ValueError("sealed severity mismatch")
                model = MLP()
                model.load_state_dict(ck["state_dict"])
                metrics = environment_metrics(model, heldout[family], labels, clean)
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
        outdir / f"highdim_heldout_{start}_{end}.csv", index=False
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("calibrate", "train", "evaluate"))
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--severity", type=float)
    a = ap.parse_args()
    out = Path(a.output_dir)
    if a.mode == "calibrate":
        calibrate(a.start, a.end, out)
    elif a.mode == "train":
        if a.severity is None:
            raise ValueError("--severity required")
        confirm_train(a.start, a.end, out, a.severity)
    else:
        if a.severity is None:
            raise ValueError("--severity required")
        confirm_evaluate(a.start, a.end, out, a.severity)


if __name__ == "__main__":
    main()

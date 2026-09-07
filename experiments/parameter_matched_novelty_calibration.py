"""Issue #83 Stage A: training-only matching-caliper calibration."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import cnn_regime_interaction as cr
import transfer_specificity as base
from common import SmallCNN, environment_parameters, head_gradient_directions, seed_everything

PILOT_REPS = tuple(range(1700, 1710))
K = 16
Q = 4
BATCH = 128
EPOCHS = 10
LR = 5e-3
WD = 1e-3
TRAIN_SEEDS = tuple(range(53000, 53064))
CALIPERS = (0.05, 0.10, 0.15, 0.20, 0.30)
MIN_GAP = 0.10
PROTOCOL = {
    "issue": 83,
    "stage": "A",
    "pilot_reps": PILOT_REPS,
    "K": K,
    "Q": Q,
    "batch": BATCH,
    "epochs": EPOCHS,
    "lr": LR,
    "wd": WD,
    "train_seeds": TRAIN_SEEDS,
    "calipers": CALIPERS,
    "minimum_mean_gradient_gap": MIN_GAP,
    "base_seed_offset": 2110000000,
    "base_seed_stride": 4099,
    "feasible_ranks": tuple(range(2, 11)),
    "parameter_coordinates": 7,
}
PROTOCOL_HASH = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def build_schedule(n: int, seed: int) -> list[tuple[torch.Tensor, list[int]]]:
    tg = torch.Generator().manual_seed(seed + 1)
    er = np.random.default_rng(seed + 2)
    out = []
    for _ in range(EPOCHS):
        for b in torch.randperm(n, generator=tg).split(BATCH):
            out.append((b, er.choice(len(TRAIN_SEEDS), K, replace=False).tolist()))
    return out


def pairnov(cos: torch.Tensor, sel: tuple[int, ...]) -> float:
    idx = torch.tensor(sel, dtype=torch.long)
    sub = cos.index_select(0, idx).index_select(1, idx)
    up = torch.triu_indices(len(sel), len(sel), offset=1)
    return float((1.0 - sub[up[0], up[1]]).mean())


def physical_diversity(env_indices: tuple[int, ...], params: np.ndarray) -> float:
    x = params[np.asarray(env_indices, dtype=int)]
    vals = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            vals.append(float(np.linalg.norm(x[i] - x[j])))
    return float(np.mean(vals))


def feasible_subsets(
    losses: torch.Tensor,
    cos: torch.Tensor,
    candidate_envs: list[int],
    params: np.ndarray,
) -> list[dict]:
    order = sorted(range(K), key=lambda i: (-float(losses[i].detach()), i))
    anchor = order[0]
    candidate_mean = float(losses.mean().detach())
    candidate_sd = float(losses.detach().double().std(unbiased=False))
    rows = []
    for others in itertools.combinations(order[1:10], 3):
        local = tuple(sorted((anchor, *others)))
        envs = tuple(int(candidate_envs[i]) for i in local)
        selected_loss = float(losses[list(local)].mean().detach())
        rows.append({
            "local": local,
            "envs": envs,
            "gradient_novelty": pairnov(cos, local),
            "z_hard": (selected_loss - candidate_mean) / (candidate_sd + 1e-8),
            "physical_diversity": physical_diversity(envs, params),
        })
    if len(rows) != 84:
        raise ValueError(f"expected 84 feasible subsets, got {len(rows)}")
    p = np.asarray([r["physical_diversity"] for r in rows], dtype=np.float64)
    psd = float(p.std(ddof=0))
    if not np.isfinite(psd) or psd <= 1e-12:
        raise ValueError("physical subset-diversity SD degenerate")
    pm = float(p.mean())
    for r in rows:
        r["z_param"] = (r["physical_diversity"] - pm) / psd
    return rows


def best_pair(rows: list[dict], caliper: float) -> dict | None:
    best = None
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            dh = abs(a["z_hard"] - b["z_hard"])
            dp = abs(a["z_param"] - b["z_param"])
            if dh > caliper + 1e-12 or dp > caliper + 1e-12:
                continue
            gap = abs(a["gradient_novelty"] - b["gradient_novelty"])
            # Maximize novelty gap; then tighter parameter match, tighter hardness match,
            # then lexicographic subset identifiers for deterministic ties.
            key = (
                -gap,
                dp,
                dh,
                tuple(a["local"]),
                tuple(b["local"]),
            )
            if best is None or key < best["key"]:
                high, low = (a, b) if a["gradient_novelty"] >= b["gradient_novelty"] else (b, a)
                best = {
                    "key": key,
                    "gradient_gap": float(high["gradient_novelty"] - low["gradient_novelty"]),
                    "hardness_diff": float(high["z_hard"] - low["z_hard"]),
                    "parameter_z_diff": float(high["z_param"] - low["z_param"]),
                    "parameter_abs_z_diff": float(abs(high["z_param"] - low["z_param"])),
                    "high_envs": ";".join(map(str, high["envs"])),
                    "low_envs": ";".join(map(str, low["envs"])),
                }
    return best


def check_range(start: int, end: int) -> None:
    if start >= end or not set(range(start, end)).issubset(PILOT_REPS):
        raise ValueError("unregistered pilot range")


def run_pilot(start: int, end: int, out: Path) -> None:
    check_range(start, end)
    base.configure_determinism(1)
    x, y, _, _, _, _, _, _ = cr.data_split()
    ty = torch.tensor(y)
    envs = base.geometric_envs(x, TRAIN_SEEDS)
    raw_params = np.stack([environment_parameters(int(s)) for s in TRAIN_SEEDS]).astype(np.float64)
    params = (raw_params - raw_params.mean(0, keepdims=True)) / (raw_params.std(0, ddof=0, keepdims=True) + 1e-12)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for rep in range(start, end):
        seed = 2110000000 + 4099 * rep
        sched = build_schedule(len(ty), seed)
        seed_everything(seed)
        model = SmallCNN()
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        model.train()
        for step, (b, cand) in enumerate(sched):
            xb = torch.cat([envs[e][b] for e in cand])
            logits, h = model(xb)
            per = torch.nn.functional.cross_entropy(
                logits, ty[b].repeat(K), reduction="none"
            ).reshape(K, -1)
            losses = per.mean(1)
            gd = head_gradient_directions(logits, h, ty[b], K)
            cos = gd @ gd.T
            subsets = feasible_subsets(losses, cos, cand, params)
            for c in CALIPERS:
                pair = best_pair(subsets, c)
                rows.append({
                    "rep": rep,
                    "step": step,
                    "caliper": c,
                    "feasible": pair is not None,
                    "gradient_gap": np.nan if pair is None else pair["gradient_gap"],
                    "hardness_diff": np.nan if pair is None else pair["hardness_diff"],
                    "parameter_z_diff": np.nan if pair is None else pair["parameter_z_diff"],
                    "parameter_abs_z_diff": np.nan if pair is None else pair["parameter_abs_z_diff"],
                })
            ref_order = sorted(range(K), key=lambda i: (-float(losses[i].detach()), i))[:Q]
            ref_order = sorted(ref_order)
            opt.zero_grad(set_to_none=True)
            per[ref_order].mean().backward()
            opt.step()
        print(f"PILOT_REFERENCE_COMPLETE rep={rep}; no arms, no heldout", flush=True)
    pd.DataFrame(rows).to_csv(out / f"parameter_match_calibration_{start}_{end}.csv.gz", index=False, compression="gzip")
    manifest = {
        "protocol": PROTOCOL,
        "protocol_hash": PROTOCOL_HASH,
        "start": start,
        "end": end,
        "split_hashes": cr.split_hashes(),
        "n_train": 988,
        "heldout_constructed": False,
        "arms_trained": False,
    }
    (out / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2))


def summarize(root: Path, out: Path) -> None:
    paths = sorted(root.rglob("parameter_match_calibration_*.csv.gz"))
    if not paths:
        raise FileNotFoundError("no calibration files")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    expected = {(r, step, c) for r in PILOT_REPS for step in range(80) for c in CALIPERS}
    observed = set(zip(df.rep.astype(int), df.step.astype(int), df.caliper.astype(float)))
    if len(df) != len(expected) or observed != expected or df.duplicated(["rep", "step", "caliper"]).any():
        raise ValueError("calibration grid mismatch")
    summaries = []
    selected = None
    for c in CALIPERS:
        d = df[np.isclose(df.caliper, c)]
        feasible = d.feasible.astype(bool)
        rate = float(feasible.mean())
        gaps = d.loc[feasible, "gradient_gap"].to_numpy(dtype=np.float64)
        mean_gap = float(gaps.mean()) if len(gaps) else float("nan")
        row = {
            "caliper": c,
            "n_steps": len(d),
            "feasible_steps": int(feasible.sum()),
            "feasibility_rate": rate,
            "mean_gradient_gap_feasible": mean_gap,
            "mean_abs_hardness_diff_feasible": float(d.loc[feasible, "hardness_diff"].abs().mean()) if feasible.any() else float("nan"),
            "mean_abs_parameter_z_diff_feasible": float(d.loc[feasible, "parameter_abs_z_diff"].mean()) if feasible.any() else float("nan"),
        }
        summaries.append(row)
        if selected is None and rate == 1.0 and mean_gap >= MIN_GAP:
            selected = c
    decision = "MATCHING CALIBRATION PASS" if selected is not None else "MATCHING CALIBRATION FAIL"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(out / "parameter_match_calibration_summary.csv", index=False)
    pd.DataFrame([{
        "decision": decision,
        "selected_caliper": selected,
        "minimum_mean_gradient_gap": MIN_GAP,
        "pilot_reps": len(PILOT_REPS),
        "pilot_steps": len(PILOT_REPS) * 80,
        "protocol_hash": PROTOCOL_HASH,
    }]).to_csv(out / "parameter_match_calibration_decision.csv", index=False)
    print(pd.DataFrame([{"decision": decision, "selected_caliper": selected}]).to_string(index=False), flush=True)


def selftest() -> None:
    base.configure_determinism(1)
    losses = torch.linspace(2.0, 0.5, K)
    g = torch.arange(K * 7, dtype=torch.float32).reshape(K, 7) + 1.0
    g = g / g.norm(dim=1, keepdim=True)
    cos = g @ g.T
    candidate_envs = list(range(K))
    params = np.arange(64 * 7, dtype=np.float64).reshape(64, 7)
    params = (params - params.mean(0)) / (params.std(0) + 1e-12)
    rows = feasible_subsets(losses, cos, candidate_envs, params)
    assert len(rows) == 84
    for c in CALIPERS:
        pair = best_pair(rows, c)
        if pair is not None:
            assert abs(pair["hardness_diff"]) <= c + 1e-12
            assert abs(pair["parameter_z_diff"]) <= c + 1e-12
            assert pair["gradient_gap"] >= -1e-12
    print("SELFTEST PASS", PROTOCOL_HASH)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("selftest", "pilot", "summarize"))
    ap.add_argument("--start", type=int)
    ap.add_argument("--end", type=int)
    ap.add_argument("--input-dir")
    ap.add_argument("--output-dir")
    a = ap.parse_args()
    if a.mode == "selftest":
        selftest()
    elif a.mode == "pilot":
        if a.start is None or a.end is None or a.output_dir is None:
            raise ValueError("pilot args required")
        run_pilot(a.start, a.end, Path(a.output_dir))
    else:
        if a.input_dir is None or a.output_dir is None:
            raise ValueError("summarize args required")
        summarize(Path(a.input_dir), Path(a.output_dir))


if __name__ == "__main__":
    main()
